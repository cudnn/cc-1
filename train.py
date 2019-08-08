
import argparse
import time
import csv
import datetime
import os

import  matplotlib.pyplot as plt

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import custom_transforms
import models
from utils import tensor2array, save_checkpoint, save_path_formatter
from inverse_warp import inverse_warp, pose2flow, flow_warp

from loss_functions import compute_joint_mask_for_depth
from loss_functions import consensus_exp_masks, consensus_depth_flow_mask
from loss_functions import\
    photometric_reconstruction_loss, \
    photometric_flow_loss,\
    explainability_loss, \
    gaussian_explainability_loss, \
    smooth_loss, \
    edge_aware_smoothness_loss

from loss_functions import compute_errors, compute_epe, compute_all_epes, flow_diff, spatial_normalize

from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter
from flowutils.flowlib import flow_to_image



def train(train_loader, disp_net, pose_net, mask_net, flow_net, optimizer,  logger=None, train_writer=None,global_vars_dict=None):
# 0. 准备
    args=global_vars_dict['args']
    n_iter = global_vars_dict['n_iter']
    device = global_vars_dict['device']

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3, w4 = args.cam_photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight, args.flow_photo_loss_weight
    w5 = args.consensus_loss_weight

    if args.robust:
        loss_camera = photometric_reconstruction_loss_robust
        loss_flow = photometric_flow_loss_robust
    else:
        loss_camera = photometric_reconstruction_loss
        loss_flow = photometric_flow_loss
#2. switch to train mode
    disp_net.train()
    pose_net.train()
    mask_net.train()
    flow_net.train()

    end = time.time()
#3. train cycle
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.to(device))
        ref_imgs_var = [Variable(img.to(device)) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.to(device))
        intrinsics_inv_var = Variable(intrinsics_inv.to(device))



    #3.1 compute output and lossfunc input valve---------------------

        #1. disp->depth(none)
        disparities = disp_net(tgt_img_var)
        if args.spatial_normalize:
            disparities = [spatial_normalize(disp) for disp in disparities]

        depth = [1/disp for disp in disparities]

        #2. pose(none)
        pose = pose_net(tgt_img_var, ref_imgs_var)
        #pose:[4,4,6]


        #3.flow_fwd,flow_bwd 全光流 (depth, pose)
        # 自己改了一点
        if args.flownet== 'Back2Future':#临近一共三帧做训练/推断
            flow_fwd, flow_bwd, _ = flow_net(tgt_img_var, ref_imgs_var[1:3])
        elif args.flownet == 'FlowNetC6':
            flow_fwd = flow_net(tgt_img_var, ref_imgs_var[2])
            flow_bwd = flow_net(tgt_img_var, ref_imgs_var[1])
        elif args.flownet == 'FlowNetS':
            print(' ')

        # flow_cam 即背景光流
        # flow - flow_s = flow_o
        flow_cam = pose2flow(depth[0].squeeze(), pose[:, 2], intrinsics_var,
                             intrinsics_inv_var)  # pose[:,2] belongs to forward frame
        flows_cam_fwd = [pose2flow(depth_.squeeze(1), pose[:, 2], intrinsics_var, intrinsics_inv_var) for depth_ in
                         depth]
        flows_cam_bwd = [pose2flow(depth_.squeeze(1), pose[:, 1], intrinsics_var, intrinsics_inv_var) for depth_ in
                         depth]



        exp_masks_target = consensus_exp_masks(flows_cam_fwd, flows_cam_bwd, flow_fwd, flow_bwd, tgt_img_var,
                                               ref_imgs_var[2], ref_imgs_var[1], wssim=args.wssim, wrig=args.wrig,
                                               ws=args.smooth_loss_weight)
        rigidity_mask_fwd = [(flows_cam_fwd_i - flow_fwd_i).abs() for flows_cam_fwd_i, flow_fwd_i in zip(flows_cam_fwd, flow_fwd)]  # .normalize()
        rigidity_mask_bwd = [(flows_cam_bwd_i - flow_bwd_i).abs() for flows_cam_bwd_i, flow_bwd_i in zip(flows_cam_bwd, flow_bwd)]  # .normalize()


        # 4.explainability_mask(none)
        explainability_mask = mask_net(tgt_img_var, ref_imgs_var)#有效区域?4??
        #list(5):item:tensor:[4,4,128,512]...[4,4,4,16] value:[0.33~0.48~0.63]
        #-------------------------------------------------


        if args.joint_mask_for_depth:
            explainability_mask_for_depth = compute_joint_mask_for_depth(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd)
        else:
            explainability_mask_for_depth = explainability_mask
        #explainability_mask_for_depth list(5) [b,2,h/ , w/]
        if args.no_non_rigid_mask:
            flow_exp_mask = [None for exp_mask in explainability_mask]
            if args.DEBUG:
                print('Using no masks for flow')
        else:
            flow_exp_mask = [1 - exp_mask[:,1:3] for exp_mask in explainability_mask]
            #list(4) [4,2,256,512]

    #3.2. compute loss重

        # minimizes the photometric loss on static scene
        loss_1 = loss_camera(tgt_img_var, ref_imgs_var, intrinsics_var, intrinsics_inv_var,
                                      depth, explainability_mask_for_depth, pose, lambda_oob=args.lambda_oob, qch=args.qch, wssim=args.wssim)
        # E_M
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask) #+ 0.2*gaussian_explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        # E_S
        if args.smoothness_type == "regular":
            loss_3 = smooth_loss(depth) + smooth_loss(flow_fwd) + smooth_loss(flow_bwd) + smooth_loss(explainability_mask)
        elif args.smoothness_type == "edgeaware":
            loss_3 = edge_aware_smoothness_loss(tgt_img_var, depth) + edge_aware_smoothness_loss(tgt_img_var, flow_fwd)
            loss_3 += edge_aware_smoothness_loss(tgt_img_var, flow_bwd) + edge_aware_smoothness_loss(tgt_img_var, explainability_mask)

        # E_F
        # minimizes photometric loss on moving regions
        loss_4 = loss_flow(tgt_img_var, ref_imgs_var[1:3], [flow_bwd, flow_fwd], flow_exp_mask,
                                        lambda_oob=args.lambda_oob, qch=args.qch, wssim=args.wssim)

        # E_C
        # drives the collaboration
        #explainagy_mask:list(6) of [4,4,4,16] rigidity_mask :list(4):[4,2,128,512]
        loss_5 = consensus_depth_flow_mask(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd,
                                        exp_masks_target, exp_masks_target, THRESH=args.THRESH, wbce=args.wbce)

        #3.2.6
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4 + w5*loss_5
        #end of loss


    #3.3
        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



    #3.4 log data

        # add scalar
        if args.scalar_freq > 0 and n_iter % args.scalar_freq == 0:
            train_writer.add_scalar('batch/cam_photometric_error', loss_1.item(), n_iter)
            if w2 > 0:
                train_writer.add_scalar('batch/explanability_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('batch/disparity_smoothness_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('batch/flow_photometric_error', loss_4.item(), n_iter)
            train_writer.add_scalar('batch/consensus_error', loss_5.item(), n_iter)
            train_writer.add_scalar('batch/total_loss', loss.item(), n_iter)

        # add_image为0 则不输出
        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

            train_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)
            train_writer.add_image('train Cam Flow Output',
                                   flow_to_image(tensor2array(flow_cam.data[0].cpu())), n_iter)

            for k, scaled_depth in enumerate(depth):
                train_writer.add_image('train Dispnet Output Normalized111 {}'.format(k),
                                       tensor2array(disparities[k].data[0].cpu(), max_value=None, colormap='bone'),
                                       n_iter)
                train_writer.add_image('train Depth Output {}'.format(k),
                                       tensor2array(1 / disparities[k].data[0].cpu(), max_value=10),
                                       n_iter)
                train_writer.add_image('train Non Rigid Flow Output {}'.format(k),
                                       flow_to_image(tensor2array(flow_fwd[k].data[0].cpu())), n_iter)
                train_writer.add_image('train Target Rigidity {}'.format(k),
                                       tensor2array(
                                           (rigidity_mask_fwd[k] > args.THRESH).type_as(rigidity_mask_fwd[k]).data[
                                               0].cpu(), max_value=1, colormap='bone'), n_iter)

                b, _, h, w = scaled_depth.size()
                downscale = tgt_img_var.size(2) / h

                tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img_var, (h, w))
                ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs_var]

                intrinsics_scaled = torch.cat((intrinsics_var[:, 0:2] / downscale, intrinsics_var[:, 2:]), dim=1)
                intrinsics_scaled_inv = torch.cat(
                    (intrinsics_inv_var[:, :, 0:2] * downscale, intrinsics_inv_var[:, :, 2:]), dim=2)

                train_writer.add_image('train Non Rigid Warped Image {}'.format(k),
                                       tensor2array(flow_warp(ref_imgs_scaled[2], flow_fwd[k]).data[0].cpu()), n_iter)

                # log warped images along with explainability mask
                for j, ref in enumerate(ref_imgs_scaled):
                    ref_warped = inverse_warp(ref, scaled_depth[:, 0], pose[:, j],
                                              intrinsics_scaled, intrinsics_scaled_inv,
                                              rotation_mode=args.rotation_mode,
                                              padding_mode=args.padding_mode)[0]
                    train_writer.add_image('train Warped Outputs {} {}'.format(k, j),
                                           tensor2array(ref_warped.data.cpu()), n_iter)
                    train_writer.add_image('train Diff Outputs {} {}'.format(k, j),
                                           tensor2array(0.5 * (tgt_img_scaled[0] - ref_warped).abs().data.cpu()),
                                           n_iter)
                    if explainability_mask[k] is not None:
                        train_writer.add_image('train Exp mask Outputs {} {}'.format(k, j),
                                               tensor2array(explainability_mask[k][0, j].data.cpu(), max_value=1,
                                                            colormap='bone'), n_iter)


        # csv file write
        with open(args.save_path / args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(
                [loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item(), loss_4.item()])
        #terminal output
        if args.log_terminal:
            logger.train_bar.update(i+1)#当前epoch 进度
            if i % args.print_freq == 0:
                logger.valid_bar_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))




    # 3.4 edge conditionsssssssssssssssssssssssss
        epoch_size = len(train_loader)
        if i >= epoch_size - 1:
            break

        n_iter += 1

    global_vars_dict['n_iter']=n_iter
    return losses.avg[0]#epoch loss
