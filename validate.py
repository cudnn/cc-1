

import argparse
import time
import csv
import datetime
import os
import copy
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
from inverse_warp import inverse_warp, pose2flow, flow2oob, flow_warp

from process_functions import compute_joint_mask_for_depth,consensus_exp_masks


from loss_functions import\
    photometric_reconstruction_loss, \
    photometric_flow_loss,\
    explainability_loss, \
    gaussian_explainability_loss, \
    smooth_loss, \
    edge_aware_smoothness_loss,\
    consensus_depth_flow_mask,\
    HistgramLoss


from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter
from flowutils.flowlib import flow_to_image


from utils import flow2rgb,compute_all_epes,flow_diff,spatial_normalize,compute_errors2
@torch.no_grad()
def validate_without_gt(val_loader,disp_net,pose_net,mask_net, flow_net, epoch, logger, tb_writer,nb_writers,global_vars_dict = None):
#data prepared
    device = global_vars_dict['device']
    n_iter_val = global_vars_dict['n_iter_val']
    args = global_vars_dict['args']
    show_samples = copy.deepcopy(args.show_samples)
    for i in range(len(show_samples)):
        show_samples[i]*=len(val_loader)
        show_samples[i] = show_samples[i]//1


    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_outputs = nb_writers > 0
    losses = AverageMeter(precision=4)


    w1, w2, w3, w4 = args.cam_photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight, args.flow_photo_loss_weight
    w5 = args.consensus_loss_weight

    loss_camera = photometric_reconstruction_loss
    loss_flow = photometric_flow_loss

# to eval model
    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    end = time.time()
    poses = np.zeros(((len(val_loader)-1) * 1 * (args.sequence_length-1),6))#init

#3. validation cycle
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics,intrinsics_inv = intrinsics.to(device),intrinsics_inv.to(device)
    #3.1 forwardpass
        #disp
        disp = disp_net(tgt_img)
        if args.spatial_normalize:
            disp = spatial_normalize(disp)
        depth = 1 / disp


        #pose
        pose = pose_net(tgt_img, ref_imgs)#[b,3,h,w]; list






        #flow----
        #制作前后一帧的
        if args.flownet == 'Back2Future':
            flow_fwd, flow_bwd, _ = flow_net(tgt_img, ref_imgs[1:3])
        elif args.flownet == 'FlowNetC6':
            flow_fwd = flow_net(tgt_img, ref_imgs[2])
            flow_bwd = flow_net(tgt_img, ref_imgs[1])
        flow_cam = pose2flow(depth.squeeze(1), pose[:, 2], intrinsics, intrinsics_inv)

        flows_cam_fwd = pose2flow(depth.squeeze(1), pose[:, 2], intrinsics, intrinsics_inv)
        flows_cam_bwd = pose2flow(depth.squeeze(1), pose[:, 1], intrinsics, intrinsics_inv)



        exp_masks_target = consensus_exp_masks(flows_cam_fwd, flows_cam_bwd, flow_fwd, flow_bwd, tgt_img,
                                               ref_imgs[2], ref_imgs[1], wssim=args.wssim, wrig=args.wrig,
                                               ws=args.smooth_loss_weight)
        no_rigid_flow = flow_fwd - flows_cam_fwd

        rigidity_mask_fwd = (flows_cam_fwd - flow_fwd).abs()#[b,2,h,w]
        rigidity_mask_bwd = (flows_cam_bwd - flow_bwd).abs()

        # mask
        # 4.explainability_mask(none)
        explainability_mask = mask_net(tgt_img, ref_imgs)  # 有效区域?4??

        # list(5):item:tensor:[4,4,128,512]...[4,4,4,16] value:[0.33~0.48~0.63]

        if args.joint_mask_for_depth:  # false
            explainability_mask_for_depth = explainability_mask

            #explainability_mask_for_depth = compute_joint_mask_for_depth(explainability_mask, rigidity_mask_bwd,
             #                                                            rigidity_mask_fwd,THRESH=args.THRESH)
        else:
            explainability_mask_for_depth = explainability_mask

        # chage

        if args.no_non_rigid_mask:
            flow_exp_mask = None
            if args.DEBUG:
                print('Using no masks for flow')
        else:
            flow_exp_mask = 1 - explainability_mask[:, 1:3]


        #3.2loss-compute
        if w1 >0:
            loss_1 = loss_camera(tgt_img, ref_imgs, intrinsics, intrinsics_inv,
                             depth, explainability_mask_for_depth, pose, lambda_oob=args.lambda_oob, qch=args.qch,
                             wssim=args.wssim)
        else:
            loss_1 = torch.tensor([0.]).to(device)

        # E_M
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)  # + 0.2*gaussian_explainability_loss(explainability_mask)
        else:
            loss_2 = 0

        #if args.smoothness_type == "regular":
        if w3>0:
            loss_3 = smooth_loss(depth) + smooth_loss(explainability_mask)+smooth_loss(flow_fwd) + smooth_loss(flow_bwd)
        else:
            loss_3 = torch.tensor([0.]).to(device)
        if w4>0:
            loss_4 = loss_flow(tgt_img, ref_imgs[1:3], [flow_bwd, flow_fwd], flow_exp_mask,
                           lambda_oob=args.lambda_oob, qch=args.qch, wssim=args.wssim)
        else:
            loss_4 = torch.tensor([0.]).to(device)
        if w5>0:
            loss_5 = consensus_depth_flow_mask(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd,
                                           exp_masks_target, exp_masks_target, THRESH=args.THRESH, wbce=args.wbce)
        else:
            loss_5 = torch.tensor([0.]).to(device)


        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3+ w4 * loss_4 + w5 * loss_5


    #3.3 data update
        losses.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()


    #3.4 check log

        #查看forward pass效果
        if args.img_freq >0 and i in show_samples:#output_writers list(3)
            if epoch == 0:#训练前的validate,目的在于先评估下网络效果
                #1.img
                # 不会执行第二次,注意ref_imgs axis0是batch的索引; axis 1是list(adjacent frame)的索引!
                tb_writer.add_image('epoch 0 Input/sample{}(img{} to img{})'.format(i,i+1,i+args.sequence_length), tensor2array(ref_imgs[0][0]), 0)
                tb_writer.add_image('epoch 0 Input/sample{}(img{} to img{})'.format(i,i+1,i+args.sequence_length), tensor2array(ref_imgs[1][0]), 1)
                tb_writer.add_image('epoch 0 Input/sample{}(img{} to img{})'.format(i,i+1,i+args.sequence_length), tensor2array(tgt_img[0]), 2)
                tb_writer.add_image('epoch 0 Input/sample{}(img{} to img{})'.format(i,i+1,i+args.sequence_length), tensor2array(ref_imgs[2][0]), 3)
                tb_writer.add_image('epoch 0 Input/sample{}(img{} to img{})'.format(i,i+1,i+args.sequence_length), tensor2array(ref_imgs[3][0]), 4)

                depth_to_show = depth[0].cpu()  # tensor disp_to_show :[1,h,w],0.5~3.1~10
                tb_writer.add_image('Disp Output/sample{}'.format(i),
                                    tensor2array(depth_to_show, max_value=None, colormap='bone'), 0)


            else:
            #2.disp
                depth_to_show = disp[0].cpu()# tensor disp_to_show :[1,h,w],0.5~3.1~10
                tb_writer.add_image('Disp Output/sample{}'.format(i), tensor2array(depth_to_show, max_value=None,colormap='bone'), epoch)
            #3. flow
                tb_writer.add_image('Flow/Flow Output sample {}'.format(i), flow2rgb(flow_fwd[0], max_value=6),epoch)
                tb_writer.add_image('Flow/cam_Flow Output sample {}'.format(i), flow2rgb(flow_cam[0], max_value=6),epoch)
                tb_writer.add_image('Flow/no rigid flow Output sample {}'.format(i), flow2rgb(no_rigid_flow[0], max_value=6),epoch)
                tb_writer.add_image('Flow/rigidity_mask_fwd{}'.format(i),flow2rgb(rigidity_mask_fwd[0],max_value=6),epoch)

            #4. mask
                tb_writer.add_image('Mask Output/mask0 sample{}'.format(i),tensor2array(explainability_mask[0][0], max_value=None, colormap='magma'), epoch)
                #tb_writer.add_image('Mask Output/mask1 sample{}'.format(i),tensor2array(explainability_mask[1][0], max_value=None, colormap='magma'), epoch)
                #tb_writer.add_image('Mask Output/mask2 sample{}'.format(i),tensor2array(explainability_mask[2][0], max_value=None, colormap='magma'), epoch)
                #tb_writer.add_image('Mask Output/mask3 sample{}'.format(i),tensor2array(explainability_mask[3][0], max_value=None, colormap='magma'), epoch)
                tb_writer.add_image('Mask Output/exp_masks_target sample{}'.format(i),
                                tensor2array(exp_masks_target[0][0], max_value=None, colormap='magma'), epoch)
                #tb_writer.add_image('Mask Output/mask0 sample{}'.format(i),
                #            tensor2array(explainability_mask[0][0], max_value=None, colormap='magma'), epoch)

        #


            #output_writers[index].add_image('val Depth Output', tensor2array(depth.data[0].cpu(), max_value=10),
             #                               epoch)

        # errors.update(compute_errors(depth, output_depth.data.squeeze(1)))
        # add scalar
        if args.scalar_freq > 0 and n_iter_val % args.scalar_freq == 0:
            tb_writer.add_scalar('val/E_R', loss_1.item(), n_iter_val)
            if w2 > 0:
                tb_writer.add_scalar('val/E_M', loss_2.item(), n_iter_val)
            tb_writer.add_scalar('val/E_S', loss_3.item(), n_iter_val)
            tb_writer.add_scalar('val/E_F', loss_4.item(), n_iter_val)
            tb_writer.add_scalar('val/E_C', loss_5.item(), n_iter_val)
            tb_writer.add_scalar('val/total_loss', loss.item(), n_iter_val)

        # terminal output
        if args.log_terminal:
            logger.valid_bar.update(i + 1)  # 当前epoch 进度
            if i % args.print_freq == 0:
                logger.valid_bar_writer.write('Valid: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))

        n_iter_val+=1


    global_vars_dict['n_iter_val'] = n_iter_val
    return losses.avg[0]#epoch validate loss






@torch.no_grad()
def validate_depth_with_gt(val_loader, disp_net, epoch, logger, tb_writer,global_vars_dict = None):
    device = global_vars_dict['device']
    args = global_vars_dict['args']
    n_iter_val_depth = global_vars_dict['n_iter_val_depth']

    show_samples = copy.deepcopy(args.show_samples)
    for i in range(len(show_samples)):
        show_samples[i] *= len(val_loader)
        show_samples[i] = show_samples[i] // 1


    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','epe']
    errors = AverageMeter(i=len(error_names))

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    fig = plt.figure(1, figsize=(8, 6))

    for i, (tgt_img, depth_gt) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)#BCHW
        depth_gt = depth_gt.to(device)

        output_disp = disp_net(tgt_img)#BCHW
        if args.spatial_normalize:
            output_disp = spatial_normalize(output_disp)

        output_depth = 1/output_disp





        errors.update(compute_errors(depth_gt.data.squeeze(1), output_depth.data.squeeze(1)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        fig = plt.figure(1,figsize=(8,6))
        if args.img_freq >0 and i in show_samples:#output_writers list(3)
            if epoch == 0:#训练前的validate,目的在于先评估下网络效果
                #1.img
                # 不会执行第二次,注意ref_imgs axis0是batch的索引; axis 1是list(adjacent frame)的索引!
                tb_writer.add_image('epoch 0 Input/sample{}'.format(i), tensor2array(tgt_img[0]), 0)
                tb_writer.add_image('epoch 0 depth_gt/sample{}'.format(i), tensor2array(depth_gt[0],colormap='bone'), 0)
                tb_writer.add_image('Depth Output/sample{}'.format(i), tensor2array(output_depth[0], max_value=None,colormap='bone'), 0)

                plt.hist(tensor2array(depth_gt[0],colormap='bone').flatten()*256,256,[0,256],color='r')
                tb_writer.add_figure(tag='histogram_gt/sample{}'.format(i), figure=fig, global_step=0)


            else:
            #2.disp
                # tensor disp_to_show :[1,h,w],0.5~3.1~10
                #disp2show = tensor2array(output_disp[0], max_value=None,colormap='bone')
                depth2show = tensor2array(output_depth[0], max_value=None, colormap='bone')
                #tb_writer.add_image('Disp Output/sample{}'.format(i), disp2show, epoch)
                tb_writer.add_image('Depth Output/sample{}'.format(i),depth2show, epoch)
                #add_figure

                plt.hist(depth2show.flatten()*256, 256, [0, 256], color='r')
                tb_writer.add_figure(tag = 'histogram_sample/sample{}'.format(i),figure=fig,global_step=epoch)

        # add scalar
        if args.scalar_freq > 0 and n_iter_val_depth % args.scalar_freq == 0:
            pass
            h_loss =HistgramLoss()(tgt_img,depth_gt)
            tb_writer.add_scalar('batch/val_h_loss' ,h_loss, n_iter_val_depth)
            #tb_writer.add_scalar('batch/' + error_names[1], errors.val[1], n_iter_val_depth)
            #tb_writer.add_scalar('batch/' + error_names[2], errors.val[2], n_iter_val_depth)
            #tb_writer.add_scalar('batch/' + error_names[3], errors.val[3], n_iter_val_depth)
            #tb_writer.add_scalar('batch/' + error_names[4], errors.val[4], n_iter_val_depth)
            #tb_writer.add_scalar('batch/' + error_names[5], errors.val[5], n_iter_val_depth)

        if args.log_terminal:
            logger.valid_bar.update(i)
            if i % args.print_freq == 0:
                logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))

        n_iter_val_depth += 1
        #end for
    #if args.log_terminal:
    #    logger.valid_bar.update(len(val_loader))

    global_vars_dict['n_iter_val_depth'] = n_iter_val_depth

    return errors.avg, error_names


def validate_flow_with_gt(val_loader, disp_net, pose_net, mask_net, flow_net, epoch, logger, output_writers=[]):
    global args
    batch_time = AverageMeter()
    error_names = ['epe_total', 'epe_rigid', 'epe_non_rigid', 'outliers', 'epe_total_with_gt_mask', 'epe_rigid_with_gt_mask', 'epe_non_rigid_with_gt_mask', 'outliers_gt_mask']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    end = time.time()

    poses = np.zeros(((len(val_loader)-1) * 1 * (args.sequence_length-1),6))

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt) in enumerate(val_loader):
        tgt_img = Variable(tgt_img.cuda(), volatile=True)
        ref_imgs = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda(), volatile=True)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), volatile=True)

        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)
        obj_map_gt_var = Variable(obj_map_gt.cuda(), volatile=True)

        # compute output-------------------------

        #1. disp fwd
        disp = disp_net(tgt_img)
        if args.spatial_normalize:
            disp = spatial_normalize(disp)

        depth = 1/disp

        #2. pose fwd
        pose = pose_net(tgt_img, ref_imgs)

        #3. mask fwd
        explainability_mask = mask_net(tgt_img, ref_imgs)

        #4. flow fwd
        if args.flownet == 'Back2Future':
            flow_fwd, flow_bwd, _ = flow_net(tgt_img, ref_imgs[1:3])#前一帧，后一阵
        elif args.flownet == 'FlowNetC6':
            flow_fwd = flow_net(tgt_img, ref_imgs[2])
            flow_bwd = flow_net(tgt_img, ref_imgs[1])
        # compute output-------------------------

        if args.DEBUG:
            flow_fwd_x = flow_fwd[:,0].view(-1).abs().data
            print("Flow Fwd Median: ", flow_fwd_x.median())
            flow_gt_var_x = flow_gt_var[:,0].view(-1).abs().data
            print("Flow GT Median: ", flow_gt_var_x.index_select(0, flow_gt_var_x.nonzero().view(-1)).median())

        flow_cam = pose2flow(depth.squeeze(1), pose[:,2], intrinsics_var, intrinsics_inv_var)
        oob_rigid = flow2oob(flow_cam)
        oob_non_rigid = flow2oob(flow_fwd)

        rigidity_mask = 1 - (1-explainability_mask[:,1])*(1-explainability_mask[:,2]).unsqueeze(1) > 0.5

        rigidity_mask_census_soft = (flow_cam - flow_fwd).abs()#.normalize()
        rigidity_mask_census_u = rigidity_mask_census_soft[:,0] < args.THRESH
        rigidity_mask_census_v = rigidity_mask_census_soft[:,1] < args.THRESH
        rigidity_mask_census = (rigidity_mask_census_u).type_as(flow_fwd) * (rigidity_mask_census_v).type_as(flow_fwd)

        rigidity_mask_combined = 1 - (1-rigidity_mask.type_as(explainability_mask))*(1-rigidity_mask_census.type_as(explainability_mask))

        #get flow
        flow_fwd_non_rigid = (rigidity_mask_combined<=args.THRESH).type_as(flow_fwd).expand_as(flow_fwd) * flow_fwd
        flow_fwd_rigid = (rigidity_mask_combined>args.THRESH).type_as(flow_fwd).expand_as(flow_fwd) * flow_cam
        total_flow = flow_fwd_rigid + flow_fwd_non_rigid

        obj_map_gt_var_expanded = obj_map_gt_var.unsqueeze(1).type_as(flow_fwd)

        if log_outputs and i % 10 == 0 and i/10 < len(output_writers):
            index = int(i//10)
            if epoch == 0:
                output_writers[index].add_image('val flow Input', tensor2array(tgt_img[0]), 0)
                flow_to_show = flow_gt[0][:2,:,:].cpu()
                output_writers[index].add_image('val target Flow', flow_to_image(tensor2array(flow_to_show)), epoch)

            output_writers[index].add_image('val Total Flow Output', flow_to_image(tensor2array(total_flow.data[0].cpu())), epoch)
            output_writers[index].add_image('val Rigid Flow Output', flow_to_image(tensor2array(flow_fwd_rigid.data[0].cpu())), epoch)
            output_writers[index].add_image('val Non-rigid Flow Output', flow_to_image(tensor2array(flow_fwd_non_rigid.data[0].cpu())), epoch)
            output_writers[index].add_image('val Out of Bound (Rigid)', tensor2array(oob_rigid.type(torch.FloatTensor).data[0].cpu(), max_value=1, colormap='bone'), epoch)
            output_writers[index].add_scalar('val Mean oob (Rigid)', oob_rigid.type(torch.FloatTensor).sum(), epoch)
            output_writers[index].add_image('val Out of Bound (Non-Rigid)', tensor2array(oob_non_rigid.type(torch.FloatTensor).data[0].cpu(), max_value=1, colormap='bone'), epoch)
            output_writers[index].add_scalar('val Mean oob (Non-Rigid)', oob_non_rigid.type(torch.FloatTensor).sum(), epoch)
            output_writers[index].add_image('val Cam Flow Errors', tensor2array(flow_diff(flow_gt_var, flow_cam).data[0].cpu()), epoch)
            output_writers[index].add_image('val Rigidity Mask', tensor2array(rigidity_mask.data[0].cpu(), max_value=1, colormap='bone'), epoch)
            output_writers[index].add_image('val Rigidity Mask Census', tensor2array(rigidity_mask_census.data[0].cpu(), max_value=1, colormap='bone'), epoch)

            for j,ref in enumerate(ref_imgs):
                ref_warped = inverse_warp(ref[:1], depth[:1,0], pose[:1,j],
                                          intrinsics_var[:1], intrinsics_inv_var[:1],
                                          rotation_mode=args.rotation_mode,
                                          padding_mode=args.padding_mode)[0]

                output_writers[index].add_image('val Warped Outputs {}'.format(j), tensor2array(ref_warped.data.cpu()), epoch)
                output_writers[index].add_image('val Diff Outputs {}'.format(j), tensor2array(0.5*(tgt_img[0] - ref_warped).abs().data.cpu()), epoch)
                if explainability_mask is not None:
                    output_writers[index].add_image('val Exp mask Outputs {}'.format(j), tensor2array(explainability_mask[0,j].data.cpu(), max_value=1, colormap='bone'), epoch)

            if args.DEBUG:
                # Check if pose2flow is consistant with inverse warp
                ref_warped_from_depth = inverse_warp(ref_imgs[2][:1], depth[:1,0], pose[:1,2],
                            intrinsics_var[:1], intrinsics_inv_var[:1], rotation_mode=args.rotation_mode,
                            padding_mode=args.padding_mode)[0]
                ref_warped_from_cam_flow = flow_warp(ref_imgs[2][:1], flow_cam)[0]
                print("DEBUG_INFO: Inverse_warp vs pose2flow",torch.mean(torch.abs(ref_warped_from_depth-ref_warped_from_cam_flow)).item())
                output_writers[index].add_image('val Warped Outputs from Cam Flow', tensor2array(ref_warped_from_cam_flow.data.cpu()), epoch)
                output_writers[index].add_image('val Warped Outputs from inverse warp', tensor2array(ref_warped_from_depth.data.cpu()), epoch)

        if log_outputs and i < len(val_loader)-1:
            step = args.sequence_length-1
            poses[i * step:(i+1) * step] = pose.data.cpu().view(-1,6).numpy()


        if np.isnan(flow_gt.sum().item()) or np.isnan(total_flow.data.sum().item()):
            print('NaN encountered')
        #
        _epe_errors = compute_all_epes(flow_gt_var, flow_cam, flow_fwd, rigidity_mask_combined) + compute_all_epes(flow_gt_var, flow_cam, flow_fwd, (1-obj_map_gt_var_expanded) )
        errors.update(_epe_errors)

        if args.DEBUG:
            print("DEBUG_INFO: EPE errors: ", _epe_errors )
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if log_outputs:
        output_writers[0].add_histogram('val poses_tx', poses[:,0], epoch)
        output_writers[0].add_histogram('val poses_ty', poses[:,1], epoch)
        output_writers[0].add_histogram('val poses_tz', poses[:,2], epoch)
        if args.rotation_mode == 'euler':
            rot_coeffs = ['rx', 'ry', 'rz']
        elif args.rotation_mode == 'quat':
            rot_coeffs = ['qx', 'qy', 'qz']
        output_writers[0].add_histogram('val poses_{}'.format(rot_coeffs[0]), poses[:,3], epoch)
        output_writers[0].add_histogram('val poses_{}'.format(rot_coeffs[1]), poses[:,4], epoch)
        output_writers[0].add_histogram('val poses_{}'.format(rot_coeffs[2]), poses[:,5], epoch)

    if args.DEBUG:
        print("DEBUG_INFO =================>")
        print("DEBUG_INFO: Average EPE : ", errors.avg )
        print("DEBUG_INFO =================>")
        print("DEBUG_INFO =================>")
        print("DEBUG_INFO =================>")

    return errors.avg, error_names

def validate_pose_with_gt():
    pass
def validate_seg_with_gt():
    pass