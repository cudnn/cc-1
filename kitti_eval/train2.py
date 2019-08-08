#魔改版
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
from inverse_warp import inverse_warp, pose2flow, flow2oob, flow_warp

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


epsilon = 1e-8

parser = argparse.ArgumentParser(description='Competitive Collaboration training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR',default='/home/roit/datasets/visdrone_raw_256512/',
                    help='path to dataset')
parser.add_argument('--kitti-dir', dest='kitti_dir', type=str, default='/home/roit/datasets/kitti_flow/',
                    help='Path to kitti2015 scene flow dataset for optical flow validation')
parser.add_argument('--name', type=str, default='visdrone_raw_256512',
                    help='name of the experiment, checpoints are stored in checpoints/name')



parser.add_argument('--DEBUG', action='store_true', help='DEBUG Mode')

parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=5)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')


parser.add_argument('--with-depth-gt', action='store_true',default=False, help='use ground truth for depth validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--with-flow-gt', action='store_true', default=False,help='use ground truth for flow validation. \
                    see data/validation_flow for an example')
parser.add_argument('--without-gt', action='store_true', default=True,help='use ground truth for flow validation. \
                    see data/validation_flow for an example')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--smoothness-type', dest='smoothness_type', type=str, default='regular', choices=['edgeaware', 'regular'],
                    help='Compute mean-std locally or globally')
parser.add_argument('--data-normalization', dest='data_normalization', type=str, default='global', choices=['local', 'global'],
                    help='Compute mean-std locally or globally')
parser.add_argument('--nlevels', dest='nlevels', type=int, default=6,
                    help='number of levels in multiscale. Options: 6')
#architecture
parser.add_argument('--dispnet', dest='dispnet', type=str, default='DispResNet6', choices=['DispNetS', 'DispNetS6', 'DispResNetS6', 'DispResNet6'],
                    help='depth network architecture.')
parser.add_argument('--posenet', dest='posenet', type=str, default='PoseNetB6', choices=['PoseNet6','PoseNetB6', 'PoseExpNet'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--masknet', dest='masknet', type=str, default='MaskNet6', choices=['MaskResNet6', 'MaskNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['Back2Future', 'FlowNetC6','FlowNetS'],
                    help='flow network architecture. Options: FlowNetC6 | Back2Future')
#modeldict
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default='/home/roit/models/cc/official/dispnet_k.pth.tar',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default='/home/roit/models/cc/official/masknet.pth.tar',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default='/home/roit/models/cc/official/posenet.pth.tar',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default='/home/roit/models/cc/official/back2future.pth.tar',
                    help='path to pre-trained Flow net model')

parser.add_argument('--spatial-normalize', dest='spatial_normalize', action='store_true', help='spatially normalize depth maps')
parser.add_argument('--robust', dest='robust', action='store_true', help='train using robust losses')
parser.add_argument('--no-non-rigid-mask', dest='no_non_rigid_mask', action='store_true', help='will not use mask on loss of non-rigid flow')
parser.add_argument('--joint-mask-for-depth', dest='joint_mask_for_depth', action='store_true', help='use joint mask from masknet and consensus mask for depth training')


parser.add_argument('--fix-masknet', dest='fix_masknet', action='store_true',default=False, help='do not train posenet')
parser.add_argument('--fix-posenet', dest='fix_posenet', action='store_true',default=False, help='do not train posenet')
parser.add_argument('--fix-flownet', dest='fix_flownet', action='store_true',default=True, help='do not train flownet')
parser.add_argument('--fix-dispnet', dest='fix_dispnet', action='store_true', help='do not train dispnet')

parser.add_argument('--alternating', dest='alternating', action='store_true', help='minimize only one network at a time')
parser.add_argument('--clamp-masks', dest='clamp_masks', action='store_true', help='threshold masks for training')
parser.add_argument('--fix-posemasknet', dest='fix_posemasknet', action='store_true', help='fix pose and masknet')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-qch', '--qch', type=float, help='q value for charbonneir', metavar='W', default=0.5)
parser.add_argument('-wrig', '--wrig', type=float, help='consensus imbalance weight', metavar='W', default=1.0)
parser.add_argument('-wbce', '--wbce', type=float, help='weight for binary cross entropy loss', metavar='W', default=0.5)
parser.add_argument('-wssim', '--wssim', type=float, help='weight for ssim loss', metavar='W', default=0.0)
parser.add_argument('-pc', '--cam-photo-loss-weight', type=float, help='weight for camera photometric loss for rigid pixels', metavar='W', default=1)
parser.add_argument('-pf', '--flow-photo-loss-weight', type=float, help='weight for photometric loss for non rigid optical flow', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--consensus-loss-weight', type=float, help='weight for mask consistancy', metavar='W', default=0.1)
parser.add_argument('--THRESH', '--THRESH', type=float, help='threshold for masks', metavar='W', default=0.01)
parser.add_argument('--lambda-oob', type=float, help='weight on the out of bound pixels', default=0)
parser.add_argument('--log-output', action='store_true',default=True, help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--log-terminal', action='store_true', default=True,help='will display progressbar at terminal')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

#freq-set

parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)#训练过程中不add_image

parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--scalar-freq', default=10, type=int,
                    metavar='N', help='add_scalar frequency')
parser.add_argument('--img-freq', default=10, type=int,
                    metavar='N', help='add_image frequency')

#global args
args = parser.parse_args()
best_error = -1
n_iter = 0
n_iter_val = 0


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    global args, best_error, n_iter

    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    args.save_path = Path('checkpoints')/args.name/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.alternating:
        args.alternating_flags = np.array([False,False,True])

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

# Data loading code
    flow_loader_h, flow_loader_w = 256, 832

    if args.data_normalization =='global':
        normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
    elif args.data_normalization =='local':
        normalize = custom_transforms.NormalizeLocally()

    if args.fix_flownet:
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])
    else:
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomRotate(),
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))

    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_depth_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data.replace('cityscapes', 'kitti'),
            transform=valid_transform
        )
    else:
        val_set = SequenceFolder(#只有图
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
        )

    if args.with_flow_gt:
        from datasets.validation_flow import ValidationFlow
        val_flow_set = ValidationFlow(root=args.kitti_dir,
                                        sequence_length=args.sequence_length,
                                      transform=valid_flow_transform)
        val_flow_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1,
                                                      # batch size is 1 since images in kitti have different sizes
                                                      shuffle=False, num_workers=args.workers, pin_memory=True,
                                                      drop_last=True)

    if args.DEBUG:
        train_set.__len__ = 32
        train_set.samples = train_set.samples[:32]

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)



    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

#1 create model
    print("=> creating model")
    #1.1 disp_net
    disp_net = getattr(models, args.dispnet)().cuda()
    output_exp = True #args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    #1.2 pose_net
    pose_net = getattr(models, args.posenet)(nb_ref_imgs=args.sequence_length - 1).cuda()

    #1.3.flow_net
    if args.flownet=='SpyNet':
        flow_net = getattr(models, args.flownet)(nlevels=args.nlevels, pre_normalization=normalize).cuda()
    elif args.flownet=='FlowNetC6':#flonwtc6
        flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()
    elif args.flownet=='FlowNetS':
        flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()
    elif args.flownet =='Back2Future':
        flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()


    # 1.4 mask_net
    mask_net = getattr(models, args.masknet)(nb_ref_imgs=args.sequence_length - 1, output_exp=True).cuda()

#2 载入参数
    #2.1 pose
    if args.pretrained_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'])
    else:
        pose_net.init_weights()

    if args.pretrained_mask:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_mask)
        mask_net.load_state_dict(weights['state_dict'])
    else:
        mask_net.init_weights()

    # import ipdb; ipdb.set_trace()
    if args.pretrained_disp:
        print("=> using pre-trained weights from {}".format(args.pretrained_disp))
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    if args.pretrained_flow:
        print("=> using pre-trained weights for FlowNet")
        weights = torch.load(args.pretrained_flow)
        flow_net.load_state_dict(weights['state_dict'])
    else:
        flow_net.init_weights()

    if args.resume:
        print("=> resuming from checkpoint")
        dispnet_weights = torch.load(args.save_path/'dispnet_checkpoint.pth.tar')
        posenet_weights = torch.load(args.save_path/'posenet_checkpoint.pth.tar')
        masknet_weights = torch.load(args.save_path/'masknet_checkpoint.pth.tar')
        flownet_weights = torch.load(args.save_path/'flownet_checkpoint.pth.tar')
        disp_net.load_state_dict(dispnet_weights['state_dict'])
        pose_net.load_state_dict(posenet_weights['state_dict'])
        flow_net.load_state_dict(flownet_weights['state_dict'])
        mask_net.load_state_dict(masknet_weights['state_dict'])


    # import ipdb; ipdb.set_trace()
    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)
    mask_net = torch.nn.DataParallel(mask_net)
    flow_net = torch.nn.DataParallel(flow_net)

    print('=> setting adam solver')

    parameters = chain(disp_net.parameters(), pose_net.parameters(), mask_net.parameters(), flow_net.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    if args.resume and (args.save_path/'optimizer_checkpoint.pth.tar').exists():
        print("=> loading optimizer from checkpoint")
        optimizer_weights = torch.load(args.save_path/'optimizer_checkpoint.pth.tar')
        optimizer.load_state_dict(optimizer_weights['state_dict'])

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_cam_loss', 'photo_flow_loss', 'explainability_loss', 'smooth_loss'])

    if args.log_terminal:
        logger = TermLogger(n_epochs=args.epochs,
                            train_size=min(len(train_loader), args.epoch_size),
                            valid_size=len(val_loader))
        logger.epoch_bar.start()
    else:
        logger=None
#3. main cycle
    for epoch in range(args.epochs):
        #3.1 四个子网络，训练哪几个
        if args.fix_flownet:
            for fparams in flow_net.parameters():
                fparams.requires_grad = False

        if args.fix_masknet:
            for fparams in mask_net.parameters():
                fparams.requires_grad = False

        if args.fix_posenet:
            for fparams in pose_net.parameters():
                fparams.requires_grad = False

        if args.fix_dispnet:
            for fparams in disp_net.parameters():
                fparams.requires_grad = False

        if args.log_terminal:
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()
        #validation data
        flow_error_names = ['no']
        flow_errors = [0]
        errors = [0]
        error_names = ['no error names depth']
        print('\nepoch [{}/{}]\n'.format(epoch+1,args.epochs))
        #3.2 train for one epoch---------
        train_loss=0
        train_loss = train(train_loader, disp_net, pose_net, mask_net, flow_net, optimizer,logger, training_writer)

        #3.3 evaluate on validation set-----

        #if args.without_gt:
         #   validate_without_gt(val_loader,disp_net,pose_net,mask_net, epoch, logger, output_writers)


        if args.with_flow_gt:
            flow_errors, flow_error_names = validate_flow_with_gt(val_flow_loader, disp_net, pose_net, mask_net, flow_net, epoch, logger, output_writers)

            for error, name in zip(flow_errors, flow_error_names):
                training_writer.add_scalar(name, error, epoch)

        if args.with_depth_gt:
            depth_errors, depth_error_names = validate_depth_with_gt(val_loader, disp_net, epoch, logger, output_writers)

            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(depth_error_names, depth_errors))

            if args.log_terminal:
                logger.valid_writer.write(' * Avg {}'.format(error_string))
            else:
                print('Epoch {} completed'.format(epoch))

            for error, name in zip(depth_errors, depth_error_names):
                training_writer.add_scalar(name, error, epoch)
        if args.without_gt:

            val_loss = 1


        #----------------------


        #3.4 Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)

        if not args.fix_posenet:
            decisive_error =0# flow_errors[-2]    # epe_rigid_with_gt_mask
        elif not args.fix_dispnet:
            decisive_error =0# errors[0]      #depth abs_diff
        elif not args.fix_flownet:
            decisive_error =0# flow_errors[-1]    #epe_non_rigid_with_gt_mask
        elif not args.fix_masknet:
            decisive_error = 0#flow_errors[3]     # percent outliers

        #3.5 log
        if args.log_terminal:
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
            logger.reset_valid_bar()


        #3.6 save model and remember lowest error and save checkpoint

        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error <= best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': mask_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': flow_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': optimizer.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    if args.log_terminal:
        logger.epoch_bar.finish()





def train(train_loader, disp_net, pose_net, mask_net, flow_net, optimizer,  logger=None, train_writer=None):
# 0. 准备
    global args, n_iter
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
        tgt_img_var = Variable(tgt_img.cuda())
        ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda())
        intrinsics_inv_var = Variable(intrinsics_inv.cuda())



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

        if args.no_non_rigid_mask:
            flow_exp_mask = [None for exp_mask in explainability_mask]
            if args.DEBUG:
                print('Using no masks for flow')
        else:
            flow_exp_mask = [1 - exp_mask[:,1:3] for exp_mask in explainability_mask]


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

    return losses.avg[0]#epoch loss


@torch.no_grad()
def validate_without_gt(val_loader,disp_net,pose_net,mask_net,  epoch, logger, output_writers=[]):
#data prepared
    global args,device,n_iter_val
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_outputs = len(output_writers) > 0
    losses = AverageMeter(precision=4)

    w1, w2, w3, w4 = args.cam_photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight, args.flow_photo_loss_weight


    loss_camera = photometric_reconstruction_loss
    loss_flow = photometric_flow_loss

# to eval model
    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    #flow_net.eval()

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
        pose = pose_net(tgt_img, ref_imgs)


        #mask
        # 4.explainability_mask(none)
        explainability_mask = mask_net(tgt_img, ref_imgs)  # 有效区域?4??
        # list(5):item:tensor:[4,4,128,512]...[4,4,4,16] value:[0.33~0.48~0.63]
        # -------------------------------------------------

        if args.joint_mask_for_depth:
            explainability_mask_for_depth = compute_joint_mask_for_depth(explainability_mask, rigidity_mask_bwd,
                                                                         rigidity_mask_fwd)
        else:
            explainability_mask_for_depth = explainability_mask


    #3.2loss-compute
        loss_1 = loss_camera(tgt_img, ref_imgs, intrinsics, intrinsics_inv,
                             depth, explainability_mask_for_depth, pose, lambda_oob=args.lambda_oob, qch=args.qch,
                             wssim=args.wssim)

        # E_M
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)  # + 0.2*gaussian_explainability_loss(explainability_mask)
        else:
            loss_2 = 0

        #if args.smoothness_type == "regular":
        loss_3 = smooth_loss(depth) + smooth_loss(explainability_mask)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3


    #3.3 data update
        losses.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()


    #3.4 check log

        #查看forward pass效果
        if log_outputs and i % 40 == 0 and i / 100 < len(output_writers):#output_writers list(3)
            index = int(i // 40)
            #disp = disp.data.cpu()[0]
            #disp = (255 * tensor2array(disp, max_value=None, colormap='bone')).astype(np.uint8)
            #disp = disp.transpose(1, 2, 0)

            if epoch == 0:
                output_writers[index].add_image('val Input', tensor2array(tgt_img[0]), 0)

                disp_to_show = disp[0].cpu()# tensor disp_to_show :[1,h,w],0.5~3.1~10
                output_writers[index].add_image('val target disp222', tensor2array(disp_to_show, max_value=None,colormap='magma'), epoch)
                save = (255 * tensor2array(disp_to_show, max_value=None, colormap='magma')).astype(np.uint8)
                save = save.transpose(1, 2, 0)
                plt.imsave('ep1_test.jpg', save, cmap='plasma')
#                depth_to_show[depth_to_show == 0] = 1000
 #               disp_to_show = (1 / depth_to_show).clamp(0, 10)
  #              output_writers[index].add_image('val target Disparity Normalized',
   #                                             tensor2array(disp_to_show, max_value=None, colormap='bone'), epoch)

            output_writers[index].add_image('val Dispnet Output Normalized123',
                                            tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone'),
                                            epoch)
            #output_writers[index].add_image('val Depth Output', tensor2array(depth.data[0].cpu(), max_value=10),
             #                               epoch)

        # errors.update(compute_errors(depth, output_depth.data.squeeze(1)))
            # add scalar
        if args.scalar_freq > 0 and n_iter_val % args.scalar_freq == 0:
            output_writers[0].add_scalar('val/cam_photometric_error', loss_1.item(), n_iter_val)
            if w2 > 0:
                output_writers[0].add_scalar('val/explanability_loss', loss_2.item(), n_iter_val)
            output_writers[0].add_scalar('val/disparity_smoothness_loss', loss_3.item(), n_iter_val)
            #output_writers[0].add_scalar('batch/flow_photometric_error', loss_4.item(), n_iter)
            #output_writers.add_scalar('batch/consensus_error', loss_5.item(), n_iter)
            output_writers[0].add_scalar('val/total_loss', loss.item(), n_iter_val)

        # terminal output
        if args.log_terminal:
            logger.valid_bar.update(i + 1)  # 当前epoch 进度
            if i % args.print_freq == 0:
                logger.valid_bar_writer.write('Valid: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))

        n_iter_val+=1


    return losses.avg[0]#epoch validate loss



if __name__ == '__main__':
    import sys
    with open("experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()
