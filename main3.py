#纯无监督


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

from loss_functions import\
    photometric_reconstruction_loss, \
    photometric_flow_loss,\
    explainability_loss, \
    gaussian_explainability_loss, \
    smooth_loss, \
    edge_aware_smoothness_loss


from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter
from flowutils.flowlib import flow_to_image
from train import  train,train_gt
from validate import validate_without_gt,validate_depth_with_gt
from args_main3 import parser_main3

#global args
args = parser_main3.parse_args()
epsilon = 1e-8

n_iter = 0#train iter
n_iter_val = 0#val_without_gt
n_iter_val_depth = 0# val_with_depth
n_iter_val_flow = 0#val_with_flow
n_iter_val_pose = 0#val with pose
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#for train.py and validate.py using
global_vars_dict={}#通过dict 可实现地址传递，可# 修改全局变量
global_vars_dict['args'] = args# = globals()

global_vars_dict['n_iter'] = n_iter
global_vars_dict['n_iter_val'] = n_iter_val
global_vars_dict['n_iter_val_depth'] = n_iter_val_depth
global_vars_dict['n_iter_val_pose'] = n_iter_val_pose
global_vars_dict['n_iter_val_flow'] = n_iter_val_flow

global_vars_dict['device']=device


def main():
    global global_vars_dict
    args = global_vars_dict['args']
    best_error = -1#best model choosing



    #mkdir
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

    args.save_path = Path('checkpoints')/Path(args.data_dir).stem/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.alternating:
        args.alternating_flags = np.array([False,False,True])
    #mk writers
    tb_writer = SummaryWriter(args.save_path)

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

    print("=> fetching scenes in '{}'".format(args.data_dir))



#train set, loader only建立一个
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder
        train_set = SequenceFolder(#mc data folder
            args.data_dir,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,#5
            target_transform = None
        )
    elif args.dataset_format == 'sequential_with_gt':  # with all possible gt
        from datasets.sequence_mc import SequenceFolder
        train_set = SequenceFolder(  # mc data folder
            args.data_dir,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,  # 5
            target_transform=None
        )
    else:
        return

    if args.DEBUG:
        train_set.__len__ = 32
        train_set.samples = train_set.samples[:32]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

#val set,loader 挨个建立

    if args.val_with_depth_gt:
        from datasets.validation_folders2 import ValidationSet

        val_set_with_depth_gt = ValidationSet(
            args.data_dir,
            transform=valid_transform
        )

        val_loader_depth = torch.utils.data.DataLoader(
            val_set_with_depth_gt, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True)






    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))



#1 create model
    print("=> creating model")
    #1.1 disp_net
    disp_net = getattr(models, args.dispnet)().cuda()
    output_exp = True #args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    #1.2 pose_net

#2 载入参数
    #2.1 pose


    # import ipdb; ipdb.set_trace()
    if args.pretrained_disp:
        print("=> using pre-trained weights from {}".format(args.pretrained_disp))
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()



    if args.resume:
        print("=> resuming from checkpoint")
        dispnet_weights = torch.load(args.save_path/'dispnet_checkpoint.pth.tar')
        disp_net.load_state_dict(dispnet_weights['state_dict'])


    # import ipdb; ipdb.set_trace()
    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)

    print('=> setting adam solver')

    parameters = chain(disp_net.parameters())
    
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


    #
    if args.log_terminal:
        logger = TermLogger(n_epochs=args.epochs,
                            train_size=min(len(train_loader), args.epoch_size),
                            valid_size=len(val_loader_depth))
        logger.epoch_bar.start()
    else:
        logger=None



#预先评估下


    if args.pretrained_disp or args.evaluate:
        logger.reset_valid_bar()
        if args.val_without_gt:
            pass
            #val_loss = validate_without_gt(val_loader,disp_net,pose_net,mask_net,flow_net,epoch=0, logger=logger, tb_writer=tb_writer,nb_writers=3,global_vars_dict = global_vars_dict)
            #val_loss =0


        if args.val_with_depth_gt:
            pass
            depth_errors, depth_error_names = validate_depth_with_gt(val_loader_depth, disp_net, epoch=0, logger=logger,tb_writer=tb_writer,global_vars_dict=global_vars_dict)



#3. main cycle
    for epoch in range(1,args.epochs):#epoch 0 在第没入循环之前已经测试了.
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
        #train_loss=0
        train_loss = train_gt(train_loader, disp_net, pose_net, mask_net, flow_net, optimizer,logger, tb_writer,global_vars_dict)

        #3.3 evaluate on validation set-----

        if args.val_without_gt:
           val_loss = validate_without_gt(val_loader,disp_net,pose_net,mask_net,flow_net, epoch=0, logger=logger, tb_writer=tb_writer,nb_writers=3,global_vars_dict=global_vars_dict)


        if args.val_with_depth_gt:
            depth_errors, depth_error_names = validate_depth_with_gt(val_loader_depth, disp_net, epoch=epoch, logger=logger, tb_writer=tb_writer,global_vars_dict=global_vars_dict)





        if args.val_with_flow_gt:
            pass
            #flow_errors, flow_error_names = validate_flow_with_gt(val_flow_loader, disp_net, pose_net, mask_net, flow_net, epoch, logger, tb_writer)

            #for error, name in zip(flow_errors, flow_error_names):
            #    training_writer.add_scalar(name, error, epoch)

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
        #eopch data log on tensorboard
        #train loss
        tb_writer.add_scalar('epoch/train_loss',train_loss,epoch)
        #val_without_gt loss
        if args.val_without_gt:
            tb_writer.add_scalar('epoch/val_loss',val_loss,epoch)

        if args.val_with_depth_gt:
        #val with depth gt
            for error, name in zip(depth_errors, depth_error_names):
                tb_writer.add_scalar('epoch/'+name, error, epoch)



        #3.6 save model and remember lowest error and save checkpoint

        if best_error < 0:
            best_error = train_loss

        is_best = train_loss <= best_error
        best_error = min(best_error, train_loss)
        save_checkpoint(
            args.save_path,
            {
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
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    if args.log_terminal:
        logger.epoch_bar.finish()





if __name__ == '__main__':
    import sys
    with open("experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()

