#纯有监督


import time

import csv
import datetime
import os

import  matplotlib.pyplot as plt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import custom_transforms
import models
from utils import tensor2array, save_checkpoint

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
from train import  train_depth_gt
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

    if args.data_normalization =='global':
        normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
    elif args.data_normalization =='local':
        normalize = custom_transforms.NormalizeLocally()


    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data_dir))

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomRotate(),
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

#train set, loader only建立一个
    from datasets.sequence_mc import SequenceFolder
    train_set = SequenceFolder(  # mc data folder
        args.data_dir,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,  # 5
        target_transform=None
    )




    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

#val set,loader 挨个建立
    #if args.val_with_depth_gt:
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




    #
    if args.log_terminal:
        logger = TermLogger(n_epochs=args.epochs,
                            train_size=min(len(train_loader), args.epoch_size),
                            valid_size=len(val_loader_depth))
        logger.reset_epoch_bar()
    else:
        logger=None



#预先评估下


    if args.pretrained_disp or args.evaluate:
        logger.reset_valid_bar()

        if args.val_with_depth_gt:
            pass
            depth_errors, depth_error_names = validate_depth_with_gt(val_loader_depth, disp_net, epoch=0, logger=logger,tb_writer=tb_writer,global_vars_dict=global_vars_dict)
            mes = ''
            for s, k in zip(depth_errors, depth_error_names):
                mes+= k
                mes+=' '
                mes+= str(s)
                mes+=' '

            logger.reset_epoch_bar()
            logger.epoch_logger_update(0,mes)
        else:
            logger.reset_epoch_bar()
            logger.epoch_logger_update(0,' ')



#3. main cycle
    for epoch in range(1,args.epochs):#epoch 0 在第没入循环之前已经测试了.



        logger.reset_train_bar()
        logger.reset_valid_bar()



        errors = [0]
        error_names = ['no error names depth']


        #3.2 train for one epoch---------
        train_loss = train_depth_gt(train_loader, disp_net, optimizer,logger, tb_writer,global_vars_dict)

        #3.3 evaluate on validation set-----
        depth_errors, depth_error_names = validate_depth_with_gt(val_loader_depth, disp_net, epoch=epoch, logger=logger, tb_writer=tb_writer,global_vars_dict=global_vars_dict)







        #3.5 log_terminal
        #if args.log_terminal:
        logger.epoch_logger_update(epoch,' * Avg Train Loss : {:.3f}'.format(train_loss))


    # tensorboard scaler
        #train loss
        tb_writer.add_scalar('epoch/train_loss',train_loss,epoch)
        #val_with_gt loss
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
            },
            {
                'epoch': epoch + 1,
                'state_dict': None
            },
            {
                'epoch': epoch + 1,
                'state_dict': None
            },
            {
                'epoch': epoch + 1,
                'state_dict': None
            },
            is_best)

    if args.log_terminal:
        logger.epoch_bar.finish()





if __name__ == '__main__':
    import sys
    with open("experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()

