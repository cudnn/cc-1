import argparse

parser_main3 = argparse.ArgumentParser(description='supervised depth estimation training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_main3.add_argument('--data_dir', metavar='DIR',default='/home/roit/datasets/MC_256512',
                    help='path to dataset')

parser_main3.add_argument('--name', type=str, default='visdrone_raw_256512',
                    help='name of the experiment, checpoints are stored in checpoints/name')



parser_main3.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')


# hyper paras
parser_main3.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser_main3.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser_main3.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser_main3.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser_main3.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser_main3.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser_main3.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser_main3.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')

parser_main3.add_argument('--smoothness-type', dest='smoothness_type', type=str, default='regular', choices=['edgeaware', 'regular'],
                    help='Compute mean-std locally or globally')
parser_main3.add_argument('--data-normalization', dest='data_normalization', type=str, default='global', choices=['local', 'global'],
                    help='Compute mean-std locally or globally')
parser_main3.add_argument('--nlevels', dest='nlevels', type=int, default=6,
                    help='number of levels in multiscale. Options: 6')
#architecture
parser_main3.add_argument('--dispnet', dest='dispnet', type=str, default='DispResNet6', choices=['DispNetS', 'DispNetS6', 'DispResNetS6', 'DispResNet6'],
                    help='depth network architecture,output[BCHW].')
#modeldict
parser_main3.add_argument('--pretrained-disp', dest='pretrained_disp', default='/home/roit/models/cc/official/dispnet_model_best.pth.tar',
                    help='path to pre-trained dispnet model')

parser_main3.add_argument('--spatial-normalize', dest='spatial_normalize', action='store_true', help='spatially normalize depth maps')
parser_main3.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=5)


parser_main3.add_argument('--alternating', dest='alternating', action='store_true', help='minimize only one network at a time')
parser_main3.add_argument('--clamp-masks', dest='clamp_masks', action='store_true', help='threshold masks for training')
parser_main3.add_argument('--fix-posemasknet', dest='fix_posemasknet', action='store_true', help='fix pose and masknet')
parser_main3.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser_main3.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')

#loss weights
parser_main3.add_argument('-pc', '--cam-photo-loss-weight', default=1,type=float, help='weight for camera photometric loss for rigid pixels', metavar='W')
parser_main3.add_argument('-pf', '--flow-photo-loss-weight', default=1, type=float, help='weight for photometric loss for non rigid optical flow', metavar='W')
parser_main3.add_argument('-m', '--mask-loss-weight',  default=0.1,type=float, help='weight for explainabilty mask loss', metavar='W')
parser_main3.add_argument('-s', '--smooth-loss-weight', default=0.1,type=float, help='weight for disparity smoothness loss', metavar='W')
parser_main3.add_argument('-c', '--consensus-loss-weight', default=0.1, type=float, help='weight for mask consistancy', metavar='W')



#freq-set
parser_main3.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)#训练过程中不add_image
parser_main3.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser_main3.add_argument('--scalar-freq', default=10, type=int,
                    metavar='N', help='add_scalar frequency')
parser_main3.add_argument('--img-freq', default=10, type=int,
                    metavar='N', help='add_image frequency')
parser_main3.add_argument('--show-samples', default=[0,0.5,0.99], help='choose three samples to show out the trainning process')

#logg
parser_main3.add_argument('--log_terminal',default=True)
#这个没有实际意义， 因为到底那些参与训练已经封装进train里面去了

parser_main3.add_argument('--resume',default=False)

parser_main3.add_argument('--val_with_depth_gt',default=True)