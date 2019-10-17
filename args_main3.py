import argparse

parser_main3 = argparse.ArgumentParser(description='Competitive Collaboration training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_main3.add_argument('--data_dir', metavar='DIR',default='/home/roit/datasets/MC_256512',
                    help='path to dataset')
parser_main3.add_argument('--kitti-dir', dest='kitti_dir', type=str, default='/home/roit/datasets/MC_256512/',
                    help='Path to kitti2015 scene flow dataset for optical flow validation')
parser_main3.add_argument('--name', type=str, default='visdrone_raw_256512',
                    help='name of the experiment, checpoints are stored in checpoints/name')



parser_main3.add_argument('--DEBUG', action='store_true', help='DEBUG Mode')


parser_main3.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=5)
parser_main3.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser_main3.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')



parser_main3.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser_main3.add_argument('--epochs', default=200, type=int, metavar='N',
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

parser_main3.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser_main3.add_argument('--smoothness-type', dest='smoothness_type', type=str, default='regular', choices=['edgeaware', 'regular'],
                    help='Compute mean-std locally or globally')
parser_main3.add_argument('--data-normalization', dest='data_normalization', type=str, default='global', choices=['local', 'global'],
                    help='Compute mean-std locally or globally')
parser_main3.add_argument('--nlevels', dest='nlevels', type=int, default=6,
                    help='number of levels in multiscale. Options: 6')
#architecture
parser_main3.add_argument('--dispnet', dest='dispnet', type=str, default='DispResNet6', choices=['DispNetS', 'DispNetS6', 'DispResNetS6', 'DispResNet6'],
                    help='depth network architecture,output[BCHW].')
parser_main3.add_argument('--posenet', dest='posenet', type=str, default='PoseNetB6', choices=['PoseNet6','PoseNetB6', 'PoseExpNet'],
                    help='pose and explainabity mask network architecture. ')
parser_main3.add_argument('--masknet', dest='masknet', type=str, default='MaskNet6', choices=['MaskResNet6', 'MaskNet6'],
                    help='pose and explainabity mask network architecture. ')
parser_main3.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['Back2Future', 'FlowNetC6','FlowNetS'],
                    help='flow network architecture. Options: FlowNetC6 | Back2Future')
#modeldict
parser_main3.add_argument('--pretrained-disp', dest='pretrained_disp', default='/home/roit/models/cc/official/dispnet_model_best.pth.tar',
                    help='path to pre-trained dispnet model')
parser_main3.add_argument('--pretrained-mask', dest='pretrained_mask', default='/home/roit/models/cc/official/masknet_model_best.pth.tar',
                    help='path to pre-trained Exp Pose net model')
parser_main3.add_argument('--pretrained-pose', dest='pretrained_pose', default='/home/roit/models/cc/official/posenet_model_best.pth.tar',
                    help='path to pre-trained Exp Pose net model')
parser_main3.add_argument('--pretrained-flow', dest='pretrained_flow', default='/home/roit/models/cc/official/flownet_model_best.pth.tar',
                    help='path to pre-trained Flow net model')

parser_main3.add_argument('--spatial-normalize', dest='spatial_normalize', action='store_true', help='spatially normalize depth maps')
parser_main3.add_argument('--robust', dest='robust', action='store_true', help='train using robust losses')
parser_main3.add_argument('--no-non-rigid-mask', dest='no_non_rigid_mask', action='store_true', help='will not use mask on loss of non-rigid flow')
parser_main3.add_argument('--joint-mask-for-depth', dest='joint_mask_for_depth', default=False, help='use joint mask from masknet and consensus mask for depth training')


parser_main3.add_argument('--fix-masknet', dest='fix_masknet', action='store_true',default=False, help='do not train posenet')
parser_main3.add_argument('--fix-posenet', dest='fix_posenet', action='store_true',default=False, help='do not train posenet')
parser_main3.add_argument('--fix-flownet', dest='fix_flownet', action='store_true',default=False, help='do not train flownet')
parser_main3.add_argument('--fix-dispnet', dest='fix_dispnet', action='store_true',default=False, help='do not train dispnet')

parser_main3.add_argument('--alternating', dest='alternating', action='store_true', help='minimize only one network at a time')
parser_main3.add_argument('--clamp-masks', dest='clamp_masks', action='store_true', help='threshold masks for training')
parser_main3.add_argument('--fix-posemasknet', dest='fix_posemasknet', action='store_true', help='fix pose and masknet')
parser_main3.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser_main3.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser_main3.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser_main3.add_argument('-qch', '--qch', type=float, help='q value for charbonneir', metavar='W', default=0.5)
parser_main3.add_argument('-wrig', '--wrig', type=float, help='consensus imbalance weight', metavar='W', default=1.0)
parser_main3.add_argument('-wbce', '--wbce', type=float, help='weight for binary cross entropy loss', metavar='W', default=0.5)
parser_main3.add_argument('-wssim', '--wssim', type=float, help='weight for ssim loss', metavar='W', default=0.0)


#loss weights
parser_main3.add_argument('-pc', '--cam-photo-loss-weight', default=1,type=float, help='weight for camera photometric loss for rigid pixels', metavar='W')
parser_main3.add_argument('-pf', '--flow-photo-loss-weight', default=1, type=float, help='weight for photometric loss for non rigid optical flow', metavar='W')
parser_main3.add_argument('-m', '--mask-loss-weight',  default=0.1,type=float, help='weight for explainabilty mask loss', metavar='W')
parser_main3.add_argument('-s', '--smooth-loss-weight', default=0.1,type=float, help='weight for disparity smoothness loss', metavar='W')
parser_main3.add_argument('-c', '--consensus-loss-weight', default=0.1, type=float, help='weight for mask consistancy', metavar='W')

#hyper parameters
parser_main3.add_argument('--THRESH', '--THRESH', type=float, help='threshold for masks', metavar='W', default=0.01)
parser_main3.add_argument('--lambda-oob', type=float, help='weight on the out of bound pixels', default=0)
parser_main3.add_argument('--log-output', action='store_true',default=True, help='will log dispnet outputs and warped imgs at validation step')
parser_main3.add_argument('--log-terminal', action='store_true', default=True,help='will display progressbar at terminal')
parser_main3.add_argument('--resume', action='store_true', help='resume from checkpoint')

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


# what with what

parser_main3.add_argument('--dataset-format', default='sequential_with_gt',choices=['sequential_with_gt','sequential','stacked'], metavar='STR',help='dataset format, stacked: stacked frames (from original TensorFlow code)sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')

#这个没有实际意义， 因为到底那些参与训练已经封装进train里面去了
parser_main3.add_argument('--train-with-depth-gt', default=True)#for mc
parser_main3.add_argument('--train-with-flow-gt', default=False)#for mc

#这个分开validate 和train集成在一起不同
parser_main3.add_argument('--val-with-depth-gt', action='store_true',default=True, help='use ground truth for depth validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser_main3.add_argument('--val-with-flow-gt', action='store_true', default=True,help='use ground truth for flow validation. \
                    see data/validation_flow for an example')
parser_main3.add_argument('--val_without_gt',default=False)


