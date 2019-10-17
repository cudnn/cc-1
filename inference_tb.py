

import argparse
import time
import datetime

from tqdm import tqdm
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

from datasets.sequence_folders import SequenceFolder


from logger import TermLogger, AverageMeter
from path import Path
from tensorboardX import SummaryWriter


from utils import flow2rgb,spatial_normalize


epsilon = 1e-8

parser = argparse.ArgumentParser(description='Competitive Collaboration training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR',default='/home/roit/datasets/visdrone_raw_256512/',
                    help='path to dataset')




parser.add_argument('--save_path',type = str,default='test_out')

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



parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')

parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--sq-name', default='uav0000013_01073_v')

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
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default='/home/roit/models/cc/MC_256512/dispnet_model_best.pth.tar',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default='/home/roit/models/cc/MC_256512/masknet_model_best.pth.tar',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default='/home/roit/models/cc/MC_256512/posenet_model_best.pth.tar',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default='/home/roit/models/cc/MC_256512/flownet_model_best.pth.tar',
                    help='path to pre-trained Flow net model')

parser.add_argument('--spatial-normalize', dest='spatial_normalize', action='store_true', help='spatially normalize depth maps')
parser.add_argument('--robust', dest='robust', action='store_true', help='train using robust losses')
parser.add_argument('--no-non-rigid-mask', dest='no_non_rigid_mask', action='store_true', help='will not use mask on loss of non-rigid flow')
parser.add_argument('--joint-mask-for-depth', dest='joint_mask_for_depth', default=True, help='use joint mask from masknet and consensus mask for depth training')



parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('-wrig', '--wrig', type=float, help='consensus imbalance weight', metavar='W', default=1.0)
parser.add_argument('-wssim', '--wssim', type=float, help='weight for ssim loss', metavar='W', default=0.0)
#global args
args = parser.parse_args()
n_iter = 0
n_iter_val = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#for train.py and validate.py using
global_vars_dict={}#通过dict 可实现地址传递，可# 修改全局变量
global_vars_dict['args'] = args# = globals()
global_vars_dict['n_iter'] = n_iter
global_vars_dict['n_iter_val'] = n_iter_val
global_vars_dict['device']=device





@torch.no_grad()
def test(val_loader,disp_net,mask_net,pose_net, flow_net, tb_writer,global_vars_dict = None):
#data prepared
    device = global_vars_dict['device']
    n_iter_val = global_vars_dict['n_iter_val']
    args = global_vars_dict['args']


    data_time = AverageMeter()


# to eval model
    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    end = time.time()
    poses = np.zeros(((len(val_loader)-1) * 1 * (args.sequence_length-1),6))#init

    disp_list = []

    flow_list = []
    mask_list = []

#3. validation cycle
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in tqdm(enumerate(val_loader)):
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
        #flow----
        #制作前后一帧的
        if args.flownet == 'Back2Future':
            flow_fwd, flow_bwd, _ = flow_net(tgt_img, ref_imgs[1:3])
        elif args.flownet == 'FlowNetC6':
            flow_fwd = flow_net(tgt_img, ref_imgs[2])
            flow_bwd = flow_net(tgt_img, ref_imgs[1])
        #FLOW FWD [B,2,H,W]
        #flow cam :tensor[b,2,h,w]
        #flow_background
        flow_cam = pose2flow(depth.squeeze(1), pose[:, 2], intrinsics, intrinsics_inv)

        flows_cam_fwd = pose2flow(depth.squeeze(1), pose[:, 2], intrinsics, intrinsics_inv)
        flows_cam_bwd = pose2flow(depth.squeeze(1), pose[:, 1], intrinsics, intrinsics_inv)

        #exp_masks_target = consensus_exp_masks(flows_cam_fwd, flows_cam_bwd, flow_fwd, flow_bwd, tgt_img,
        #                                       ref_imgs[2], ref_imgs[1], wssim=args.wssim, wrig=args.wrig,
        #                                       ws=args.smooth_loss_weight)

        rigidity_mask_fwd = (flows_cam_fwd - flow_fwd).abs()#[b,2,h,w]
        rigidity_mask_bwd = (flows_cam_bwd - flow_bwd).abs()

        # mask
        # 4.explainability_mask(none)
        explainability_mask = mask_net(tgt_img, ref_imgs)  # 有效区域?4??

        # list(5):item:tensor:[4,4,128,512]...[4,4,4,16] value:[0.33~0.48~0.63]
        end = time.time()


    #3.4 check log

        #查看forward pass效果
    # 2 disp
        disp_to_show =tensor2array(disp[0].cpu(), max_value=None,colormap='bone')# tensor disp_to_show :[1,h,w],0.5~3.1~10
        tb_writer.add_image('Disp/disp0', disp_to_show,i)
        disp_list.append(disp_to_show)

        if i == 0:
            disp_arr =  np.expand_dims(disp_to_show,axis=0)
        else:
            disp_to_show = np.expand_dims(disp_to_show,axis=0)
            disp_arr = np.concatenate([disp_arr,disp_to_show],0)


    #3. flow
        tb_writer.add_image('Flow/Flow Output', flow2rgb(flow_fwd[0], max_value=6),i)
        tb_writer.add_image('Flow/cam_Flow Output', flow2rgb(flow_cam[0], max_value=6),i)
        tb_writer.add_image('Flow/rigid_Flow Output', flow2rgb(rigidity_mask_fwd[0], max_value=6),i)
        tb_writer.add_image('Flow/rigidity_mask_fwd',flow2rgb(rigidity_mask_fwd[0],max_value=6),i)
        flow_list.append(flow2rgb(flow_fwd[0], max_value=6))
    #4. mask
        tb_writer.add_image('Mask /mask0',tensor2array(explainability_mask[0][0], max_value=None, colormap='magma'), i)
        #tb_writer.add_image('Mask Output/mask1 sample{}'.format(i),tensor2array(explainability_mask[1][0], max_value=None, colormap='magma'), epoch)
        #tb_writer.add_image('Mask Output/mask2 sample{}'.format(i),tensor2array(explainability_mask[2][0], max_value=None, colormap='magma'), epoch)
        #tb_writer.add_image('Mask Output/mask3 sample{}'.format(i),tensor2array(explainability_mask[3][0], max_value=None, colormap='magma'), epoch)
        mask_list.append(tensor2array(explainability_mask[0][0], max_value=None, colormap='magma'))
    #

    return disp_list,disp_arr,flow_list,mask_list

def main():

    global global_vars_dict
    args = global_vars_dict['args']


    normalize = custom_transforms.NormalizeLocally()
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    args.save_path = Path('test_out')/ Path(args.sq_name)
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    tb_writer = SummaryWriter(args.save_path)

    val_set = SequenceFolder(  # 只有图
        args.data,
        transform=valid_transform,
        seed=None,
        train=False,
        sequence_length=args.sequence_length,
        target_transform=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print("=> creating model")
    # 1.1 disp_net
    disp_net = getattr(models, args.dispnet)().cuda()
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'])


    # 1.2 pose_net
    pose_net = getattr(models, args.posenet)(nb_ref_imgs=args.sequence_length - 1).cuda()
    weights = torch.load(args.pretrained_pose)
    pose_net.load_state_dict(weights['state_dict'])

    # 1.3.flow_net
    flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()
    weights = torch.load(args.pretrained_flow)
    flow_net.load_state_dict(weights['state_dict'])

    # 1.4 mask_net
    mask_net = getattr(models, args.masknet)(nb_ref_imgs=args.sequence_length - 1, output_exp=True).cuda()
    weights = torch.load(args.pretrained_mask)
    mask_net.load_state_dict(weights['state_dict'])

    disp_list,disp_arr,flow_list,mask_list= test(val_loader,disp_net,mask_net,pose_net, flow_net, tb_writer,global_vars_dict = global_vars_dict)

    print('over')









if __name__=="__main__":
    main()