
#主要是火车和飞行棋视觉里程计
from random import random
import torch

from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import models
from models import PoseNetB6
from utils import tensor2array
from inverse_warp import pose_vec2mat
from mpl_toolkits.mplot3d import Axes3D

from datasets.sequence_folders2 import SequenceFolder
import custom_transforms
parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument("--pretrained",  type=str, help="pretrained DispNet path",default='/home/roit/models/cc/official/dispnet_k.pth.tar')
parser.add_argument("--pretrained",  type=str, help="pretrained DispNet path",default='/home/roit/models/cc/official/posenet_model_best.pth.tar')

parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=512, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir",
                    default='/home/roit/datasets/2019c_256512/', type=str, help="Dataset directory")
                    #default='/home/roit/datasets/MC/2019_08_26_10_05_26/imgs', type=str,help="Dataset directory")

parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--trace-dir", action='store_true', help="save disparity img",default='trace')
#parser.add_argument("--output-depth", action='store_true', help="save depth img",default='output_depth/')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args = parser.parse_args()

def save_pose_all():

    if len(val_loader)==0:
        print('载入数据出错')
        return
    poses_all = None

    print(len(val_loader))
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        poses = pose_net(tgt_img, ref_imgs)

        poses = torch.cat([poses[:, :len(ref_imgs) // 2, :],
                           torch.zeros(1, 1, 6).float().to(device),
                           poses[:, len(ref_imgs) // 2:, :]], dim=1)  # add 0
        if i == 0:
            poses_all = poses
        else:
            poses_all = torch.cat([poses_all, poses])

        #if poses_all.shape[0]>200:
        #    ret_file_name = dataset_name +'_'+ str(i)+'_poses_all.npy'
        #    np.save(ret_file_name, poses_all.detach().cpu().numpy())
    ret_file_name = dataset_name+'_poses_all.npy'
    np.save(ret_file_name, poses_all.detach().cpu().numpy())
    return ret_file_name

def pose_vec2points(file):
    '''
        把所有的pose变成其次坐标点
    :return:
    '''
    #origin = torch.tensor([0,0,0])
    poses_all = np.load(file)#[b,4,6]


    frame_poses_list = [[] for i in range(poses_all.shape[0]) ]
    for i in range(poses_all.shape[0]):#b
        for j in range(i,i+poses_all.shape[1]):#0~5,1~6
            if j<poses_all.shape[0]:
                frame_poses_list[j].append(poses_all[i,j-i,:])
            else:
                break
    #去除空值
    for i in range(2,len(frame_poses_list)):
        if i ==2:
            frame_poses_list[i].pop(0)
        elif i ==3:
            frame_poses_list[i].pop(1)
        else:
            frame_poses_list[i].pop(2)

    #求和平均，得到一个list
    batch_pose_vec=None#at last [b(96),6]
    for i in range(len(frame_poses_list)):
        nump=np.zeros(6)
        for j in range(len(frame_poses_list[i])):
            nump += np.array(frame_poses_list[i][j])
        nump/=len(frame_poses_list[i])
        if i==0:
            batch_pose_vec = nump.reshape(1,-1)
        else:
            batch_pose_vec=np.concatenate([batch_pose_vec,nump.reshape(1,-1)])

    #6d-tensor 2 matrix
    batch_pose_vec = torch.tensor(batch_pose_vec)
    batch_pose_mat = pose_vec2mat(batch_pose_vec)


    origin = torch.tensor([[0.], [0.], [0.], [1.]]).double()
    point=origin
    points=None#last [b,4]齐次坐标
    for i in range(batch_pose_mat.shape[0]):
        point= batch_pose_mat[i]@point
        point = torch.cat([point,torch.ones([1,1]).double()])
        if i ==0:
            points=point.unsqueeze(0)
        else:
            points=torch.cat([points,point.unsqueeze(0)])
    ret_file_name = dataset_name+'_corrds.npy'
    np.save(ret_file_name,points.detach().numpy())
    return  ret_file_name



def draw(file):

    corrs = np.load(file)#

    x=corrs[:,0,0]
    y=corrs[:,1,0]
    z=corrs[:,2,0]

    plt.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    ax.plot(x, y, z, label='path line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('The result on train100a')
    #ax.plot(x, y, z, label='parametric curve2')
    #ax.plot(x*1.3*random()+random()*.1, y*1.3+random()*0.01, z+random()*0.1, label='parametric curve3')


    ax.legend()

    plt.show()
    print('ok')
def draw2():
    file1 = 'train100_corrds.npy'
    file2 = 'train281_corrds.npy'
    corrs = np.load(file1)#
    corrs2 = np.load(file2)#

    x=corrs[:,0,0]
    y=corrs[:,1,0]
    z=corrs[:,2,0]
    x2 = corrs2[:, 0, 0]
    y2 = corrs2[:, 1, 0]
    z2 = corrs2[:, 2, 0]

    plt.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    ax.plot(x, y, z, label='train100 path')
    ax.plot(x2, y2, z2, label='train281 path')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('The result on train')
    #ax.plot(x, y, z, label='parametric curve2')
    #ax.plot(x*1.3*random()+random()*.1, y*1.3+random()*0.01, z+random()*0.1, label='parametric curve3')


    ax.legend()

    plt.show()
    print('ok')
if __name__ == '__main__':


    weights = torch.load(args.pretrained)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = getattr(models, 'PoseNetB6')(nb_ref_imgs=seq_length - 1).cuda()
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    trace_dir = output_dir / args.trace_dir  # 轨迹
    trace_dir.makedirs_p()
    # data prepare

    normalize = custom_transforms.NormalizeLocally()
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    val_set = SequenceFolder(  # 只有图
        args.dataset_dir,
        transform=valid_transform,
        seed=None,
        train=False,
        sequence_length=5,
        target_transform=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=128, pin_memory=True, drop_last=True)
    dataset_name = val_set.scenes[0].stem



    #file = save_pose_all()# out dataset_.npy
    #file = pose_vec2points(file)#out

    #file='uav100_corrds.npy'
    draw2()

    # This import registers the 3D projection, but is otherwise unused.

