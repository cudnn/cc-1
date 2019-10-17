
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
parser.add_argument("--pretrained",  type=str, help="pretrained mask path",default='/home/roit/models/cc/official/masknet_model_best.pth.tar')

parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=512, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir",
                    default='/home/roit/datasets/2019c_256512', type=str, help="Dataset directory")
                    #default='/home/roit/datasets/MC/2019_08_26_10_05_26/imgs', type=str,help="Dataset directory")

parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--trace-dir", action='store_true', help="save disparity img",default='trace')
#parser.add_argument("--output-depth", action='store_true', help="save depth img",default='output_depth/')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_mask_all():
    args = parser.parse_args()

    weights = torch.load(args.pretrained)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    mask_net = getattr(models, 'MaskNet6')(nb_ref_imgs=5 - 1, output_exp=True).cuda()
    mask_net.load_state_dict(weights['state_dict'], strict=False)
    mask_net.eval()

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
    if len(val_set)==0:
        print('读取错误')
        return
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)
    mask_all = None
    print(len(val_loader))
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        explainability_mask = mask_net(tgt_img, ref_imgs)

        #explainability_mask = torch.cat([explainability_mask[:, :len(ref_imgs) // 2, :],
        #                   torch.zeros(1, 1, 6).float().to(device),
        #                   explainability_mask[:, len(ref_imgs) // 2:, :]], dim=1)  # add 0
        if i == 0:
            mask_all = explainability_mask
        else:
            mask_all = torch.cat([mask_all, explainability_mask])

    np.save('mask_all.npy', mask_all.detach().cpu().numpy())



def imsave(path):
    imgs = np.load(path)
    dump_path_dir = Path('./mask_out').makedirs_p()
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            plt.imsave(dump_path_dir/(str(i)+'_'+str(j)+'_'+'.jpg'), imgs[i,j,:,:])
    print('ok')


if __name__ == '__main__':
    save_mask_all()
    #pose_vec2points()
    #draw()
    imsave('mask_all.npy')

    # This import registers the 3D projection, but is otherwise unused.

