import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random

import torch

from .utils import load_as_float,load_depth,crawl_folders,crawl_folders2,crawl_folders_gt

from utils import tensor2array
import matplotlib.pyplot as plt

#文件写的有点毛病， gt的transpose不知道如何处理因为一开始的代码就是无监督



class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None,depth_format='png', target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        if train:
            self.samples = crawl_folders_gt(self.scenes, sequence_length,interval_frame=0,sample_gap=0,shuffle=True,depth_format=depth_format)
        else:
            self.samples = crawl_folders_gt(self.scenes, sequence_length,interval_frame=0,sample_gap=0,shuffle=False,depth_format=depth_format)
        self.transform = transform
        print('train-set-init-ok')


    def __getitem__(self, index):
        sample = self.samples[index]

        item = {'tgt_img':None,
                'ref_imgs':None,
                'intrinsics':None,
                'tgt_depth':None,
                'pose':None,
                'flow':None}

        tgt_img = load_as_float(sample['tgt'])#直到现在才载入数据，之前都是路径
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        tgt_depth = load_depth(sample['tgt_depth'])#(h,w,1),tensor[0,1.]

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]

        else:
            intrinsics = np.copy(sample['intrinsics'])



        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics),tgt_depth

    def __len__(self):
        return len(self.samples)
