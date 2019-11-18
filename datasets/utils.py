import numpy as np
from scipy.misc import imread
import torch
import random

def load_depth(path,format='png'):
    if format=='npy':

        tgt_depth = np.expand_dims(np.load(path), axis=0)
    elif format=='png':
        tgt_depth =np.expand_dims( imread(path), axis=0)
    return torch.from_numpy(tgt_depth).float() / 255


def load_as_float(path):
    return imread(path).astype(np.float32)


#这里跳帧
def crawl_folders(folders_list, sequence_length,shuffle = False):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        for folder in folders_list:
            intrinsics = np.genfromtxt(folder/'cam.txt', delimiter=',')#分隔符空格
            intrinsics = intrinsics.astype(np.float32).reshape((3, 3))
            imgs = sorted(folder.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in range(-demi_length, demi_length + 1):
                    if j != 0:
                        sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        if shuffle:
            random.shuffle(sequence_set)
        else:
            pass
        return sequence_set

#增加跳帧功能
def crawl_folders2(folders_list, sequence_length,interval_frame=0,sample_gap = 0, shuffle=False):
    sequence_set = []
    demi_length = (sequence_length - 1) // 2
    for folder in folders_list:
        intrinsics = np.genfromtxt(folder / 'cam.txt', delimiter=',')  # 分隔符空格
        intrinsics = intrinsics.astype(np.float32).reshape((3, 3))
        imgs = sorted(folder.files('*.jpg'))
        if len(imgs) < sequence_length:#frame太少, 放弃这个folder
            continue
        #插孔抽出
        for i in range(len(imgs)):
            if i % (interval_frame+1) != 0 :
                imgs[i]=None
        while None in imgs:
            imgs.remove(None)

        for i in range(demi_length, len(imgs) - demi_length):#在一个folder里
            sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
            for j in range(-demi_length, demi_length + 1):
                if j != 0:
                    sample['ref_imgs'].append(imgs[i + j])
            sequence_set.append(sample)



    if shuffle:
        random.shuffle(sequence_set)
    else:
        pass

    # 插空减少样本，提升训练速度
    for i in range(len(sequence_set)):
        if i % (sample_gap+1) != 0:
            sequence_set[i] = None
    while None in sequence_set:
        sequence_set.remove(None)


    return sequence_set

#跳帧且加载gt
def crawl_folders_gt(folders_list, sequence_length,interval_frame=0,sample_gap = 0,depth_format='png', shuffle=False):
    sequence_set = []
    demi_length = (sequence_length - 1) // 2
    for folder in folders_list:
        intrinsics = np.genfromtxt(folder / 'cam.txt', delimiter=',')  # 分隔符空格
        intrinsics = intrinsics.astype(np.float32).reshape((3, 3))

        depths_folder = folder / 'depths'
        imgs_folder = folder/'imgs'

        # all paths
        imgs = sorted(imgs_folder.files('*.png'))

        if depth_format=='npy':
            depths = sorted(depths_folder.files('*.npy'))
        elif depth_format=='png':
            depths = sorted(depths_folder.files('*.png'))

        if len(imgs) < sequence_length:#frame太少, 放弃这个folder
            continue


        #插孔抽出
        for i in range(len(imgs)):
            if i % (interval_frame+1) != 0 :
                imgs[i]=None
                depths[i]=None
                #pose[i]=None
                #flow[i]=None
        while None in imgs:
            imgs.remove(None)
            depths.remove(None)


        for i in range(demi_length, len(imgs) - demi_length):#在一个folder里
            sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],'tgt_depth':depths[i]}

            #ref imgs precess
            for j in range(-demi_length, demi_length + 1):
                if j != 0:
                    sample['ref_imgs'].append(imgs[i + j])
            #flow precess

            #pose precess

            sequence_set.append(sample)



    if shuffle:
        random.shuffle(sequence_set)
    else:
        pass

    # 插空减少样本，提升训练速度
    for i in range(len(sequence_set)):
        if i % (sample_gap+1) != 0:
            sequence_set[i] = None
    while None in sequence_set:
        sequence_set.remove(None)


    return sequence_set

