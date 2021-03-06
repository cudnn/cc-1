import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import torch
from .utils import load_depth,load_as_float
#with depth_gt

#with depth_gt
class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/imgs/
                    0000000.jpg
                    0000001.jpg
                    0000002.jpg
                    ...

        root/scene_1/depths/
                    0000000.npy
                    0000001.npy
                    0000002.npy



        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None,depth_format='png'):

        def crawl_folders(folders_list):
            imgs = []
            depth = []
            for folder in folders_list:
                #all are paths

                current_imgs = sorted((folder/'imgs').files('*.png'))
                if depth_format=='npy':
                    current_depth = sorted((folder/'depths').files('*.npy'))
                elif depth_format=='png':
                    current_depth = sorted((folder/'depths').files('*.png'))
                else:
                    current_depth=None
                    print('error')
                    exit(-1)

                imgs.extend(current_imgs)
                depth.extend(current_depth)
            return imgs, depth


        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.imgs, self.depth = crawl_folders(self.scenes)
        self.transform = transform

    def __getitem__(self, index):
        img = load_as_float(self.imgs[index])
        #depth = np.load(self.depth[index]).astype(np.float32)
        depth = load_depth(self.depth[index])
        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]
        return img, depth

    def __len__(self):
        return len(self.imgs)

#查看连续帧
class ValidationSetSeq(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None):
        def crawl_folders_seq(folders_list, sequence_length):
            imgs1 = []
            imgs2 = []
            depth = []
            for folder in folders_list:
                current_imgs = sorted(folder.files('*.png'))
                current_imgs1 = current_imgs[:-1]
                current_imgs2 = current_imgs[1:]
                current_depth = []
                for (img1, img2) in zip(current_imgs1, current_imgs2):
                    d = img1.dirname() / (img1.name[:-4] + '.npy')
                    assert (d.isfile()), "depth file {} not found".format(str(d))
                    depth.append(d)
                imgs1.extend(current_imgs1)
                imgs2.extend(current_imgs2)
                depth.extend(current_depth)
            return imgs1, imgs2, depth

        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.imgs1, self.imgs2, self.depth = crawl_folders_seq(self.scenes)
        self.transform = transform

    def __getitem__(self, index):
        img1 = load_as_float(self.imgs1[index])
        img2 = load_as_float(self.imgs2[index])
        depth = np.load(self.depth[index]).astype(np.float32)
        if self.transform is not None:
            img, _ = self.transform([img1, img2], None)
            img1, img2 = img[0], img[1]
        return (img1, img2), depth

    def __len__(self):
        return len(self.imgs1)


