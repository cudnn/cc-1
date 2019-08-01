import argparse
import torch
import numpy as np
from path import Path
import scipy.misc
from collections import Counter
import torch.utils.data
import custom_transforms
import models
from datasets.sequence_folders import SequenceFolder


parser = argparse.ArgumentParser(description='Competitive Collaboration training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--data', metavar='DIR',default='/home/roit/datasets/visdrone_raw128512/',help='path to dataset')
parser.add_argument('--sequence_length',default=3)
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')


def main():
    global args, best_error, n_iter
    args = parser.parse_args()

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.Compose([
            custom_transforms.RandomRotate(),
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])


    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    print(len(train_loader))


if __name__ == "__main__":
    main()
    print('ok')
