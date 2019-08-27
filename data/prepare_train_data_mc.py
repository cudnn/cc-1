from __future__ import division
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                   # help='path to original dataset',default='/home/roit/datasets/kitti_small/')
                   #help = 'path to original dataset', default = '/home/roit/datasets/VisDrone_prep_input')
                   help = 'path to original dataset', default = '/home/roit/datasets/MC')

parser.add_argument("--dataset-format", type=str, default='minecraft', choices=["kitti", "cityscapes","visdrone",'minecraft'])
parser.add_argument("--static-frames", default=None,
                    help="list of imgs to discard for being static, if not set will discard them based on speed \
                    (careful, on KITTI some frames have incorrect speed)")
parser.add_argument("--with-gt", action='store_true',default=True,
                    help="If available (e.g. with KITTI), will store ground truth along with images, for validation")
parser.add_argument("--height", type=int, default=256, help="image height")
#parser.add_argument("--dump-root", type=str, default='/home/roit/datasets/kitti_256512', help="Where to dump the data")
parser.add_argument("--dump-root", type=str, default=None, help="Where to dump the data")

parser.add_argument("--width", type=int, default=512, help="image width")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

args = parser.parse_args()


def dump_example(scene_path):
    '''
    生成处理后数据文件
    :param scene:
    :return:
    '''
    global data_loader
    scene_data = data_loader.collect_scenes(scene_path)

    #abs path
    dump_dir = args.dump_root/scene_data['rel_path']

    dump_imgs = dump_dir/'imgs'
    dump_depths = dump_dir / 'depths'

    dump_dir.makedirs_p()
    dump_imgs.makedirs_p()
    dump_depths.makedirs_p()

    #inc
    intrinsics = scene_data['intrinsics'].reshape(3,3)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    #camfile
    dump_cam_file = dump_dir/'cam.txt'
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

    #imgs and depth
    for sample in data_loader.get_scene_imgs(scene_data=scene_data):#该函数是生成器, 使用yield返回而非return

        dump_img_file = dump_imgs/'{}.jpg'.format(sample['f_name'])

        scipy.misc.imsave(dump_img_file, sample['imgs'])

        if data_loader.gt_depth:
            dump_depth_file = dump_depths/'{}.npy'.format(sample['f_name'])
            np.save(dump_depth_file, sample['depth'])
        '''
        if sample['pose']:
            pass
        if sample['flow']:
            pass
        
        '''

    if len(dump_imgs.files('*.jpg')) < 3:
        dump_imgs.rmtree()


def main():
    dataset_dir = Path(args.dataset_dir)
    if args.dump_root==None:
        args.dump_root = Path('/home/roit/datasets')/dataset_dir.stem+'_256512'
    else:
        args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader


    if args.dataset_format == 'visdrone':
        from  visdrone_raw_loader import VisDroneRawLoader
        data_loader = VisDroneRawLoader(args.dataset_dir,
                                     static_frames_file=args.static_frames,
                                     img_height=args.height,
                                     img_width=args.width,
                                     get_gt=args.with_gt)
    if args.dataset_format == 'minecraft':
        from minecraft_loader import MCLoader
        data_loader = MCLoader(args.dataset_dir,
                                        static_frames_file=args.static_frames,
                                        img_height=args.height,
                                        img_width=args.width,
                                        gt_depth=args.with_gt)


    print('Retrieving frames')#joblib.delayed


    #Parallel(n_jobs=args.num_threads)(delayed(dump_example)(scene) for scene in tqdm(data_loader.scenes))
    totally_imgs = 0
    for scene_path in tqdm(data_loader.scenes):
        #dump_example(scene_path)
        scene_data = data_loader.collect_scenes(scene_path)

        # abs path
        dump_dir = args.dump_root / scene_data['rel_path']

        dump_imgs = dump_dir / 'imgs'
        dump_depths = dump_dir / 'depths'

        dump_dir.makedirs_p()
        dump_imgs.makedirs_p()
        dump_depths.makedirs_p()

        # inc
        intrinsics = scene_data['intrinsics'].reshape(3, 3)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # camfile
        dump_cam_file = dump_dir / 'cam.txt'
        with open(dump_cam_file, 'w') as f:
            f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

        # imgs and depth
        for sample in data_loader.get_scene_imgs(scene_data=scene_data):  # 该函数是生成器, 使用yield返回而非return

            dump_img_file = dump_imgs / '{}.jpg'.format(sample['f_name'])

            scipy.misc.imsave(dump_img_file, sample['imgs'])

            if data_loader.gt_depth:
                dump_depth_file = dump_depths / '{}.npy'.format(sample['f_name'])
                np.save(dump_depth_file, sample['depth'])
            '''
            if sample['pose']:
                pass
            if sample['flow']:
                pass

            '''

        if len(dump_imgs.files('*.jpg')) < 3:
            dump_imgs.rmtree()

        totally_imgs+=scene_data['nums_frame']
    #end for

    print('get {} scences and totally {} imgs'.format(len(data_loader.scenes), totally_imgs))

    # Split into train/val
    print('Generating train val lists')
    np.random.seed(8964)
    subfolders = args.dump_root.dirs()
    with open(args.dump_root / 'train.txt', 'w') as tf:
        with open(args.dump_root / 'val.txt', 'w') as vf:
            for s in tqdm(subfolders):
                if np.random.random() < 0.1:#随机分割
                    vf.write('{}\n'.format(s.name))
                else:
                    tf.write('{}\n'.format(s.name))
                    # remove useless groundtruth data for training comment if you don't want to erase it
                    for gt_file in s.files('*.npy'):
                        gt_file.remove_p()


if __name__ == '__main__':
    main()
    print('over')
