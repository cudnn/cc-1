from __future__ import division
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from minecraft_loader import MCLoader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                   # help='path to original dataset',default='/home/roit/datasets/kitti_small/')
                   #help = 'path to original dataset', default = '/home/roit/datasets/VisDrone_prep_input')
                   #help = 'path to original dataset', default = '/home/roit/datasets/MC')
                   help = 'path to original dataset', default = '/home/roit/datasets/MC')
parser.add_argument('--prep_list',default='prep_train_data_mc.txt')
parser.add_argument("--dataset-format", type=str, default='minecraft', choices=["kitti", "cityscapes","visdrone",'minecraft'])

parser.add_argument("--with-gt", default=True,help="")
#output
parser.add_argument("--height", type=int, default=128, help="image height")
parser.add_argument("--width", type=int, default=192, help="image width")
parser.add_argument("--postfix",default='_128192')

#parser.add_argument("--dump-root", type=str, default='/home/roit/datasets/kitti_256512', help="Where to dump the data")
parser.add_argument("--dump-root", type=str, default=None, help="Where to dump the data")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")
parser.add_argument("--depth_format",default='png',choices=['npy','png'],help = "depth saving format")

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
        args.dump_root = Path('/home/roit/datasets')/dataset_dir.stem+args.postfix
    else:
        args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader



    data_loader = MCLoader(args.dataset_dir,
                            static_frames_file=None,
                            img_height=args.height,
                            img_width=args.width,
                            gt_depth=args.with_gt)


    print('Retrieving frames')#joblib.delayed


    #Parallel(n_jobs=args.num_threads)(delayed(dump_example)(scene) for scene in tqdm(data_loader.scenes))
    totally_imgs = 0

    #表里面的才处理
    seqs = open(args.prep_list)
    seq_names = []
    for line in seqs:
        if line[0] != '#':
            seq_names.append(line.strip('\n'))
    seq_names.sort()

    data_loader.scenes=[]
    for s in seq_names:
        data_loader.scenes.append(dataset_dir/s)


    for scene_path in tqdm(data_loader.scenes) :
        #dump_example(scene_path)
        scene_data = data_loader.collect_scenes(scene_path)

        dump_dir = args.dump_root / scene_data['rel_path']
        dump_imgs = dump_dir / 'imgs'
        dump_depths = dump_dir / 'depths'

        dump_dir.makedirs_p()
        dump_imgs.makedirs_p()
        dump_depths.makedirs_p()

        # inc
        intrinsics = scene_data['intrinsics'].reshape(3, 3)
        #fx = intrinsics[0, 0]
        #fy = intrinsics[1, 1]
        #cx = intrinsics[0, 2]
        #cy = intrinsics[1, 2]

        cx=args.width/2
        cy=args.height/2
        fx = 91.42  # fov = 70d, cx/tan(35d)
        fy = 137.14

        # camfile
        dump_cam_file = dump_dir / 'cam.txt'
        with open(dump_cam_file, 'w') as f:
            f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

        # imgs and depth




        for sample in data_loader.get_scene_imgs(scene_data=scene_data):  # 该函数是生成器, 使用yield返回而非return
            #imgs
            dump_img_file = dump_imgs / '{}.png'.format(sample['f_name'])
            scipy.misc.imsave(dump_img_file, sample['imgs'])
            #depths
            if data_loader.gt_depth:
                if args.depth_format=='npy':
                    dump_depth_file = dump_depths / '{}.npy'.format(sample['f_name'])
                    np.save(dump_depth_file, sample['depth'])
                if args.depth_format=='png':
                    depth_map = dump_depths / '{}.png'.format(sample['f_name'])
                    scipy.misc.imsave(depth_map, sample['depth'])

            '''
            if sample['pose']:
                pass
            if sample['flow']:
                pass

            '''



        totally_imgs+=scene_data['nums_frame']
    #end for

    print('get {} scences and totally {} imgs'.format(len(data_loader.scenes), totally_imgs))

    # Split into train/val




def Split():
    print('Generating train val lists')
    dump_root = Path(args.dataset_dir+args.postfix)
    np.random.seed(8964)
    subfolders = dump_root.dirs()
    with open(dump_root / 'train.txt', 'w') as tf:
        with open(dump_root / 'val.txt', 'w') as vf:
            for s in tqdm(subfolders):
                if np.random.random() < 0.1:  # 随机分割
                    vf.write('{}\n'.format(s.stem))
                else:
                    tf.write('{}\n'.format(s.stem))
                    # remove useless groundtruth data for training comment if you don't want to erase it
                    for gt_file in s.files('*.npy'):
                        gt_file.remove_p()


    pass

if __name__ == '__main__':
    main()
    Split()
    print('over')
