
import numpy as np
from path import Path
import scipy.misc
import os

'''
    基本等于sequence—folders
'''
class MCLoader(object):
    def __init__(self,
                 dataset_dir,
                 static_frames_file=None,
                 img_height=256,
                 img_width=512,
                 min_speed=2,
                 gt_depth=False):
        dir_path = Path(__file__).realpath().dirname()
        test_scene_file = dir_path / 'visdrone_test_scenes.txt'
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]#最后要留空白行?
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height#
        self.img_width = img_width#out-put size
        self.cam_ids = ['00']#单目
        self.min_speed = min_speed
        self.from_speed = None
        self.gt_depth = gt_depth
        self.gt_pose = False

        self.collect_train_folders()
        self.nums_scenes = 0
        self.subfile = {'imgs':'imgs',
                    'depths':'depths',
                    'camfile':'cam.txt',
                    'pose':'pose.txt'}
        print('init ok')



    #public static method
    def collect_train_folders(self):
        self.scenes = []
        drive_set = self.dataset_dir.dirs()
        for dr in drive_set:
            if dr.name not in self.test_scenes:#如果不在test列表内,用来处理. 实质是一个禁忌表, 表内的用来inference
                self.scenes.append(dr)

    #public called
    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_name):
            sample = {}
            sample['f_name'] = frame_name
            sample['imgs'] = self.load_image(scene_data, i)[0]
            if self.gt_depth:
                #sample['depth'] = self.generate_depth_map(scene_data, i)
                sample['depth']= self.load_depth(scene_data, i)[0]
            return  sample

        for i in range(scene_data['nums_frame']):
            frame_name = scene_data['frame_names'][i]
            yield construct_sample(scene_data, i, frame_name)



    def load_depth(self,scene_data, tgt_idx,format = '.png'):
        depth_file = scene_data['root'] / self.subfile['depths'] / scene_data['frame_names'][tgt_idx] + format
        depth = scipy.misc.imread(depth_file)
        zoom_y = self.img_height / depth.shape[0]# 256/600
        zoom_x = self.img_width / depth.shape[1]#512 / 800
        depth = scipy.misc.imresize(depth, (self.img_height, self.img_width))
        return depth, zoom_x, zoom_y

    def load_image(self, scene_data, tgt_idx,format = '.jpg'):
        #img_file = scene_data['dir']/scene_data['frame_id'][tgt_idx]+format
        img_file = scene_data['root']/self.subfile['imgs']/scene_data['frame_names'][tgt_idx]+format
        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def collect_scenes(self, root):

        #oxts = sorted((drive / 'oxts' / 'data').files('*.txt'))
        scene_data = {'root': root,
                      'speed': [],
                      'rel_path': root.name,
                      'frame_names':[],
                      'nums_frame':0}#这里就不用cam号了，单目

        img_files = sorted((root/self.subfile['imgs']).files('*.jpg'))
        #1.load imgs
        for n, f in enumerate(img_files):

            #scene_data['frame_id'].append('{:07d}'.format(n+1))#frameid严格七位数，不然没法读取，切帧的时候需要注意
            scene_data['frame_names'].append(f.stem)

        scene_data['nums_frame'] = len(img_files)

        sample = self.load_image(scene_data, 0)
        if sample is None:
            print('load imgs failed')
            return []

        #2. load intrinsics
        if(os.path.exists(root/self.subfile['camfile'])==False):#内参矩阵先用kitti的，其实一样
            scene_data['intrinsics']=np.array([241.674463,0.,204.168010,
                                               0.,246.284868,59.000832,
                                               0.,0.,1.])
        else:
            scene_data['intrinsics']  = np.genfromtxt(root/self.subfile['camfile'], delimiter=',')#分隔符空格

        return scene_data

    def generate_depth_map(self, scene_data, tgt_idx):
       #load depth map

        return None




        return depth
