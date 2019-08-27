
import numpy as np
from path import Path
import scipy.misc
import os

'''
    基本等于sequence—folders
'''
class VisDroneRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 static_frames_file=None,
                 img_height=256,
                 img_width=512,
                 min_speed=2,
                 get_gt=False):
        dir_path = Path(__file__).realpath().dirname()
        test_scene_file = dir_path / 'visdrone_test_scenes.txt'
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]#最后要留空白行?
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['00']#单目
        self.min_speed = min_speed
        self.from_speed = None
        self.get_gt = get_gt
        self.collect_train_folders()
        print('init ok')



    #static method
    def collect_train_folders(self):
        self.scenes = []
        drive_set = self.dataset_dir.dirs()
        for dr in drive_set:
            if dr.name not in self.test_scenes:#如果不在test列表内,用来处理. 实质是一个禁忌表, 表内的用来inference
                self.scenes.append(dr)


    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = [self.load_image(scene_data, i)[0], frame_id]
            if self.get_gt:
                #sample.append(self.generate_depth_map(scene_data, i))#no gt
                pass
            return sample
        for i in range(len(scene_data['frame_id'])):
            frame_id = scene_data['frame_id'][i]
            yield construct_sample(scene_data, i, frame_id)




    def load_image(self, scene_data, tgt_idx,format = '.jpg'):
        #img_file = scene_data['dir']/scene_data['frame_id'][tgt_idx]+format
        img_file = scene_data['dir']/scene_data['frame_name'][tgt_idx]+format
        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def collect_scenes(self, drive):
        train_scenes = []
        for c in self.cam_ids:
            img_files = sorted((drive).files('*.jpg'))
            #oxts = sorted((drive / 'oxts' / 'data').files('*.txt'))
            scene_data = {'cid': c, 'dir': drive, 'speed': [], 'frame_id': [], 'rel_path': drive.name,'frame_name':[] }#这里就不用cam号了，单目
            for n, f in enumerate(img_files):

                scene_data['frame_id'].append('{:07d}'.format(n+1))#frameid严格七位数，不然没法读取，切帧的时候需要注意
                scene_data['frame_name'].append(f.stem)
            sample = self.load_image(scene_data, 0)
            if sample is None:
                return []
            #scene_data['P_rect'] = self.get_P_rect(scene_data, sample[1], sample[2])
            if(os.path.exists(drive/'cam.txt')==False):#内参矩阵先用kitti的，其实一样
                scene_data['intrinsics']=np.array([241.674463,0.,204.168010,
                                                   0.,246.284868,59.000832,
                                                   0.,0.,1.])
            else:
                scene_data['intrinsics']  = np.genfromtxt(drive/'cam.txt', delimiter=',')#分隔符空格

            train_scenes.append(scene_data)
        return train_scenes

