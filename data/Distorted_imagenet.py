from dis import dis
import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
from utils import distort_image
import random
import warnings
import torch 
import torchvision.transforms as T
from glob import glob
import pickle as pkl
import torch.nn as nn
from datetime import datetime
import random
from pyinstrument import Profiler
from sphericaldistortion import distort, undistort

profiler = Profiler(interval=0.0001)

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

pil = T.ToPILImage()
t = []

t.append(T.ToTensor())
# t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
trans =  T.Compose(t)


def random_1():
    return 1 if random.random() < 0.5 else -1



def random_direction_normal(dim, n):
    p = np.abs(np.random.standard_normal((dim, n)))
    norm = np.sqrt(np.sum(p**2, axis=0))
    return p / norm

def random_direction_uniform(dim, n):
    p = np.random.uniform(0, 1, (dim, n))
    norm = np.sqrt(np.sum(p**2, axis=0))
    return p / norm

def random_magnitude_uniform(points, high=1):
    scale = np.random.uniform(0, high, points.shape[1])
    return points * scale

def random_magnitude_custom(points, high=1):
    scale = np.random.uniform(0, high, points.shape[1])**(1/points.shape[0])
    return points * scale




class M_distort(data.Dataset):
    def __init__(self, root, distortion, low, high, transform=None, task='train', target_transform=None, radius_cuts=16, azimuth_cuts=64, in_chans=3, img_size=(64, 64)):
        super(M_distort, self).__init__()
        self.data_path = root
        # self.in_chans = in_chans
        # self.subdiv = (radius_cuts, azimuth_cuts)
        # self.n_radius = 15
        # self.n_azimuth = 15
        self.img_size = img_size
        self.dist = distortion
        self.low = low
        self.high = high 

        with open(self.data_path + '/classes.pkl', 'rb') as f:
            classes = pkl.load(f)
        
        self.classes = classes
        # with open('distortion_file.pkl') as f: ####
        #     distortion  = pkl.load(f)
        # lst = []
        if task == 'train':
            with open(self.data_path + '/train/train.pkl', 'rb') as f:
                data = pkl.load(f)
            # with open(self.data_path + '/train_data.pkl', 'rb') as f:
            #     data = pkl.load(f)
        elif task == 'val':
            with open(self.data_path + '/val/val.pkl', 'rb') as f:
                data = pkl.load(f)
        elif task == 'test_1' or task == 'test_2' or task == 'test_3' or task == 'test_4' or task == 'test_5':
            with open(self.data_path + '/test_cls/test_cls.pkl', 'rb') as f:
                data = pkl.load(f)
            if distortion == 'spherical':
                with open(self.data_path + '/test_cls/' + task  + '.pkl', 'rb') as f:
                    test_dist = pkl.load(f)
                    self.test_dist = test_dist
            elif distortion == 'polynomial':
                with open(self.data_path + '/dist_params.pkl', 'rb') as f:
                    test_dist = pkl.load(f)
                    self.test_dist = test_dist


                # import pdb;pdb.set_trace()

            # with open(self.data_path + '/val_data.pkl', 'rb') as f:
            #     data = pkl.load(f)
        




        # classes = list(data.keys())

        # self.test = data

        # data_img = {}

        # for i in range(len(data)):


        #     data_img[i] = (keys[i], keys[i].split('/')[1])

        # for i in range(len(data)):
        #     for j in range(len(data[classes[i]])):
        #         data_img[idx] = (data[classes[i]][j], i, data[classes[i]][j].split('/')[-2])
        #         idx = idx+1
        # self.test_dist = test_dist
        self.task = task 
        # self.data_img = data_img
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        # # id & label: https://github.com/google-research/big_transfer/issues/7
        # # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
        # self.database = json.load(open(self.ann_path))

    # def _load_image(self, path):
    #     try:
    #         im = Image.open(path)
    #     except:
    #         print("ERROR IMG LOADED: ", path)
    #         random_img = np.random.rand(224, 224, 3) * 255
    #         im = Image.fromarray(np.uint8(random_img))
    #     return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.

            return sampling points for each image and targets :) 
        """
        # import pdb;pdb.set_trace()

        images = Image.open(self.data_path + '/' + self.data[index])

        if self.dist == 'polynomial':
            # print("polynomial")
            if self.task == 'train' or self.task == 'val':
                # print("train")
                points = random_direction_normal(4, 1)
                D = random_magnitude_uniform(points, high=5).T
                D = np.array([33.21885116,5.86361501, 17.73762952, 13.39959067]) + D[0]
            elif self.task == 'test_1' or self.task == 'test_2' or self.task == 'test_3' or self.task == 'test_4' or self.task == 'test_5' or self.task == 'test':
                D = self.test_dist[self.data[index]]
            images = distort_image(images, D)


        elif self.dist == 'spherical':
            # print("polynomial")
            if self.task == 'train' or self.task == 'val':
                xi = random.uniform(self.low, self.high) # change the higher limit depending on the group
            elif self.task == 'test_1' or self.task == 'test_2' or self.task == 'test_3' or self.task == 'test_4' or self.task == 'test_5' or self.task == 'test':
                xi = self.test_dist[self.data[index]]

            # import pdb;pdb.set_trace()
            images, new_f, new_xi, new_fov  = distort(images, xi, f=9)
            D = np.array([new_xi, new_f, new_fov]).astype(np.float32)

        target = int(self.classes.index(self.data[index].split('/')[1]))

        # distortion = self.data_img[index][2]
        # print(distortion)
        # import pdb;pdb.set_trace()
        res = 64
        cartesian = torch.cartesian_prod(
            torch.linspace(-1, 1, res),
            torch.linspace(1, -1, res)
        ).reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)
        radius = cartesian.norm(dim=0)
        mask = (radius > 0.0) & (radius < 1) 
        mask1 = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0) * 1.0, (res), mode="area")
        

        if self.transform is not None:
            images = self.transform(images)
        # images = mask1*images
        return images, target, D, self.data[index]

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    profiler.start()
    m = M_distort('/home-local2/akath.extra.nobkp/imagenet_2010', task='val', distortion='spherical', low = 0.5, high = 0.7, transform=trans)
    # import pdb;pdb.set_trace()
    img = m[1]
    img = pil(img[0])
    img.save('undistort.png')
    profiler.stop()
    profiler.print()
    

