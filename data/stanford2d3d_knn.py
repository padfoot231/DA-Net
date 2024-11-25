from envmap import EnvironmentMap
from envmap import rotation_matrix
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import cv2
import random
import numpy as np
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.transform import resize
import torchvision
import json
import pickle as pkl 
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn

from utils_curve import concentric_dic_sampling_origin


T = transforms.ToTensor()
pil = transforms.ToPILImage()

#while training sparseconv try normalizing depth (divide by max_depth of each image)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = torch.rot90(image, k, dims=[1, 2])
    label = torch.rot90(label, k, dims=[0, 1])
    axis = np.random.randint(1, 3)
    image = torch.flip(image, dims=[1,2])
    label = torch.flip(label, dims=[0,1])
    return image, label


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img
    
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = transforms.functional.rotate(image, angle)
    label = transforms.functional.rotate(label.reshape(1, label.shape[0], label.shape[1]), angle)
    return image, label[0]

#'png'
def load_color(filename: str) -> torch.Tensor:    
    return {'color': torchvision.io.read_image(filename) / 255.0 }    

#'depth' and 'exr'
# def load_depth(filename: str, max_depth: float=8.0) -> torch.Tensor:
#     depth_filename = filename.replace('.png', '.exr')
#     depth = torch.from_numpy(
#         cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
#     ).unsqueeze(0)
#     #NOTE: add a micro meter to allow for thresholding to extact the valid mask
#     depth[depth > max_depth] = max_depth + 1e-6 #replace inf value by max_depth #without any impact on wood 
#     return {
#         'depth': depth
#     }

def sph2cart(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

#rad
def compute_focal(fov, xi, width):
    return width / 2 * (xi + np.cos(fov/2)) / np.sin(fov/2)

#order 0:nearest 1:bilinear
def warpToFisheye(pano,outputdims,viewingAnglesPYR=[np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)],xi=0.9, fov=150, order=1):

    outputdims1=outputdims[0]
    outputdims2=outputdims[1]
   
    pitch, yaw, roll = np.array(viewingAnglesPYR)
    #print(pano.shape)
    # breakpoint()
    e = EnvironmentMap(pano, format_='latlong')
    e = e.rotate(rotation_matrix(yaw, -pitch, -roll).T)
    r_max = max(outputdims1/2,outputdims2/2)

    h= min(outputdims1,outputdims2)
    f = compute_focal(np.deg2rad(fov),xi,h)
    
    t = np.linspace(0,fov/2, 100)
   

    #test spherical
    # print('xi  {}, f {}'.format(xi,f))
    theta= np.deg2rad(t)
    funT = (f* np.sin(theta))/(np.cos(theta)+xi)
    funT= funT/r_max


    #creates the empty image
    [u, v] = np.meshgrid(np.linspace(-1, 1, outputdims1), np.linspace(-1, 1, outputdims2))
    r = np.sqrt(u ** 2 + v ** 2)
    phi = np.arctan2(v, u)
    validOut = r <= 1
    # interpolate the _inverse_ function!
    fovWorld = np.deg2rad(np.interp(x=r, xp=funT, fp=t))
    # fovWorld = np.pi / 2 - np.arccos(r)
    FOV = np.rad2deg((fovWorld))

    el = fovWorld + np.pi / 2

    # convert to XYZ
    #ref
    x, y, z = sph2cart(phi, fovWorld + np.pi / 2, 1)

    x = -x
    z = -z

    #return values in [0,1]
    #the source pixel from panorama 
    [u1, v1] = e.world2image(x, y, z)
    # breakpoint()
    # Interpolate
    #validout to set the background to black (the circle part)
    eOut= e.interpolate(u1, v1, validOut, order)
    #eOut= e.interpolate(u1, v1)

    return eOut.data, f

def segm_transform(segm):
    # to tensor, -1 to 149
    segm = torch.from_numpy(np.array(segm)).long()
    return segm
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        #wether add the mask in the dataloader or no aug
        if random.random() > 0.5: #corrupt the mask (rotation 90)
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5: #corrupt the mask (random rotation)
            image, label = random_rotate(image, label)
        
        """
        x, y, c  = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        """
        
        image = torch.from_numpy(image.astype(np.float32)) #.unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        return sample

#distortion params: polynomial K1,k2,k3,k4
"""
FV: Front CAM
RV: Rear CAM
MVL: Mirror Left CAM
MVR: Mirror Right CAM
"""
Distortion= {
'MVL': np.array([342.234, -18.6659, 23.1572, 4.28064]),
'FV' : np.array([341.725, -26.4448, 32.7864, 0.50499]),
'MVR' : np.array([340.749, -16.9704, 20.9909, 4.60924]),
'RV' : np.array([342.457, -22.4772, 28.5462, 1.78203])
}


#woodscape
#mean= [0.1867, 0.1694, 0.1573]
#std= [0.2172, 0.2022, 0.1884]


#CVRGpano
# mean = [0.3742, 0.3776, 0.3574]
# std = [0.2792, 0.2748, 0.2866]

#CVRGpano highdistort
mean = [0.3977, 0.4041, 0.4012]
std = [0.2794, 0.2807, 0.2894]

#Matterport
# mean= [0.2217, 0.1939, 0.1688]
# std= [0.1884, 0.1744, 0.1835]

#normalize = None
normalize= transforms.Normalize(
            mean= mean ,
            std= std )
#normalize = None
class Stanford_da_knn(Dataset):
    def __init__(self, base_dir, split, grp, n_rad,  xi=0.8, model= "spherical", img_size = 128, fov = 170, high=0.0, low=0.35, transform=None):
        self.fov = fov
        self.transform = transform  # using transform in torch!
        self.split = split
        self.model= model
        self.img_size= img_size
        self.n_rad = 5
        self.data_dir = base_dir
        self.calib = None
        self.low = low
        self.xi = xi
        self.high = high
        self.subdiv = ((img_size*n_rad)//2, (img_size*n_rad)//2)
        
        if split == 'train':
            with open(base_dir + '/train.pkl', 'rb') as f:
                img = pkl.load(f)
            with open(base_dir + '/train_sem.pkl', 'rb') as f:
                sem = pkl.load(f)
            with open(base_dir + '/12NN_3_128_el.pkl', 'rb') as f:
                self.cl = pkl.load(f)

        elif split == 'val':
            with open(base_dir + '/val.pkl', 'rb') as f:
                img = pkl.load(f)
            with open(base_dir + '/val_sem.pkl', 'rb') as f:
                sem = pkl.load(f)
            with open(base_dir + '/12NN_3_128_el.pkl', 'rb') as f:
                self.cl = pkl.load(f)

        elif split == 'test':
            with open(base_dir + '/test.pkl', 'rb') as f:
                img = pkl.load(f)
                img = img
            with open(base_dir + '/test_sem.pkl', 'rb') as f:
                sem = pkl.load(f)
                sem = sem

            with open(base_dir + '/deg_stan.pkl', 'rb') as f:
                deg = pkl.load(f)
                self.deg = deg
            with open(base_dir + '/12NN_5_128_el_test.pkl', 'rb') as f:
                self.cl = pkl.load(f)



        self.img = img #['1LXtFkjw3qL/85_spherical_1_emission_center_0.png'] #data[:5]
        self.sem = sem


    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        
        # elif self.model== "spherical":
        # breakpoint()
        if self.split == 'train' or self.split == 'val':
            # breakpoint()/
            img_path = self.data_dir + '/' + self.img[idx]
            sem_path = self.data_dir + '/' + self.sem[idx]
        elif self.split == 'test':
            img_path = self.data_dir + '/' + self.img[idx]
            sem_path = self.data_dir + '/' + self.sem[idx]
                
                
        # image= load_color(img_path)['color']
        # print(len(self.dist))
        image = Image.open(img_path)
        image = image.resize((1024, 512), resample = 2)
        image = np.array(image)
        image = image[80:-80]
        image = Image.fromarray(image)
        image = image.resize((704, 352), resample = 2)
        # image.save("pano.png")
        image = np.array(image)
        # breakpoint()
        # im.save("pano.png")
        # image = resize(image, (704, 352), order = 1).astype(np.uint8)
        # image = np.transpose(image, (1, 0, 2))
        # im = Image.fromarray(image)
        # im.save("pano.png")
        segm= Image.open(sem_path).convert('L')
        segm = segm.resize((1024, 512), resample = 0)
        segm = np.array(segm)
        segm = segm[80:-80]
        segm = Image.fromarray(segm)
        segm = segm.resize((704, 352), resample = 0)
        segm = np.array(segm)
        # from matplotlib import pyplot as plt
        # plt.imshow(segm)
        # plt.savefig("segm.png")
        # segm= resize(segm, (704, 352), order = 0)
        
        
        segm = segm.reshape(352, 704, 1)
        # mat_path= img_path.replace('png','npy')
        #cl= np.load(mat_path)

        # image=image.permute(1,2,0)
        # segm=segm.permute(1,2,0)
        if self.model == "spherical":
            # h= self.img_size
            h = 512
            fov=self.fov
            # print("field of view", fov)
            if self.split=='train' or self.split=='val':
                i = random.randint(0, len(self.img)-1)
                cl = self.cl[i][0]
                xi = self.cl[i][2]
                # print(xi)
                deg = random.uniform(0, 360)
                # print(xi, fov, deg)
            elif self.split=='test':
                xi= self.xi
                # print(xi)
                cl = self.cl[xi][0]
                deg = self.deg[idx]


            # print(xi, deg)
            image, f = warpToFisheye(image[:, :, :3], viewingAnglesPYR=[np.deg2rad(0), np.deg2rad(deg), np.deg2rad(0)], outputdims=(h,h),xi=xi, fov=fov, order=1)
            segm,_= warpToFisheye(segm, viewingAnglesPYR=[np.deg2rad(0), np.deg2rad(deg), np.deg2rad(0)], outputdims=(h,h),xi=xi, fov=fov, order=0)
            dist= np.array([xi, f/(h/self.img_size), np.deg2rad(fov)]).astype(np.float32)
      
            segm = segm.astype(np.uint8)


        image = resize(image, (self.img_size,self.img_size), order = 1).astype(np.uint8)
        label= resize(segm, (self.img_size,self.img_size), order = 0)

        image = T(image)

        #################################### image to DA transformation ############################################################

        image = image.unsqueeze(0)
        # pil(image[0]).save("polar_f.png")
        dist = torch.tensor(dist).unsqueeze(-1)
        # breakpoint()
        xc, yc = concentric_dic_sampling_origin(
            subdiv=self.subdiv,
            img_size=(self.img_size, self.img_size),
            distortion_model = "spherical",
            D = dist)

        B, n_r, n_a = xc.shape
        x_ = xc.reshape(B, n_r, n_a, 1).float()
        x_ = x_/(self.img_size//2)
        y_ = yc.reshape(B, n_r, n_a, 1).float()
        y_ = y_/(self.img_size//2)

        out = torch.cat((y_, x_), dim = 3)
        # out = out.cuda()
        pil(image[0]).save("fish_8.png")
        # import pdb;pdb.set_trace()

        # image = nn.functional.grid_sample(image, out, align_corners = True)
        # # print(image.shape)
        # image = nn.functional.interpolate(image, size=(self.img_size*self.n_rad, self.img_size*self.n_rad), mode='bilinear', align_corners = True)
        # breakpoint()
        # pil(image[0]).save("cds_8.png")
        # import pdb;pdb.set_trace()

        #################################### image to DA transformation ############################################################
        
        label = segm_transform(label)
        ############################################# masks ############################################################
        res = 1024
        # breakpoint()
        # cartesian = torch.cartesian_prod(
        #     torch.linspace(-1, 1, res),
        #     torch.linspace(1, -1, res)
        # ).reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)

        cartesian = torch.meshgrid(
            torch.linspace(-1, 1, res),
            torch.linspace(1, -1, res), indexing='ij'
        )
        cartesian = torch.stack((cartesian[0], cartesian[1]), dim=-1).transpose(2, 1).transpose(1, 0).transpose(1, 2)
        # breakpoint()
        
        
        # .reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)

        radius = cartesian.norm(dim=0)
        mask = (radius > 0.0) & (radius < 1) 
        mask1 = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0) * 1.0, (self.img_size), mode="nearest")
        ############################################# masks ############################################################
        #sample = {'image': image, 'label': label, 'path':b_path.replace('png','npy')}
        sample = {'image': image, 'label': label}
        # print(self.transform)
        # if self.transform:
        if None:    
            sample = self.transform(sample)
        else:
            sample['image']= image.type(torch.float32)
            sample['label']= label.type(torch.uint8)
        

        one_hot = F.one_hot(sample['label'].to(torch.int64), num_classes=14).to(torch.float32)
        # sample['image']= sample['image'].permute(2,0,1)
        # sample['label']= sample['label']
        # breakpoint()
        
        # print(sample['label'].shape)


        # sample['label']= sample['label'].squeeze(0)

        if normalize is not None:
            sample['image']= normalize(sample['image'])

        sample['image'] = nn.functional.grid_sample(sample['image'], out, align_corners = True)
        # print(image.shape)
        sample['image'] = nn.functional.interpolate(sample['image'], size=(self.img_size*self.n_rad, self.img_size*self.n_rad), mode='bilinear', align_corners = True)

        sample['one_hot'] = one_hot
        sample['mask'] = mask1[0].to(torch.long)

        #print(sample.keys())
        # breakpoint()
        return sample['image'][0], sample['label'][:, :, 0],  cl, sample['mask'], one_hot[:, :, 0]

def get_mean_std(base_dir ):
    db= CVRG(base_dir, split="train", transform=None)
    print(len(db))
    #sample= db.__getitem__(0)
    #print(sample['image'].shape)
    #print(sample['label'].shape)
    #print(sample['dist'].shape)
    loader = DataLoader(db, batch_size=len(db), shuffle=False,num_workers=0)
    im_lab_dict = next(iter(loader))
    images, labels, dist, cls, mask, one_hot = im_lab_dict
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    print("mean",mean)
    print("std", std)
    return mean , std


if __name__ == "__main__":
    root_path= '/localscratch/prongs.52548387.0/data_new/semantic2d3d'
    # breakpoint()
    db= Stanford_da(root_path, split="test", grp="vlow", n_rad=3, transform=None)
    
    # mean,std= get_mean_std(root_path)
    db[20]
    breakpoint()
    print("end")
    
    data_loader_train = torch.utils.data.DataLoader(
        db,
        batch_size=8,
        num_workers=3,
        pin_memory=True,
        drop_last=True,
    )
    # mean,std= get_mean_std(root_path)
    for idx, (samples, targets, dist, cls, mask, one_hot) in enumerate(data_loader_train):
        breakpoint()
        print(samples)

#     mean tensor([0.9735, 1.2531, 1.3141])
# std tensor([1.4566, 1.5787, 1.5510])