from envmap import EnvironmentMap
from envmap import rotation_matrix
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
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

T = transforms.ToTensor()

#while training sparseconv try normalizing depth (divide by max_depth of each image)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img
    
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

#'png'
def load_color(filename: str) -> torch.Tensor:    
    return {'color': torchvision.io.read_image(filename) / 255.0 }    

#'depth' and 'exr'
def load_depth(filename: str, max_depth: float=8.0) -> torch.Tensor:
    depth_filename = filename.replace('.png', '.exr')
    depth = torch.from_numpy(
        cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    ).unsqueeze(0)
    #NOTE: add a micro meter to allow for thresholding to extact the valid mask
    depth[depth > max_depth] = max_depth + 1e-6 #replace inf value by max_depth #without any impact on wood 
    return {
        'depth': depth
    }

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
mean = [-1.1691, -1.1035, -0.9122]
std = [0.0057, 0.0061, 0.0060]

#Matterport
# mean= [0.2217, 0.1939, 0.1688]
# std= [0.1884, 0.1744, 0.1835]

#normalize = None
normalize= transforms.Normalize(
            mean= mean ,
            std= std )
#normalize = None
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, model= "spherical", img_size = 128, low = 0.2, high=0.35, xi=0.0, transform=None):
        self.xi = xi
        self.transform = transform  # using transform in torch!
        self.split = split
        self.model= model
        self.img_size= img_size
        self.data_dir = base_dir
        self.calib = None
        self.low = low
        self.high = high
        
        if split == 'train':
            with open(base_dir + '/train.pkl', 'rb') as f:
                data = pkl.load(f)
        elif split == 'val':
            with open(base_dir + '/val.pkl', 'rb') as f:
                data = pkl.load(f)
        elif split == 'test':
            with open(base_dir + '/test.pkl', 'rb') as f:
                data = pkl.load(f)

            # with open(self.data_dir + '/test_calib.pkl', 'rb') as f:
            #     self.calib = pkl.load(f)

        self.data = data #['1LXtFkjw3qL/85_spherical_1_emission_center_0.png'] #data[:5]

        # if self.calib is None and os.path.exists(self.data_dir+ '/calib_gp2.pkl') :
        #     with open(self.data_dir + '/calib_gp2.pkl', 'rb') as f:
        #         self.calib = pkl.load(f)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        b_path= self.data[idx]
        if self.model =="polynomial":
            img_path = self.data_dir + '/rgb_images/' + self.data[idx]
            depth_path = self.data_dir + '/depth_maps/' + self.data[idx].replace('png','exr')
            # max_depth=1000.0
            dist= Distortion[(b_path.split('.')[0]).split('_')[1]]
        elif self.model== "spherical":
            if self.split == 'train' or self.split == 'val':
                img_path = self.data_dir + '/train/rgb/' + self.data[idx]
                sem_path = self.data_dir + '/train/mask/' + self.data[idx]
            elif self.split == 'test':
                img_path = self.data_dir + '/test/rgb/' + self.data[idx]
                sem_path = self.data_dir + '/test/mask/' + self.data[idx]
        # image= load_color(img_path)['color']
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)
        image = image.permute(1, 2, 0)
        segm= Image.open(sem_path).convert('L')
        segm = segm_transform(segm)
        segm = segm.reshape(832, 1664, 1)
        # mat_path= img_path.replace('png','npy')
        #cl= np.load(mat_path)

        # image=image.permute(1,2,0)
        # segm=segm.permute(1,2,0)
        if self.model == "spherical":
            h= self.img_size
            fov=90
            if self.split=='train' or self.split=='val':
                xi= random.uniform(self.low,self.high)
                deg = random.uniform(0, 360)
            elif self.split=='test':
                # print("this is test", self.xi)
                xi= 0
                deg = 0
            # print(xi, deg)
            image, f = warpToFisheye(image.numpy(), viewingAnglesPYR=[np.deg2rad(0), np.deg2rad(deg), np.deg2rad(0)], outputdims=(h,h),xi=xi, fov=fov, order=1)
            segm,_= warpToFisheye(segm.numpy(), viewingAnglesPYR=[np.deg2rad(0), np.deg2rad(deg), np.deg2rad(0)], outputdims=(h,h),xi=xi, fov=fov, order=0)
            dist= np.array([xi, f/(h/self.img_size), np.deg2rad(fov)])
            segm = segm.astype(np.uint8)
            # print(xi, f, fov, h, deg)
        #resizing to image_size
        #image = resize(image,(self.img_size, self.img_size), order=1)
        #label= resize(depth,(self.img_size, self.img_size), order=0)
        breakpoint()
        image = cv2.resize(image, (self.img_size,self.img_size),interpolation = cv2.INTER_LINEAR)
        label= cv2.resize(segm, (self.img_size,self.img_size), interpolation = cv2.INTER_NEAREST)

        im = Image.fromarray(image)
        label 
        # ############################################# masks ############################################################
        # res = 1024
        # cartesian = torch.cartesian_prod(
        #     torch.linspace(-1, 1, res),
        #     torch.linspace(1, -1, res)
        # ).reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)
        # radius = cartesian.norm(dim=0)
        # mask = (radius > 0.0) & (radius < 1) 
        # mask1 = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0) * 1.0, (self.img_size), mode="nearest")
        # ############################################# masks ############################################################
        # #sample = {'image': image, 'label': label, 'path':b_path.replace('png','npy')}
        # sample = {'image': image, 'label': label}
        
        # if self.transform:
        #     sample = self.transform(sample)
        # else:
        #     sample['image']= torch.from_numpy(image.astype(np.float32)/(255.0))
        #     sample['label']= torch.from_numpy(label.astype(np.uint8))
        
        # one_hot = F.one_hot(sample['label'].to(torch.int64), num_classes=20)
        # sample['image']= sample['image'].permute(2,0,1)
        # # sample['label']= sample['label']

        # sample['dist'] = dist
        # #sample['cl'] = cl 
        # #print(sample['dist'])x``
        

        # #sample['label']= sample['label'].squeeze(0)

        # if normalize is not None:
        #     sample['image']= normalize(sample['image'])
        # sample['one_hot'] = one_hot
        # sample['mask'] = mask1[0].to(torch.long)

        #print(sample.keys())
        return image, lable

def get_mean_std(base_dir ):
    db= Synapse_dataset(base_dir, split="train", transform=None)
    print(len(db))
    #sample= db.__getitem__(0)
    #print(sample['image'].shape)
    #print(sample['label'].shape)
    #print(sample['dist'].shape)
    loader = DataLoader(db, batch_size=len(db), shuffle=False,num_workers=0)
    im_lab_dict = next(iter(loader))
    images, labels = im_lab_dict['image'], im_lab_dict['label']
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    print("mean",mean)
    print("std", std)
    return mean , std


if __name__ == "__main__":
    root_path= '/home/prongs/scratch/CVRG-Pano'
    db= Synapse_dataset(root_path, split="train", transform=None)
    # mean,std= get_mean_std(root_path)
    img = db[0]
    breakpoint()

    print("end")

#     mean tensor([0.9735, 1.2531, 1.3141])
# std tensor([1.4566, 1.5787, 1.5510])