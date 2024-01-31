import os
import random
import h5py
import numpy as np
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from PIL import Image
import pickle as pkl
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, rotate


# Transpose.FLIP_LEFT_RIGHT 

normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img

def segm_transform(segm):
    # to tensor, -1 to 149
    segm = torch.from_numpy(np.array(segm)).long()
    return segm

# H, W = (128, 128)
# x = torch.linspace(0, H, H+1) - H//2 - 0.5
# y = torch.linspace(0, W, W+1) - H//2 - 0.5
# grid_x, grid_y = torch.meshgrid(x[1:], y[1:])
# x_ = grid_x.reshape(H*H, 1)
# y_ = grid_y.reshape(W*W, 1)
# grid_pix = torch.cat((x_, y_), dim=1)
# grid_pix = grid_pix.reshape(1, H*W, 2)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Cityscape_dataset(Dataset):
    def __init__(self, base_dir, split, img_size, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        
        self.pad = (121, 121,633, 633) 
        if split == 'train':
            with open(base_dir + '/leftImg8bit/train.pkl', 'rb') as f:
                img_data = pkl.load(f)
            with open(base_dir + '/gtFine/train.pkl', 'rb') as f:
                sem_data = pkl.load(f)
        elif split == 'val':
            with open(base_dir + '/leftImg8bit/val.pkl', 'rb') as f:
                img_data = pkl.load(f)
            with open(base_dir + '/gtFine/val.pkl', 'rb') as f:
                sem_data = pkl.load(f)
        elif split == 'test':
            with open(base_dir + '/leftImg8bit/test.pkl', 'rb') as f:
                img_data = pkl.load(f)
            with open(base_dir + '/gtFine/test.pkl', 'rb') as f:
                sem_data = pkl.load(f)
        self.img_data = img_data
        self.sem_data = sem_data
        self.data_dir = base_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.data_dir + '/leftImg8bit/' + self.img_data[idx]
        sem_path = self.data_dir + '/gtFine/' + self.sem_data[idx]
        mat_path = self.data_dir + '/key_4t4_1.pkl.npy'
        img = Image.open(img_path).convert('RGB')
        segm = Image.open(sem_path).convert('L')
        cls = np.load(mat_path)

        # dist = self.calib[key].astype(np.float32)
        dist= np.array([1.0, 0.0, 0.0, 0.0]).astype(np.float32)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

            # random_flip
        if np.random.choice([0, 1]):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
        # image transform, to torch float tensor 3xHxW
        img = img_transform(img)

        # segm transform, to torch long tensor HxW
        segm = segm_transform(segm)

        img = F.pad(img, self.pad, "constant", 0)
        segm = F.pad(segm, self.pad, "constant", 3)
        segm = segm.reshape(1, segm.shape[0], segm.shape[1])
        # import pdb;pdb.set_trace()
        # note that each sample within a mini batch has different scale param
        # img = imresize(img, (self.img_size, self.img_size), interp='bilinear')
        # segm = imresize(segm, (self.img_size,self.img_size), interp='nearest')
        img = transforms.Resize((self.img_size,self.img_size), interpolation=InterpolationMode.BILINEAR)(img)
        segm = transforms.Resize((self.img_size,self.img_size), interpolation=InterpolationMode.NEAREST)(segm)
        segm = segm[0]
        # image transform, to torch float tensor 3xHxW
        # img = img_transform(img)

        # segm transform, to torch long tensor HxW
        # segm = segm_transform(segm)
        one_hot = F.one_hot(segm.to(torch.int64), num_classes=34).to(torch.float32)

        ########################## mask ##################################
        res = 1024
        cartesian = torch.cartesian_prod(
            torch.linspace(-1, 1, res),
            torch.linspace(1, -1, res)
        ).reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)
        radius = cartesian.norm(dim=0)
        mask = (radius > 0.0) & (radius < 1) 
        mask1 = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0) * 1.0, (self.img_size), mode="nearest").to(torch.long)
        # sample = {'image': img, 'label': segm, 'dist':dist, 'class':cls}
        return img, segm, dist, cls, mask1[0, 0], one_hot


if __name__=='__main__':
    db_train = Cityscape_dataset(base_dir="/home-local2/akath.extra.nobkp/cityscapes", split="train", img_size=128)
    # trainloader = DataLoader(db_train, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    # for i_batch, sampled_batch in enumerate(trainloader):
    #     import pdb;pdb.set_trace()
    #     print("ass")
    m = db_train[0]
    import pdb;pdb.set_trace()
    print("ass")
