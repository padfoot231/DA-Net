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

# Transpose.FLIP_LEFT_RIGHT 

T = transforms.ToTensor()


normalize = transforms.Normalize(
            mean=[0.2151, 0.2235, 0.2283],
            std=[0.2300, 0.2334, 0.2419])

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
    image = torch.rot90(image, k, dims=[1, 2])
    label = torch.rot90(label, k, dims=[0, 1])
    axis = np.random.randint(1, 3)
    image = torch.flip(image, dims=[1,2])
    label = torch.flip(label, dims=[0,1])
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = transforms.functional.rotate(image, angle)
    label = transforms.functional.rotate(label.reshape(1, label.shape[0], label.shape[1]), angle)
    return image, label[0]


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img))
    img = img.transpose((2, 0, 1))
    # img = normalize(torch.from_numpy(img.copy()))
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
        # print(image.shape, label.shape, type(image), type(label))
        # x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image.type(torch.float32), 'label': label.type(torch.uint8)}
        return sample


class Woodscape_dataset(Dataset):
    def __init__(self, base_dir, split, img_size = (768, 768), transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        
        if split == 'train':
            with open(base_dir + '/train.json', 'r') as f:
                data = json.load(f)
        elif split == 'val':
            with open(base_dir + '/val.json', 'r') as f:
                data = json.load(f)
        elif split == 'test':
            with open(base_dir + '/test.json', 'r') as f:
                data = json.load(f)
        with open(base_dir + '/calib.pkl', 'rb') as f:
            calib = pkl.load(f)

        self.calib = calib
        self.data = data
        self.data_dir = base_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data[idx]
        img_path = self.data_dir + '/rgb_images/' + self.data[idx]
        lbl_path = self.data_dir + '/gtLabels/' + self.data[idx]
        img = Image.open(img_path).convert('RGB')
        segm = Image.open(lbl_path).convert('L')
        key_calib = self.data[idx][:-4] + "_img" + ".png"
        # breakpoint()
        dist = self.calib[key_calib]
    

        # dist = torch.tensor([k1, k2, k3, k4], dtype=torch.float32)

        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

            # random_flip
        # if np.random.choice([0, 1]):
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        # import pdb;pdb.set_trace()
        # note that each sample within a mini batch has different scale param
        img = imresize(img, (self.img_size), interp='bilinear')
        segm = imresize(segm, (self.img_size[0] + 1, self.img_size[1] + 1), interp='nearest')

        image = T(img)
        label = segm_transform(segm)

        ############################################# masks ############################################################
        res = 1024
        cartesian = torch.cartesian_prod(
            torch.linspace(-1, 1, res),
            torch.linspace(1, -1, res)
        ).reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)
        radius = cartesian.norm(dim=0)
        mask = (radius > 0.0) & (radius < 1) 
        mask1 = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0) * 1.0, (self.img_size[0] + 1, self.img_size[1] + 1), mode="nearest")
        ############################################# masks ############################################################

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['image']= image.type(torch.float32)
            sample['label']= label.type(torch.uint8)
        one_hot = F.one_hot(sample['label'].to(torch.int64), num_classes=10).to(torch.float32)
        sample['dist'] = dist
        if normalize is not None:
            sample['image']= normalize(sample['image'])
        sample['one_hot'] = one_hot
        sample['mask'] = mask1[0].to(torch.long)
        # sample = {'image': img, 'label': segm, 'dist':dist, 'class':cls}
        # breakpoint()
        return sample['image'], sample['label'], sample['dist'] , sample['mask'], one_hot

def get_mean_std(base_dir ):
    db= Woodscape_dataset(base_dir, split="train", transform=None)
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



if __name__=='__main__':
    db_train = Woodscape_dataset(base_dir="/localscratch/prongs.50875538.0/data_new/woodscapes", split="train")
    # trainloader = DataLoader(db_train, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    # for i_batch, sampled_batch in enumerate(trainloader):
    #     import pdb;pdb.set_trace()
    #     print("ass")
    # mean,std= get_mean_std(base_dir="/localscratch/prongs.45335371.0/data/woodscapes")
    m = db_train[0]
    import pdb;pdb.set_trace()
    print("ass")
