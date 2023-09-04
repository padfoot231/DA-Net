from PIL import Image
import torch
import torch.nn as nn
import pickle as pkl
import random
import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
import numpy as np
use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"
# from utils import get_sample_params_from_subdiv, get_sample_locations
import numpy as np
from utils import get_sample_params_from_subdiv, get_sample_locations, distort_image
from torchvision.transforms import transforms
t2pil = transforms.ToTensor()
pil = transforms.ToPILImage()

def KMeans(x, c, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    B, N, D = x.shape  # Number of samples, dimension of the ambient space

    x_i = LazyTensor(x.view(B, N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(B, 1, K, D))  # (1, K, D) centroids
    breakpoint()
    D_ij = ((x_i - c_j) ** 2).sum(B, -1)  # (N, K) symbolic squared distances
    cl = D_ij.argmin(dim=2).long().view(B, -1)  # Points -> Nearest cluster

    return cl, c

def KNN(x, c, P=10, k = 1, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

#     start = time.time()
    B, N, D = x.shape  # Number of samples, dimension of the ambient space
    x_i = LazyTensor(x.view(B, 1, N, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(B, P, 1, D))  # (1, K, D) centroids

    D_ij = ((x_i - c_j) ** 2).sum(B, -1)
#     import pdb;pdb.set_trace()
#     .sum(B, -1)  # (N, K) symbolic squared distances
    
    cl = D_ij.argKmin(k, dim=2)  # Points -> Nearest cluster

    return cl

def resample(grid, grid_pix, H, B):
    B, N, D, K = grid.shape[0], grid.shape[1], 2, grid_pix.shape[1]
    cl, c = KMeans(grid/(H//2), grid_pix/(H//2), K)
#     import pdb;pdb.set_trace()
    # ind = torch.arange(N).reshape(1, -1)
    # ind = torch.repeat_interleave(ind, B, 0)
    # mat = torch.zeros(B, K, N)
    # mat[:, cl, ind] = 1
#     output = output.reshape(B, L, -1).transpose(1, 2)
#     pixel_out = torch.matmul(mat, output)
#     div = mat.sum(-1).unsqueeze(2)
#     div[div == 0] = 1
#     pixel_out = torch.div(pixel_out, div)
#     pixel_out = pixel_out.transpose(2, 1).reshape(B, 3, H, H)
    return cl

with open('/home-local2/akath.extra.nobkp/woodscapes/calib.pkl', 'rb') as f:
    data = pkl.load(f)

key = list(data.keys())

H, W = 64, 64
x = torch.linspace(0, H, H+1) - H//2 + 0.5
y = torch.linspace(0, W, W+1) - W//2 + 0.5
grid_x, grid_y = torch.meshgrid(x[1:], y[1:])
x_ = grid_x.reshape(H*W, 1)
y_ = grid_y.reshape(H*W, 1)
grid_pix = torch.cat((x_, y_), dim=1)
grid_pix = grid_pix.reshape(1, H*W, 2).cuda("cuda:0")

for i in range(len(key)):
    print(i)
    D = torch.tensor(data[key[i]].reshape(1,4).transpose(1,0)).cuda("cuda:0")
    
    azimuth_subdiv = H
    radius_subdiv = W//4
    subdiv = (radius_subdiv, azimuth_subdiv)
    # subdiv = 3
    n_radius = 4
    n_azimuth = 4
    img_size = (64, 64)
    radius_buffer, azimuth_buffer = 0, 0
    params, D_s = get_sample_params_from_subdiv(
        subdiv=subdiv,
        img_size=img_size,
        D = D, 
        n_radius=n_radius,
        n_azimuth=n_azimuth,
        radius_buffer=radius_buffer,
        azimuth_buffer=azimuth_buffer, 
        distortion_model = 'polynomial')

    sample_locations = get_sample_locations(**params)  ## B, azimuth_cuts*radius_cuts, n_radius*n_azimut
    B, n_p, n_s = sample_locations[0].shape
    x = sample_locations[0].reshape(1, 1, -1).transpose(1,2).cuda("cuda:0")
    # x = torch.repeat_interleave(x, 2, 0).cuda("cuda:2")
    y = sample_locations[1].reshape(1, 1, -1).transpose(1,2).cuda("cuda:0")
    # y = torch.repeat_interleave(y, 2, 0).cuda("cuda:2")
#     out = torch.cat((y_, x_), dim = 3)
    grid_ = torch.cat((x, y), dim=2)
    # grid_ = grid_.reshape(1, -1, 2)
    # x = resample(grid_, grid_pix, 128, 2)
    B, N, D, P, k = grid_.shape[0], grid_.shape[1], 2, grid_pix.shape[1], 1
    cl = KNN(grid_/(H//2), grid_pix/(W//2), P, k)
    # cl = np.array(cl.cpu())
    cl = cl[0].cpu()
    # breakpoint()
    np.save('/home-local2/akath.extra.nobkp/woodscapes/index_16s_1k/' + key[i][:-4], cl)
    print(i)






# grid = grid.reshape(8234, -1, 2)
# B, N, D = grid.shape
# B, N_p, D = grid_pix.shape
# dic = {}
# breakpoint()
# for i in range(len(key)):
#     g = grid[i].reshape(1, N, D)
#     g_p = grid_pix[i].reshape(1, N_p, D)
#     x = resample(g, g_p, 128, 2)
#     x = np.array(x)
#     np.save('/home-local2/akath.extra.nobkp/woodscape/mat/' + key[i][:-4], x)
#     print(i)