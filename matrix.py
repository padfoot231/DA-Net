from PIL import Image
from envmap import EnvironmentMap
from envmap import rotation_matrix
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
from utils_tan import get_sample_params_from_subdiv
from torchvision.transforms import transforms
t2pil = transforms.ToTensor()
pil = transforms.ToPILImage()



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

def KMeans(x, c, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    B, N, D = x.shape  # Number of samples, dimension of the ambient space

    x_i = LazyTensor(x.view(B, N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(B, 1, K, D))  # (1, K, D) centroids
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

# key = list(data.keys())

H, W = 128, 128
x = torch.linspace(0, H, H+1) - H//2 + 0.5
y = torch.linspace(0, W, W+1) - W//2 + 0.5
grid_x, grid_y = torch.meshgrid(x[1:], y[1:])
x_ = grid_x.reshape(H*W, 1)
y_ = grid_y.reshape(H*W, 1)
grid_pix = torch.cat((x_, y_), dim=1).type(torch.float32)
grid_pix = grid_pix.reshape(1, H*W, 2).cuda("cuda:0")
image = Image.open('/localscratch/prongs.46754485.0/data/CVRG-Pano/all-rgb/img-260.png')
image = np.array(image)
with open('/localscratch/prongs.46754485.0/data/CVRG-Pano/val.pkl', 'rb') as f:
    data = pkl.load(f)
h= 512
fov= 175
D_xi = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dist = {}
dist = []

for i in range(len(data)):
# for i in range(10):
    # print(key[i])
    xi = np.random.uniform(0.83, 0.9)
    # xi = D_xi[i]
    im, f = warpToFisheye(image, viewingAnglesPYR=[np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)], outputdims=(h,h),xi=xi, fov=fov, order=1)
    # D = torch.tensor(data[key[i]].reshape(1,4).transpose(1,0)).cuda("cuda:0")
    # D = np.array([1.0, 0.0, 0.0, 0.0])
    # D = torch.tensor(D.reshape(1,4).transpose(1,0)).cuda("cuda:0")
    D = torch.tensor([ xi, f/(h/128),  2.9671]).reshape(1, 3).transpose(0,1).cuda()
    azimuth_subdiv = H
    radius_subdiv = W//4
    n_radius = 28
    n_azimuth = 4
    subdiv = (radius_subdiv*n_radius, azimuth_subdiv*n_azimuth)
    # subdiv = 3
    # breakpoint()

    img_size = (H, W)
    radius_buffer, azimuth_buffer = 0, 0
    xc, yc, theta_max = get_sample_params_from_subdiv(
        subdiv=subdiv,
        img_size=img_size,
        distortion_model = 'spherical',
        D = D, 
        n_radius=n_radius,
        n_azimuth=n_azimuth,
        radius_buffer=radius_buffer,
        azimuth_buffer=azimuth_buffer)
    
    B, n_p, n_s = xc.shape
    x = xc.reshape(1, 1, -1).transpose(1,2).cuda("cuda:0")
    # x = torch.repeat_interleave(x, 2, 0).cuda("cuda:2")
    y = yc.reshape(1, 1, -1).transpose(1,2).cuda("cuda:0")
    # y = torch.repeat_interleave(y, 2, 0).cuda("cuda:2")
#     out = torch.cat((y_, x_), dim = 3)
    grid_ = torch.cat((x, y), dim=2).type(torch.float32)
    # grid_ = grid_.reshape(1, -1, 2)
    # x = resample(grid_, gid_pix, 128, 2)
    B, N, D, P, k = grid_.shape[0], grid_.shape[1], 2, grid_pix.shape[1], 4
    # breakpoint()
    cl = KNN(grid_/(H//2), grid_pix/(W//2), P, k)
    # cl = np.array(cl.cpu())
    cl = cl[0].cpu()    
    # breakpoint()
    # breakpoint()/
    dist.append([cl, f/(h/128), xi])
    # dist[xi] = [cl, f, xi]
    # np.save('/home-local2/akath.extra.nobkp/woodscape/matrix_8/8_8_' + str(key[i]) + '.pkl', cl)
    print(i)

# breakpoint()
with open('/home/prongs/scratch/32_5_cl_tan_val.pkl', 'wb') as f:
    data = pkl.dump(dist, f)



 


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