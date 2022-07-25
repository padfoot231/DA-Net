from torchvision import datasets, transforms
from IPython.display import clear_output, display
import time 
import pylab as pl
import numpy as np
import pickle as pkl
from utils import get_sample_params_from_subdiv, get_sample_locations
import torch 
import cProfile

with open('/home-local2/akath.extra.nobkp/all-distorted-tiny-imagenet/distortion_val_file.pkl', 'rb') as f:
    dist = pkl.load(f)


keys = list(dist.keys())
output = {}

# ax.set_title("Sampling locations")

# cProfile.run(get_sample_params_from_subdiv())
# subdiv = 3
radius_subdiv = 16
azimuth_subdiv = 64
subdiv = (radius_subdiv, azimuth_subdiv)
# subdiv = 3
n_radius = 20
n_azimuth = 20
img_size = (64,64)
radius_buffer, azimuth_buffer = 0, 0
# get_optimal_buffers(subdiv, n_radius, n_azimuth, img_size)
for k in range(len(keys)):
    print(dist[keys[k]])
    params = get_sample_params_from_subdiv(
        subdiv=subdiv,
        img_size=img_size,
        D = dist[keys[k]], 
        n_radius=n_radius,
        n_azimuth=n_azimuth,
        radius_buffer=radius_buffer,
        azimuth_buffer=azimuth_buffer
    )

    # cProfile.run(get_sample_params_from_subdiv())

    
    x = []
    y = []
    s = []
    sample_locations = get_sample_locations(**params)
    # cProfile.run(get_sample_locations())
    x_ = torch.tensor(sample_locations[0]).reshape(1024, 1, n_radius*n_azimuth, 1).float()
    x_ = x_/ 32
    x_.long()
    y_ = torch.tensor(sample_locations[1]).reshape(1024, 1, n_radius*n_azimuth, 1).float()
    y_ = y_/32
    y_.long()
    t = torch.cat((x_, -(y_)), dim = 3)
    # import pdb;pdb.set_trace()
          
    torch.save(t, '/home-local2/akath.extra.nobkp/sampling_val/' + keys[k].split('/')[-1] + '.pt')

# with open('sampling_poiints_val_val.pkl', 'wb') as f:
#     pkl.dump(output, f)
