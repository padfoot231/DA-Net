import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image

breakpoint()
im = np.load('im.npy')
lbl = np.load('lb.npy')

image = Image.fromarray((im*255).astype(np.uint8))
image.save('img.png')

plt.imshow(lbl)
plt.savefig('label.png')