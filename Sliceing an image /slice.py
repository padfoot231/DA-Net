import torch
from torchvision.utils import save_image
from PIL import Image, ImageOps
import numpy as np 
from torchvision import transforms
import wandb
wandb.init(project='images')

im = Image.open('/gel/usr/akath/tiny-imagenet-200/train/n02837789/images/n02837789_110.JPEG')
# im = ImageOps.grayscale(im)
trans = transforms.ToTensor()
transform = transforms.ToPILImage()

im_ = trans(im)

# im = np.array(im)
res=1024

cartesian = torch.cartesian_prod(
    torch.linspace(-1, 1, res),
    torch.linspace(1, -1, res)
).reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)
radius = cartesian.norm(dim=0)
y = cartesian[1]
x = cartesian[0]
theta = torch.atan2(cartesian[1], cartesian[0])
image = torch.randn(res, res, 3)
mask = (radius > 0.4) & (radius < 0.8) 

mask1 = mask & (theta >0) & (theta<3.14/2)
mask2 = mask & (theta >3.14/2) & (theta<3.14)


# import pdb;pdb.set_trace()

mask3 = mask & (theta > -3.14) & (theta < -3.14/2)

mask4 = mask & (theta > -3.14/2) & (theta<0)
lst = [mask1, mask2, mask3, mask4]
img = []
for i in range(4):
    lst[i] = torch.nn.functional.interpolate(lst[i].unsqueeze(0).unsqueeze(0) * 1.0, (64), mode="area")
    flag = lst[i]*im_
    # import pdb;pdb.set_trace()
    image = transform(flag[0])
    img.append(image)
    lst[i] = transform(lst[i][0])

wandb.log({"masks" : [wandb.Image(image) for image in lst]})

wandb.log({"segments" : [wandb.Image(image) for image in img]})
wandb.log({"original image" : wandb.Image(im)})






# mask = mask.reshape(64,64)
# # mask = mask.repeat(1, 3)screen 
# # mask = mask.reshape(3, 64, 64)
# import pdb;pdb.set_trace()
# im = im[0, :, :][mask.bool()]
# im = im.unsqueeze(0).unsqueeze(0)
# image1 = mask1*im
# image2 = mask2*im
# image3 = mask3*im
# image4 = mask4*im

# a1 = torch.nonzero(image1).transpose(0,1)
# a2 = torch.nonzero(image2).transpose(0,1)
# a3 = torch.nonzero(image3).transpose(0,1)
# a4 = torch.nonzero(image4).transpose(0,1)
# x1 = a1[0]
# y1 = a1[1]
# z1 = a1[2]
# x2 = a2[0]
# y2 = a2[1]
# z2 = a2[2]
# import pdb;pdb.set_trace()
# save_image(mask1, "masking.png")
# save_image(mask2, "masking_1.png")
# save_image(mask3, "masking2.png")
# save_image(mask4, "masking3.png")