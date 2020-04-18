from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
'

def make_kernels(kernel_size=21, mean=0.0, sigma=1.0, device='cpu'):

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    d = np.sqrt(x * x + y * y)
    d_g = (1/(2 * np.power(sigma, 2.)) * np.sqrt(2.0 * 3.1415)) * np.exp(-np.power(y - mu, 2.) / (2 * np.power(sigma, 2.))) #1D-gaussian-func
    dd_g = np.exp(-((d - mean) ** 2 / (2.0 * sigma ** 2)))  #2D-gaussian-func

    dd_z_g = dd_g * np.power((y-mean), 2.)   #2D-2d_Gauss*(x-mean)**2-func

    t_dd_g = torch.from_numpy(dd_g)
    t_dd_z_g = torch.from_numpy(dd_z_g)

    t_dd_g = torch.unsqueeze(t_dd_g, 0)
    t_dd_g = torch.unsqueeze(t_dd_g, 0)
    t_dd_g = t_dd_g.to(device)

    t_dd_z_g = torch.unsqueeze(t_dd_z_g, 0)
    t_dd_z_g = torch.unsqueeze(t_dd_z_g, 0)
    t_dd_z_g = t_dd_z_g.to(device)

    return (t_dd_g.float(), t_dd_z_g.float())


class f_block(torch.nn.Module):
    def __init__(self, kernels, FILTER_TYPE, MASK_P, device):
        super(f_block, self).__init__()
        self.MASK_P = MASK_P
        self.FILTER_TYPE = FILTER_TYPE
        self.dim_change = None
        self.device = device
        self.g_kernel = kernels[0]
        self.x_g_kernel = kernels[1]
        self.padding = (kernels[0].shape[3] )//2   ##keep kernel size odd!

    def normalize_image(self,sample):
        im_min = sample.min()
        im_max = sample.max()
        sub = im_max - im_min
        res = (sample - im_min) / (sub)

        return res


    def apply_mask(self,x, mask_percent, device):
      k = 1 + round(.01 * float(mask_percent) * (x.numel() - 1))
      per_val = x.view(-1).kthvalue(k).values.item()
      ones = torch.ones(x.shape[0] ,x.shape[1], x.shape[2], x.shape[3])
      zeros = torch.zeros(x.shape[0] ,x.shape[1], x.shape[2], x.shape[3])
      res = torch.where(x > per_val, ones.to(device), zeros.to(device))
      # res = res *x
      return res


    def forward(self, x, TO_PRINT=0):

        copy = x.clone()

        if (TO_PRINT):
          print("orig:")
          plt.imshow(x[0,0,:,:],cmap = 'gray')
          plt.show()

        psy= torch.nn.functional.conv2d(input=x, weight=self.g_kernel, padding=self.padding)
        psy[torch.isnan(psy)] = 0
        if (TO_PRINT):
          print(psy.min(),psy.max())
          print("gauss:")
          plt.imshow(psy[0,0,:,:], cmap = 'gray')
          plt.show()

        x_psy= torch.nn.functional.conv2d(input=x, weight=self.x_g_kernel,padding=self.padding)
        x_psy[torch.isnan(x_psy)] = 0
        if (TO_PRINT):
          print(x_psy.min(),x_psy.max())
          print("x*gauss:")
          plt.imshow(x_psy[0,0,:,:], cmap = 'gray')
          plt.show()

        x = torch.div(psy, x_psy)
        x[torch.isnan(x)] = 0

        if (TO_PRINT):
          print("divided:")
          plt.imshow(x[0,0,:,:], cmap = 'gray')
          plt.show()

        x = self.normalize_image(x)

        if (self.FILTER_TYPE =="V"):
            x = self.apply_mask(x, self.MASK_P, device)

        if (self.FILTER_TYPE == "S"):
            x = torch.exp(-x)

        if (self.FILTER_TYPE == "S_psy"):
            x = torch.exp(-x)
            x = psy * x

        if (self.FILTER_TYPE == "S_img"):
            x = torch.exp(-x)
            x = torch.pow(x ,2)
            x = copy * x

        if (TO_PRINT):
          print("result:")
          plt.imshow(x[0,0,:,:], cmap = 'gray')
          plt.show()

        return x

