from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
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



def make_kernels(kernel_size=21, mean=0.0, sigma=1.0, device='cpu'):

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    d = np.sqrt(x * x + y * y)
    d_g = (1/(2 * np.power(sigma, 2.)) * np.sqrt(2.0 * 3.1415)) * np.exp(-np.power(y - mean, 2.) / (2 * np.power(sigma, 2.))) #1D-gaussian-func
    plt.imshow(d_g, cmap = 'gray')
    plt.show()
    dd_g = np.exp(-((d - mean) ** 2 / (2.0 * sigma ** 2)))  #2D-gaussian-func

    plt.imshow(dd_g, cmap = 'gray')
    plt.show()

    dd_z_g = dd_g * np.power((d-mean), 2.)   #2D-2d_Gauss*(x-mean)**2-func

    plt.imshow(dd_z_g, cmap='gray')
    plt.show()

    t_dd_g = torch.from_numpy(dd_g)
    t_dd_z_g = torch.from_numpy(dd_z_g)

    t_dd_g = torch.unsqueeze(t_dd_g, 0)
    t_dd_g = torch.unsqueeze(t_dd_g, 0)
    t_dd_g = np.repeat(t_dd_g, 3, axis=1)  # duplicate grayscale image to 3 channels
    t_dd_g = np.repeat(t_dd_g, 3, axis=0)  # duplicate grayscale image to 3 channels
    t_dd_g = t_dd_g.to(device)

    t_dd_z_g = torch.unsqueeze(t_dd_z_g, 0)
    t_dd_z_g = torch.unsqueeze(t_dd_z_g, 0)
    t_dd_z_g = np.repeat(t_dd_z_g, 3, axis=1)  # duplicate grayscale image to 3 channels
    t_dd_z_g = np.repeat(t_dd_z_g, 3, axis=0)  # duplicate grayscale image to 3 channels
    t_dd_z_g = t_dd_z_g.to(device)

    return (t_dd_g.float(), t_dd_z_g.float())


class f_block(torch.nn.Module):
    def __init__(self, kernels, FILTER_TYPE, MASK_P, device, crop_num):
        super(f_block, self).__init__()
        self.MASK_P = MASK_P
        self.FILTER_TYPE = FILTER_TYPE
        self.dim_change = None
        self.device = device
        self.crop_num = crop_num
        self.g_kernel = kernels[0]
        self.x_g_kernel = kernels[1]
        self.padding = (kernels[0].shape[3]) // 2  ##keep kernel size odd!

    def normalize_image(self, sample):
        im_min = sample.min()
        im_max = sample.max()
        sub = im_max - im_min
        res = (sample - im_min) / (sub)

        return res

    def apply_mask(self, x, mask_percent, device):
        k = 1 + round(.01 * float(mask_percent) * (x.numel() - 1))
        per_val = x.view(-1).kthvalue(k).values.item()
        ones = torch.ones(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        res = torch.where(x > per_val, ones.to(device), zeros.to(device))
        # res = res *x
        return res

    def apply_Otsus(self, x, device):
        entropy_res = torch.zeros(x.shape)  # placeholders for results
        binary_res = torch.zeros(x.shape)

        for i in [0, 1, 2]:  # RGB
            x_one_channel = img[:, i, :, :]
            x_one_channel = x_one_channel.numpy()  # convert to numpy array
            x_one_channel = x_one_channel[0, :, :]
            entropy_img = entropy(x_one_channel, disk(10))  # entropy filter
            thresh = threshold_otsu(entropy_img)
            binary = entropy_img <= thresh

            entropy_img = torch.from_numpy(entropy_img)  # return to tensor
            binary = torch.from_numpy(binary)

            entropy_res[0, i, :, :] = entropy_img  # build back RGB images
            binary_res[0, i, :, :] = binary

        return (entropy_res, binary_res)

    def forward(self, x, TO_PRINT=0):

        copy = x.clone()

        if (TO_PRINT):
            print("orig:")
            plt.imshow(x[0, 0, :, :].cpu(), cmap='gray')
            plt.show()

        psy = torch.nn.functional.conv2d(input=x, weight=self.g_kernel, padding=self.padding)
        psy[torch.isnan(psy)] = 0
        psy = self.normalize_image(psy)
        if (TO_PRINT):
            print(psy.min(), psy.max())
            print("gauss:")
            plt.imshow(psy[0, 0, :, :].cpu(), cmap='gray')
            plt.show()

        x_psy = torch.nn.functional.conv2d(input=x, weight=self.x_g_kernel, padding=self.padding)
        x_psy[torch.isnan(x_psy)] = 0
        x_psy = self.normalize_image(x_psy)
        if (TO_PRINT):
            print(x_psy.min(), x_psy.max())
            print("x*gauss:")
            plt.imshow(x_psy[0, 0, :, :].cpu(), cmap='gray')
            plt.show()

        x = torch.div(psy, x_psy)
        x[torch.isnan(x)] = 0

        if (TO_PRINT):
            print("divided:")
            plt.imshow(x[0, 0, :, :].cpu(), cmap='gray')
            plt.show()


        if (self.FILTER_TYPE == "V"):
            (masked_env, masked_bin) = self.apply_Otsus(x, self.device)
            if (self.MASK_P == 0) :
                x = masked_env
            else:
                x = masked_bin

        if (self.FILTER_TYPE == "S"):
            x = torch.exp(-x)

        if (self.FILTER_TYPE == "S_psy"):
            x = torch.exp(-x)
            if (TO_PRINT):
                print("S:")
                plt.imshow(x[0, 0, :, :].cpu(), cmap='gray')
                plt.show()
            x = psy * x

        if (self.FILTER_TYPE == "S_img"):
            x = torch.exp(-x)
            if (TO_PRINT):
                print("S:")
                plt.imshow(x[0, 0, :, :].cpu(), cmap='gray')
                plt.show()
            x = torch.pow(x, 2)
            if (TO_PRINT):
                print("S^2:")
                plt.imshow(x[0, 0, :, :].cpu(), cmap='gray')
                plt.show()
            x = copy * x

        if self.crop_num > 0:
            BS, CH, H, W = x.shape
            x = x[:, :, self.crop_num:H - self.crop_num, self.crop_num:W - self.crop_num]

        x = self.normalize_image(x)

        if (TO_PRINT):
            print("result:")
            plt.imshow(x[0, 0, :, :].cpu(), cmap='gray')
            plt.show()

        return x