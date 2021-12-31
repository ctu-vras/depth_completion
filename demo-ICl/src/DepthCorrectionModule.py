import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np


# --------- simple depth correction module - linear transformation --------- #
class LinTransModel(nn.Module):

    def __init__(self, batch_size=1):
        super(LinTransModel, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.]).view(batch_size, 1))
        self.b = nn.Parameter(torch.tensor([0.]).view(batch_size, 1))

    def forward(self, x):
        # input x is torch tensor of depth image, output is tensor in the same shape
        assert x.dim() == 5
        # x.shape == (B, S, H, W, 1)
        y = (x * self.w) + self.b
        assert y.shape == x.shape
        return y


# --------- simple depth correction module - multiple layers --------- #
def ConvLay(in_channels, out_channels):
    kernel = (3, 3)
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )
    return layer


class ModelV1(nn.Module):

    def __init__(self, batch_size=1):
        super(ModelV1, self).__init__()
        self.layer1 = ConvLay(1, 3)
        self.layer2 = ConvLay(3, 3)
        self.layer3 = ConvLay(3, 1)

    def forward(self, x):
        # input x is torch tensor of depth image, output is tensor in the same shape
        # x.shape == (B, S, H, W, 1)
        assert x.dim() == 5

        # transform input for convolutional layer to shape (B, CH, H, W)
        B, S, H, W, CH = x.shape
        x = x.view(B*S, H, W, CH)
        x = torch.permute(x, (0, 3, 1, 2))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        y = self.layer3(x2)

        # return depth image to original shape
        y = torch.permute(y, (0, 2, 3, 1))
        y = y.view(B, S, H, W, CH)
        assert y.shape == (B, S, H, W, CH)
        return y

    def weight_init(self):
        # Xavier initialization
        for lay in self.modules():
            if type(lay) in [nn.Conv2d]:
                torch.nn.init.xavier_uniform_(lay.weight)


# --------- dataset used for simulation data --------- #
class Dataset(Dataset):
    def __init__(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.root_dir = path
        self.rgbs = [os.path.join(path, 'rgb', f) for f in os.listdir(os.path.join(path, 'rgb')) if '.npy' in f]
        self.depths = [os.path.join(path, 'depth', f) for f in os.listdir(os.path.join(path, 'depth')) if '.npy' in f]
        self.points = [os.path.join(path, 'point_clouds', f) for f in os.listdir(os.path.join(path, 'point_clouds')) if '.npy' in f]
        self.normals = [os.path.join(path, 'normals', f) for f in os.listdir(os.path.join(path, 'normals')) if '.npy' in f]
        self.length = len(self.depths)

    def __getitem__(self, i):
        sample = {'rgb': np.asarray(np.load(self.rgbs[i]), dtype=np.uint8),
                  'depth': np.load(self.depths[i]),
                  'points': np.load(self.points[i]),
                  'normals': np.load(self.normals[i])}
        return sample

    def __len__(self):
        return self.length
