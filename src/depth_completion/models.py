import torch
import torch.nn as nn


class DnCNN_c(nn.Module):
    """
    Model mde for noise removal
    source: https://github.com/yzhouas/PD-Denoising-pytorch
    """
    def __init__(self, channels, num_of_layers=17, num_of_est=0):
        super(DnCNN_c, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels + num_of_est, out_channels=features, kernel_size=kernel_size,
                                padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        B, S, H, W, CH = x.shape
        x = x.view(B * S, H, W, CH)
        x = torch.permute(x, (0, 3, 1, 2))
        out = self.dncnn(x)
        #out = out + x

        y = torch.permute(out, (0, 2, 3, 1))
        y = y.view(B, S, H, W, CH)
        assert y.shape == (B, S, H, W, CH)
        return y


###################################################################################
# Sparsity Invariant CNN
# source paper: http://www.cvlibs.net/publications/Uhrig2017THREEDV.pdf
# Tensorflow code: https://github.com/code-xD/Sparsity-Invariant-CNNs
# source code: https://github.com/chenxiaoyu523/Sparsity-Invariant-CNNs-pytorch
# Code has been changed to better accommodate gradslam tensor shape
###################################################################################


class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels),
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(
            kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x, mask):

        # do forward pass
        x = x * mask
        x = self.conv(x)
        normalizer = 1 / (self.sparsity(mask) + 1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.relu(x)

        mask = self.max_pool(mask)

        return x, mask


class SparseConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.SparseLayer1 = SparseConv(1, 16, 11)
        self.SparseLayer2 = SparseConv(16, 16, 7)
        self.SparseLayer3 = SparseConv(16, 16, 5)
        self.SparseLayer4 = SparseConv(16, 16, 3)
        self.SparseLayer5 = SparseConv(16, 16, 3)
        self.SparseLayer6 = SparseConv(16, 1, 1)

    def forward(self, x, mask):
        # reshape inputs from gradslam format to net format
        B, S, H, W, CH = x.shape
        x = x.view(B * S, H, W, CH)
        x = torch.permute(x, (0, 3, 1, 2))
        mask = mask.view(B * S, H, W, CH)
        mask = torch.permute(mask, (0, 3, 1, 2))

        # do a forward pass
        x, mask = self.SparseLayer1(x, mask)
        x, mask = self.SparseLayer2(x, mask)
        x, mask = self.SparseLayer3(x, mask)
        x, mask = self.SparseLayer4(x, mask)
        x, mask = self.SparseLayer5(x, mask)
        x, mask = self.SparseLayer6(x, mask)

        # reshape outputs to gradslam format
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.view(B, S, H, W, CH)
        mask = torch.permute(mask, (0, 2, 3, 1))
        mask = mask.view(B, S, H, W, CH)
        assert x.shape == (B, S, H, W, CH)
        assert mask.shape == (B, S, H, W, CH)

        return x
