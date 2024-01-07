import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import cv2
import sys
sys.path.append('../')
import timeit
from tqdm import tqdm, trange
import lpips

""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SimpleUnetLight(nn.Module):
    def __init__(self, cfg, n_channels=3, n_classes=3, bilinear=True):
        super(SimpleUnetLight, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        input_channels = 64
        self.inc = DoubleConv(n_channels, input_channels)
        self.cfg = cfg
        self.down1 = Down(input_channels, input_channels*2)
        factor = 2 if bilinear else 1
        self.down2 = Down(input_channels*2, input_channels*4 // factor)

        self.up1 = Up(input_channels*4, input_channels*2 // factor, bilinear)
        self.up2 = Up(input_channels*2, input_channels, bilinear)
        self.outc = OutConv(input_channels, n_classes)

    def forward(self, x, x_level1=None, x_level2=None):
        # print(x.shape) # torch.Size([1, 3, 500, 500])
        x1 = self.inc(x) # torch.Size([1, 64, 500, 500])
        
        x2 = self.down1(x1) # torch.Size([1, 128, 250, 250])
        x3 = self.down2(x2) # torch.Size([1, 128, 125, 125])
    
        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        logits = self.outc(x)
        return logits
  