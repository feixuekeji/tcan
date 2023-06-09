
import torch.nn as nn
import torch.nn.functional as F
import torch

from .dfcan import DFCAN
from .unet import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.unet = UNet(1,1)
        self.dfcan = DFCAN(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=0)

    def forward(self, img):
        img = self.conv1(img)
        img = self.relu(img)
        img1 = self.unet(img)
        img2 = self.dfcan(img)
        img = img1 + img2
        img = self.conv2(img)
        img = self.relu(img)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.AvgPool2d(2))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([32, 64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters

        layers.append(nn.Flatten())
        layers.append(nn.Linear(8192, 1024))
        # layers.append(nn.Linear(32768, 1024))
        layers.append(nn.Linear(1024, 128))
        layers.append(nn.Linear(128, 1))

        self.model = nn.Sequential(*layers)


    def forward(self, img):
        return self.model(img)
