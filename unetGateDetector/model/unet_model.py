""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def  __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        # self.up2 = Up(512, 128, bilinear)
        # self.up3 = Up(256, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.inc = DoubleConv(n_channels, 8, kernel_size=3)
        self.down1 = Down(8, 16, kernel_size=3)
        self.down2 = Down(16, 16, kernel_size=3)
        self.down3 = Down(16, 16, kernel_size=5)
        self.down4 = Down(16, 16, kernel_size=7)
        self.down5 = Down(16, 16, kernel_size=7)
        self.up1 = Up(32, 16, bilinear)
        self.up2 = Up(32, 16, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.up5 = Up(24, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits



if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1)
    print(net)
