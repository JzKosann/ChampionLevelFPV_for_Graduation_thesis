import torch
import torch.nn as nn
import torch.nn.functional as torchFun
import cv2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    MaxPool--->double conv
    """
    def __int__(self, in_channel, out_channel, kernel_size):
        super().__int__()
        self.downSample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channel, 
                       out_channels=out_channel, 
                       kernel_size=kernel_size)
        )


class UNetModel(nn.Module):
    def __int__(self):
        super(UNetModel, self).__init__()
        # 第一层
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)

    # 下采样  twice conv -> once maxPool
