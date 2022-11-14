import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision

class Vgg16(nn.Module):
    """
    based on kaggle winner (translated from tensorflow to pytorch)
    https://www.kaggle.com/code/kmader/attention-on-pretrained-vgg16-for-bone-age
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.VGG16_Weights.DEFAULT if pretrained else None
        self.vgg16 = torchvision.models.vgg16(weights=weights)
        # self.vgg16.features[0] = nn.Conv2d(1, 64, 3)
        # self.vgg16.features[30] = nn.Identity()
        # self.vgg16.avgpool = nn.Identity()
        self.vgg16.classifier = nn.Identity()
        # print(self.vgg16)
        self.bn = nn.BatchNorm2d(512)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 1),
            nn.ReLU(),
            LocallyConnected2d(16, 1, 7, 1, 1),
            nn.Sigmoid(),
            nn.Conv2d(1, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        # add in relu shit
        x = self.vgg16(x)
        x = x.reshape(-1, 512, 7, 7)
        xBn = self.bn(x)
        xConv = self.conv(xBn)
        xBn = xBn*xConv
        xBn = self.pool(xBn)
        xConv = self.pool(xConv)
        x = xBn/xConv
        x = self.fc(torch.squeeze(x))
        return x

class LocallyConnected2d(nn.Module): #https://github.com/ptrblck/pytorch_misc/blob/master/LocallyConnected2d.py
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out