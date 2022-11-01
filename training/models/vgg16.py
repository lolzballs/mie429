import torch
import torch.nn as nn
import torch.nn.functional as F
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
        vgg16_out_size = self.vgg16.fc.out_features #12,12,512?
        self.bn = nn.BatchNorm2d(vgg16_out_size)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1), #??
            nn.Conv2d(64, 16, 1),
            LocalLinear(16, 1, 1),
            nn.Conv2d(1, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1)) #??
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.vgg16(x)
        xBn = self.bn(x)
        xConv = self.conv(xBn)
        xBn = xBn*xConv
        xBn = self.pool(xBn)
        xConv = self.pool(xConv)
        x = xBn/xConv
        x = self.fc(x)
        return x

class LocalLinear(nn.Module): # https://stackoverflow.com/questions/59455386/local-fully-connected-layer-pytorch
    def __init__(self,in_features,local_features,kernel_size,padding=0,stride=1,bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fold_num = (in_features+2*padding-self.kernel_size)//self.stride+1
        self.weight = nn.Parameter(torch.randn(fold_num,kernel_size,local_features))
        self.bias = nn.Parameter(torch.randn(fold_num,local_features)) if bias else None

    def forward(self, x:torch.Tensor):
        x = F.pad(x,[self.padding]*2,value=0)
        x = x.unfold(-1,size=self.kernel_size,step=self.stride)
        x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)+self.bias
        return x