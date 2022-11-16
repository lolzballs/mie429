import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.nn.modules.utils import _pair
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from models import Vgg16
# import data 

# from torchvision.models import vgg19
# test = vgg19(pretrained=True)
# print('test', test.features)
# print(test.features[:36])

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='../test_img/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

# train_dp, val_dp = data.RSNA(root=args.data)
# train_dp = train_dp.map(apply_to_image(transforms))
# val_dp = val_dp.map(apply_to_image(transforms))
# train_loader = torch.utils.data.DataLoader(dataset=train_dp, batch_size=hyperparams['batch_size'])
# dataloader = torch.utils.data.DataLoader(dataset=val_dp)

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
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

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
        # register the hook
        h = x.register_hook(self.activations_hook)
        x = self.fc(torch.squeeze(x))
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, features_conv, x):
        return features_conv(x)

model = Vgg16()
model.load_state_dict(torch.load('vgg16-withpreprocessing_1109_002607.pt', map_location=torch.device('cpu')), strict=False)
model.eval()
print(model)
# print(model.vgg16.features)
# print("-------------------------- START --------------------------")
# i = 1
# for module in model.modules():
#     if not isinstance(module, nn.Sequential):
#         print("-------------------------- {} --------------------------".format(i))
#         print(module, "\n")
#         i+=1
layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)] # i found this on stack overflow: https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model

features_conv = nn.Sequential(*layers[2:43]) #holy shit i think this is IT
# print(features_conv)

# get the image from the dataloader
img, _ = next(iter(dataloader))

# get the most likely prediction of the model
pred = model(img)
pred.backward()

# pull the gradients out of the model
gradients = model.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = model.get_activations(features_conv, img).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
    
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
# heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())
plt.savefig('../XAI_vgg16/heatmap.jpg')

heatmap = heatmap.numpy()

img = cv2.imread('../test_img/Elephant/sample_hand.png')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('../XAI_vgg16/hand_map.jpg', superimposed_img)