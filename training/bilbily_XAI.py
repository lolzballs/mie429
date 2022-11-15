import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from models import Bilbily

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = torch.load_state_dict('bilbily_grayscale_1101_152328.pt', map_location=torch.device('cpu'))
# print(type(model))

model = Bilbily()
model.load_state_dict(torch.load('bilbily_grayscale_1101_152328.pt', map_location=torch.device('cpu')), strict=False)
model.eval()
# print(model.get_features())
print(model.layers)
# features = getattr(model, net)(pretrained=pretrained).get_features()
# print(features)

# class Bilbily_XAI(nn.Module):
#     def __init__(self):
#         super(Bilbily_XAI, self).__init__()
        
#         self.model = model(pretrained=True)
        
#         # disect the network to access its last convolutional layer
#         self.features_conv = self.model.features[:36]
        
#         # get the max pool of the features stem
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
#         # get the classifier of the vgg19
#         self.classifier = self.vgg.classifier
        
#         # placeholder for the gradients
#         self.gradients = None
    
#     # hook for the gradients of the activations
#     def activations_hook(self, grad):
#         self.gradients = grad
        
#     def forward(self, x):
#         x = self.features_conv(x)
        
#         # register the hook
#         h = x.register_hook(self.activations_hook)
        
#         # apply the remaining pooling
#         x = self.max_pool(x)
#         x = x.view((1, -1))
#         x = self.classifier(x)
#         return x
    
#     # method for the gradient extraction
#     def get_activations_gradient(self):
#         return self.gradients
    
#     # method for the activation exctraction
#     def get_activations(self, x):
#         return self.features_conv(x)
