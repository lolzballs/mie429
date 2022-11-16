import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
# from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from models import Bilbily
import cv2
from data import RSNA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = torch.load_state_dict('bilbily_grayscale_1101_152328.pt', map_location=torch.device('cpu'))
# print(type(model))

model = Bilbily(grayscale=True)
model.load_state_dict(torch.load('bilbily_grayscale_1101_152328.pt', map_location=torch.device('cpu'))['model_state_dict'])
model.eval()
# print(model.get_features())
# print(model._modules.get('inception')._modules.get('Mixed_7c'))

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((512, 512)), 
                                transforms.ToTensor()
                                ])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='../test_img/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

data, _ = next(iter(dataloader))
# print(data.shape)
data = data[:,0,:,:].unsqueeze(1)
# print(data.shape)

class SaveFeatures():
    """ Extract pretrained activations"""
    features=None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

final_layer = model._modules.get('inception')._modules.get('Mixed_7c')
activated_features = SaveFeatures(final_layer)

# get the most likely prediction of the model
pred = model(data, torch.Tensor([1]))
activated_features.remove() #let's do this to be safe

# print(activated_features.features.shape)

def getCAM(feature_conv, weight_fc, final_fc_weights):
    _, nc, h, w = feature_conv.shape

    cam = weight_fc[:,None,:nc].dot(feature_conv.reshape((-1, nc, h*w))).squeeze()
    # print("cam: {}, final_fc_weights: {}".format(cam.shape, final_fc_weights.shape))
    cam = cam.T@final_fc_weights.T
    cam = cam.reshape(h, w)
    
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img

weight_softmax_params = list(model._modules.get('fc').parameters())
# print('hello i am here \n', weight_softmax_params)
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
# weight_softmax_params
final_fc_weights = weight_softmax_params[2].data.numpy()

cur_images = data.cpu().numpy().transpose((0, 2, 3, 1))
heatmaps = []

data = data.expand(-1,3,-1,-1).squeeze().numpy().transpose((1, 2, 0))
data = data * 255

# for i in range(0, 1000, 50):

i = 'notbaby'
img = getCAM(activated_features.features, weight_softmax, final_fc_weights)
heatmap = cv2.resize(img, (data.shape[1], data.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + data * 0.6
cv2.imwrite('bilbily_XAI/didwedoitPRAYDGE_{}.png'.format(i), superimposed_img)

# ben has idea

    
# print(cur_images.shape, len(heatmaps))

# breakpoint()



# heatmap = heatmaps[0]

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
