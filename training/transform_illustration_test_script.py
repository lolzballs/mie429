from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T

class AddGaussianNoise(object):
    # Custom class to add gaussian noise to input images as data augmentation
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        print(tensor)
        print('out tensor:',tensor + torch.randn(tensor.size()) * self.std + self.mean)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

#Self defined functions must be defined for adjust contrast and sharpness because they are functional pytorch augmentations that cannot be chained via compose normally
def adjust_contrast_compose_transform(cf=1.5):
    def _func(img):
        return T.functional.adjust_contrast(img,contrast_factor=cf) # 1 for original image, 2 for increasing constrast by factor of 2, will have to be tuned to our data
    return _func 


def adjust_sharpness_compose_transform(cf=1.5):
    def _func(img):
        return T.functional.adjust_sharpness(img,sharpness_factor=cf) # 1 for original image, 2 for increasing sharpness by factor of 2, will have to be tuned to our data
    return _func 


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(T.functional.to_pil_image(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig('transform_test_'+row_title+'.png', bbox_inches='tight')

if __name__ == "__main__":
    '''
    Unit test script to visualize and plot transformations
    '''
    plt.rcParams["savefig.bbox"] = 'tight'
    orig_img = Image.open('data/Bone+Age+Training+Set/boneage-training-dataset/1417.png') #choose any specific image to test transformations on
    # if you change the seed, make sure that the randomly-applied transforms
    # properly show that the image can be both transformed and *not* transformed!
    torch.manual_seed(2)
    orig_img = T.ToTensor()(orig_img)
    print(orig_img)
    # Call transform(s) you want to visualize 
    orig_img = adjust_contrast_compose_transform(cf=1.2)(orig_img)
    orig_img = T.functional.normalize(orig_img,mean=0.1875,std=0.198)
    orig_img = AddGaussianNoise(0.,0.1)(orig_img)

    # Multiple transformed images can be added to list below to plot multiple images in output file in a row
    affine_imgs = [orig_img]
    # render plot to a file with name row_title
    plot(affine_imgs,row_title="img1417contrast12+norm+gaussNoise01")