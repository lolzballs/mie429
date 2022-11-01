from typing import List

import torch.nn as nn
import torchvision


def apply_to_nn(function, instances: List = [nn.Linear, nn.Conv2d]):
    def apply(m):
        if any(isinstance(m, i) for i in instances):
            function(m.weight)
    return apply


def apply_to_image(function):
    def apply(entry):
        *rest, image = entry
        return *rest, function(image)
    return apply


def crop_image(entry):
    *rest, image = entry
    return *rest, torchvision.transforms.functional.center_crop(image, (1024, 1024))
