import torchvision
import matplotlib.pyplot as plt


def crop_image(entry):
    *rest, image = entry
    return *rest, torchvision.transforms.functional.center_crop(image, (1024, 1024))


