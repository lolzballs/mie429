from torchvision.models import resnet50, ResNet50_Weights,resnet34, ResNet34_Weights
import models
import torchvision.transforms as transforms

class AddGaussianNoise(object):
    # Custom class to add gaussian noise to input images as data augmentation
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

#Self defined functions must be defined for adjust contrast and sharpness because they are functional pytorch augmentations that cannot be chained via compose normally
def adjust_contrast_compose_transform(cf):
    def _func(img):
        return transforms.functional.adjust_contrast(img,contrast_factor=cf) # 1 for original image, 2 for increasing constrast by factor of 2, will have to be tuned to our data
    return _func 


class ModelManager():
    def __init__(self,input_size=512,contrast_factor = 1.2, gaussian_std = 0.1, affine_rotation=30, affine_translate_ratio=0.2):
        """
        This class should manage all the models we want to experiment with, 
        either from direct import from pytorch or from the models folder
        """
        self.pretraining_source = ["imagenet",'random']
        # NOTE IF adding noise, MUST add noise AFTER normalize, or else image will become all noise
        # Current hyperparams are determined to be best after empirical tests
        # Standard transformation stack should be contrast -> normalize -> gaussianNoise. 
        # Resizing can be done before or after contrast
        self.transform_string_key_base = {'gaussiannoise':AddGaussianNoise(0.,gaussian_std), #MUST set std to a value between 0 - 0.2 or else image becomes TOO much noise since tensor pixels are autoscaled to 0-1
                                        'normalize':transforms.Normalize((0.1875,),(0.198,)),
                                        'resize':transforms.Resize(input_size),
                                        'adjust_contrast':adjust_contrast_compose_transform(contrast_factor),
                                        'random_affine':transforms.RandomAffine(degrees=affine_rotation,translate=(affine_translate_ratio,affine_translate_ratio))} #random image rotation by arg degree and translate by arg.decimal% of input_image size


    def get_model(self, model_name="resnet34", pretrain_source="imagenet", **kwargs):
        if model_name == "resnet34":
            return self.pretrained_resnet34(pretrain_source)
        elif model_name == "resnet50":
            return self.pretrained_resnet50(pretrain_source)
        elif model_name == "simpleconv":
            return models.Simpleconv(), None
        elif model_name == "inceptionv3":
            return models.InceptionV3(), None
        elif model_name == "bilbily":
            return models.Bilbily(**kwargs), None
        else:
            raise ValueError("Wrong model name")

    def pretrained_resnet50(self,pretrain_source="imagenet"):

        if pretrain_source not in self.pretraining_source:
            raise NotImplementedError(f"The specified pretrain_source argument {pretrain_source} is not supported, please use one of the following options: {' '.join(self.pretraining_source)}")
        
        if pretrain_source == "imagenet":

            resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Some pretrained models have packaged preprocessing transforms that should be applied/wrapped to input to ensure performance
            return resnet50_model, ResNet50_Weights.DEFAULT.transforms()
            
        elif pretrain_source == "random":

            resnet50_model = resnet50(weights=None)
            return resnet50_model, None
 
    def pretrained_resnet34(self,pretrain_source="imagenet"):

            if pretrain_source not in self.pretraining_source:
                raise NotImplementedError(f"The specified pretrain_source argument {pretrain_source} is not supported, please use one of the following options: {' '.join(self.pretraining_source)}")
            
            if pretrain_source == "imagenet":

                resnet34_model = resnet34(weights=ResNet34_Weights.DEFAULT)
                # Some pretrained models have packaged preprocessing transforms that should be applied/wrapped to input to ensure performance
                return resnet34_model, ResNet34_Weights.DEFAULT.transforms()
                
            elif pretrain_source == "random":

                resnet34_model = resnet34(weights=None)
                return resnet34_model, None

    def get_data_transform(self,transform_string_keys):
        compose_list = [transforms.ToTensor()]
        for t in transform_string_keys:
            if t.lower() not in self.transform_string_key_base.keys():
                raise NotImplementedError(f"The specified data transform  {t} is not supported, please use one of the following options: {' '.join(self.transform_string_key_base.keys())}")
            compose_list.append(self.transform_string_key_base[t])
        
        composed_transforms = transforms.Compose(compose_list)
        return composed_transforms

