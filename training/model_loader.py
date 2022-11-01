from torchvision.models import resnet50, ResNet50_Weights,resnet34, ResNet34_Weights
import models

class ModelManager():
    def __init__(self):
        """
        This class should manage all the models we want to experiment with, 
        either from direct import from pytorch or from the models folder
        """
        self.pretraining_source = ["imagenet",'random']

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
        elif model_name == "vgg16":
            return models.Vgg16(), None
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
