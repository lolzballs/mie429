from torchvision.models import resnet50, ResNet50_Weights,resnet34, ResNet34_Weights

class ModelManager():
    def __init__(self):
        """
        This class should manage all the models we want to experiment with, 
        either from direct import from pytorch or from the models folder
        """
        self.pretraining_source = ["imagenet",'random']

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