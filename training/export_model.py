import argparse
import copy
from typing import Tuple
import yaml

import torch
import torchvision

from model_loader import ModelManager


def save_activations(module, _input: Tuple[torch.Tensor], output: torch.Tensor):
    module.activations = output.data


class ModelWithTransforms(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.transforms = torch.nn.Sequential(*model_manager.get_data_transform([
            'resize',
            'normalize',
        ]).transforms)

        final_layer = model._modules.get('inception')._modules.get('Mixed_7c')
        final_layer.activations = torch.zeros((1, 2048, 14, 14))
        final_layer.register_forward_hook(save_activations)

    def forward(self, x: torch.Tensor, sex: torch.Tensor):
        with torch.no_grad():
            x = self.transforms(x)
            x = torchvision.transforms.functional.adjust_contrast(x, contrast_factor=1.2)
        return self.model(x, sex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exports a checkpoint .pt "
                                     "file to a self-contained model")
    parser.add_argument("pt")
    parser.add_argument("hyperparams")
    parser.add_argument("out")
    args = parser.parse_args()

    with open(args.hyperparams, 'r') as stream:
        hp = yaml.safe_load(stream)

    model_params = copy.deepcopy(hp['model'])
    del model_params['name']
    model_manager = ModelManager()

    model, _ = model_manager.get_model(
        model_name=hp['model']['name'],
        **model_params,
    )
    model.load_state_dict(torch.load(args.pt, map_location=torch.device('cpu'))['model_state_dict'])

    model_with_transforms = ModelWithTransforms(model)
    model_with_transforms.eval()
    torchscript = torch.jit.script(model_with_transforms)
    torchscript.save(args.out)
