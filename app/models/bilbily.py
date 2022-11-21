from typing import Literal

import torch
import torch.nn as nn
import torchvision

import utils


class Bilbily(nn.Module):
    """
    Based on Alexander Bilbily and Mark Cicero's submission to the 2017 RSNA
    Pediatric Bone Age Challenge.
    """

    def __init__(self, sex_encoding: Literal['onehot', 'binary'] = 'onehot',
                 grayscale: bool = True, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.Inception_V3_Weights.DEFAULT if pretrained else None
        self.inception = torchvision.models.inception_v3(weights=weights)
        inception_out_size = self.inception.fc.in_features

        if grayscale:
            self.inception.Conv2d_1a_3x3 = \
                    torchvision.models.inception.BasicConv2d(1, 32, kernel_size=3, stride = 2)
            self.inception.transform_input = False

        # remove aux_logits
        self.inception.aux_logits = False

        # replace the last layer with identity
        self.inception.fc = nn.Identity() # type: ignore

        self.sex_encoding = sex_encoding
        if sex_encoding == 'onehot':
            sex_input_size = 2
        elif sex_encoding == 'binary':
            sex_input_size = 1
        else:
            raise ValueError(f'{sex_encoding} must be either onehot or binary')
        self.sex_fc = nn.Linear(sex_input_size, 32)

        self.fc = nn.Sequential(
            nn.Linear(inception_out_size + 32, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )

        if not pretrained:
            self.apply(utils.apply_to_nn(torch.nn.init.orthogonal_))

    def forward(self, x, sex):
        if self.sex_encoding == 'onehot':
            sex = nn.functional.one_hot(sex.long(), 2).float()
        elif self.sex_encoding == 'binary':
            sex = torch.unsqueeze(sex,1)
        else:
            raise NotImplementedError("incorrect sex_encoding argument")

        x = self.inception(x)
        sex = self.sex_fc(sex)

        if torch.jit.is_scripting():
            x = x.logits
        x = torch.cat((x, sex), 1)
        x = self.fc(x)
        return x
