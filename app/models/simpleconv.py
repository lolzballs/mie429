import torch
import torch.nn as nn
import torch.nn.functional as F


class Simpleconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.conv2 = nn.Conv2d(4, 5, 4)
        self.conv3 = nn.Conv2d(5, 6, 4)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(6 * 15 * 15, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #4,255,255
        x = self.pool(F.relu(self.conv2(x))) #5,63,63
        x = self.pool(F.relu(self.conv3(x))) #6,15,15
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x