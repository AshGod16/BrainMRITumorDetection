import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import numpy as np
import os
import sys
import copy
from tqdm import tqdm
from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)

        self.dropout = nn.Dropout(p=0.2)

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            )

        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3), # 224x224
            nn.MaxPool2d(2, stride=2), # 112x112
            nn.ReLU(True),
            nn.Conv2d(16, 24, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),  # 56x56
            nn.ReLU(True),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),  # 28x28
            nn.ReLU(True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),  # 14x14
            nn.ReLU(True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),  # 7x7
            nn.ReLU(True)
        )

        self.resnet_fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
            )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()

    # Spatial transformer network forward function
    def stn(self, x):
        x = self.pretrained(x)
        x = self.dropout(x)
        theta = self.resnet_fc(x)
        theta = nn.Flatten(theta)

        return theta

    def forward(self, x):
        # transform the input
        theta = self.stn(x)
        return theta