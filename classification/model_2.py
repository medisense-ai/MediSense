import os
import numpy as np
import torch
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class MammoClassificationResNet50(nn.Module):
    def __init__(self, categories=6, pretrained=False):
        super().__init__()
        # self.conv1 = torch.nn.Conv2d(1, 3, stride=2, kernel_size=(5, 5), padding=2)
        if pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        self.resnet50 = torchvision.models.resnet50(weights=weights)
        self.resnet50.fc = nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.base = nn.Sequential(
            # self.conv1,
            self.dropout,
            self.relu,
            self.resnet50,
        )
        self.fc1 = nn.Linear(4 * 512, 256)
        self.fc2 = nn.Linear(256, categories)

    def forward(self, x):
        y = []
        for k in ["L_CC", "R_CC", "L_MLO", "R_MLO"]:
            y.append(self.base(x[k]))
        y = torch.concat(y, dim=1)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        return y
