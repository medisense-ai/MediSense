import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

from common.local.dataset import MammographyDataset


class MammographyModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MammographyModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

    def train_model(self, dataloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}"
            )

        print("Training complete")
