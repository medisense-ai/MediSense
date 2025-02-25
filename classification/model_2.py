import os
import numpy as np
import torch
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, f1_score

from common.local.logger import Logger

class MammoClassificationResNet50(nn.Module):
    def __init__(self, categories=4, pretrained=False, logger=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if pretrained: 
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        else: 
            weights = None

        self.conv1 = torch.nn.Conv2d(1, 3, stride=2, kernel_size=(5,5), padding=2)
        self.dropout = nn.Dropout(0.2)
    
        self.resnet50 = torchvision.models.resnet50(weights=weights)
        self.resnet50.fc = nn.Identity()
        self.fc1 = nn.Linear(4*512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, categories)
        
        self.model = nn.Sequential(
            self.conv1,
            self.dropout,
            self.relu,
            self.resnet50,
            self.fc1,
            self.relu,
            self.fc2
        )

        self.logger = logger

    def forward(self, x):
        y = self.model(x)
        return y

    def train_model(self, dataloader):
        if self.logger:
                self.logger.info(
                    f"Training model"
                )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.to(self.device)

        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1}"
                )
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if self.logger:
                self.logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}"
                )

        if self.logger:
            self.logger.info("Training complete")
        else:
            print("Training complete")

    def evaluate_model(self, dataloader):
        '''
        Evaluates following metrics of model: 
            precision, recall, F1 - per class
            MCC, F1 (multi-class macro average)
        Returns a dictionary with these metrics
        '''

        # set model to evaluation mode
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        # disable gradient calculation for evaluation
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # forward pass
                outputs = self.model(inputs)
                
                # get predictions
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # calculate precision, recall, f1-score per class
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
        
        # calculate multi-class macro average F1
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        # calculate Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        metrics = {
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'f1_macro': f1_macro,
            'mcc': mcc
        }
        
        return metrics
        
