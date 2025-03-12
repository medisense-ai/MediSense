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
    def __init__(self, categories=4, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, stride=2, kernel_size=(9,9), padding=4)
        self.conv2 = nn.Conv2d(4, 8, stride=1, kernel_size=(5,5), padding=2)
        self.conv3 = nn.Conv2d(8, 16, stride=1, kernel_size=(5,5), padding=2)
        self.conv4 = nn.Conv2d(16, 32, stride=1, kernel_size=(3,3), padding=1)
        self.conv5 = nn.Conv2d(32, 32, stride=1, kernel_size=(3,3), padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.base = nn.Sequential(
            self.conv1, self.dropout, self.relu, self.pool,
            self.conv2, self.dropout, self.relu, self.pool,
            self.conv3, self.dropout, self.relu, self.pool,
            self.conv4, self.dropout, self.relu, self.pool,
            self.conv5, self.dropout, self.relu, self.pool,
            self.flatten,
            self.fc1, self.relu,
            self.fc2, self.relu,
        )
        self.fc3 = nn.Linear(4*64, 24)
        self.fc4 = nn.Linear(24, categories)

    def forward(self, x):
        y = []
        for k in ['CC_L', 'CC_R', 'LO_L', 'LO_R']:
            y.append(self.base(x[k]))
        y = torch.concat(y, dim=1)
        y = self.relu(self.fc3(y))
        y = self.fc4(y)
        return y
    
    def load_model(self, model_path):
        state_dict = torch.load(model_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        current_state_dict = self.model.state_dict()

        print(set(current_state_dict.keys()) - set(new_state_dict.keys()))
        print(set(new_state_dict.keys()) - set(current_state_dict.keys()))

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

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
        
