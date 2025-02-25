import os
import numpy as np
import torch
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from typing import Dict, List, Tuple, Union, Optional

#######################################
# Classification modeli (ostaju isti) #
#######################################

class MammoBaseClassificationNet(nn.Module):
    def __init__(self, categories=6, pretrained=False):
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
    
    
class MammoClassificationResNet18(nn.Module):
    def __init__(self, categories=6, pretrained=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, stride=2, kernel_size=(5,5), padding=2)
        if pretrained: 
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        else: 
            weights = None
        self.resnet18 = torchvision.models.resnet18(weights=weights)
        self.resnet18.fc = nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.base = nn.Sequential(
            self.conv1, self.dropout, self.relu,
            self.resnet18,
        )
        self.fc1 = nn.Linear(4*512, 256)
        self.fc2 = nn.Linear(256, categories)

    def forward(self, x):
        y = []
        for k in ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']:
            y.append(self.base(x[k]))
        y = torch.concat(y, dim=1)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        return y


class RandomCategoryClassifier:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        # Calculate probability distribution of categories
        self.category_distribution = self.calculate_category_distribution()
        # Store unique categories
        self.categories = sorted(self.df['category'].unique())
        
    def calculate_category_distribution(self) -> dict:
        """Calculate the probability distribution of categories in the training data"""
        category_counts = self.df.groupby('case_id')['category'].first().value_counts()
        total_cases = len(self.df['case_id'].unique())
        probabilities = (category_counts / total_cases).to_dict()
        return probabilities
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict categories and generate probabilities for each class.
        Returns DataFrame with case_id, predictions, and probabilities for each class.
        """
        # Get unique case_ids
        case_ids = df['case_id'].unique()
        
        # Generate predictions and probabilities for each case
        predictions = []
        for case_id in case_ids:
            # Generate random class based on category distribution
            pred_class = random.choices(
                population=list(self.category_distribution.keys()),
                weights=list(self.category_distribution.values()),
                k=1
            )[0]
            random_values = np.random.rand(len(self.category_distribution.keys()) - 1)
            random_values /= random_values.sum()
            max_value = np.random.uniform(0.5, 0.9)  # Ensure it's the highest
            scaled_values = (1 - max_value) * random_values
            
            predictions.append({
                'case_id': case_id,
                'preds': pred_class,
                'probabilityC0': scaled_values[0],
                'probabilityC1': scaled_values[1],
                'probabilityC2': scaled_values[2],
                'probabilityC3': scaled_values[3]
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)    
        return results_df


##########################################
# NOVI SEGMENTACIJSKI MODEL ZA LOKALIZACIJU #
##########################################

# Definicija osnovnih blokova za U-Net arhitekturu

class DoubleConv(nn.Module):
    """Dva uzastopna konvolucijska sloja s BatchNorm i ReLU aktivacijom."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling (max pool) slijed konvolucije."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling i spajanje s odgovarajućim encoder značajkama."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Poravnaj dimenzije ako se ne podudaraju
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Završni konvolucijski sloj koji mapira na broj izlaznih kanala (klasa)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class MammoSegmentationUNet(nn.Module):
    """
    Segmentacijski model za lokalizaciju lezija na rendgenskim slikama dojke.
    Ovaj model koristi U-Net arhitekturu kako bi generirao piksel-po-piksel masku.
    
    Ulaz: jednokanalna slika (npr. 1xHxW)
    Izlaz: maska s 'n_classes' kanala, gdje se za svaki kanal primjenjuje sigmoid
           (multi-label segmentation – piksel može pripadati više klasa).
    """
    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16)
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        logits = self.outc(x) # (B, n_classes, H, W)
        # Za multi-label segmentation koristimo sigmoid (ne softmax)
        return torch.sigmoid(logits)


##############################
# MODEL MAPA I POMOĆNE FUNKCIJE #
##############################

MODEL_NAMES = [
    "mammo-cls-resnet18", 
    "mammo-cls-base", 
    "mammo-random",
    "mammo-segmentation-unet"  # novi model za segmentaciju
]

MODEL_MAP = {
    "mammo-cls-resnet18": MammoClassificationResNet18,
    "mammo-cls-base": MammoBaseClassificationNet,
    "mammo-random": RandomCategoryClassifier,
    "mammo-segmentation-unet": MammoSegmentationUNet,
}

def get_model(model_name: str = MODEL_NAMES[0], categories: int = 6):
    """Get model by model name."""
    return MODEL_MAP[model_name]
    
    
#############################################
# Ostale klase (npr. RandomLocalization) ostaju #
#############################################

class RandomLocalization():
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.unique_images_per_category = self.df.groupby('category')['image_id'].nunique()
        # Divide by total number of images to get ratios
        self.probabilities_dict = self.get_probabilities_dict()
        self.average_area_dict = self.get_average_area_dict()
        
    def get_probabilities_dict(self):   
        probabilities_series = self.unique_images_per_category / self.df['image_id'].nunique()
        return probabilities_series.to_dict()
        
    def get_average_area_dict(self) -> dict:
        average_area_dict = {}
        self.df['area'] = (self.df['xmax'] - self.df['xmin']) * (self.df['ymax'] - self.df['ymin'])
        self.df['area'] = self.df['area'].fillna(0)
        for category in self.df['category'].unique():
            category_df = self.df[self.df['category'] == category]
            sum_area = category_df[category_df['area'] > 0]['area'].sum()
            count_area = category_df[category_df['area'] > 0].shape[0]
            average_area = sum_area / count_area if count_area > 0 else 0
            average_area_dict[category] = average_area
        return average_area_dict
    
    def generate_random_category(self) -> str:
        categories = list(self.probabilities_dict.keys())
        probabilities = list(self.probabilities_dict.values())
        return random.choices(categories, probabilities)[0]
    
    def predict(self, df) -> pd.DataFrame:
        df['category'] = df.apply(lambda row: self.generate_random_category(), axis=1)
        df['laterality'] = df['image_id'].apply(lambda x: x.split('_')[1])
        df['view'] = df['image_id'].apply(lambda x: x.split('_')[2].split('.')[0])
        df_tmp = df[df['category'] != 'No_Finding']
        df_tmp_copy = df_tmp.copy()
        df_tmp_copy['view'] = df_tmp_copy['view'].apply(lambda x: 'MLO' if x == 'CC' else 'CC')
        df_tmp_copy['image_id'] = df_tmp_copy.apply(
            lambda row: f"{row['case_id']}_{row['laterality']}_{row['view']}", axis=1)
        df = df[~df['image_id'].isin(df_tmp_copy['image_id'])]
        df = pd.concat([df, df_tmp_copy], ignore_index=True)
        df = df.sort_values(by='case_id').reset_index(drop=True)
        return df
