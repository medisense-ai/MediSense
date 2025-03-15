"""Script to load data.

File structure, existing in evaluation environment:
    -data/
        -test/
            -images/
                -...
            -localization.csv
            -classification.csv
        -train/
            -images/
                -...
            -localization.csv
            -classification.csv
"""
import os
import random
import torch
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from collections import Counter
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

class MammoClassificationDataset(Dataset):
    """Mammo classification dataset.
    
    Based on SRC: https://pytorch.org/tutorials/beginner/data_loading_tutorial
    """        
    def __init__(self, data_dir=None, transform=None):
        """
        Arguments:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = Path(data_dir) / 'images'         # Mandatory     
        self.classification_df = pd.read_csv(               # Mandatory
            os.path.join(data_dir, 'classification.csv'))   # Mandatory
        r = range(len(self.classification_df))
        self.transform = transform
        self.images = {}
        self.cases = []
        for idx in r:
            case_dir = self.images_dir / f"{self.classification_df.loc[idx, 'case_id']}"
            self.images[case_dir.stem] = {}
            image_paths = os.listdir(case_dir)
            if len(image_paths) == 4:
                for img in image_paths:
                    # Save the path to the image as a string in the dictionary
                    # e.g. images['00123']['CC_L'] = 'path/to/00123/CC_L.jpg'
                    self.images[case_dir.stem][img[-8:-4] if 'CC' in img else img[-9:-4]] = str(case_dir / str(img))
                self.cases.append(case_dir.stem)
            else:
                pass
        
    def __len__(self):
        return len(self.classification_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        patient_dir = self.images_dir / f"{self.cases[idx]}"
        for k in self.images[patient_dir.stem]:
            sample[k] = io.imread(self.images[patient_dir.stem][k])
            
        if self.transform:
            sample = self.transform(sample)

        return f"{self.cases[idx]}", sample
    
class MammoLocalizationDataset(Dataset):
    """Mammo Localization dataset.
    
    Based on SRC: https://pytorch.org/tutorials/beginner/data_loading_tutorial
    """        
    def __init__(self, data_dir=None, transform=None):
        """
        Arguments:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = Path(data_dir) / 'images'         # Mandatory     
        self.classification_df = pd.read_csv(               # Mandatory
            os.path.join(data_dir, 'localization.csv'))     # Mandatory
        r = range(0, len(self.classification_df))
        self.transform = transform
        self.images = {}
        self.cases = []
        for idx in r:
            case_dir = self.images_dir / f"{self.classification_df.loc[idx, 'case_id']}"
            self.images[case_dir.stem] = {}
            image_paths = os.path.join(case_dir, f"{self.classification_df.loc[idx, 'image_id']}.jpg")
            self.cases.append(image_paths)

        
    def __len__(self):
        return len(self.classification_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = io.imread(self.cases[idx])
        if self.transform:
            sample = self.transform(sample)

        return os.path.basename(os.path.dirname(self.cases[idx])), sample
    

class ToTensorMammoClassification(object):
    """Convert ndarrays in sample to Tensors.
    
    Based on SRC: https://pytorch.org/tutorials/beginner/data_loading_tutorial
    """

    def __call__(self, sample):
        return {'L_CC': torch.from_numpy(sample['L_CC']).float().unsqueeze(dim=0),
                'R_CC': torch.from_numpy(sample['R_CC']).float().unsqueeze(dim=0),
                'L_MLO': torch.from_numpy(sample['L_MLO']).float().unsqueeze(dim=0),
                'R_MLO': torch.from_numpy(sample['R_MLO']).float().unsqueeze(dim=0)}


class NormalizeMammoClassification(object):
    """Normalize images to [-1,1].
    
    Based on SRC: https://pytorch.org/tutorials/beginner/data_loading_tutorial
    """

    def __call__(self, sample):
        return {'L_CC': sample['L_CC'] / 255 * 2 - 1,
                'R_CC': sample['R_CC'] / 255 * 2 - 1,
                'L_MLO': sample['L_MLO'] / 255 * 2 - 1,
                'R_MLO': sample['R_MLO'] / 255 * 2 - 1}


class ResizeMammoClassification(object):
    """Resize images to a given size.
    
    Based on SRC: https://pytorch.org/tutorials/beginner/data_loading_tutorial
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        for k in sample:
            sample[k] = transform.resize(sample[k], self.output_size)
        return sample
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_dataloader(dataset, batch_size=64, shuffle=False,
                num_workers=0, seed=1234, use_sampler=False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g)
    return dataloader