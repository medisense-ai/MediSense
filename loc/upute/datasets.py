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

#############################
# TRANSFORMACIJE (RE-SIZE)  #
#############################

class ResizeMammoClassification(object):
    """Resize images to a given size.
    
    Based on SRC: https://pytorch.org/tutorials/beginner/data_loading_tutorial
    """
    def __init__(self, output_size):
        # output_size can be an int or a tuple (height, width)
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        # sample is expected to be a numpy array (an image)
        return transform.resize(sample, self.output_size)

#############################
# DATASET ZA KLASIFIKACIJU #
#############################

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
        self.images_dir = Path(data_dir) / 'images'         
        self.classification_df = pd.read_csv(os.path.join(data_dir, 'classification.csv'))
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
                    # Save the path to the image as a string in the dictionary.
                    # For example: images['00123']['CC_L'] = 'path/to/00123/CC_L.jpg'
                    key = img[-8:-4] if 'CC' in img else img[-9:-4]
                    self.images[case_dir.stem][key] = str(case_dir / str(img))
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
            sample = {k: self.transform(sample[k]) for k in sample}
            
        return f"{self.cases[idx]}", sample

#############################
# DATASET ZA LOKALIZACIJU  #
#############################

import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict

class MammoLocalizationDataset(Dataset):
    def __init__(self, data_dir=None, transform=None, resize_output_size=None):
        self.images_dir = Path(data_dir) / 'images'
        self.localization_df = pd.read_csv(os.path.join(data_dir, 'localization.csv'))
        
        # Sort DataFrame by 'case_id' and 'image_id'
        self.localization_df.sort_values(by=['case_id', 'image_id'], inplace=True)
        
        self.transform = transform
        self.resize_output_size = resize_output_size  
        self.cases = defaultdict(list)  # Dictionary to store multiple annotations per image
        
        for _, row in self.localization_df.iterrows():
            case_id = row['case_id']
            image_id = row['image_id']
            image_path = self.images_dir / f"{case_id}" / f"{image_id}.jpg"

            if not image_path.exists():
                print(f"Warning: Image {image_path} not found. Skipping.")
                continue  # Skip missing images

            # Store all annotations per image
            ann = row[['category', 'xmin', 'ymin', 'xmax', 'ymax']].fillna(0).to_dict()
            self.cases[(case_id, image_id, image_path)].append(ann)

        # Convert dict keys to list to preserve order
        self.image_keys = list(self.cases.keys())

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        case_id, image_id, image_path = self.image_keys[idx]
        sample = io.imread(str(image_path))  # Load image
        
        # if sample.ndim == 2:  # Convert grayscale to 3 channels
        #     sample = np.stack([sample] * 3, axis=-1)
        
        orig_shape = sample.shape[:2]
        annotations = self.cases[(case_id, image_id, image_path)]

        if self.resize_output_size is not None:
            new_H, new_W = self.resize_output_size
            scale_y = new_H / orig_shape[0]
            scale_x = new_W / orig_shape[1]

            sample = transform.resize(sample, self.resize_output_size)

            # Scale all bounding boxes
            scaled_annotations = []
            for ann in annotations:
                scaled_ann = {
                    'category': ann['category'],
                    'xmin': ann['xmin'] * scale_x,
                    'ymin': ann['ymin'] * scale_y,
                    'xmax': ann['xmax'] * scale_x,
                    'ymax': ann['ymax'] * scale_y,
                }
                scaled_annotations.append(scaled_ann)
        else:
            scaled_annotations = annotations  # Keep original annotations

        if self.transform:
            sample = self.transform(sample)

        return case_id, sample, scaled_annotations



#############################
# DODATNE TRANSFORMACIJE   #
#############################

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

#############################
# POMOĆNE FUNKCIJE         #
#############################

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