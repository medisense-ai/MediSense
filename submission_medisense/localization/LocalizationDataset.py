import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Tuple, Dict, Any

# Define a mapping from lesion category to numeric labels
CATEGORY_MAP = {
    "Mass": 1,
    "Suspicious_Calcification": 2,
    "Focal_Asymmetry": 3,
    "Architectural_Distortion": 4,
    "Suspicious_Lymph_Node": 5,
    "Other": 6,
}

class ResizeWithBBoxes(object):
    """
    Custom transform that resizes images and scales bounding boxes.
    Note: The target size should be provided as (width, height) to match PIL conventions.
    """
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        # Store original size (width, height)
        orig_w, orig_h = image.size
        
        # Resize the image
        image = transforms.functional.resize(image, self.size)
        
        # Retrieve new size (should match self.size)
        new_w, new_h = image.size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        # Scale bounding boxes
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] *= scale_x  # Adjust x coordinates
        boxes[:, [1, 3]] *= scale_y  # Adjust y coordinates
        
        target["boxes"] = boxes
        return image, target

class NormalizeImageNet(object):
    """
    Normalize image tensor using ImageNet statistics.
    
    This transform expects the image to be a tensor with values in [0,1]
    and then applies the normalization expected by a ResNet backbone.
    """
    def __call__(self, image, target):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        return image, target

class MammographyLocalizationDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with columns:
                [case_id, patient_id, study_id, image_id, category, xmin, ymin, xmax, ymax].
            img_dir (str): Directory containing the images.
            transform (callable, optional): Function/transform taking in a PIL image and target dict.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Group annotations by image (using case_id and image_id as unique identifiers)
        self.grouped = self.data.groupby(["case_id", "image_id"])
        self.image_keys = list(self.grouped.groups.keys())

    def __len__(self) -> int:
        return len(self.image_keys)

    def __getitem__(self, idx: int):
        case_id, image_id = self.image_keys[idx]
        img_file = f'{case_id}/{image_id}.jpg'
        img_path = os.path.join(self.img_dir, img_file)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Get all annotations for this image
        annotations = self.grouped.get_group((case_id, image_id))
        boxes = []
        labels = []
        for _, row in annotations.iterrows():
            if row["category"] == "No_Finding":
                continue  # Skip No_Finding rows
            # Only process rows with valid bounding box coordinates
            if pd.isnull(row["xmin"]) or pd.isnull(row["ymin"]) or pd.isnull(row["xmax"]) or pd.isnull(row["ymax"]):
                continue
            boxes.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            label = CATEGORY_MAP.get(row["category"], 0)  # Map the category to a numeric label
            labels.append(label)
            
        # If no valid boxes found, return empty tensors (image with no lesions)
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target
