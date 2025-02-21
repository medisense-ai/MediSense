import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def auto_crop(image, threshold=10, margin=20):
    """
    Auto-crops the image by removing regions with very low intensity,
    but leaves a margin around the detected area to retain some background.
    Returns the cropped image along with the x and y offsets.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Create a binary mask: non-black pixels become white
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Find coordinates of non-black pixels
    coords = cv2.findNonZero(thresh_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Expand the bounding box by a margin, ensuring we remain within the image bounds
        x_new = max(x - margin, 0)
        y_new = max(y - margin, 0)
        x_end = min(x + w + margin, image.shape[1])
        y_end = min(y + h + margin, image.shape[0])
        cropped = image[y_new:y_end, x_new:x_end]
        return cropped, x_new, y_new
    else:
        # If the entire image is black, return the original image with zero offsets
        return image, 0, 0

def get_transform(train):
    if train:
        return A.Compose(
            [
                # Use cv2 constants for border/interpolation
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,  # replaces 'border_mode=0'
                    p=0.5
                ),
                A.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.5
                ),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
    else:
        return A.Compose(
            [
                A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

class MammographyLocalizationDataset(Dataset):
    def __init__(self, csv_file, img_dir, laterality, view, transform=None, auto_crop_enabled=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.auto_crop_enabled = auto_crop_enabled

        # Extract laterality and view from the image_id (expects format like "20_L_CC")
        self.data['laterality'] = self.data['image_id'].apply(lambda x: x.split('_')[1])
        self.data['view'] = self.data['image_id'].apply(lambda x: x.split('_')[2])
        
        # Filter dataset by the specified laterality and view
        self.data = self.data[(self.data['laterality'] == laterality) & (self.data['view'] == view)]
        
        # Each sample corresponds to a unique image_id
        self.image_ids = self.data['image_id'].unique()

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Identify the row(s) corresponding to this image_id
        image_id = self.image_ids[idx]
        rows = self.data[self.data['image_id'] == image_id]
        
        # We assume 'case_id' matches the subfolder name (0, 1, 2, etc.)
        folder = rows['case_id'].values[0]  # Make sure your CSV has this column
        img_file = f"{image_id}.jpg"
        img_path = os.path.join(self.img_dir, str(folder), img_file)

        image = np.array(Image.open(img_path).convert("RGB"))
        
        crop_offset_x, crop_offset_y = 0, 0
        if self.auto_crop_enabled:
            # Auto-crop to remove large black regions, retaining some background with a margin
            image, crop_offset_x, crop_offset_y = auto_crop(image, threshold=10, margin=20)
        
        boxes = []
        labels = []
        for _, row in rows.iterrows():
            if pd.notna(row['xmin']) and pd.notna(row['ymin']) and pd.notna(row['xmax']) and pd.notna(row['ymax']):
                # Adjust bounding box coordinates by subtracting the crop offsets
                bbox = [
                    float(row['xmin']) - crop_offset_x,
                    float(row['ymin']) - crop_offset_y,
                    float(row['xmax']) - crop_offset_x,
                    float(row['ymax']) - crop_offset_y
                ]
                boxes.append(bbox)
                labels.append(row['category'])
        
        # Apply transformations (which also adjust bounding boxes)
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Convert bounding boxes to a tensor (if no boxes, create an empty tensor)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        detection_flag = 1 if boxes.shape[0] > 0 else 0
        
        return image, boxes, labels, detection_flag

# Example usage:
if __name__ == "__main__":
    # Example: reading from data/train/ with subfolders for each case_id
    transform = get_transform(train=True)
    
    dataset = MammographyLocalizationDataset(
        csv_file='/home/ja/new_env/AI4Health/testing/train/localization.csv',
        img_dir='/home/ja/new_env/AI4Health/testing/train/images',
        laterality='L',   # Example: left breast
        view='CC',        # Example: craniocaudal view
        transform=transform,
        auto_crop_enabled=True
    )
    
    # Use a simple collate function to handle variable numbers of bounding boxes per image
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)
    
    # Example iteration over one batch
    for sample in dataloader:
        for image, boxes, labels, detection_flag in sample:
            print("Image shape:", image.shape)       # Expected: (3, 224, 224)
            print("Boxes:", boxes)                   # Tensor of shape (N, 4) or (0, 4)
            print("Labels:", labels)                 # List of label strings
            print("Detection flag:", detection_flag) # 1 if lesion is present, 0 if not
        break
