import os
import torch
import numpy as np
import pandas as pd
import random
import math
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def rotate_image_and_boxes(image, boxes, angle):
    """
    Rotate the PIL image around its center by `angle` degrees.
    Update bounding boxes accordingly (pixel coords).
    """
    width, height = image.size
    theta = math.radians(angle)
    rotated_image = image.rotate(angle, fillcolor=(0, 0, 0))
    center_x, center_y = width / 2.0, height / 2.0
    rotated_boxes = []
    for (xmin, ymin, xmax, ymax) in boxes:
        corners = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ], dtype=np.float32)
        rotated_corners = []
        for cx, cy in corners:
            tx = cx - center_x
            ty = cy - center_y
            rx = tx * math.cos(theta) - ty * math.sin(theta)
            ry = tx * math.sin(theta) + ty * math.cos(theta)
            fx = rx + center_x
            fy = ry + center_y
            rotated_corners.append([fx, fy])
        rotated_corners = np.array(rotated_corners)
        x_min_new = rotated_corners[:, 0].min()
        x_max_new = rotated_corners[:, 0].max()
        y_min_new = rotated_corners[:, 1].min()
        y_max_new = rotated_corners[:, 1].max()
        rotated_boxes.append([x_min_new, y_min_new, x_max_new, y_max_new])
    return rotated_image, rotated_boxes

def resize_image_and_boxes(image, boxes, target_size=(224, 224)):
    """
    Resize the PIL image to target_size and scale bounding boxes accordingly.
    """
    orig_width, orig_height = image.size
    new_width, new_height = target_size
    resized_image = image.resize(target_size, Image.BILINEAR)
    x_scale = new_width / orig_width
    y_scale = new_height / orig_height
    resized_boxes = []
    for (xmin, ymin, xmax, ymax) in boxes:
        xmin_resized = xmin * x_scale
        xmax_resized = xmax * x_scale
        ymin_resized = ymin * y_scale
        ymax_resized = ymax * y_scale
        resized_boxes.append([xmin_resized, ymin_resized, xmax_resized, ymax_resized])
    return resized_image, resized_boxes

class MammographyLocalizationDataset(Dataset):
    """
    Dataset for Mammography Localization.
      - Reads bounding boxes from a CSV.
      - Applies optional random rotation.
      - Resizes images to (224, 224) and converts to tensor.
      - Does not perform horizontal flipping.
    """
    def __init__(self, csv_file, img_dir, laterality, view, train=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train = train

        # Extract laterality and view from image_id (e.g., "20_L_CC")
        self.data['laterality'] = self.data['image_id'].apply(lambda x: x.split('_')[1])
        self.data['view'] = self.data['image_id'].apply(lambda x: x.split('_')[2])
        self.data = self.data[(self.data['laterality'] == laterality) & (self.data['view'] == view)]
        self.image_ids = self.data['image_id'].unique()

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        rows = self.data[self.data['image_id'] == image_id]
        folder = rows['case_id'].values[0]
        img_file = f"{image_id}.jpg"
        img_path = os.path.join(self.img_dir, str(folder), img_file)

        # Error handling for image loading
        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a dummy image if loading fails
            image_pil = Image.new("RGB", (224, 224), color=(0, 0, 0))
        
        # Collect bounding boxes and labels
        boxes = []
        labels = []
        for _, row in rows.iterrows():
            if pd.notna(row['xmin']) and pd.notna(row['ymin']) and pd.notna(row['xmax']) and pd.notna(row['ymax']):
                boxes.append([
                    float(row['xmin']),
                    float(row['ymin']),
                    float(row['xmax']),
                    float(row['ymax'])
                ])
                labels.append(row['category'])
        detection_flag = 1 if len(boxes) > 0 else 0

        # Apply small random rotation if training
        if self.train:
            angle = random.uniform(-5, 5)
            image_pil, boxes = rotate_image_and_boxes(image_pil, boxes, angle)

        # Resize image and boxes to (224, 224)
        image_pil, boxes = resize_image_and_boxes(image_pil, boxes, target_size=(224, 224))
        image_tensor = T.ToTensor()(image_pil)
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        
        return image_tensor, boxes_tensor, labels, detection_flag

def custom_collate(batch):
    """
    Custom collate function to handle variable-sized targets.
    Returns:
      - images: stacked tensor
      - boxes: list of bounding boxes tensors
      - labels: list of labels lists
      - detection_flags: tensor of detection flags
    """
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    detection_flags = torch.tensor([item[3] for item in batch], dtype=torch.int)
    return images, boxes, labels, detection_flags

# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = MammographyLocalizationDataset(
        csv_file='/home/team11/dev/loc/dataset/train/localization.csv',
        img_dir='/home/team11/dev/loc/dataset/train/images',
        laterality='R',
        view='MLO',
        train=True
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)
    for images, boxes, labels, detection_flags in dataloader:
        print("Batch images shape:", images.shape)
        print("Batch boxes:", boxes)
        print("Batch labels:", labels)
        print("Batch detection flags:", detection_flags)
        break
