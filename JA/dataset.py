import os
import torch
import numpy as np
import pandas as pd
import random
import math

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def rotate_image_and_boxes(image, boxes, angle):
    """
    Rotate the PIL image around its center by `angle` degrees.
    Update bounding boxes accordingly (pixel coords).
    boxes: list of [xmin, ymin, xmax, ymax] in absolute pixel coords.
    angle: degrees to rotate (positive = counterclockwise).
    """
    width, height = image.size
    
    # Convert degrees -> radians
    theta = math.radians(angle)

    # PIL-based rotation (about the center)
    rotated_image = image.rotate(angle, fillcolor=(0,0,0))  # black fill

    # Center coords
    center_x, center_y = width / 2.0, height / 2.0

    rotated_boxes = []
    for (xmin, ymin, xmax, ymax) in boxes:
        corners = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ], dtype=np.float32)

        # Rotate each corner around (center_x, center_y)
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
        x_coords = rotated_corners[:,0]
        y_coords = rotated_corners[:,1]

        x_min_new = x_coords.min()
        x_max_new = x_coords.max()
        y_min_new = y_coords.min()
        y_max_new = y_coords.max()

        rotated_boxes.append([x_min_new, y_min_new, x_max_new, y_max_new])
    
    return rotated_image, rotated_boxes

def resize_image_and_boxes(image, boxes, target_size=(224,224)):
    """
    Resize the PIL image to target_size and scale bounding boxes accordingly.
    boxes in pixel coords.
    """
    orig_width, orig_height = image.size
    new_width, new_height = target_size

    # Resize
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
    Dataset that:
      - Reads bounding boxes from a CSV in pixel coords.
      - Optionally applies a small random rotation (±5°) for augmentation.
      - Then resizes to (224, 224) and converts to tensor.
      - DOES NOT perform horizontal flipping (to preserve L/R orientation).
    """
    def __init__(self, csv_file, img_dir, laterality, view, train=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train = train

        # Extract laterality, view from e.g. "20_L_CC"
        self.data['laterality'] = self.data['image_id'].apply(lambda x: x.split('_')[1])
        self.data['view'] = self.data['image_id'].apply(lambda x: x.split('_')[2])

        # Filter by laterality/view
        self.data = self.data[(self.data['laterality'] == laterality) & (self.data['view'] == view)]

        # Unique image_ids
        self.image_ids = self.data['image_id'].unique()

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        rows = self.data[self.data['image_id'] == image_id]

        folder = rows['case_id'].values[0]
        img_file = f"{image_id}.jpg"
        img_path = os.path.join(self.img_dir, str(folder), img_file)

        # Load the image
        image_pil = Image.open(img_path).convert("RGB")

        # Collect bounding boxes
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

        # SMALL RANDOM ROTATION if train=True
        if self.train:
            angle = random.uniform(-5, 5)
            image_pil, boxes = rotate_image_and_boxes(image_pil, boxes, angle)

        # Resize to 224×224
        image_pil, boxes = resize_image_and_boxes(image_pil, boxes, target_size=(224,224))

        # Convert to tensor
        image_tensor = T.ToTensor()(image_pil)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.empty((0,4), dtype=torch.float32)

        return image_tensor, boxes_tensor, labels, detection_flag

# Example usage:
if __name__ == "__main__":
    dataset = MammographyLocalizationDataset(
        csv_file='/home/data/train/localization.csv',
        img_dir='/home/data/train/images',
        laterality='R',
        view='MLO',
        train=True  # True => apply random rotation, no flipping
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
    
    for batch_idx, batch in enumerate(dataloader):
        for image, boxes, labels, detection_flag in batch:
            print("Image shape:", image.shape)
            print("Boxes:", boxes)
            print("Labels:", labels)
            print("Detection flag:", detection_flag)
        break
