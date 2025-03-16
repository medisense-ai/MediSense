#!/usr/bin/env python
import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

# Add the directory two levels up to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from LocalizationDataset import MammographyLocalizationDataset, ResizeWithBBoxes, NormalizeImageNet
from logger import Logger, log_cuda_memory
from LocalizationModel import MammoLocalizationResNet50

# Custom Compose for transforms that take (image, target) as input.
class ComposeDouble:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# Custom ToTensor transform for (image, target) pairs.
class ToTensorDouble:
    def __call__(self, image, target):
        image = transforms.ToTensor()(image)
        return image, target

# Custom collate function to handle (image, target) pairs.
def collate_fn(batch):
    return tuple(zip(*batch))

# Function to check bounding boxes for potential issues.
def check_boxes(boxes, image_shape):
    warnings = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        if x_max <= x_min or y_max <= y_min:
            warnings.append(f"Degenerate box with non-positive area: {box}")
        if x_min < 0 or y_min < 0 or x_max > image_shape[1] or y_max > image_shape[0]:
            warnings.append(f"Box out of bounds: {box} for image shape {image_shape}")
    return warnings

def main():
    # Enable anomaly detection to help trace autograd errors.
    torch.autograd.set_detect_anomaly(True)

    # Set up logging and device.
    log = Logger(log_dir='/home/team11/dev/MediSense/localization/debug_logs', log_file='debug_localization.log')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    log.info(f"Training on device: {device}")

    # Define the transform pipeline.
    transform = ComposeDouble([
        ResizeWithBBoxes((1024, 1024)),
        ToTensorDouble(),
        NormalizeImageNet()
    ])

    # Create the dataset.
    train_dataset = MammographyLocalizationDataset(
        csv_file="/home/team11/data/train/localization.csv",
        img_dir="/home/data/train/images",
        transform=transform
    )

    # -------------------------------
    # Compute weights for oversampling positive examples
    # -------------------------------
    num_positive = 0
    num_negative = 0

    for i in range(len(train_dataset)):
        _, target = train_dataset[i]
        if target["boxes"].size(0) > 0:
            num_positive += 1
        else:
            num_negative += 1

    log.info(f"Positive images: {num_positive}, Negative images: {num_negative}")
    print(f"Positive images: {num_positive}, Negative images: {num_negative}")

    # Compute weight ratio for positive samples.
    weight_ratio = (num_negative / num_positive) if num_positive > 0 else 1.0

    # Create a list of weights for each sample.
    weights = []
    for i in range(len(train_dataset)):
        _, target = train_dataset[i]
        if target["boxes"].size(0) > 0:
            weights.append(weight_ratio)
        else:
            weights.append(1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Create the DataLoader with the sampler (no shuffle because the sampler handles randomness).
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        sampler=sampler,
        collate_fn=collate_fn
    )

    # Create the localization model.
    model = MammoLocalizationResNet50(num_classes=7, pretrained=True, logger=log)
    model.model.to(device)
    model.model.train()

    optimizer = torch.optim.SGD(
        model.model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Debug training: Process only a few batches for faster debugging.
    max_batches = 5
    batch_count = 0

    for images, targets in train_loader:
        print(f"Processing batch {batch_count}")
        log.info(f"Processing batch {batch_count}")

        # Check bounding boxes for each image in the batch.
        for i, target in enumerate(targets):
            boxes = target["boxes"]
            print(f"Image {i} target boxes: {boxes}")
            log.info(f"Image {i} target boxes: {boxes}")
            # Get image dimensions (assuming tensor shape [C, H, W]).
            image_tensor = images[i]
            image_shape = image_tensor.shape[1:]  # (height, width)
            warnings = check_boxes(boxes.numpy(), image_shape)
            for warning in warnings:
                print("Warning:", warning)
                log.warning(warning)

        # Move images and targets to the device.
        images = [img.to(device) for img in images]
        new_targets = []
        for t in targets:
            if isinstance(t, dict):
                new_targets.append({k: v.to(device) for k, v in t.items()})
            else:
                new_targets.append({
                    "boxes": torch.empty((0, 4), dtype=torch.float32).to(device),
                    "labels": torch.empty((0,), dtype=torch.int64).to(device)
                })

        optimizer.zero_grad()
        try:
            loss_dict = model.model(images, new_targets)
            print("Loss dictionary keys:", loss_dict.keys())
            log.info(f"Loss dictionary keys: {loss_dict.keys()}")
            total_loss = 0.0
            for key, loss_value in loss_dict.items():
                loss_item = loss_value.item()
                print(f"Loss component {key}: {loss_item}")
                log.info(f"Loss component {key}: {loss_item}")
                if torch.isnan(loss_value):
                    log.error(f"Loss component {key} is NaN")
                total_loss += loss_value

            print("Total loss:", total_loss.item())
            log.info(f"Total loss: {total_loss.item()}")

            if torch.isnan(total_loss):
                log.error("Total loss is NaN. Skipping backward pass for this batch.")
                continue

            total_loss.backward()
            optimizer.step()

        except Exception as e:
            print("Exception during forward/backward pass:", e)
            log.error(f"Exception during forward/backward pass: {e}")

        batch_count += 1
        if batch_count >= max_batches:
            break

    # Quick inference: run one batch to inspect predicted boxes.
    model.model.eval()
    with torch.no_grad():
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            outputs = model.model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output.get('boxes', 'No boxes')
                print(f"Predicted boxes for image {i}: {pred_boxes}")
                log.info(f"Predicted boxes for image {i}: {pred_boxes}")
            break

    # Optionally log CUDA memory usage.
    log_cuda_memory(log)
    print("Debugging script complete.")

if __name__ == "__main__":
    main()
