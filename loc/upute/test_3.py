#!/usr/bin/env python3

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

# Import your dataset and model
from datasets import MammoLocalizationDataset, ResizeMammoClassification
from models import MammoSegmentationUNet, get_model
from train import generate_combined_gt_mask  # Or wherever this function is defined

# 7-class color map (background + 6 lesion classes)
# Adjust colors as desired
COLOR_MAP = [
    [0, 0, 0],        # class 0 = No_Finding (black)
    [255, 0, 0],      # class 1 = Mass (red)
    [0, 255, 0],      # class 2 = Suspicious_Calcification (green)
    [0, 0, 255],      # class 3 = Focal_Asymmetry (blue)
    [255, 255, 0],    # class 4 = Architectural_Distortion (yellow)
    [255, 0, 255],    # class 5 = Suspicious_Lymph_Node (magenta)
    [0, 255, 255]     # class 6 = Other (cyan)
]

def colorize_mask(mask):
    """
    Convert a 2D integer mask (H,W) with values in [0..6] into an RGB image.
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(COLOR_MAP)):
        colored[mask == cls_id] = COLOR_MAP[cls_id]
    return colored

def visualize_samples(
    model_path,
    data_dir,
    resize_size=(256, 256),
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_lesion_samples=2,
    num_no_finding_samples=2
):
    """
    Load the trained segmentation model and dataset, pick random images with and
    without lesions, run inference, and display the results side-by-side:
      1) Original Image
      2) Ground-Truth Mask
      3) Predicted Mask
    """

    # 1) Load the dataset
    dataset = MammoLocalizationDataset(
        data_dir=data_dir,
        transform=ResizeMammoClassification(resize_size),
        resize_output_size=resize_size
    )
    
    # 2) Separate indices for images with lesions vs. no finding
    lesion_indices = []
    no_finding_indices = []
    for i in range(len(dataset)):
        case_id, sample, annotations = dataset[i]
        # Check if there's any annotation that isn't "No_Finding"
        if any(ann["category"] != "No_Finding" for ann in annotations):
            lesion_indices.append(i)
        else:
            no_finding_indices.append(i)
    
    # Pick random samples from each category
    chosen_lesion = random.sample(lesion_indices, min(num_lesion_samples, len(lesion_indices)))
    chosen_no_finding = random.sample(no_finding_indices, min(num_no_finding_samples, len(no_finding_indices)))
    chosen_indices = chosen_lesion + chosen_no_finding

    # 3) Load the trained segmentation model
    model = MammoSegmentationUNet(n_channels=1, n_classes=7)  # or get_model("mammo-segmentation-unet", categories=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4) Run inference and display
    for idx in chosen_indices:
        case_id, sample, annotations = dataset[idx]
        
        # Convert image to numpy if needed
        if torch.is_tensor(sample):
            image_np = sample.numpy()
        else:
            image_np = sample
        
        # If the image is 3D (H,W,3), convert to grayscale
        # or if it's (3,H,W), pick channel 0
        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            image_np = color.rgb2gray(image_np)
        elif image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = image_np[0, :, :]
        
        # Build ground truth mask (2D, shape=(H,W))
        gt_mask = generate_combined_gt_mask(image_np, annotations)
        
        # Prepare image for inference: shape (1,1,H,W)
        im_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            logits = model(im_tensor)  # shape (1,7,H,W)
            pred_label = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Colorize GT and predicted masks
        gt_mask_colored = colorize_mask(gt_mask)
        pred_mask_colored = colorize_mask(pred_label)

        # 5) Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_np, cmap="gray")
        axes[0].set_title(f"Original Image\nCase ID: {case_id}")
        axes[0].axis("off")

        axes[1].imshow(gt_mask_colored)
        axes[1].set_title("Ground-Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(pred_mask_colored)
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage:
    # Adjust the paths, model checkpoint name, and number of samples as needed
    model_ckpt_path = "best_mammo_segmentation_unet_2.pth"  # path to your trained model
    data_dir = "/home/team11/data/train/"                 # or wherever your dataset is
    visualize_samples(
        model_path=model_ckpt_path,
        data_dir=data_dir,
        resize_size=(256, 256),
        num_lesion_samples=3,
        num_no_finding_samples=2
    )