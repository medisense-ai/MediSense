#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import color

from datasets import MammoLocalizationDataset, ResizeMammoClassification
from models import MammoSegmentationUNet
from train import generate_combined_gt_mask  # or wherever your mask-creation function lives

def overlay_mask(image_gray, mask, alpha=0.5):
    """
    Overlays a binary lesion mask (mask != 0) in red on top of the grayscale image.
    Returns an RGB image suitable for plotting with matplotlib.
    """
    # Ensure grayscale image is 2D
    if image_gray.ndim == 3:
        # If last dimension is 3, convert to grayscale
        if image_gray.shape[-1] == 3:
            image_gray = color.rgb2gray(image_gray)
        else:
            image_gray = image_gray[0, :, :]  # if shape (3, H, W)

    # Normalize image to [0,1] for display if needed
    if image_gray.max() > 1.0:
        image_gray = image_gray / 255.0

    # Convert single-channel grayscale to 3-channel RGB
    h, w = image_gray.shape
    base_rgb = np.stack([image_gray, image_gray, image_gray], axis=-1)

    # Create an RGB overlay for the predicted lesion region
    overlay = base_rgb.copy()
    # Any non-zero class in the mask is considered "lesion"
    lesion_pixels = (mask != 0)
    # Color those pixels red
    overlay[lesion_pixels, 0] = 1.0  # Red channel
    overlay[lesion_pixels, 1] = 0.0  # Green channel
    overlay[lesion_pixels, 2] = 0.0  # Blue channel

    # Blend the overlay and base images
    blended = (1 - alpha) * base_rgb + alpha * overlay
    return blended

def quick_reality_check(
    model_ckpt,
    data_dir,
    resize_size=(256, 256),
    device="cuda" if torch.cuda.is_available() else "cpu",
    samples_with_lesions=2,
    samples_no_finding=2
):
    """
    Quickly check model predictions by overlaying the predicted mask (any lesion) in red
    on the original grayscale image. Picks random samples that do and do not have lesions.
    """

    # 1) Build dataset
    dataset = MammoLocalizationDataset(
        data_dir=data_dir,
        transform=ResizeMammoClassification(resize_size),
        resize_output_size=resize_size
    )

    # Separate indices for images with lesions vs. no_finding
    lesion_indices = []
    no_finding_indices = []
    for i in range(len(dataset)):
        case_id, sample, annotations = dataset[i]
        # If any annotation is not "No_Finding", we treat it as a lesion image
        if any(ann["category"] != "No_Finding" for ann in annotations):
            lesion_indices.append(i)
        else:
            no_finding_indices.append(i)

    # Randomly pick some indices
    chosen_lesion = random.sample(lesion_indices, min(samples_with_lesions, len(lesion_indices)))
    chosen_no_finding = random.sample(no_finding_indices, min(samples_no_finding, len(no_finding_indices)))
    chosen_indices = chosen_lesion + chosen_no_finding
    random.shuffle(chosen_indices)

    # 2) Load model
    model = MammoSegmentationUNet(n_channels=1, n_classes=7)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # 3) Inference and visualization
    for idx in chosen_indices:
        case_id, sample, annotations = dataset[idx]

        # Convert to numpy if it's a tensor
        if torch.is_tensor(sample):
            image_np = sample.numpy()
        else:
            image_np = sample

        # Convert image to grayscale if needed
        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            image_np = color.rgb2gray(image_np)
        elif image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = image_np[0, :, :]

        # Prepare tensor for model
        im_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(im_tensor)  # (1, 7, H, W)
            pred_label = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # Create overlay
        overlayed = overlay_mask(image_np, pred_label)

        # Show side-by-side: Original vs. Overlay
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np, cmap='gray')
        plt.title(f"Original (Case ID: {case_id})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlayed)
        # If there's a lesion in the ground truth, label the title accordingly
        label_str = "Has Lesion" if any(a["category"] != "No_Finding" for a in annotations) else "No Finding"
        plt.title(f"Prediction Overlay ({label_str})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage:
    # Provide the path to your checkpoint, the data directory, etc.
    model_checkpoint = "best_mammo_segmentation_unet_2.pth"
    data_directory = "/home/team11/data/train/"  # adjust as needed
    quick_reality_check(
        model_ckpt=model_checkpoint,
        data_dir=data_directory,
        resize_size=(256, 256),
        samples_with_lesions=3,
        samples_no_finding=2
    )