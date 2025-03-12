import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, transform, io
from pathlib import Path

# Import your dataset, model definitions, and helper functions.
from datasets import MammoLocalizationDataset, ResizeMammoClassification
from models import get_model
# If generate_combined_gt_mask is not available as an import, you can copy it here.
from train import generate_combined_gt_mask  # Assumes this is importable from train.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a fixed random seed for reproducibility
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define a simple color mapping for the 7 channels:
COLOR_MAP = {
    0: [255, 255, 255],  # White for No_Finding
    1: [255, 0, 0],      # Red
    2: [0, 255, 0],      # Green
    3: [0, 0, 255],      # Blue
    4: [255, 255, 0],    # Yellow
    5: [255, 0, 255],    # Magenta
    6: [0, 255, 255]     # Cyan
}

def label_to_color(label_img):
    """
    Convert a single-channel label image (H x W, with values 0-6) into a color image using COLOR_MAP.
    """
    H, W = label_img.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in COLOR_MAP.items():
        color_img[label_img == cls] = col
    return color_img

def create_overlay(original_gray, color_mask, alpha=0.5):
    """
    Blend the original grayscale image (converted to RGB) with the color mask.
    """
    original_rgb = cv2.cvtColor((original_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, color_mask, alpha, 0)
    return overlay

def get_label_from_mask(mask):
    """
    Convert a multi-label binary mask of shape (7, H, W) into a single-channel label image
    by taking argmax along the channel axis.
    """
    label_img = np.argmax(mask, axis=0)
    return label_img

def visualize_segmentation(test_dataset, model, num_samples=5):
    model.eval()
    indices = random.sample(range(len(test_dataset)), num_samples)
    
    for idx in indices:
        # Retrieve a test sample: case_id, image, annotations.
        case_id, image, annotations = test_dataset[idx]
        
        # Convert image to grayscale if needed.
        if image.ndim == 3:
            if image.shape[-1] == 3:
                image_gray = color.rgb2gray(image)
            else:
                image_gray = image[0, :, :]
        else:
            image_gray = image
        
        H, W = image_gray.shape
        
        # Generate ground truth mask (7, H, W) using your function.
        gt_mask = generate_combined_gt_mask(image_gray, annotations)
        gt_label = get_label_from_mask(gt_mask)
        gt_color = label_to_color(gt_label)
        
        # Prepare image tensor for model prediction.
        im_tensor = torch.from_numpy(image_gray).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(im_tensor)  # (1, 7, H, W)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).long().squeeze(0).cpu().numpy()  # (7, H, W)
        
        # Convert predicted multi-label mask to single-channel label using argmax.
        pred_label = get_label_from_mask(pred_mask)
        pred_color = label_to_color(pred_label)
        
        # Create an overlay of the predicted mask on the original image.
        overlay = create_overlay(image_gray, pred_color, alpha=0.5)
        
        # Plot the original image, ground truth mask, and overlay of predicted mask.
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(image_gray, cmap="gray")
        axs[0].set_title(f"Original Image\nCase: {case_id}")
        axs[0].axis("off")
        
        axs[1].imshow(gt_color)
        axs[1].set_title("Ground Truth Segmentation")
        axs[1].axis("off")
        
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay: Predicted Mask")
        axs[2].axis("off")
        
        plt.tight_layout()
        plt.show()

def main():
    # Use the same data directory as in your train.py.
    data_dir = "/home/team11/data/train/"
    resize_size = (256, 256)
    transform_fn = ResizeMammoClassification(resize_size)
    
    # Load the full dataset.
    full_dataset = MammoLocalizationDataset(data_dir=data_dir, transform=transform_fn, resize_output_size=resize_size)
    total_samples = len(full_dataset)
    
    # Define split sizes as in train.py.
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    
    # Recreate the same splits (using a fixed seed ensures consistency).
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Load the trained segmentation model.
    model_class = get_model("mammo-segmentation-unet", categories=1)
    model = model_class(n_channels=1, n_classes=7)
    model_path = "best_mammo_segmentation_unet.pth"  # Update path if needed.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Visualize a few random test samples.
    visualize_segmentation(test_dataset, model, num_samples=5)

if __name__ == "__main__":
    main()