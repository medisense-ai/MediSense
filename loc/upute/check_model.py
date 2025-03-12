#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, color

# Import your custom modules
from datasets import MammoLocalizationDataset, ResizeMammoClassification
from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# Utility Functions
########################################
def generate_gt_mask(image, annotation):
    """
    Generates ground truth mask based on bounding box and 'birads' value.
    If annotation['category'] is "No_Finding", then the mask is all zeros (background).
    Otherwise, inside the bounding box, the mask is filled with (birads + 1),
    so that background is 0 and BIRADS values [0..5] become [1..6].
    
    Expects a 2D image (H x W) and returns a mask with values in [0, 6].
    """
    if image.ndim != 2:
        raise ValueError("generate_gt_mask expects a 2D image.")
    H, W = image.shape
    mask = np.zeros((H, W), dtype=np.int64)
    
    if annotation['category'] == "No_Finding":
        return mask  # background

    try:
        birads_val = int(float(annotation['birads']))  # expected 0..5
    except (ValueError, TypeError):
        return mask

    try:
        xmin = int(float(annotation['xmin']))
        ymin = int(float(annotation['ymin']))
        xmax = int(float(annotation['xmax']))
        ymax = int(float(annotation['ymax']))
    except (ValueError, TypeError):
        return mask

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W, xmax)
    ymax = min(H, ymax)
    
    class_val = birads_val + 1  # shift range: 0..5 -> 1..6
    mask[ymin:ymax, xmin:xmax] = class_val
    return mask

def multiclass_iou(pred, target, num_classes=7):
    """
    Computes the average Intersection over Union (IoU) for multi-class segmentation.
    Both pred and target should be 2D tensors (H x W) with integer labels in [0,6],
    where 0 = background and 1..6 are the segmentation classes.
    """
    pred = pred.long()
    target = target.long()
    ious = []
    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float() - intersection
        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    return np.mean(ious)

def evaluate_segmentation(model, dataloader, num_classes=7):
    """
    Evaluates the model on the validation set and returns the average CrossEntropyLoss and IoU.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()  # For multi-class segmentation.
    total_loss = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for case_id_batch, images_batch, annotations_batch in dataloader:
            # Convert batch to numpy if needed.
            if isinstance(images_batch, torch.Tensor):
                images_np = images_batch.cpu().numpy()
            else:
                images_np = images_batch

            # Handle batch shape: expected (B, H, W) or (B, 1, H, W)
            if images_np.ndim == 4:
                images_np = images_np[:, 0, :, :]
            elif images_np.ndim == 3 and images_np.shape[-1] == 1:
                images_np = images_np[..., 0]

            B = images_np.shape[0]
            batch_loss = 0.0
            batch_iou = 0.0

            for j in range(B):
                im = images_np[j]
                ann = {k: annotations_batch[k][j] for k in annotations_batch}
                gt_mask = generate_gt_mask(im, ann)

                im_tensor = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
                gt_mask_tensor = torch.from_numpy(gt_mask).long().to(device)  # (H,W)

                logits = model(im_tensor)  # (1,7,H,W)
                loss = criterion(logits, gt_mask_tensor.unsqueeze(0))
                batch_loss += loss.item()

                pred_mask = logits.argmax(dim=1).squeeze(0)  # (H,W)
                iou_val = multiclass_iou(pred_mask, gt_mask_tensor, num_classes=num_classes)
                batch_iou += iou_val

            batch_loss /= B
            batch_iou /= B
            total_loss += batch_loss
            total_iou += batch_iou
            count += 1

    avg_loss = total_loss / count if count > 0 else float('inf')
    avg_iou = total_iou / count if count > 0 else 0.0
    return avg_loss, avg_iou

def visualize_predictions(dataset, model, num_samples=5):
    """
    Visualizes predictions by randomly selecting samples from the dataset.
    For each sample, it displays the original image, the ground truth mask,
    and the model's predicted mask.
    """
    model.eval()
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        case_id, image, annotation = dataset[idx]
        
        # Generate ground truth mask
        gt_mask = generate_gt_mask(image, annotation)
        
        # Prepare image for inference
        im = image
        if isinstance(im, torch.Tensor):
            im = im.cpu().numpy()
        # If the image has a channel dimension, remove it.
        if im.ndim == 3 and im.shape[0] == 1:
            im = im[0]
        elif im.ndim == 3 and im.shape[-1] == 1:
            im = im[..., 0]
            
        im_tensor = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            logits = model(im_tensor)  # (1,7,H,W)
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Plot the image, ground truth, and prediction side by side.
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(im, cmap='gray')
        plt.title(f"Original Image (Case {case_id})")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='jet', vmin=0, vmax=6)
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=6)
        plt.title("Predicted Mask")
        plt.axis('off')
        
        plt.show()

def check_data_leakage(train_dataset, val_dataset):
    """
    Checks if there is any overlap between training and validation splits.
    Assumes that the underlying dataset is a Subset and has an attribute 'indices'.
    """
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    overlap = train_indices.intersection(val_indices)
    if overlap:
        print("Warning: Data leakage detected! Overlap found in training and validation sets.")
    else:
        print("No overlap between training and validation sets.")

def main():
    ################################################################################
    # 1. Load the trained model
    ################################################################################
    model_class = get_model("mammo-segmentation-unet", categories=1)
    model = model_class(n_channels=1, n_classes=7)
    checkpoint_path = "best_mammo_segmentation_unet.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    ################################################################################
    # 2. Prepare the validation set (and check for leakage)
    ################################################################################
    data_dir = "/home/team11/data/train/"  # Update to your actual path
    resize_size = (224, 224)  # Make sure this matches your training configuration
    transform_fn = ResizeMammoClassification(resize_size)
    
    full_dataset = MammoLocalizationDataset(
        data_dir=data_dir,
        transform=transform_fn,
        resize_output_size=resize_size
    )
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    # For reproducibility, set a fixed seed for splitting
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Check for overlap between train and val splits
    check_data_leakage(train_dataset, val_dataset)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    ################################################################################
    # 3. Evaluate the model on the validation set
    ################################################################################
    val_loss, val_iou = evaluate_segmentation(model, val_loader, num_classes=7)
    print("=== Evaluation on Validation Set ===")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation IoU : {val_iou:.4f}")

    ################################################################################
    # 4. Visual Inspection of a few validation samples
    ################################################################################
    print("Displaying a few random validation samples for visual inspection...")
    visualize_predictions(val_dataset, model, num_samples=5)

if __name__ == "__main__":
    main()

