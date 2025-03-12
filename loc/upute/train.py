#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import color
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# --- Import your dataset and model modules ---
from datasets import MammoLocalizationDataset, ResizeMammoClassification
from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Logger Setup
###############################################################################
logger = logging.getLogger("train_segmentation")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler("training.log", mode="w")
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

###############################################################################
# Focal Loss Definition (unchanged from earlier)
###############################################################################
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    gamma: focusing parameter (default=2.0)
    alpha: can be None, a single float, or a list/array of per-class weights
           matching the number of classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (N, C, H, W)
        targets: (N, H, W) with class IDs in [0..C-1].
        """
        # Cross-entropy per pixel: shape (N, H, W)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # pt is the probability of the true class (per pixel)
        pt = torch.exp(-ce_loss)

        # Focal Loss = (1 - pt)^gamma * ce_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Optional: apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray, torch.Tensor)):
                # Convert alpha to a tensor of shape [C], then index by target
                alpha_t = torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device)[targets]
                focal_loss = alpha_t * focal_loss
            else:
                # alpha is just a single float
                focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

###############################################################################
# Data Utilities: Collate + Mask Generation
###############################################################################
def custom_collate(batch):
    """
    Custom collate function that returns the batch as-is:
    (list_of_case_ids, list_of_samples, list_of_annotations)
    """
    case_ids = [item[0] for item in batch]
    samples = [item[1] for item in batch]
    annotations = [item[2] for item in batch]
    return case_ids, samples, annotations

def generate_combined_gt_mask(image, annotations):
    """
    Generate a single-channel ground truth mask with shape (H, W) for 7 classes:
      0 = No_Finding
      1 = Mass
      2 = Suspicious_Calcification
      3 = Focal_Asymmetry
      4 = Architectural_Distortion
      5 = Suspicious_Lymph_Node
      6 = Other
    Each pixel gets exactly one label.
    """
    if image.ndim != 2:
        raise ValueError("generate_combined_gt_mask expects a 2D image.")
    H, W = image.shape
    
    # Initialize all pixels to 0 (No_Finding)
    target = np.zeros((H, W), dtype=np.int64)
    
    mapping = {
        "No_Finding": 0,
        "Mass": 1,
        "Suspicious_Calcification": 2,
        "Focal_Asymmetry": 3,
        "Architectural_Distortion": 4,
        "Suspicious_Lymph_Node": 5,
        "Other": 6,
    }
    
    for ann in annotations:
        cat = ann['category']
        if cat not in mapping:
            continue
        cls_id = mapping[cat]
        
        try:
            xmin = int(float(ann['xmin']))
            ymin = int(float(ann['ymin']))
            xmax = int(float(ann['xmax']))
            ymax = int(float(ann['ymax']))
        except (ValueError, TypeError):
            continue
        
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(W, xmax)
        ymax = min(H, ymax)
        
        # Fill bounding box region with the lesion class ID
        target[ymin:ymax, xmin:xmax] = cls_id
    
    return target

def multiclass_iou(pred, target, num_classes=7):
    """
    Compute the average IoU for multi-class segmentation.
    pred and target are both shape (H, W), with integer class labels in [0..num_classes-1].
    IoU is computed per class, then averaged.
    """
    pred = pred.long()
    target = target.long()
    ious = []
    
    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            # If there are no pixels of this class in both pred & target, consider IoU=1
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    return np.mean(ious)

###############################################################################
# Validation Loop (Now receives alpha as argument)
###############################################################################
def validate_segmentation(model, dataloader, alpha, num_classes=7):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    
    # Create a FocalLoss instance with the computed alpha
    criterion = FocalLoss(
        alpha=alpha,
        gamma=2.0,       # or tune further
        reduction='mean'
    )
    
    with torch.no_grad():
        for case_id, images, annotations in dataloader:
            images_np = []
            for im in images:
                if isinstance(im, torch.Tensor):
                    images_np.append(im.cpu().numpy())
                else:
                    images_np.append(im)
            
            batch_loss = 0.0
            batch_iou = 0.0
            B = len(images_np)
            
            for j in range(B):
                im = images_np[j]
                # Convert image to grayscale if needed
                if im.ndim == 3 and im.shape[-1] == 3:
                    im = color.rgb2gray(im)
                elif im.ndim == 3 and im.shape[0] == 3:
                    im = im[0, :, :]
                
                ann_list = annotations[j]
                gt_mask = generate_combined_gt_mask(im, ann_list)  # (H, W) with class IDs
                
                # Move to torch
                im_tensor = torch.from_numpy(im).unsqueeze(0).float().to(device)  # (1, H, W)
                im_tensor = im_tensor.unsqueeze(0)  # (1, 1, H, W)
                
                logits = model(im_tensor)  # shape (1, 7, H, W)
                gt_mask_tensor = torch.from_numpy(gt_mask).long().to(device)  # (H, W)
                
                loss = criterion(logits, gt_mask_tensor.unsqueeze(0))
                batch_loss += loss.item()
                
                # Get predicted label per pixel
                pred_label = torch.argmax(logits, dim=1).squeeze(0)  # (H, W)
                
                iou_val = multiclass_iou(pred_label, gt_mask_tensor, num_classes=num_classes)
                batch_iou += iou_val
            
            # Average over images in this batch
            batch_loss /= B
            batch_iou /= B
            
            total_loss += batch_loss
            total_iou += batch_iou
            count += 1
    
    model.train()
    avg_loss = total_loss / count if count > 0 else float('inf')
    avg_iou = total_iou / count if count > 0 else 0.0
    return avg_loss, avg_iou

###############################################################################
# Training Loop (Now receives alpha as argument)
###############################################################################
def train_segmentation(model, train_loader, val_loader, alpha, num_epochs, writer,
                       patience=5, num_classes=7, lr=1e-4, accumulation_steps=4):
    """
    Note we pass 'alpha' (the array of per-class weights) into here and 
    create the FocalLoss instance with it.
    """
    criterion = FocalLoss(
        alpha=alpha,
        gamma=2.0,      # can tune further
        reduction='mean'
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        optimizer.zero_grad()
        
        for i, (case_id, images, annotations) in enumerate(train_loader):
            images_np = []
            for im in images:
                if isinstance(im, torch.Tensor):
                    images_np.append(im.cpu().numpy())
                else:
                    images_np.append(im)
            
            # Convert images to grayscale if needed
            processed_images = []
            for im in images_np:
                if im.ndim == 3 and im.shape[-1] == 3:
                    processed_images.append(color.rgb2gray(im))
                elif im.ndim == 3 and im.shape[0] == 3:
                    processed_images.append(im[0, :, :])
                else:
                    processed_images.append(im)
            
            B = len(processed_images)
            batch_loss = 0.0
            batch_iou = 0.0
            
            for j in range(B):
                im = processed_images[j]
                ann_list = annotations[j]
                gt_mask = generate_combined_gt_mask(im, ann_list)  # shape (H, W), int labels [0..6]
                
                im_tensor = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().to(device)  
                # => shape (1,1,H,W)
                logits = model(im_tensor)  # => (1,7,H,W)
                
                gt_mask_tensor = torch.from_numpy(gt_mask).long().to(device)  # (H,W)
                
                loss = criterion(logits, gt_mask_tensor.unsqueeze(0))
                batch_loss += loss
                
                # Compute IoU
                pred_label = torch.argmax(logits, dim=1).squeeze(0)  # (H,W)
                sample_iou = multiclass_iou(pred_label, gt_mask_tensor, num_classes=num_classes)
                batch_iou += sample_iou
            
            # Average over B images in this mini-batch
            batch_loss = batch_loss / B
            batch_iou = batch_iou / B
            
            # Gradient accumulation
            batch_loss = batch_loss / accumulation_steps
            batch_loss.backward()
            
            running_loss += batch_loss.item() * accumulation_steps
            running_iou += batch_iou
            
            # Update optimizer every accumulation_steps mini-batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar("Train/AccumulatedStepLoss", running_loss / accumulation_steps, global_step)
                global_step += 1
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
                        f"Loss: {batch_loss.item()*accumulation_steps:.4f}, IoU: {batch_iou:.4f}")
        
        # If the last batch is not a multiple of accumulation_steps, do one more step
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_train_iou = running_iou / len(train_loader)
        writer.add_scalar("Train/EpochLoss", avg_train_loss, epoch)
        writer.add_scalar("Train/EpochIoU", avg_train_iou, epoch)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}, Training IoU: {avg_train_iou:.4f}")
        
        # Validation
        val_loss, val_iou = validate_segmentation(model, val_loader, alpha, num_classes=num_classes)
        writer.add_scalar("Val/EpochLoss", val_loss, epoch)
        writer.add_scalar("Val/EpochIoU", val_iou, epoch)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save new checkpoint name
            torch.save(model.state_dict(), "best_mammo_segmentation_unet_mfbal.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_val_loss

###############################################################################
# Median-Frequency Balancing Computation
###############################################################################

def compute_median_frequency_alpha(dataset, num_classes=7, epsilon=1e-6):
    """
    Goes through the ENTIRE 'dataset' to build masks and count pixel occurrences.
    Then applies the median-frequency balancing formula:
        alpha_c = median_freq / freq_c
    where freq_c = (#pixels of class c) / (total #pixels)
    
    Returns a list 'alpha' of length num_classes.
    """
    # Accumulate total pixel counts for each class
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    
    # Iterate through the dataset (NOT the dataloader) to get every sample
    for idx in range(len(dataset)):
        case_id, image, annotations = dataset[idx]
        
        # 'image' might be resized or grayscale, but we only need shape
        # Check if transform means image is float in [0..1], etc.
        # Usually it's shape (H, W), or (H, W, 3). Convert to gray if needed:
        if isinstance(image, torch.Tensor):
            # if a transform turned it to a tensor, convert back to numpy
            image_np = image.cpu().numpy()
        else:
            image_np = image
        
        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            image_np = color.rgb2gray(image_np)
        elif image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = image_np[0, :, :]
        
        gt_mask = generate_combined_gt_mask(image_np, annotations)  # shape (H, W)

        # Count pixels of each class:
        # For large images, np.bincount might be more efficient:
        flat_mask = gt_mask.flatten()
        counts = np.bincount(flat_mask, minlength=num_classes)
        class_counts += counts
        total_pixels += flat_mask.size
    
    # Now compute frequency f_c = (#pixels c) / (total_pixels)
    # but note that "background" might be huge
    frequencies = class_counts / (total_pixels + epsilon)

    # The standard median-frequency formula uses only classes with freq>0
    # But we might include class 0 as well (some references exclude background).
    # Typically we do include all classes. Filter out truly 0 freq to avoid nonsense:
    nonzero_freqs = [f for f in frequencies if f > 0]
    median_freq = np.median(nonzero_freqs)

    alpha = []
    for c in range(num_classes):
        freq_c = frequencies[c]
        if freq_c > 0:
            alpha_c = median_freq / (freq_c + epsilon)
        else:
            # If class never appears, set alpha_c to something high:
            alpha_c = 5.0  # or something large
        alpha.append(alpha_c)
    
    return alpha

###############################################################################
# Main Script
###############################################################################
def main():
    # Hyperparameters
    learning_rate = 1e-4
    num_epochs = 15
    patience = 5
    batch_size = 32
    resize_size = (256, 256)
    accumulation_steps = 4  # Number of mini-batches to accumulate gradients over

    data_dir = "/home/team11/data/train/"
    transform_fn = ResizeMammoClassification(resize_size)

    # Build dataset
    full_dataset = MammoLocalizationDataset(
        data_dir=data_dir,
        transform=transform_fn,
        resize_output_size=resize_size
    )
    
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # ------------------------------------------------------------------------
    # 1) Compute alpha using Median-Frequency Balancing on *train_dataset*
    #    This can be computationally expensive if your dataset is huge.
    #    If so, consider sampling or caching results.
    # ------------------------------------------------------------------------
    logger.info("Computing median-frequency alpha values for the training dataset...")
    alpha_mf = compute_median_frequency_alpha(train_dataset, num_classes=7, epsilon=1e-6)
    logger.info(f"Median-frequency alpha: {alpha_mf}")

    # Build DataLoaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )

    # Create U-Net model with 7 output classes
    model_class = get_model("mammo-segmentation-unet", categories=1)
    model = model_class(n_channels=1, n_classes=7)
    model.to(device)

    writer = SummaryWriter(log_dir="runs/segmentation_experiment_mfbal")
    
    # Pass the alpha array to the trainer
    best_val_loss = train_segmentation(
        model,
        train_loader,
        val_loader,
        alpha=alpha_mf,
        num_epochs=num_epochs,
        writer=writer,
        patience=patience,
        num_classes=7,
        lr=learning_rate,
        accumulation_steps=accumulation_steps
    )

    # Evaluate on test set
    model.load_state_dict(torch.load("best_mammo_segmentation_unet_mfbal.pth"))
    test_loss, test_iou = validate_segmentation(model, test_loader, alpha_mf, num_classes=7)
    logger.info(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")

    writer.add_scalar("Test/Loss", test_loss)
    writer.add_scalar("Test/IoU", test_iou)
    writer.close()

if __name__ == "__main__":
    main()