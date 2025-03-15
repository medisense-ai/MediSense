#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import optuna  # only if you need it, otherwise remove

import matplotlib.pyplot as plt

# Import your custom modules
from datasets import MammoLocalizationDataset, ResizeMammoClassification
from models import get_model
from skimage import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# Same IoU/Validation function from your training code
########################################
def multiclass_iou(pred, target, num_classes=7):
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

def generate_gt_mask(image, annotation):
    """
    Same function you used during training.
    0 = NoFinding; if annotation['category']=='No_Finding', mask is all zeros.
    else, label is birads_val+1 in bounding box region.
    """
    if image.ndim != 2:
        raise ValueError("generate_gt_mask expects a 2D image.")

    H, W = image.shape
    mask = np.zeros((H, W), dtype=np.int64)

    if annotation['category'] == "No_Finding":
        return mask  # all 0

    try:
        birads_val = int(float(annotation['birads']))  # 0..5
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

    class_val = birads_val + 1  # so 0..5 => 1..6
    mask[ymin:ymax, xmin:xmax] = class_val
    return mask

def evaluate_segmentation(model, val_loader, num_classes=7):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for case_id_batch, images_batch, annotations_batch in val_loader:
            # Convert batch to numpy if needed
            if isinstance(images_batch, torch.Tensor):
                images_np = images_batch.cpu().numpy()
            else:
                images_np = images_batch

            # Suppose shape is (B, H, W) or (B, 1, H, W)
            if images_np.ndim == 4:
                images_np = images_np[:, 0, :, :]
            elif images_np.ndim == 3 and images_np.shape[-1] == 1:
                images_np = images_np[..., 0]

            B = images_np.shape[0]
            batch_loss = 0.0
            batch_iou = 0.0

            for j in range(B):
                im = images_np[j]
                # Convert annotation dictionary
                ann = {k: annotations_batch[k][j] for k in annotations_batch}

                # Generate ground truth mask
                gt_mask = generate_gt_mask(im, ann)

                # Prepare tensors
                im_tensor = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().to(device)
                gt_mask_tensor = torch.from_numpy(gt_mask).long().to(device)

                logits = model(im_tensor)  # shape: (1,7,H,W)
                loss = criterion(logits, gt_mask_tensor.unsqueeze(0))
                batch_loss += loss.item()

                pred_mask = logits.argmax(dim=1).squeeze(0)
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

def main():
    ################################################################################
    # 1. Load model
    ################################################################################
    # This matches your multi-class approach with 7 channels: 0..6
    model_class = get_model("mammo-segmentation-unet", categories=1)
    model = model_class(n_channels=1, n_classes=7)
    model.load_state_dict(torch.load("best_mammo_segmentation_unet.pth"))
    model.eval()
    model.to(device)

    ################################################################################
    # 2. Create validation set (same as in train_optuna.py)
    ################################################################################
    data_dir = "/home/team11/data/train/"
    resize_size = (224, 224)
    transform_fn = ResizeMammoClassification(resize_size)
    
    full_dataset = MammoLocalizationDataset(
        data_dir=data_dir,
        transform=transform_fn,
        resize_output_size=resize_size
    )
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    ################################################################################
    # 3. Evaluate on the validation set
    ################################################################################
    val_loss, val_iou = evaluate_segmentation(model, val_loader, num_classes=7)

    # 4. Print the final results in the terminal
    print("=== Validation Results ===")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation IoU : {val_iou:.4f}")

if __name__ == "__main__":
    main()

