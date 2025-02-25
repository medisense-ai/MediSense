#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, color
from torch.utils.tensorboard import SummaryWriter

import optuna

# Uvoz dataset i modela (podesi putanje prema svojoj strukturi)
from datasets import MammoLocalizationDataset, ResizeMammoClassification
from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_gt_mask(image, annotation):
    """
    Generira ground truth masku na temelju bounding boxa i vrijednosti 'birads'.
    Ako je 'birads' jednaka 0 (No_Finding), maska ostaje prazna.
    Očekuje se da je image 2D (H x W).
    """
    if image.ndim != 2:
        raise ValueError("generate_gt_mask očekuje 2D sliku.")
    H, W = image.shape
    mask = np.zeros((H, W), dtype=np.float32)
    try:
        birads_val = int(float(annotation['birads']))
    except (ValueError, TypeError):
        return mask
    if birads_val != 0:
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
        mask[ymin:ymax, xmin:xmax] = 1.0
    return mask

def validate_segmentation(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for case_id, image, annotation in dataloader:
            # Ako je image tensor, konvertiraj ga u numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = image
            # Ako je image_np 3D, provjeri format
            if image_np.ndim == 3:
                if image_np.shape[0] == 1:
                    image_np = image_np[0]
                elif image_np.shape[-1] == 1:
                    image_np = image_np[..., 0]
            if image_np.ndim != 2:
                raise ValueError(f"Očekivana 2D slika, dobiveno: {image_np.shape}")
            gt_mask = generate_gt_mask(image_np, annotation)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
            gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float().to(device)
            output = model(image_tensor)
            loss = criterion(output, gt_mask_tensor)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count if count > 0 else float('inf')

def train_segmentation(model, train_loader, val_loader, optimizer, criterion, num_epochs, writer, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (case_id, image, annotation) in enumerate(train_loader):
            # Ako je image tensor, konvertiraj ga u numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = image

            # Ako je image_np 3D, provjeri da li je u channel-first ili channel-last formatu
            if image_np.ndim == 3:
                if image_np.shape[0] == 1:
                    image_np = image_np[0]
                elif image_np.shape[-1] == 1:
                    image_np = image_np[..., 0]
            if image_np.ndim != 2:
                raise ValueError(f"Očekivana 2D slika, dobiveno: {image_np.shape}")
            
            gt_mask = generate_gt_mask(image_np, annotation)
            if gt_mask.shape != image_np.shape:
                raise ValueError(f"Dimenzije maske {gt_mask.shape} ne odgovaraju dimenzijama slike {image_np.shape}")
            
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
            gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float().to(device)
            
            optimizer.zero_grad()
            output = model(image_tensor)
            loss = criterion(output, gt_mask_tensor)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar("Train/StepLoss", loss.item(), global_step)
            global_step += 1
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Train/EpochLoss", avg_train_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")
        
        # Validacija nakon svake epohe
        val_loss = validate_segmentation(model, val_loader, criterion)
        writer.add_scalar("Val/EpochLoss", val_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_mammo_segmentation_unet.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    return best_val_loss

def objective(trial):
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    num_epochs = trial.suggest_int("num_epochs", 5, 15)
    
    data_dir = "/home/data/train/"
    resize_size = (512, 512)
    transform_fn = ResizeMammoClassification(resize_size)
    
    # Podijeli dataset na trening i validacijski skup (80/20)
    full_dataset = MammoLocalizationDataset(data_dir=data_dir, transform=transform_fn, resize_output_size=resize_size)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model_class = get_model("mammo-segmentation-unet", categories=1)
    model = model_class(n_channels=1, n_classes=1)
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    trial_id = trial.number
    writer = SummaryWriter(log_dir=f"runs/trial_{trial_id}")
    
    best_val_loss = train_segmentation(model, train_loader, val_loader, optimizer, criterion, num_epochs, writer, patience=5)
    writer.add_scalar("Train/BestValLoss", best_val_loss, num_epochs)
    writer.close()
    
    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Loss: {:.4f}".format(best_trial.value))
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Za pregled TensorBoarda, pokreni: tensorboard --logdir=runs
