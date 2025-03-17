import sys
import os
import torch
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

# Custom ToTensor transform for (image, target).
class ToTensorDouble:
    def __call__(self, image, target):
        image = transforms.ToTensor()(image)
        return image, target

# Custom collate function to handle (image, target) pairs.
def collate_fn(batch):
    return tuple(zip(*batch))

# Hyperparameters
BATCH_SIZE = 1     # Detection → smaller batch sizes.
WORKERS = 4
NUM_EPOCHS = 4
LR = 0.001       # Lower learning rate to help stabilize training

# Set up logging and device.
log = Logger(log_dir='/home/team11/dev/MediSense/localization/temp', log_file='localization.log')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Training on device: {device}")
print(f"Training on device: {device}")

# Define a transform pipeline that includes resizing, converting to tensor, and normalization.
transform = ComposeDouble([
    ResizeWithBBoxes((1024, 1024)),
    ToTensorDouble(),
    NormalizeImageNet()  # Note: This applies ImageNet normalization.
])

# Create the dataset.
train_dataset = MammographyLocalizationDataset(
    csv_file="/home/team11/data/train/localization.csv",
    #csv_file="/home/team11/dev/MediSense/loc/upute/test/localization.csv",
    img_dir="/home/data/train/images",
    #img_dir="/home/team11/dev/MediSense/loc/upute/test/",
    transform=transform
)

# -------------------------------
# Compute weights for oversampling positive examples
# -------------------------------
num_positive = 1411
num_negative = 14585

# Loop over the dataset to count positive and negative images.
#for i in range(len(train_dataset)):
#    _, target = train_dataset[i]
    # If the annotation indicates no lesion, ensure boxes/labels are empty.
#    if isinstance(target, dict) and "finding" in target:
#        if target["finding"] in ["No_finding", "No_finiding"]:
 #           target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
 #           target["labels"] = torch.empty((0,), dtype=torch.int64)
  #  if target["boxes"].size(0) > 0:
   #     num_positive += 1
  #  else:
   #     num_negative += 1

log.info(f"Positive images: {num_positive}, Negative images: {num_negative}")
print(f"Positive images: {num_positive}, Negative images: {num_negative}")

# Compute the ratio to use as weight for positive samples.
weight_ratio = (num_negative / num_positive) if num_positive > 0 else 1.0

# Create a list of weights for each sample.
weights = []
for i in range(len(train_dataset)):
    _, target = train_dataset[i]
    # Check again for the 'finding' flag in case it wasn't handled inside the dataset.
    if isinstance(target, dict) and "finding" in target:
        if target["finding"] in ["No_finding", "No_finiding"]:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
    if target["boxes"].size(0) > 0:
        weights.append(weight_ratio)
    else:
        weights.append(1.0)

# Create a WeightedRandomSampler with the computed weights.
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Create the DataLoader with the sampler (remove shuffle because sampler handles randomness).
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    sampler=sampler,
    collate_fn=collate_fn
)

# Create the localization model.
model = MammoLocalizationResNet50(num_classes=7, pretrained=True, logger=log)
model.model.to(device)

# Set up optimizer.
optimizer = torch.optim.SGD(
    model.model.parameters(),
    lr=LR,
    momentum=0.9,
    weight_decay=0.0005
)

# Training loop with IoU evaluation per epoch.
for epoch in range(NUM_EPOCHS):
    model.model.train()
    epoch_loss = 0.0
    log.info(f"Epoch {epoch+1}/{NUM_EPOCHS} starting...")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} starting...")

    for images, targets in train_loader:
        # Move images to the device.
        images = [img.to(device) for img in images]
        new_targets = []
        for t in targets:
            if isinstance(t, dict):
                # Check for the 'finding' flag and override boxes/labels if no lesion.
                if "finding" in t and t["finding"] in ["No_finding", "No_finiding"]:
                    new_targets.append({
                        "boxes": torch.empty((0, 4), dtype=torch.float32).to(device),
                        "labels": torch.empty((0,), dtype=torch.int64).to(device)
                    })
                else:
                    new_targets.append({k: v.to(device) for k, v in t.items()})
            else:
                new_targets.append({
                    "boxes": torch.empty((0, 4), dtype=torch.float32).to(device),
                    "labels": torch.empty((0,), dtype=torch.int64).to(device)
                })
        targets = new_targets

        optimizer.zero_grad()
        loss_dict = model.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()

    avg_loss = epoch_loss / len(train_loader)
    log.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # Evaluate IoU metric on the training set (you may choose a separate validation set if available)
    iou_metrics = model.evaluate_model_iou(train_loader)
    mean_iou = iou_metrics.get("mean_iou", 0.0)
    log.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Mean IoU: {mean_iou:.4f}")
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Mean IoU: {mean_iou:.4f}")

log.info("Training complete")
print("Training complete")

# Save the trained model.
model_path = "/home/team11/dev/MediSense/localization/models/localization_model.pth"
torch.save(model.model.state_dict(), model_path)
log.info(f"Saved model to {model_path}")
print(f"Saved model to {model_path}")

# -------------------------------
# Inference Debugging: Lower Detection Threshold and Log Predicted Scores
# -------------------------------
detection_threshold = 0.1  # Lower detection threshold to capture low-confidence detections

model.model.eval()
with torch.no_grad():
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        outputs = model.model(images)
        for i, output in enumerate(outputs):
            scores = output.get('scores', torch.tensor([]))
            boxes = output.get('boxes', torch.tensor([]))
            if scores.numel() > 0:
                mask = scores > detection_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]
            else:
                filtered_boxes, filtered_scores = boxes, scores

            print(f"Image {i} predicted boxes: {filtered_boxes}")
            print(f"Image {i} predicted scores: {filtered_scores}")
            log.info(f"Image {i} predicted boxes: {filtered_boxes}")
            log.info(f"Image {i} predicted scores: {filtered_scores}")
        break  # Process one batch for debugging

# Optionally log CUDA memory usage.
log_cuda_memory(log)