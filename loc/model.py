import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import dataset and custom collate function
from dataset import MammographyLocalizationDataset, custom_collate

class MammographyDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(MammographyDetectionModel, self).__init__()
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_detect = nn.Linear(num_ftrs, 1)
        self.fc_bbox = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        features = self.backbone(x)
        detect = self.fc_detect(features)
        detect = torch.sigmoid(detect)
        bbox = self.fc_bbox(features)
        return detect, bbox

    def evaluate(self, dataloader, criterion_detection, criterion_bbox, device):
        self.eval()
        running_val_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, boxes, labels, detection_flags in dataloader:
                images = images.to(device)
                detection_targets = detection_flags.float().to(device)
                detect_out, bbox_out = self.forward(images)
                loss_detection = criterion_detection(detect_out.view(-1), detection_targets)
                
                # Compute bbox loss for samples with lesion present
                bbox_loss_total = 0.0
                count_bbox = 0
                for i in range(len(detection_flags)):
                    if detection_flags[i] == 1 and boxes[i].shape[0] > 0:
                        bbox_target = boxes[i][0].unsqueeze(0).to(device)
                        bbox_loss_total += criterion_bbox(bbox_out[i].unsqueeze(0), bbox_target)
                        count_bbox += 1
                loss_bbox = bbox_loss_total / count_bbox if count_bbox > 0 else torch.tensor(0.0, device=device)
                loss = loss_detection + loss_bbox
                batch_size = detection_flags.size(0)
                running_val_loss += loss.item() * batch_size
                total_samples += batch_size
        avg_val_loss = running_val_loss / total_samples if total_samples > 0 else 0.0
        self.train()
        return avg_val_loss

    def train_model(self, 
                    train_csv_file, 
                    train_img_dir,
                    val_csv_file=None,
                    val_img_dir=None,
                    laterality="R", 
                    view="MLO", 
                    num_epochs=20, 
                    learning_rate=0.001,
                    weight_decay=1e-5,
                    detection_loss_weight=1.0,
                    bbox_loss_weight=1.0,
                    batch_size=1,
                    patience=5,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.to(device)
        train_dataset = MammographyLocalizationDataset(
            csv_file=train_csv_file,
            img_dir=train_img_dir,
            laterality=laterality,
            view=view,
            train=True
        )
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        
        if val_csv_file and val_img_dir:
            val_dataset = MammographyLocalizationDataset(
                csv_file=val_csv_file,
                img_dir=val_img_dir,
                laterality=laterality,
                view=view,
                train=False
            )
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        else:
            val_dataloader = None

        criterion_detection = nn.BCELoss()
        criterion_bbox = nn.SmoothL1Loss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_model_wts = None
        total_train_samples = len(train_dataset)

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for images, boxes, labels, detection_flags in train_dataloader:
                images = images.to(device)
                detection_targets = detection_flags.float().to(device)
                detect_out, bbox_out = self.forward(images)
                loss_detection = criterion_detection(detect_out.view(-1), detection_targets)
                bbox_loss_total = 0.0
                count_bbox = 0
                for i in range(len(detection_flags)):
                    if detection_flags[i] == 1 and boxes[i].shape[0] > 0:
                        bbox_target = boxes[i][0].unsqueeze(0).to(device)
                        bbox_loss_total += criterion_bbox(bbox_out[i].unsqueeze(0), bbox_target)
                        count_bbox += 1
                loss_bbox = bbox_loss_total / count_bbox if count_bbox > 0 else torch.tensor(0.0, device=device)
                loss = (detection_loss_weight * loss_detection + bbox_loss_weight * loss_bbox)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = detection_flags.size(0)
                running_loss += loss.item() * batch_size

            avg_train_loss = running_loss / total_train_samples
            log_msg = f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}"
            
            if val_dataloader:
                avg_val_loss = self.evaluate(val_dataloader, criterion_detection, criterion_bbox, device)
                log_msg += f", Val Loss: {avg_val_loss:.4f}"
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_model_wts = self.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(log_msg)
                        print("Early stopping triggered.")
                        if best_model_wts is not None:
                            self.load_state_dict(best_model_wts)
                        return
            print(log_msg)
        
        if best_model_wts is not None:
            self.load_state_dict(best_model_wts)
        print("Training complete.")

if __name__ == "__main__":
    # Example usage:
    model = MammographyDetectionModel(pretrained=True)
    model.train_model(
        train_csv_file="/home/team11/dev/loc/dataset/train/localization.csv",
        train_img_dir="/home/team11/dev/loc/dataset/train/images",
        val_csv_file="/home/team11/dev/loc/dataset/subset-train-eval/localization.csv",
        val_img_dir="/home/team11/dev/loc/dataset/subset-train-eval/images",
        laterality="R",
        view="MLO",
        num_epochs=20,
        learning_rate=0.001,
        weight_decay=1e-5,
        detection_loss_weight=1.0,
        bbox_loss_weight=1.0,
        batch_size=1,
        patience=5
    )
