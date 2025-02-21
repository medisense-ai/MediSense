import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import the modified localization dataset and transformation pipeline from dataset.py
from dataset import MammographyLocalizationDataset, get_transform

class MammographyDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(MammographyDetectionModel, self).__init__()
        # Use a pretrained ResNet50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        num_ftrs = self.backbone.fc.in_features
        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()
        # Detection head: outputs a single value (probability of lesion)
        self.fc_detect = nn.Linear(num_ftrs, 1)
        # Regression head: outputs 4 values (bounding box coordinates)
        self.fc_bbox = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        # Detection branch: predict lesion presence
        detect = self.fc_detect(features)
        detect = torch.sigmoid(detect)  # Output probability between 0 and 1
        # Regression branch: predict bounding box coordinates
        bbox = self.fc_bbox(features)
        return detect, bbox

    def evaluate(self, dataloader, criterion_detection, criterion_bbox, device):
        """
        Evaluates the model on a validation dataloader.
        Returns the average validation loss.
        """
        self.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                image, boxes, labels, detection_flag = batch[0]
                image = image.unsqueeze(0).to(device)
                detection_target = torch.tensor([detection_flag], dtype=torch.float32).to(device)
                
                detect_out, bbox_out = self.forward(image)

                # Detection loss
                loss_detection = criterion_detection(detect_out.view(-1), detection_target)

                # BBox loss if lesion present
                if detection_flag == 1 and boxes.shape[0] > 0:
                    bbox_target = boxes[0].unsqueeze(0).to(device)
                    loss_bbox = criterion_bbox(bbox_out, bbox_target)
                else:
                    loss_bbox = torch.tensor(0.0, device=device)

                loss = loss_detection + loss_bbox
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(dataloader)
        self.train()  # Return to training mode
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
                    weight_decay=1e-5,     # L2 regularization strength
                    detection_loss_weight=1.0,
                    bbox_loss_weight=1.0,
                    batch_size=1,
                    patience=5,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Trains the detection/localization model using the MammographyLocalizationDataset.
        Optionally uses a validation set for early stopping and logging.
        
        Parameters:
          train_csv_file: Path to the CSV file with training annotations.
          train_img_dir: Directory with training x-ray images.
          val_csv_file: (Optional) Path to the CSV file with validation annotations.
          val_img_dir: (Optional) Directory with validation x-ray images.
          laterality, view: Filter criteria for the dataset.
          num_epochs: Number of training epochs.
          learning_rate: Learning rate for the optimizer.
          weight_decay: Weight decay (L2 regularization) for Adam.
          detection_loss_weight: Weighting factor for the detection loss.
          bbox_loss_weight: Weighting factor for the bounding box regression loss.
          batch_size: Batch size to use during training.
          patience: Number of epochs with no improvement in validation loss before early stopping.
          device: Device to run training on (CPU or GPU).
        """
        self.to(device)
        # Albumentations pipelines for training and validation
        train_transform = get_transform(train=True)
        train_dataset = MammographyLocalizationDataset(
            csv_file=train_csv_file,
            img_dir=train_img_dir,
            laterality=laterality,
            view=view,
            transform=train_transform,
            auto_crop_enabled=True
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

        if val_csv_file and val_img_dir:
            val_transform = get_transform(train=False)
            val_dataset = MammographyLocalizationDataset(
                csv_file=val_csv_file,
                img_dir=val_img_dir,
                laterality=laterality,
                view=view,
                transform=val_transform,
                auto_crop_enabled=True
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
        else:
            val_dataloader = None

        # Loss functions
        criterion_detection = nn.BCELoss()
        criterion_bbox = nn.SmoothL1Loss()

        # Adam optimizer with weight decay for L2 regularization
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = np.inf
        epochs_no_improve = 0
        best_model_wts = None

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0

            for batch in train_dataloader:
                image, boxes, labels, detection_flag = batch[0]
                image = image.unsqueeze(0).to(device)
                detection_target = torch.tensor([detection_flag], dtype=torch.float32).to(device)

                # Forward pass
                detect_out, bbox_out = self.forward(image)

                # Detection loss
                loss_detection = criterion_detection(detect_out.view(-1), detection_target)
                
                # BBox loss if lesion present
                if detection_flag == 1 and boxes.shape[0] > 0:
                    bbox_target = boxes[0].unsqueeze(0).to(device)
                    loss_bbox = criterion_bbox(bbox_out, bbox_target)
                else:
                    loss_bbox = torch.tensor(0.0, device=device)

                # Weighted combined loss
                loss = (detection_loss_weight * loss_detection 
                        + bbox_loss_weight * loss_bbox)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_dataloader)
            log_msg = f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}"

            # Validation, if available
            if val_dataloader:
                avg_val_loss = self.evaluate(val_dataloader, criterion_detection, criterion_bbox, device)
                log_msg += f", Val Loss: {avg_val_loss:.4f}"

                # Early Stopping Logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_model_wts = self.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(log_msg)
                        print("Early stopping triggered.")
                        # Load the best model weights before exiting
                        if best_model_wts is not None:
                            self.load_state_dict(best_model_wts)
                        return

            print(log_msg)

        # If training completes without early stopping, optionally load best weights
        if best_model_wts is not None:
            self.load_state_dict(best_model_wts)
        print("Training complete.")

if __name__ == "__main__":
    # Instantiate the detection/localization model
    model = MammographyDetectionModel(pretrained=True)
    # Train the model using training and (optional) validation data
    model.train_model(
        train_csv_file="data/train_localization.csv",  # Path to your training CSV
        train_img_dir="data/train_images",             # Directory with training images
        val_csv_file="data/val_localization.csv",      # Path to your validation CSV (optional)
        val_img_dir="data/val_images",                 # Directory with validation images (optional)
        laterality="R",                                # Example: right breast
        view="MLO",                                    # Example: mediolateral oblique view
        num_epochs=20,                                 # Total number of epochs
        learning_rate=0.001,
        weight_decay=1e-5,                             # L2 regularization strength
        detection_loss_weight=1.0,
        bbox_loss_weight=1.0,
        batch_size=1,
        patience=5                                      # Early stopping patience
    )
