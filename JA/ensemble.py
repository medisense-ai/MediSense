import torch
import torch.nn as nn
import numpy as np

# Updated detection model and dataset references
from model import MammographyDetectionModel
from dataset import get_transform, MammographyLocalizationDataset

class WeightedEnsembleModel(nn.Module):
    """
    Ensemble of multiple detection/localization models. Each model returns:
      - detect: Probability of lesion presence (shape: [batch_size, 1])
      - bbox:   Predicted bounding box coordinates (shape: [batch_size, 4])

    We apply different weights for each model to produce a weighted average:
      avg_detect = sum(w_i * detect_i) / sum(w_i)
      avg_bbox   = sum(w_i * bbox_i)   / sum(w_i)

    detection_weights and bbox_weights are lists of floats, one per model. If not provided,
    defaults to uniform weighting.
    """
    def __init__(self, models, detection_weights=None, bbox_weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)

        # If no weights specified, default to equal weighting
        num_models = len(self.models)
        if detection_weights is None:
            detection_weights = [1.0] * num_models
        if bbox_weights is None:
            bbox_weights = [1.0] * num_models

        self.detection_weights = detection_weights
        self.bbox_weights = bbox_weights

    def forward(self, x):
        detect_outputs = []
        bbox_outputs = []

        # Gather outputs from each model and weight them
        for model, w_det, w_box in zip(self.models, self.detection_weights, self.bbox_weights):
            detect, bbox = model(x)
            detect_outputs.append(detect * w_det)
            bbox_outputs.append(bbox * w_box)

        # Sum the weighted outputs
        sum_detect = torch.stack(detect_outputs, dim=0).sum(dim=0)
        sum_bbox = torch.stack(bbox_outputs, dim=0).sum(dim=0)

        # Compute total weights
        total_detection_weight = sum(self.detection_weights)
        total_bbox_weight = sum(self.bbox_weights)

        # Weighted average
        avg_detect = sum_detect / total_detection_weight
        avg_bbox = sum_bbox / total_bbox_weight

        return avg_detect, avg_bbox

def evaluate_detection_accuracy(model, csv_file, img_dir, laterality, view, device=torch.device("cpu")):
    """
    Computes the detection accuracy of `model`:
      - For each image, we do a forward pass.
      - We threshold detection probability at 0.5 to decide if a lesion is predicted.
      - Compare that prediction to the ground truth detection_flag (0 or 1).
    Returns:
      detection_accuracy (float): ratio of correct predictions over total images.
    """
    val_transform = get_transform(train=False)
    val_dataset = MammographyLocalizationDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        laterality=laterality,
        view=view,
        transform=val_transform,
        auto_crop_enabled=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x
    )

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_dataloader:
            image, boxes, labels, detection_flag = batch[0]
            image = image.unsqueeze(0).to(device)

            detect_out, _ = model(image)  # (batch_size=1, 1), (batch_size=1, 4)
            predicted_flag = (detect_out.item() >= 0.5)  # True/False
            predicted_flag = int(predicted_flag)         # 1 or 0

            if predicted_flag == detection_flag:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def train_single_model(laterality, view, train_csv, train_img_dir, val_csv, val_img_dir,
                       device=torch.device("cpu")):
    """
    Creates and trains a MammographyDetectionModel for a specific laterality/view subset.
    Returns the trained model plus its detection accuracy on the validation set.
    """
    model = MammographyDetectionModel(pretrained=True)
    # Adjust as needed; reference new directories:
    model.train_model(
        train_csv_file=train_csv,
        train_img_dir=train_img_dir,
        val_csv_file=val_csv,
        val_img_dir=val_img_dir,
        laterality=laterality,
        view=view,
        num_epochs=10,
        learning_rate=0.001,
        weight_decay=1e-5,
        detection_loss_weight=1.0,
        bbox_loss_weight=1.0,
        batch_size=1,
        patience=5,
        device=device
    )
    # Evaluate detection accuracy on validation set
    accuracy = evaluate_detection_accuracy(
        model, val_csv, val_img_dir, laterality, view, device=device
    )
    return model, accuracy

def train_ensemble():
    """
    Trains four separate detection/localization models (L_MLO, L_CC, R_MLO, R_CC),
    computes their validation detection accuracies, normalizes these into weights,
    and creates a weighted ensemble model.

    Data structure:
      - train set:    data/train/localization.csv, data/train/images/[case_id]/image_name.jpg
      - subset eval:  data/subset_train_eval/localization.csv, data/subset_train_eval/images/[case_id]/image_name.jpg
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths for training and validation
    train_csv = "/home/team11/data/train/localization.csv"
    train_img_dir = "/home/team11/data/train/images"
    val_csv = "/home/team11/data/subset_train_eval/localization.csv"
    val_img_dir = "/home/team11/data/subset_train_eval/images"

    # Train individual models
    model_L_MLO, acc_L_MLO = train_single_model("L", "MLO", train_csv, train_img_dir, val_csv, val_img_dir, device=device)
    model_L_CC, acc_L_CC = train_single_model("L", "CC", train_csv, train_img_dir, val_csv, val_img_dir, device=device)
    model_R_MLO, acc_R_MLO = train_single_model("R", "MLO", train_csv, train_img_dir, val_csv, val_img_dir, device=device)
    model_R_CC, acc_R_CC = train_single_model("R", "CC", train_csv, train_img_dir, val_csv, val_img_dir, device=device)

    print("Validation detection accuracies for each model:")
    print(f"L_MLO: {acc_L_MLO:.3f}, L_CC: {acc_L_CC:.3f}, R_MLO: {acc_R_MLO:.3f}, R_CC: {acc_R_CC:.3f}")

    # Save each model's weights
    torch.save(model_L_MLO.state_dict(), "model_L_MLO.pth")
    torch.save(model_L_CC.state_dict(), "model_L_CC.pth")
    torch.save(model_R_MLO.state_dict(), "model_R_MLO.pth")
    torch.save(model_R_CC.state_dict(), "model_R_CC.pth")

    # Compute normalized weights from accuracies
    detection_accuracies = [acc_L_MLO, acc_L_CC, acc_R_MLO, acc_R_CC]
    total_acc = sum(detection_accuracies)
    if total_acc == 0:
        # Fallback: if all accuracies are zero, use equal weights
        detection_weights = [1.0, 1.0, 1.0, 1.0]
    else:
        detection_weights = [acc / total_acc for acc in detection_accuracies]

    # For simplicity, we'll use the same weights for bounding box outputs
    bbox_weights = detection_weights

    # Reload each model (optional) so the ensemble can start fresh
    model_L_MLO.load_state_dict(torch.load("model_L_MLO.pth"))
    model_L_CC.load_state_dict(torch.load("model_L_CC.pth"))
    model_R_MLO.load_state_dict(torch.load("model_R_MLO.pth"))
    model_R_CC.load_state_dict(torch.load("model_R_CC.pth"))

    # Create weighted ensemble
    ensemble_model = WeightedEnsembleModel(
        [model_L_MLO, model_L_CC, model_R_MLO, model_R_CC],
        detection_weights=detection_weights,
        bbox_weights=bbox_weights
    )

    # Save the ensemble model weights
    torch.save(ensemble_model.state_dict(), "ensemble_model.pth")

    print("\nWeighted Ensemble Model created.")
    print("Weights based on validation detection accuracy:")
    print(f"Detection Weights: {detection_weights}")
    print("Ensemble training complete.")

if __name__ == "__main__":
    train_ensemble()
