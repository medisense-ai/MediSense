import torch
import torch.nn as nn
import numpy as np
FileNotFoundError
from model import MammographyDetectionModel
# Now we import the simpler dataset class (no Albumentations)
from dataset import MammographyLocalizationDataset

class WeightedEnsembleModel(nn.Module):
    """
    Ensemble of multiple detection/localization models. 
    Averages detection outputs and bounding boxes by specified weights.
    """
    def __init__(self, models, detection_weights=None, bbox_weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
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
        for model, w_det, w_box in zip(self.models, self.detection_weights, self.bbox_weights):
            detect, bbox = model(x)
            detect_outputs.append(detect * w_det)
            bbox_outputs.append(bbox * w_box)
        sum_detect = torch.stack(detect_outputs, dim=0).sum(dim=0)
        sum_bbox = torch.stack(bbox_outputs, dim=0).sum(dim=0)
        avg_detect = sum_detect / sum(self.detection_weights)
        avg_bbox = sum_bbox / sum(self.bbox_weights)
        return avg_detect, avg_bbox

def evaluate_detection_accuracy(model, csv_file, img_dir, laterality, view, device=torch.device("cpu")):
    """
    Computes the detection accuracy of `model`:
      - We iterate over the dataset, do a forward pass with images.
      - If model outputs detect_out >= 0.5 => predicted_flag=1, else 0.
      - Compare to detection_flag for each sample.
    """
    val_dataset = MammographyLocalizationDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        laterality=laterality,
        view=view,
        train=False  # no random rotation
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

            detect_out, _ = model(image)
            predicted_flag = int(detect_out.item() >= 0.5)
            if predicted_flag == detection_flag:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def train_single_model(laterality, view, train_csv, train_img_dir, val_csv, val_img_dir, device=torch.device("cpu")):
    """
    Creates and trains a MammographyDetectionModel for one (laterality, view) subset.
    Returns the trained model plus detection accuracy on val set.
    """
    model = MammographyDetectionModel(pretrained=True)
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
    accuracy = evaluate_detection_accuracy(model, val_csv, val_img_dir, laterality, view, device=device)
    return model, accuracy

def train_ensemble():
    """
    Trains four detection/localization models:
        L_MLO, L_CC, R_MLO, R_CC
    Normalizes detection accuracies into weights, builds a WeightedEnsembleModel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train + Val data paths
    train_csv = "/home/data/train/localization.csv"
    train_img_dir="/home/data/train/images"       
    val_csv = "/home/data/subset-train-eval/localization.csv"
    val_img_dir = "/home/data/subset-train-eval/images"

    # Train individual models
    model_L_MLO, acc_L_MLO = train_single_model("L", "MLO", train_csv, train_img_dir, val_csv, val_img_dir, device=device)
    model_L_CC, acc_L_CC = train_single_model("L", "CC", train_csv, train_img_dir, val_csv, val_img_dir, device=device)
    model_R_MLO, acc_R_MLO = train_single_model("R", "MLO", train_csv, train_img_dir, val_csv, val_img_dir, device=device)
    model_R_CC, acc_R_CC = train_single_model("R", "CC", train_csv, train_img_dir, val_csv, val_img_dir, device=device)

    print("Validation detection accuracies for each model:")
    print(f"L_MLO: {acc_L_MLO:.3f}, L_CC: {acc_L_CC:.3f}, R_MLO: {acc_R_MLO:.3f}, R_CC: {acc_R_CC:.3f}")

    # Save weights
    torch.save(model_L_MLO.state_dict(), "model_L_MLO.pth")
    torch.save(model_L_CC.state_dict(), "model_L_CC.pth")
    torch.save(model_R_MLO.state_dict(), "model_R_MLO.pth")
    torch.save(model_R_CC.state_dict(), "model_R_CC.pth")

    # Compute normalized detection weights
    detection_accuracies = [acc_L_MLO, acc_L_CC, acc_R_MLO, acc_R_CC]
    total_acc = sum(detection_accuracies)
    if total_acc == 0:
        detection_weights = [1.0, 1.0, 1.0, 1.0]
    else:
        detection_weights = [acc / total_acc for acc in detection_accuracies]

    # For simplicity, use same weights for bounding box outputs
    bbox_weights = detection_weights

    # Reload each model for the ensemble
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
    torch.save(ensemble_model.state_dict(), "ensemble_model.pth")

    print("\nWeighted Ensemble Model created.")
    print("Weights based on validation detection accuracy:", detection_weights)
    print("Ensemble training complete.")

if __name__ == "__main__":
    train_ensemble()