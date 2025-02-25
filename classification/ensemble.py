import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from common.local.dataset import MammographyDataset
from model import MammographyModel

import torch.nn as nn
import torch.optim as optim


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output


    def train_ensemble(self, csv_file, img_dir):
        # Train individual models with different parameters
        model_L_MLO = MammographyModel(num_classes=4)
        model_L_CC = MammographyModel(num_classes=4)
        model_R_MLO = MammographyModel(num_classes=4)
        model_R_CC = MammographyModel(num_classes=4)

        model_L_MLO.train_model(laterality="L", view="MLO")
        model_L_CC.train_model(laterality="L", view="CC")
        model_R_MLO.train_model(laterality="R", view="MLO")
        model_R_CC.train_model(laterality="R", view="CC")

        # Save the trained models
        torch.save(model_L_MLO.state_dict(), "/home/team11/dev/MediSense/classification/t1/model_L_MLO.pth")
        torch.save(model_L_CC.state_dict(), "/home/team11/dev/MediSense/classification/t1/model_L_CC.pth")
        torch.save(model_R_MLO.state_dict(), "/home/team11/dev/MediSense/classification/t1/model_R_MLO.pth")
        torch.save(model_R_CC.state_dict(), "/home/team11/dev/MediSense/classification/t1/model_R_CC.pth")

        # Load the trained models
        model_L_MLO.load_state_dict(torch.load("/home/team11/dev/MediSense/classification/t1/model_L_MLO.pth"))
        model_L_CC.load_state_dict(torch.load("/home/team11/dev/MediSense/classification/t1/model_L_CC.pth"))
        model_R_MLO.load_state_dict(torch.load("/home/team11/dev/MediSense/classification/t1/model_R_MLO.pth"))
        model_R_CC.load_state_dict(torch.load("/home/team11/dev/MediSense/classification/t1/model_R_CC.pth"))

        # Create ensemble model
        ensemble_model = EnsembleModel([model_L_MLO, model_L_CC, model_R_MLO, model_R_CC])

        # Save the ensemble model
        torch.save(ensemble_model.state_dict(), "/home/team11/dev/MediSense/classification/t1/ensemble_model.pth")

        print("Ensemble training complete")


if __name__ == "__main__":
    train_ensemble()
