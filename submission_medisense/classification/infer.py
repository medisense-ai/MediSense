import sys
import os

# Add the directory two levels up to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pickle

from common.local.logger import Logger, log_cuda_memory
from common.local.dataset import InferenceMammographyDataset
from classification.model_resnet import MammoClassificationResNet50
from classification.ensemble_LR import WeightedEnsembleModel
from classification.ensemble_combine import CombinedEnsemble


def get_model(
    task: str = "t1",
    file: str = "/home/team11/dev/MediSense/classification/t1/train_labels.csv",
    ens_file: str = "/home/team11/dev/MediSense/classification/t1/ensemble_weights.pkl",
):
    # Load models

    # BATCH_SIZE = 32
    # WORKERS = 4

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #     ]
    # )

    # ds = InferenceMammographyDataset(
    #     csv_file=file,
    #     img_dir="/home/data/train/images",
    #     transform=transform,
    # )
    # dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=WORKERS)

    ensembles = torch.nn.ModuleDict({})
    num_categories = 4 if task == "t1" else 6

    for laterality in ["L", "R"]:
        models = torch.nn.ModuleDict({})

        for view in ["CC", "MLO"]:
            model = MammoClassificationResNet50(categories=num_categories)
            model.load_model(
                f"/home/team11/dev/MediSense/classification/{task}/model_{laterality}_{view}.pth"
            )
            models[view] = model

        with open(ens_file, "rb") as f:
            detection_weights = pickle.load(f)

        ensembles[laterality] = WeightedEnsembleModel(
            laterality=laterality, models=models, detection_weights=detection_weights
        )

    ens_combined = CombinedEnsemble(ensembles)
    return ens_combined

    probs, labels = ens_combined(dataloaders)
