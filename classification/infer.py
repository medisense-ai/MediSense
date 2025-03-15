import sys
import os

# Add the directory two levels up to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pickle

from common.local.logger import Logger, log_cuda_memory
from common.local.dataset import MammographyDataset, MammographyDataset2
from classification.model_resnet import MammoClassificationResNet50
from classification.ensemble_LR import WeightedEnsembleModel
from classification.ensemble_combine import CombinedEnsemble


def main(
    task: str = "t1", 
    file: str = "/home/team11/dev/MediSense/classification/t1/train_labels.csv",
    ens_file: str = "/home/team11/dev/MediSense/classification/t1/ensemble_weights.pkl"
):
    # Load models

    BATCH_SIZE = 32
    WORKERS = 4

    log = Logger(log_dir='/home/team11/dev/MediSense/classification/temp', log_file='ensemble2.log')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")


    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataloaders = {}
    ensembles = torch.nn.ModuleDict({})

    for laterality in ["L", "R"]:
        models = torch.nn.ModuleDict({})
        dataloaders[laterality] = {}
        
        for view in ["CC", "MLO"]:
            log.info(f"Creating model for {laterality} {view}")
            
            if task == "t1":
                ds = MammographyDataset(
                    csv_file=file,
                    img_dir="/home/data/train/images",
                    laterality=laterality,
                    view=view,
                    transform=transform,
                )
            else:
                ds = MammographyDataset2(
                    csv_file=file,
                    img_dir="/home/data/train/images",
                    laterality=laterality,
                    view=view,
                    transform=transform,
                )
            dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=False)
            dataloaders[laterality][view] = dataloader


            model = MammoClassificationResNet50(categories=6, logger=log)
            models[view] = model
        
        with open(ens_file, 'rb') as f:
            detection_weights = pickle.load(f)

        ensembles[laterality] = WeightedEnsembleModel(laterality = laterality, models = models, detection_weights=detection_weights)

    ens_combined = CombinedEnsemble(ensembles)
    log.info(f"Inference of ensemble for {task}")

    probs, labels = ens_combined(dataloaders)