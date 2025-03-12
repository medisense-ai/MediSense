import sys
import os

# Add the directory two levels up to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from common.local.dataset import MammographyDataset2
from common.local.splitting import split_by_case
from common.local.logger import Logger, log_cuda_memory
from classification.model_resnet import MammoClassificationResNet50
from classification.ensemble_LR import WeightedEnsembleModel

BATCH_SIZE = 32
WORKERS = 4

log = Logger(log_dir='/home/team11/dev/MediSense/classification/temp', log_file='ensemble2.log')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Training on device: {device}")

# 1) Split the dataset

split_by_case(
    csv_path = "/home/team11/dev/MediSense/common/local/classification.csv",
    output_train_csv = "/home/team11/dev/MediSense/classification/temp/train_labels_3.csv",
    output_val_csv = "/home/team11/dev/MediSense/classification/temp/val_labels_3.csv",
    test_size = 0.2,
    random_state = 42,
    logger = log
)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


for laterality in ["L", "R"]:
    models = []
    dataloaders_train = []

    for view in ["CC", "MLO"]:

        log.info(f"Creating model for {laterality} {view}")
        ds = MammographyDataset2(
            csv_file="/home/team11/dev/MediSense/classification/temp/train_labels_3.csv",
            img_dir="/home/data/train/images",
            laterality=laterality,
            view=view,
            transform=transform,
        )
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=False)
        dataloaders_train.append(dataloader)


        model = MammoClassificationResNet50(categories=6, logger=log)
        models.append(model)

    log_cuda_memory(log)
        
    ensemble = WeightedEnsembleModel(models)
    log.info(f"Training ensemble for {laterality}")
    ensemble.train_ensemble(dataloaders_train)
    ensemble.save_models(laterality, ["CC", "MLO"], "/home/team11/dev/MediSense/classification/t2")

    log_cuda_memory(log)
    
log.info("Done")


    