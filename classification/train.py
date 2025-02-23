from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from common.dataset import MammographyDataset
from common.splitting import split_by_case


# 1) Split the dataset

split_by_case(
    csv_path = "/home/team11/dev/MediSense/common/classification.csv",
    output_train_csv = "/home/team11/dev/MediSense/classification/temp/train_labels.csv",
    output_val_csv = "/home/team11/dev/MediSense/classification/temp/val_labels.csv",
    test_size = 0.2,
    random_state = None
)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)



for laterality in ["L", "R"]:
    for view in ["CC", "MLO"]:
        ds = MammographyDataset(
            csv_file="/home/team11/dev/MediSense/classification/temp/train_labels.csv",
            img_dir="/home/data/train/images",
            laterality=laterality,
            view=view,
            transform=transform,
        )
        dataloader = DataLoader(ds, batch_size=8, shuffle=True)
        