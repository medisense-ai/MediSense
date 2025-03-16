from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MammographyDataset(Dataset):

    def __init__(self, csv_file, img_dir, laterality, view, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.laterality = laterality
        self.view = view
        self.transform = transform

        self.unique_case_ids = self.data["case_id"].unique()
        # self.case_id_mapping = {old_id : new_id for new_id, old_id in enumerate(unique_case_ids)}

    def __len__(self):
        return len(
            self.data[
                (self.data["laterality"] == self.laterality)
                & (self.data["view"] == self.view)
            ]
        )

    def __getitem__(self, idx):
        actual_idx = self.unique_case_ids[idx]
        row = self.data[
            (self.data["case_id"] == actual_idx)
            & (self.data["laterality"] == self.laterality)
            & (self.data["view"] == self.view)
        ]
        if row.empty:
            raise ValueError(
                f"Case ID {actual_idx} not found for laterality {self.laterality} and view {self.view}"
            )
        label = row["category"].values[0]
        img_file = f'{actual_idx}/{row["image_id"].values[0]}.jpg'
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class MammographyDataset2(Dataset):

    def __init__(self, csv_file, img_dir, laterality, view, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.laterality = laterality
        self.view = view
        self.transform = transform

        self.unique_case_ids = self.data["case_id"].unique()
        # self.case_id_mapping = {old_id : new_id for new_id, old_id in enumerate(unique_case_ids)}

    def __len__(self):
        return len(
            self.data[
                (self.data["laterality"] == self.laterality)
                & (self.data["view"] == self.view)
            ]
        )

    def __getitem__(self, idx):
        actual_idx = self.unique_case_ids[idx]
        row = self.data[
            (self.data["case_id"] == actual_idx)
            & (self.data["laterality"] == self.laterality)
            & (self.data["view"] == self.view)
        ]
        if row.empty:
            raise ValueError(
                f"Case ID {actual_idx} not found for laterality {self.laterality} and view {self.view}"
            )
        label = row["birads"].values[0]
        img_file = f'{actual_idx}/{row["image_id"].values[0]}.jpg'
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class InferenceMammographyDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.unique_case_ids = self.data["case_id"].unique()
        # self.case_id_mapping = {old_id : new_id for new_id, old_id in enumerate(unique_case_ids)}

    def __len__(self):
        return len(self.data) / 4

    def __getitem__(self, idx):
        actual_idx = self.unique_case_ids[idx]
        rows = self.data[(self.data["case_id"] == actual_idx)]
        if rows.empty:
            raise ValueError(
                f"Case ID {actual_idx} not found"
            )

        case_id = rows["case_id"].values[0]
        out = {"L" : {}, "R" : {}}
        for laterality in ["L", "R"]:
            for view in ["CC", "MLO"]:
                img_name = rows[(rows["laterality"] == laterality) & (rows["view"] == view)]["image_id"].values[0]
                img_file = f'{actual_idx}/{img_name}.jpg'
                img_path = os.path.join(self.img_dir, img_file)
                image = Image.open(img_path)
                
				if self.transform:
					image = self.transform(image)
                         
				out[laterality][view] = image

        return case_id, out
