"""Script to generate submission file.

File structure, existing in evaluation environment:
    -data/
        -test/
            -images/
                - ...
            -localization.csv
            -classification.csv
        -train/
            -images/
                - ...
            -localization.csv
            -classification.csv

File structure, expected output of submission.py script:
    -output/
        -classification_results.csv
        -localization_results/
            -4/
                -4_L_CC/
                    -Architectural_Distortion.png
                    -Focal_Asymmetry.png
                    -Mass.png
                    -Other.png
                    -Suspicious_Calcification.png
                    -Suspicious_Lymph_Node.png
                -4_L_MLO/
                    -...
                -4_R_CC/
                    -...
                -4_R_MLO/
                    -...
            -...
"""

import os
from tqdm import tqdm
import argparse
import random

import torch
import models
import datasets
import utils
import cv2

import pandas as pd
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import numpy as np

from classification.infer import get_model
from common.local.dataset import InferenceMammographyDataset


def predict(
    model: nn.Module,
    dataloader: datasets.DataLoader,
    device: str = "cuda:0",
    test: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Make predictions on a dataloader with a model.

    Args:
        model: The model to use for prediction.
        dataloader: The dataloader containing the data.
        device: The device to use for prediction. Defaults to 'cuda:0'.

    Returns:
        A dictionary with the following keys:
            preds: The predicted labels.
            logits: The raw output of the model.
            softmax: The output of the model after applying softmax.
    """
    model = model.to(device)
    model.eval()
    logits = []
    preds = []
    case_ids = []
    softmax_outputs = []
    with torch.no_grad():
        for case_id, data in tqdm(dataloader):
            inputs = {
                "L": {k: data[k].to(device) for k in ["CC", "MLO"]},
                "R": {k: data[k].to(device) for k in ["CC", "MLO"]},
            }
            # inputs = {k: data[k].to(device) for k in ["L_CC", "R_CC", "L_MLO", "R_MLO"]}
            # # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            if softmax_outputs == []:
                softmax_outputs = nn.functional.softmax(outputs, dim=1).detach().cpu()
            else:
                softmax_outputs = torch.concat(
                    (
                        softmax_outputs,
                        nn.functional.softmax(outputs, dim=1).detach().cpu(),
                    )
                )
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.detach().cpu().tolist())
            case_ids.extend(list(case_id))
            if test:
                break

    return {
        "case_id": case_ids,  # Mandatory - outputs should look like this
        "preds": preds,  # Mandatory - outputs should look like this
        "probabilityC0": softmax_outputs[
            :, 0
        ],  # Mandatory - outputs should look like this
        "probabilityC1": softmax_outputs[
            :, 1
        ],  # Mandatory - outputs should look like this
        "probabilityC2": softmax_outputs[
            :, 2
        ],  # Mandatory - outputs should look like this
        "probabilityC3": softmax_outputs[:, 3],
    }  # Mandatory - outputs should look like this


# You can save the images to disk after pre-preprocessing them
# os.makedirs('tmp', exist_ok=True)


def arg_parse():  # Mandatory
    parser = argparse.ArgumentParser(description="Mammo Classification")  # Mandatory
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,  # Mandatory
        help="data directory containing test data, organizers set this path.",
    )  # Mandatory
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,  # Mandatory
        help="data directory containing train data, organizers set this path.",
    )  # Mandatory
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,  # Mandatory
        help="output directory, organizers set this path.",
    )  # Mandatory
    parser.add_argument(
        "--test",
        action="store_true",  # Mandatory
        help="Test the code on a single batch.",
    )  # Mandatory

    return parser.parse_args()


LABELS_LOCALIZATION = [
    "Architectural_Distortion",
    "Focal_Asymmetry",
    "Mass",
    "Other",
    "Suspicious_Calcification",
    "Suspicious_Lymph_Node",
]

if __name__ == "__main__":
    # !------------- Task 1 classification problem -------------!
    # !---- Mandatory make output directory ----!
    args = arg_parse()  # Mandatory
    os.makedirs(args.output_dir, exist_ok=True)  # Mandatory
    TEST_DATA_DIR = os.path.join(args.test_dir)  # Mandatory
    TRAIN_DATA_DIR = os.path.join(args.train_dir)  # Mandatory
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = "cuda:0"
    else:
        device = "cpu"
        print("!!!! CUDA is NOT available!!!!")
    # -------- Data directory --------

    print("Preprocessing done!")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ds = InferenceMammographyDataset(
        csv_file=str(TEST_DATA_DIR),
        img_dir="/home/data/train/images",
        transform=transform,
    )
    dataloader = DataLoader(ds, batch_size=16, num_workers=4)

    model = get_model(task="t1", file=str(TEST_DATA_DIR))
    output_dictionary = predict(model, dataloader, device=device, test=args.test)
    out_df = pd.DataFrame(output_dictionary)

    # !---- Output of task 1 classification problem ----!
    # !---- Output is per case not per image ----!
    out_df.drop_duplicates(subset=["case_id"], inplace=True, keep="first")
    out_df.reset_index(drop=True, inplace=True)
    #!---- Output of task 1 classification problem ----!
    out_df.to_csv(
        os.path.join(args.output_dir, "classification_results.csv"),  # Mandatory
        index=False,
    )  # Mandatory
    print("Dummy classification done!")
    # ****************************************************************************
    # !------------- Task 2 localization problem -------------!
    # ****************************************************************************
    dataset = datasets.MammoLocalizationDataset(data_dir=str(TEST_DATA_DIR))
    df = pd.read_csv(os.path.join(TRAIN_DATA_DIR, "localization.csv"))
    model_localization = models.RandomLocalization(df)
    dfl = pd.read_csv(os.path.join(TEST_DATA_DIR, "localization.csv"))
    # print(dfl.shape)
    # ---- You can store some intermediate results ----!
    if os.path.isfile(os.path.join(args.output_dir, "results_with_image_info.csv")):
        dfl = pd.read_csv(os.path.join(args.output_dir, "results_with_image_info.csv"))
        print("Resuming from previous progress!")
    else:
        dfl = utils.get_images_dim_and_contour(dfl, dataset)
        dfl.to_csv(
            os.path.join(args.output_dir, "results_with_image_info.csv"), index=False
        )

    # !---- Output of task 2 segmentation problem and multi-label----!
    results_df = model_localization.predict(dfl)
    for i, row in results_df.iterrows():
        # !------------------ The expected file system structure is as: -------------------!# Mandatory
        per_image_id_output_dirs = (
            args.output_dir
            + "/localization_results/{}/{}".format(row.case_id, row.image_id)
        )  # Mandatory
        os.makedirs(per_image_id_output_dirs, exist_ok=True)  # Mandatory
        for category in LABELS_LOCALIZATION:  # Mandatory
            if category == "No_Finding":  # Mandatory
                pass
            else:  # Mandatory
                mask = np.zeros(
                    (row["height"], row["width"]), dtype=np.uint8
                )  # Mandatory
                cv2.imwrite(
                    per_image_id_output_dirs + f"/{category}.png",  # Mandatory
                    mask,
                    [cv2.IMWRITE_PNG_BILEVEL, 1],
                )  # Mandatory
        # Heuristically generated localizations --> segmentation maps
        if row["category"] != "No_Finding":
            xmin, ymin = utils.get_xmin_ymin(row)
            area = model_localization.average_area_dict[
                row["category"]
            ] * random.uniform(0.8, 1.2)

            side_length = int(area**0.5)
            width, height = row["width"], row["height"]
            mask = np.zeros((height, width), dtype=np.uint8)
            xmax, ymax = xmin + side_length, ymin + side_length
            mask[int(ymin) : int(ymax), int(xmin) : int(xmax)] = 1
            # !---------------- Save all category masks in .png format even zeros ---------------!                                                                  # Mandatory
            cv2.imwrite(
                per_image_id_output_dirs + f'/{row["category"]}.png',  # Mandatory
                mask,
                [cv2.IMWRITE_PNG_BILEVEL, 1],
            )  # Mandatory
