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
    import pandas as pd
    import numpy as np
    import cv2
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    # Import your custom classes from your localization code.
    from LocalizationModel import MammoLocalizationResNet50
    from LocalizationDataset import MammographyLocalizationDataset, ResizeWithBBoxes, NormalizeImageNet

    # Define helper transforms (same as in your BestThreshold.py)
    class ComposeDouble:
        def __init__(self, transforms_list):
            self.transforms = transforms_list

        def __call__(self, image, target):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

    class ToTensorDouble:
        def __call__(self, image, target):
            image = transforms.ToTensor()(image)
            return image, target

    # -------- Load best threshold value --------
    # You can either hardcode this value or load it from a file (e.g., best_threshold.txt).
    # For example:
    best_threshold = 0.5  # Replace with your computed best threshold or load it from file:
    # with open("best_threshold.txt", "r") as f:
    #     best_threshold = float(f.read().strip())

    # -------- Prepare test dataset for localization --------
    # Here we assume that the test CSV is located in TEST_DATA_DIR and images are in a subfolder "images"
    transform_localization = ComposeDouble([
        ResizeWithBBoxes((1024, 1024)),
        ToTensorDouble(),
        NormalizeImageNet(),
    ])
    test_dataset = MammographyLocalizationDataset(
        csv_file=os.path.join(TEST_DATA_DIR, "localization.csv"),
        img_dir=os.path.join(TEST_DATA_DIR, "images"),
        transform=transform_localization
    )

    # Define a simple collate function (as in your training script)
    def collate_fn(batch):
        return tuple(zip(*batch))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # -------- Load the trained localization model --------
    model_localization = MammoLocalizationResNet50(num_classes=7, pretrained=False)
    localization_model_path = "/home/team11/dev/MediSense/localization/models/localization_model.pth"  # Update with the correct path
    model_localization.load_model(localization_model_path)
    model_localization.model.to(device)
    model_localization.model.eval()

    # -------- Run inference to get detection outputs --------
    detections, _ = model_localization.infer(test_loader)
    
    # -------- Get image info (dimensions and identifiers) --------
    # We use the CSV from the test directory and augment it with image dimensions using your utils function.
    dfl = pd.read_csv(os.path.join(TEST_DATA_DIR, "localization.csv"))
    dfl = utils.get_images_dim_and_contour(dfl, test_dataset)
    dfl.to_csv(os.path.join(args.output_dir, "results_with_image_info.csv"), index=False)

    # -------- Build a results dataframe from detection outputs --------
    # Here we create one row per image. If no detection meets the best threshold, we mark it as "No_Finding".
    results = []
    # Create an inverse mapping for category labels.
    inv_category_map = {1: "Mass", 2: "Suspicious_Calcification", 3: "Focal_Asymmetry",
                          4: "Architectural_Distortion", 5: "Suspicious_Lymph_Node", 6: "Other"}
    # Iterate over test images (assumes same order as in dfl)
    for i, ((image, target), detection) in enumerate(zip(test_dataset, detections)):
        info = dfl.iloc[i]
        case_id = info["case_id"]
        image_id = info["image_id"]
        height = info["height"]
        width = info["width"]

        boxes = detection["boxes"]
        scores = detection["scores"]
        labels = detection["labels"]

        # Filter detections by best_threshold
        keep = scores >= best_threshold
        if keep.sum() > 0:
            # Choose the detection with the highest score among those above threshold.
            filtered_indices = keep.nonzero(as_tuple=True)[0]
            best_idx = filtered_indices[torch.argmax(scores[keep])]
            pred_box = boxes[best_idx].cpu().numpy()
            pred_label = labels[best_idx].item()
            pred_category = inv_category_map.get(pred_label, "No_Finding")
        else:
            pred_box = None
            pred_category = "No_Finding"

        results.append({
            "case_id": case_id,
            "image_id": image_id,
            "height": height,
            "width": width,
            "category": pred_category,
            "xmin": pred_box[0] if pred_box is not None else np.nan,
            "ymin": pred_box[1] if pred_box is not None else np.nan,
            "xmax": pred_box[2] if pred_box is not None else np.nan,
            "ymax": pred_box[3] if pred_box is not None else np.nan,
        })
    results_df = pd.DataFrame(results)

    # !---- Output of task 2 segmentation problem and multi-label----!
    # The following code block (with all Mandatory comments) remains unchanged.
    results_df = results_df  # (using our results_df from above)
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
            # Use the predicted bounding box information
            xmin, ymin = row["xmin"], row["ymin"]
            # Here we use the box size from the detection; you could also use a fixed or computed area.
            side_length = int(((row["xmax"] - row["xmin"]) * (row["ymax"] - row["ymin"])) ** 0.5)
            mask = np.zeros((row["height"], row["width"]), dtype=np.uint8)
            xmax, ymax = xmin + side_length, ymin + side_length
            mask[int(ymin) : int(ymax), int(xmin) : int(xmax)] = 1
            # !---------------- Save all category masks in .png format even zeros ---------------!                                                                  # Mandatory
            cv2.imwrite(
                per_image_id_output_dirs + f'/{row["category"]}.png',  # Mandatory
                mask,
                [cv2.IMWRITE_PNG_BILEVEL, 1],
            )  # Mandatory
