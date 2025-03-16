#!/usr/bin/env python
import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Add the parent directory to the Python path if needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import your custom dataset, transforms, model, and logger.
from LocalizationDataset import MammographyLocalizationDataset, ResizeWithBBoxes, NormalizeImageNet
from LocalizationModel import MammoLocalizationResNet50
from logger import Logger

# --- Helper Classes for Transforms ---
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

# --- Helper Function to Compute IoU ---
def compute_iou(box1, box2, eps=1e-8):
    """
    Compute the Intersection over Union (IoU) between two boxes.
    Each box is in the format [xmin, ymin, xmax, ymax].
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / (area_box1 + area_box2 - inter_area + eps)
    return iou

# --- Evaluation Function for a Given Threshold ---
def evaluate_threshold(model, dataloader, device, score_threshold, iou_threshold=0.5):
    """
    For each image in the dataloader, filter predictions using `score_threshold`
    and match predicted boxes to ground truth boxes using an IoU threshold.
    Returns overall precision, recall, F1 score, and mean IoU.
    """
    total_TP = 0
    total_FP = 0
    total_FN = 0
    iou_values = []  # Collect IoU values for true positives

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        # Move ground truth boxes to CPU for evaluation.
        gt_boxes_list = [t['boxes'].cpu() for t in targets]

        with torch.no_grad():
            outputs = model(images)  # Model returns list of dicts

        for gt_boxes, output in zip(gt_boxes_list, outputs):
            if 'scores' in output:
                scores = output['scores']
                pred_boxes = output['boxes']
                # Filter predictions by the score threshold.
                keep_idx = scores >= score_threshold
                pred_boxes = pred_boxes[keep_idx].cpu()
            else:
                pred_boxes = output['boxes'].cpu()

            matched_gt = set()
            TP = 0
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    iou_val = compute_iou(pred_box.numpy(), gt_box.numpy())
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = i
                if best_iou >= iou_threshold:
                    TP += 1
                    matched_gt.add(best_gt_idx)
                    iou_values.append(best_iou)
            FP = len(pred_boxes) - TP
            FN = len(gt_boxes) - TP

            total_TP += TP
            total_FP += FP
            total_FN += FN

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(iou_values) if iou_values else 0.0
    return precision, recall, f1, mean_iou

# --- Test Transform for Inference (No ground truth boxes) ---
class TestTransform:
    def __init__(self, size):
        self.resize = ResizeWithBBoxes(size)  # Expects a target; we create a dummy one.
        self.to_tensor = transforms.ToTensor()
        self.normalize = NormalizeImageNet()

    def __call__(self, image):
        dummy_target = {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}
        image, _ = self.resize(image, dummy_target)
        image = self.to_tensor(image)
        image, _ = self.normalize(image, dummy_target)
        return image

def main():
    # -------------------------------
    # Setup logging and device
    # -------------------------------
    # You can adjust the log directory and file name as needed.
    log = Logger(log_dir='./logs', log_file='merged_evaluation.log')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Running evaluation and inference on device: {device}")
    print(f"Running on device: {device}")

    # -------------------------------
    # Load the trained model
    # -------------------------------
    # Choose the model path. If you need to switch paths, adjust accordingly.
    MODEL_PATH = "/home/team11/dev/MediSense/localization/models/localization_model.pth"
    # MODEL_PATH = "/home/ja/new_env/AI4Health/localization/models/localization_model.pth"

    model = MammoLocalizationResNet50(num_classes=7, pretrained=True, logger=log)
    model.load_model(MODEL_PATH)
    model.model.to(device)
    model.model.eval()
    log.info("Model loaded and set to evaluation mode.")
    print("Model loaded and set to evaluation mode.")

    # -------------------------------
    # Evaluate thresholds on evaluation dataset
    # -------------------------------
    # Define evaluation transform pipeline.
    eval_transform = ComposeDouble([
        ResizeWithBBoxes((1024, 1024)),
        ToTensorDouble(),
        NormalizeImageNet()
    ])

    # Evaluation dataset and loader (adjust paths as needed)
    EVAL_CSV = "/home/team11/data/train/localization.csv"
    EVAL_IMG_DIR = "/home/data/train/images"
    eval_dataset = MammographyLocalizationDataset(
        csv_file=EVAL_CSV,
        img_dir=EVAL_IMG_DIR,
        transform=eval_transform
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,  # Process one image at a time for evaluation.
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Evaluate for a range of detection score thresholds.
    thresholds = [0.1 * i for i in range(1, 10)]
    results = {}
    print("Evaluating thresholds on evaluation set:")
    for thresh in thresholds:
        precision, recall, f1, mean_iou = evaluate_threshold(model.model, eval_loader, device, score_threshold=thresh)
        results[thresh] = (precision, recall, f1, mean_iou)
        log.info(f"Threshold {thresh:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Mean IoU={mean_iou:.4f}")
        print(f"Threshold {thresh:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Mean IoU={mean_iou:.4f}")

    # Determine the best threshold based on mean IoU, using F1 as a tie breaker.
    max_mean_iou = max(results.values(), key=lambda x: x[3])[3]
    epsilon = 1e-6
    best_candidates = [thresh for thresh, metrics in results.items() if abs(metrics[3] - max_mean_iou) < epsilon]
    if len(best_candidates) == 1:
        best_thresh = best_candidates[0]
    else:
        best_thresh = max(best_candidates, key=lambda t: results[t][2])
    print(f"\nSelected best threshold: {best_thresh:.2f} with Mean IoU = {results[best_thresh][3]:.4f} and F1 = {results[best_thresh][2]:.4f}")
    log.info(f"Selected best threshold: {best_thresh:.2f} with Mean IoU = {results[best_thresh][3]:.4f} and F1 = {results[best_thresh][2]:.4f}")

    # -------------------------------
    # Run inference on test dataset using the best threshold
    # -------------------------------
    # Define test transform.
    test_transform = TestTransform((1024, 1024))

    # Test dataset and loader (adjust paths as needed)
    TEST_CSV = "/home/team11/dev/MediSense/localization/testing/images/localization.csv"
    TEST_IMG_DIR = "/home/team11/dev/MediSense/localization/testing/ima"
    test_dataset = MammographyLocalizationDataset(
        csv_file=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=lambda img, target: (test_transform(img), target)  # Only transform the image.
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results_list = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(test_loader):
            images = [img.to(device) for img in images]
            outputs = model.model(images)
            output = outputs[0]
            # Use the best threshold found during evaluation.
            detection_threshold = best_thresh
            scores = output.get("scores", torch.tensor([]))
            boxes = output.get("boxes", torch.tensor([]))
            labels = output.get("labels", torch.tensor([]))
            if scores.numel() > 0:
                mask = scores > detection_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
            else:
                boxes, scores, labels = torch.tensor([]), torch.tensor([]), torch.tensor([])

            # Retrieve image identification info.
            case_id, image_id = test_dataset.image_keys[idx]
            
            # Optionally, get image dimensions.
            img_path = os.path.join(TEST_IMG_DIR, f'{case_id}/{image_id}.jpg')
            with Image.open(img_path) as pil_img:
                width, height = pil_img.size

            result_entry = {
                "case_id": case_id,
                "image_id": image_id,
                "width": width,
                "height": height,
                "boxes": boxes.cpu().numpy().tolist(),       # List of bounding boxes
                "scores": scores.cpu().numpy().tolist(),       # Confidence scores
                "labels": labels.cpu().numpy().tolist()        # Predicted labels (numeric)
            }
            results_list.append(result_entry)
            log.info(f"Processed {case_id}/{image_id}: {len(boxes)} detections.")

    # Save the results as CSV.
    results_df = pd.DataFrame(results_list)
    output_csv = os.path.join("./", "/home/team11/dev/MediSense/localization/results_with_image_info.csv")
    results_df.to_csv(output_csv, index=False)
    log.info(f"Saved prepared results to {output_csv}")
    print(f"\nPrepared localization results saved to {output_csv}")

if __name__ == "__main__":
    main()
