import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
import concurrent.futures

# Add the parent directory to the Python path if needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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

# --- Vectorized IoU Evaluation ---
def evaluate_threshold_cached(cached_data, threshold, iou_threshold=0.5):
    """
    Evaluate detection metrics (precision, recall, F1, mean IoU) on cached data for one threshold.
    Uses torchvision.ops.box_iou for fast IoU computation.
    """
    total_TP = 0
    total_FP = 0
    total_FN = 0
    iou_values = []
    
    for gt_boxes, pred_boxes, pred_scores in cached_data:
        # Filter predictions based on the confidence threshold.
        keep_idx = pred_scores >= threshold
        pred_boxes_filtered = pred_boxes[keep_idx]
        
        # If no predictions pass the threshold, count all GT boxes as false negatives.
        if len(pred_boxes_filtered) == 0:
            total_FN += len(gt_boxes)
            continue
        
        # Compute IoU matrix between predicted boxes and ground truth boxes.
        ious = box_iou(pred_boxes_filtered, gt_boxes)
        
        # Greedy matching: assign each predicted box to a GT box if IoU >= threshold,
        # ensuring that each GT box is only matched once.
        assigned_gt = set()
        for i in range(ious.shape[0]):
            best_iou, best_idx = torch.max(ious[i], dim=0)
            if best_iou >= iou_threshold and best_idx.item() not in assigned_gt:
                total_TP += 1
                assigned_gt.add(best_idx.item())
                iou_values.append(best_iou.item())
            else:
                total_FP += 1
        total_FN += (len(gt_boxes) - len(assigned_gt))
        
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(iou_values) if iou_values else 0.0
    return precision, recall, f1, mean_iou

# --- Bootstrapping Iteration Function ---
def bootstrap_iteration(i, cached_data, thresholds, iou_threshold=0.5):
    """
    Performs one bootstrapping iteration:
    - Resamples cached_data with replacement.
    - Computes metrics for each threshold.
    Returns a dictionary mapping each threshold to its evaluation metrics.
    """
    n_samples = len(cached_data)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    sample = [cached_data[j] for j in indices]
    iteration_results = {}
    for thr in thresholds:
        iteration_results[thr] = evaluate_threshold_cached(sample, threshold=thr, iou_threshold=iou_threshold)
    return iteration_results

def bootstrap_evaluation(cached_data, thresholds, n_iterations=100, iou_threshold=0.5, parallel=True):
    """
    Performs bootstrapping over cached predictions.
    If parallel=True, iterations are run concurrently using ProcessPoolExecutor.
    Returns a dictionary mapping each threshold to a list of metric tuples from each iteration.
    """
    bootstrap_results = {thr: [] for thr in thresholds}
    
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(bootstrap_iteration, i, cached_data, thresholds, iou_threshold)
                for i in range(n_iterations)
            ]
            for future in concurrent.futures.as_completed(futures):
                iteration_result = future.result()
                for thr in thresholds:
                    bootstrap_results[thr].append(iteration_result[thr])
    else:
        for i in range(n_iterations):
            iteration_result = bootstrap_iteration(i, cached_data, thresholds, iou_threshold)
            for thr in thresholds:
                bootstrap_results[thr].append(iteration_result[thr])
                
    return bootstrap_results

def main():
    # Setup logging and device.
    log = Logger(log_dir='./', log_file='bootstrap_evaluation.log')
    if not torch.cuda.is_available():
        log.warning("CUDA is not available. Running on CPU!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Running bootstrap evaluation on device: {device}")
    print(f"Running bootstrap evaluation on device: {device}")

    # Define transforms and load dataset.
    transform = ComposeDouble([
        ResizeWithBBoxes((1024, 1024)),
        ToTensorDouble(),
        NormalizeImageNet()
    ])

    dataset = MammographyLocalizationDataset(
        csv_file="/home/team11/data/train/localization.csv",
        img_dir="/home/data/train/images",
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time.
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Load the trained model and move it to GPU.
    model = MammoLocalizationResNet50(num_classes=7, pretrained=True, logger=log)
    model_path = "/home/team11/dev/MediSense/localization/models/localization_model.pth"
    model.load_model(model_path)
    model.model.to(device)
    model.model.eval()
    log.info("Model loaded and set to evaluation mode on GPU.")
    print("Model loaded and set to evaluation mode on GPU.")

    # --- Cache Predictions ---
    # Run inference once on the entire dataset and store results.
    cached_data = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            # Move ground truth boxes to CPU for evaluation.
            gt_boxes_list = [t['boxes'].cpu() for t in targets]
            outputs = model.model(images)
            for gt_boxes, output in zip(gt_boxes_list, outputs):
                if 'scores' in output:
                    scores = output['scores'].cpu()
                    pred_boxes = output['boxes'].cpu()
                else:
                    scores = torch.tensor([])
                    pred_boxes = output['boxes'].cpu()
                cached_data.append((gt_boxes, pred_boxes, scores))
    print(f"Cached predictions for {len(cached_data)} images.")
    log.info(f"Cached predictions for {len(cached_data)} images.")

    # --- Bootstrap Evaluation ---
    thresholds = [0.1 * i for i in range(1, 10)]
    n_iterations = 100  # You can adjust this as needed.
    bootstrap_results = bootstrap_evaluation(
        cached_data, thresholds, n_iterations=n_iterations, iou_threshold=0.5, parallel=True
    )

    # --- Aggregate Bootstrap Results ---
    avg_metrics = {}
    for thr in thresholds:
        metrics_list = bootstrap_results[thr]
        avg_precision = np.mean([m[0] for m in metrics_list])
        avg_recall = np.mean([m[1] for m in metrics_list])
        avg_f1 = np.mean([m[2] for m in metrics_list])
        avg_iou = np.mean([m[3] for m in metrics_list])
        avg_metrics[thr] = (avg_precision, avg_recall, avg_f1, avg_iou)
        print(f"Threshold {thr:.2f}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}, Mean IoU={avg_iou:.4f}")
        log.info(f"Threshold {thr:.2f}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}, Mean IoU={avg_iou:.4f}")

    # --- Select Best Threshold ---
    # Best threshold is chosen based on mean IoU (with F1 as a tie-breaker).
    best_thr = max(avg_metrics.items(), key=lambda x: (x[1][3], x[1][2]))[0]
    best_metrics = avg_metrics[best_thr]
    print(f"\nSelected best threshold: {best_thr:.2f} with Mean IoU = {best_metrics[3]:.4f} and F1 = {best_metrics[2]:.4f}")
    log.info(f"Selected best threshold: {best_thr:.2f} with Mean IoU = {best_metrics[3]:.4f} and F1 = {best_metrics[2]:.4f}")

if __name__ == "__main__":
    main()