import os
import cv2
import argparse

import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from imcp import plot_mcp_curve, mcp_score 
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Iterable, Callable
from sklearn.metrics import (matthews_corrcoef, 
                            confusion_matrix, 
                            precision_recall_curve, 
                            auc,
                            roc_auc_score
                            )
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def create_localization_csv(localization_dir):
    """
    Process localization_results directory and create localization.csv
    Returns DataFrame with columns: case_id, image_id, and one-hot encoded predictions
    """
    # List of all possible finding types (based on localization classes)
    finding_types = [
        'Architectural_Distortion',
        'Focal_Asymmetry',
        'Mass',
        'Other',
        'Suspicious_Calcification',
        'Suspicious_Lymph_Node'
    ]
    
    results = []
    localization_path = Path(localization_dir)
    
    # Walk through the directory structure
    for case_dir in localization_path.iterdir():
        if case_dir.is_dir():
            case_id = case_dir.name
            # Process each image directory within the case directory
            for image_dir in case_dir.iterdir():
                if image_dir.is_dir():
                    image_id = image_dir.name
                    
                    # Initialize prediction vector
                    predictions = {finding: 0 for finding in finding_types}
                    
                    # Check each PNG file
                    for finding in finding_types:
                        png_path = image_dir / f"{finding}.png"
                        if png_path.exists():
                            mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
                            predictions[finding] = 1 if np.any(mask) else 0
                        else:
                            predictions[finding] = 0
                    # TODO: IOU calculation
                    # Add case_id and image_id to predictions
                    predictions['case_id'] = case_id
                    predictions['image_id'] = image_id
                    results.append(predictions)
    
    # Create DataFrame and organize columns
    df = pd.DataFrame(results)
    # Reorder columns to put case_id and image_id first
    cols = ['case_id', 'image_id'] + finding_types
    df = df[cols]
    
    return df

def get_base_iou(gt: np.ndarray, pred: np.ndarray, eps=1e-8):
    intersection = (gt * pred).sum()
    union = (gt + pred).astype(bool).sum()
    return (intersection + eps) / (union + eps)

def load_and_prepare_data(args):
    """Load and prepare both classification and localization data."""
    input_dir = Path(args.input_results)
    output_dir = Path(args.output_metrics)
    os.makedirs(output_dir, exist_ok=True)
    # Load gold labels
    classification_labels_df = pd.read_csv(args.gold_labels + '/classification.csv')
    classification_labels_df = classification_labels_df[['case_id', 'category']]
    localization_labels_df = create_localization_csv(args.localization_gt_path)
    localization_labels_df['case_id'] = localization_labels_df['case_id'].astype(int)
    localization_labels_df.sort_values(by=['case_id', 'image_id'], inplace=True)
    # Load classification results
    classification_path = input_dir / 'classification_results.csv'
    classification_df = pd.read_csv(classification_path)
    
    localization_df = create_localization_csv(args.localization_path)
    localization_df['case_id'] = localization_df['case_id'].astype(int)
    localization_df.sort_values(by=['case_id', 'image_id'], inplace=True)
    localization_csv_path = output_dir / 'localization_results.csv'
    localization_df.to_csv(localization_csv_path, index=False)
        
    return classification_df, localization_df, classification_labels_df, localization_labels_df

def multilabel_accuracy(preds, gts):
    """Calculate multilabel accuracy."""
    return np.ndarray((preds == gts).float())

def multilabel_auprc(logits, gts, average='macro'):
    """Calculate multilabel Average Precision."""
    gts_np = gts.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    
    n_classes = logits_np.shape[1]
    auprc_scores = []
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(gts_np[:, i], logits_np[:, i])
        auprc_scores.append(auc(recall, precision))
    
    if average == 'macro':
        return np.mean(auprc_scores)
    return auprc_scores


def multiclass_accuracy(preds, gts, num_classes, average=None):
    """Calculate multiclass accuracy."""
    preds_np = np.array(preds)
    gts_np = np.array(gts)
    
    # Per-class accuracy
    accuracies = []
    for i in range(num_classes):
        class_mask = (gts_np == i)
        accuracies.append(np.mean(preds_np[class_mask] == gts_np[class_mask]))
    
    if average == 'macro':
        return np.mean(accuracies)
    return accuracies

def multiclass_auprc(gts, logits, num_classes, average=None):
    """Calculate multiclass AUPRC."""
    gts_np = np.array(gts)
    logits_np = np.array(logits)
    
    # One-hot encode ground truth
    gts_onehot = label_binarize(gts_np, classes=range(num_classes))
    
    # Calculate AUPRC for each class
    auprc_scores = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(gts_onehot[:, i], logits_np[:, i])
        auprc_scores.append(auc(recall, precision))
    
    if average == 'macro':
        return np.mean(auprc_scores)
    return auprc_scores

def multiclass_auroc(gts, logits, num_classes, average=None):
    """Calculate multiclass AUROC."""
    gts_np = np.array(gts)
    logits_np = np.array(logits)
    
    # One-hot encode ground truth
    gts_onehot = label_binarize(gts_np, classes=range(num_classes))
    
    # Calculate AUROC for each class
    auroc_scores = []
    for i in range(num_classes):
        auroc_scores.append(roc_auc_score(gts_onehot[:, i], logits_np[:, i]))
    
    if average == 'macro':
        return np.mean(auroc_scores)
    return auroc_scores

def multiclass_confusion_matrix(preds, gts, num_classes):
    """Calculate confusion matrix."""
    preds_np = np.array(preds)
    gts_np = np.array(gts)
    
    return confusion_matrix(gts_np, preds_np, labels=range(num_classes))

def process_metrics(args,
                    classification_df, 
                    localization_df, 
                    classification_labels, 
                    localization_labels,):
    """
    Process and calculate metrics for both classification and localization.
    
    Args:
    - classification_df: DataFrame with classification predictions
    - localization_df: DataFrame with localization predictions
    - classification_labels: DataFrame with classification ground truth labels
    - localization_labels: DataFrame with localization ground truth labels (generated from png files)
    
    Returns:
    - Dict with classification and localization metrics
    """
    # Prepare classification metrics
    cls_metrics = {}
    
    # Classification inputs (assuming these are available or can be converted)
    cls_preds = classification_df.preds.values
    cls_proba = classification_df[['probabilityC0', 'probabilityC1', 
                                'probabilityC2', 'probabilityC3']].values
    
    cls_proba = cls_proba / np.concat([np.expand_dims(cls_proba.sum(axis=1), axis=0)]*4, axis=0).T
    cls_gts = classification_labels.category.values
    num_classes = 4  # Adjust as needed
    if args.verbose:
        print("Confusion Matrix:")
        print(multiclass_confusion_matrix(cls_preds, cls_gts, num_classes))
        plot_mcp_curve(cls_gts, cls_proba)
    binary_cls_pred = label_binarize(cls_preds, classes=range(num_classes))
    binary_cls_gts = label_binarize(cls_gts, classes=range(num_classes))
    # Classification metrics
    cls_metrics['cls_macro_f1'] = f1_score(cls_gts, cls_preds, average='macro')
    cls_metrics['cls_macro_auprc'] = multiclass_auprc(cls_gts, cls_proba, num_classes=num_classes, average='macro')
    cls_metrics['cls_macro_auroc'] = roc_auc_score(cls_gts, cls_proba, multi_class='ovr', average='macro')
    cls_metrics['cls_mcp'] = mcp_score(cls_gts, cls_proba)
    # Per-class metrics
    f1_scores, precision_scores, recall_scores, auroc_scores= [], [], [], []
    for i in range(num_classes):
        f1_scores.append(f1_score(binary_cls_gts[:,i], binary_cls_pred[:,i]))
        precision_scores.append(precision_score(binary_cls_gts[:,i], binary_cls_pred[:,i]))
        recall_scores.append(recall_score(binary_cls_gts[:,i], binary_cls_pred[:,i]))
    
    auroc_scores = multiclass_auroc(cls_gts, cls_proba, num_classes=num_classes, average=None)
    auprc_scores = multiclass_auprc(cls_gts, cls_proba, num_classes=num_classes, average=None)

    # Add per-class metrics
    for i in range(num_classes):
        cls_metrics[f'cls_{i}_f1'] = f1_scores[i]
        cls_metrics[f'cls_{i}_precision'] = precision_scores[i]
        cls_metrics[f'cls_{i}_recall'] = recall_scores[i]
        cls_metrics[f'cls_{i}_auprc'] = auprc_scores[i]
        cls_metrics[f'cls_{i}_auroc'] = auroc_scores[i]
    
    # Matthews Correlation Coefficient
    mcc_scores = []
    for class_idx in range(num_classes):
        binary_true = (cls_gts == class_idx).astype(int)
        binary_pred = (cls_preds == class_idx).astype(int)
        mcc = matthews_corrcoef(binary_true, binary_pred)
        cls_metrics[f'cls_{class_idx}_mcc'] = mcc
        mcc_scores.append(mcc)
    
    cls_metrics['cls_macro_mcc'] = np.mean(mcc_scores)
    
    # Localization metrics
    loc_metrics = {}
    global LABELS_LOCALIZATION
    # Localization inputs
    loc_preds = localization_df[LABELS_LOCALIZATION].values
    loc_gts = localization_labels[LABELS_LOCALIZATION].values
    # Calculate MCC for each label
    mcc_scores = []
    for i, label in enumerate(LABELS_LOCALIZATION):
        # Calculate MCC
        mcc = matthews_corrcoef(loc_gts[:, i], loc_preds[:, i])
        loc_metrics[f'loc_{label}_mcc'] = mcc
        mcc_scores.append(mcc)
    
        # Calculate Precision, Recall, and F1 score
        precision = precision_score(loc_gts[:, i], loc_preds[:, i])
        recall = recall_score(loc_gts[:, i], loc_preds[:, i])
        f1 = f1_score(loc_gts[:, i], loc_preds[:, i])
        loc_metrics[f'loc_{label}_precision'] = precision
        loc_metrics[f'loc_{label}_recall'] = recall
        loc_metrics[f'loc_{label}_f1'] = f1
        if args.verbose:
            print("Confusion Matrix {}:".format(label))
            print(multiclass_confusion_matrix(loc_preds[:,i], loc_gts[:,i], num_classes=2))
    # Calculate IoU
    loc_metrics['iou_unified'], mean_per_label, mean_per_label_gt = iou(args, 
                                                                        localization_df, 
                                                                        localization_labels)
    # Calculate macro MCC
    for i, label in enumerate(LABELS_LOCALIZATION):
        # loc_metrics[f'loc_{label}_iou'] = mean_per_label[i]
        loc_metrics[f'loc_{label}_iou_gt'] = mean_per_label_gt[i]
    # loc_metrics['iou_macro'] = np.mean(mean_per_label)
    loc_metrics['iou_macro_gt'] = np.mean(mean_per_label_gt)
    loc_metrics['loc_macro_mcc'] = np.mean(mcc_scores)
    
    return {**cls_metrics, **loc_metrics}

def get_mask(png_path):
    if os.path.exists(png_path):
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(bool)
        if mask.any():
            return mask
    return None

LABELS_LOCALIZATION = [
        'Architectural_Distortion',
        'Focal_Asymmetry',
        'Mass',
        'Other',
        'Suspicious_Calcification',
        'Suspicious_Lymph_Node'
    ]


def get_iou_image(gt: List[Union[np.ndarray, None]], pred: List[Union[np.ndarray, None]], 
            unify=False, remove_no_findings=False, eps=1e-8):
    # pred = (pred).astype(bool)
    # gt = (gt).astype(bool)
    if unify:
        # pred = [p for p, g in zip(pred, gt) if g is not None]
        pred = [p for p in pred if p is not None]
        gt = [g for g in gt if g is not None]
        pred = np.concatenate(np.expand_dims(pred, axis=0), axis=0).astype(bool)
        gt = np.concatenate(np.expand_dims(gt, axis=0), axis=0).astype(bool)
        pred = pred.sum(axis=0).astype(bool)
        gt = gt.sum(axis=0).astype(bool)
        return get_base_iou(gt, pred, eps=eps)
    
    iou = []
    if remove_no_findings:
        # findings = {}
        for i in range(len(gt)):
            if gt[i] is not None:
                if pred[i] is not None:
                    iou.append(get_base_iou(gt[i], pred[i], eps=eps))
                else:
                    iou.append(0)
            else:
                iou.append(None)
                pass
    else:
        for i in range(len(gt)):
            if gt[i] is not None and pred[i] is not None:
                iou.append(get_base_iou(gt[i], pred[i], eps=eps))
            elif gt[i] is None and pred[i] is None:
                iou.append(1)
            else:
                iou.append(0)
                
    return iou


def iou(args, localization_df: pd.DataFrame, localization_labels: pd.DataFrame,  
            eps=1e-8) -> Tuple[float, List[float], List[float]]:
    global LABELS_LOCALIZATION
    results = []
    # Walk through the directory structure
    localization_df['relative_path'] = localization_df['case_id'].astype(str) + \
        '/' + localization_df['image_id']
    localization_labels['relative_path'] = localization_labels['case_id'].astype(str) + \
        '/' + localization_labels['image_id']
    iou_per_image_micro = []
    iou_per_image_macro = []
    iou_per_image_macro_gt = []
    for img_path in tqdm(localization_df['relative_path']):
        if os.path.isdir(args.localization_path / img_path) and os.path.isdir(args.localization_gt_path / img_path):
            predictions = []
            labels = []
            for finding in LABELS_LOCALIZATION:
                png_label_path = args.localization_gt_path / img_path / f"{finding}.png"
                png_pred_path = args.localization_path / img_path / f"{finding}.png"
                labels.append(get_mask(png_label_path))
                predictions.append(get_mask(png_pred_path))
            if all(mask is None for mask in predictions) or all(mask is None for mask in labels):
                iou_per_image_macro.append([1 for _ in range(len(LABELS_LOCALIZATION))])
                # iou_per_image_macro_gt
                # continue # this completely skips TP of No_Finding label
            else:
                try:
                    # list of single iou
                    iou_per_image_micro.append(get_iou_image(
                        labels, predictions, unify=True)) 
                    # list of iou per finding
                    iou_per_image_macro.append(get_iou_image(labels, predictions))
                    # list of iou per finding
                    iou_per_image_macro_gt.append(get_iou_image(
                        labels, predictions, remove_no_findings=True))
                except Exception as e:
                        # print(f"Error in image {img_path}")
                        print(e)
        else:
            print(f"Image {img_path} not found")
            pass
    # Calculate IoU
    print("Calculating IoU...")
    iou_per_image_macro = np.concatenate(np.expand_dims(np.array(iou_per_image_macro), axis=0), axis=0)
    iou_per_image_macro = np.where(iou_per_image_macro == None, np.nan, iou_per_image_macro)
    iou_per_image_macro_gt = np.concatenate(np.expand_dims(np.array(iou_per_image_macro_gt), axis=0), axis=0)
    iou_per_image_macro_gt = np.where(iou_per_image_macro_gt == None, np.nan, iou_per_image_macro_gt)
    iou_per_class_mean = [np.nan for _ in range(len(LABELS_LOCALIZATION))]
    iou_per_class_mean_gt = [np.nan for _ in range(len(LABELS_LOCALIZATION))]
    for i in range(len(LABELS_LOCALIZATION)):
        if len(iou_per_image_macro.shape) < 2:
            pass
        elif all(mask is None for mask in iou_per_image_macro[:,i]):
            pass
        else:
            iou_per_class_mean[i] = np.nanmean(iou_per_image_macro[:,i])
            
        if len(iou_per_image_macro_gt.shape) < 2:
            pass
        elif all(mask is None for mask in iou_per_image_macro_gt[:,i]):
            pass
        else:
            iou_per_class_mean_gt[i] = np.nanmean(iou_per_image_macro_gt[:,i])

    return sum(iou_per_image_micro) / (len(iou_per_image_micro) + eps), \
        iou_per_class_mean, iou_per_class_mean_gt

def main():
    # Parse arguments
    args = parse_arguments()
    os.makedirs(args.output_metrics, exist_ok=True)
    # Create and save localization results
    output_dir = Path(args.output_metrics)
    input_dir = Path(args.input_results)
    if os.path.isdir(input_dir / 'localization_results'):
        localization_dir = input_dir / 'localization_results'
    elif os.path.isdir(input_dir / 'localization_results_png'):
        localization_dir = input_dir / 'localization_results_png'
    args.localization_path = localization_dir
    args.localization_gt_path = Path(args.gold_labels) / 'localization'
    # Load and prepare data
    print("Loading and preparing data...")
    classification_df, localization_df, classification_labels, localization_labels = load_and_prepare_data(args)
    # Calculate metrics
    print("Calculating metrics... ")
    metrics = process_metrics(args, classification_df, localization_df, 
                            classification_labels, localization_labels)
    evaluation_metrics = ['cls_macro_mcc', 'loc_macro_mcc', 'loc_macro_iou', 'iou_unified']

    metrics_df = pd.DataFrame([metrics])
    evaluation_metrics = ['cls_macro_mcc', 'loc_macro_mcc', 'iou_unified']
    cols = evaluation_metrics + [col for col in metrics_df.columns if col not in evaluation_metrics]
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(args.output_metrics + '/scores.csv', index=False)

    print("Calculated Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process ML results and calculate metrics')
    parser.add_argument('--input_results', required=True, 
        help='Path classification.csv and directory localization_results/')
    parser.add_argument('--output_metrics', required=True,
            help='Path to save calculated metrics')
    parser.add_argument('--gold_labels', required=True,
            help='Path to ground truth labels')
    parser.add_argument('--test', action='store_true',
                        help='Test the code on a single batch.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print more information')
    return parser.parse_args()

if __name__ == "__main__":
    main()
