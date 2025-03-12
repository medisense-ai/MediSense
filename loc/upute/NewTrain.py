import os
import cv2
import pandas as pd
import random
from pathlib import Path
from ultralytics import YOLO
import textwrap  # For dedenting YAML
import numpy as np

# Set a seed for reproducibility
random.seed(42)

# Mapping for lesion classes (0-indexed for YOLO)
category_mapping = {
    "Mass": 0,
    "Suspicious_Calcification": 1,
    "Focal_Asymmetry": 2,
    "Architectural_Distortion": 3,
    "Suspicious_Lymph_Node": 4,
    "Other": 5,
}

def clamp(value, min_val, max_val):
    """Clamp a float value between min_val and max_val."""
    return max(min_val, min(value, max_val))

def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert bounding box from (xmin, ymin, xmax, ymax) to YOLO format:
    (x_center, y_center, width, height), normalized by image w/h.
    """
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    box_width = xmax - xmin
    box_height = ymax - ymin
    return (x_center / img_w,
            y_center / img_h,
            box_width / img_w,
            box_height / img_h)

def letterbox_image(image, desired_size=(640, 640), color=(114, 114, 114)):
    """
    Letterbox-resize image while preserving aspect ratio. The rest is filled with `color`.
    Returns:
      letterbox_resized (np.ndarray), scale factor `r`, and offsets (dw, dh).
    """
    h, w = image.shape[:2]
    new_w, new_h = desired_size
    r = min(new_w / w, new_h / h)  # scale factor
    
    # New size (no padding yet)
    nw, nh = int(w * r), int(h * r)
    
    # Create blank image
    letterbox_resized = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    
    # Resize the original image
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Offsets for centering
    dw = (new_w - nw) // 2
    dh = (new_h - nh) // 2
    
    # Place it into the letterbox
    letterbox_resized[dh:dh+nh, dw:dw+nw] = resized_image
    
    return letterbox_resized, r, dw, dh

def process_annotations(csv_path, images_base_dir, output_base_dir, target_size=(640, 640)):
    """
    Processes the CSV annotations, grouping by (case_id, image_id) so that all bounding 
    boxes for each image go into the same label file. Splits by case_id (80% train, 10% val, 10% test).
    
    Steps:
      1. Group CSV by (case_id, image_id).
      2. Load and letterbox-resize each image to target_size.
      3. Convert bounding boxes to letterbox coordinates and YOLO format.
      4. Write *all* boxes into a single .txt per image.
      5. Split data by case_id to keep all images from a case in the same set.
    """
    sets = ['train', 'val', 'test']
    out_dirs = {}
    for s in sets:
        out_dirs[s] = {
            "images": os.path.join(output_base_dir, s, "images"),
            "labels": os.path.join(output_base_dir, s, "labels")
        }
        os.makedirs(out_dirs[s]["images"], exist_ok=True)
        os.makedirs(out_dirs[s]["labels"], exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # 1) Assign each case_id to a split
    unique_case_ids = df["case_id"].unique()
    case_id_to_split = {}
    for case_id in unique_case_ids:
        rnd = random.random()
        if rnd < 0.8:
            case_id_to_split[case_id] = 'train'
        elif rnd < 0.9:
            case_id_to_split[case_id] = 'val'
        else:
            case_id_to_split[case_id] = 'test'
    
    # 2) Group by (case_id, image_id)
    grouped = df.groupby(["case_id", "image_id"])
    
    target_w, target_h = target_size
    
    for (case_id, image_id), group in grouped:
        image_path = os.path.join(images_base_dir, str(case_id), f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found, skipping.")
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load {image_path}, skipping.")
            continue
        
        # Determine split for this case_id
        split = case_id_to_split[case_id]
        
        # Letterbox-resize
        img_resized, r, dw, dh = letterbox_image(img, desired_size=(target_w, target_h))
        out_img_path = os.path.join(out_dirs[split]["images"], f"{case_id}_{image_id}.jpg")
        cv2.imwrite(out_img_path, img_resized)
        
        orig_h, orig_w = img.shape[:2]
        
        # 3) Write YOLO labels
        label_file_path = os.path.join(out_dirs[split]["labels"], f"{case_id}_{image_id}.txt")
        
        # Open once for all bounding boxes in this group
        with open(label_file_path, "w") as f:
            for _, row in group.iterrows():
                # Skip incomplete bounding boxes
                if pd.isna(row['xmin']) or pd.isna(row['ymin']) or pd.isna(row['xmax']) or pd.isna(row['ymax']):
                    continue
                
                category = row['category']
                if category not in category_mapping:
                    continue
                class_id = category_mapping[category]
                
                # Clamp original coords
                xmin = clamp(float(row['xmin']), 0, orig_w - 1)
                ymin = clamp(float(row['ymin']), 0, orig_h - 1)
                xmax = clamp(float(row['xmax']), 0, orig_w - 1)
                ymax = clamp(float(row['ymax']), 0, orig_h - 1)
                
                # Scale by r
                xmin_resized = xmin * r
                xmax_resized = xmax * r
                ymin_resized = ymin * r
                ymax_resized = ymax * r
                
                # Add offsets
                xmin_lb = xmin_resized + dw
                xmax_lb = xmax_resized + dw
                ymin_lb = ymin_resized + dh
                ymax_lb = ymax_resized + dh
                
                # Convert to YOLO format
                x_center, y_center, w_box, h_box = convert_to_yolo_format(
                    xmin_lb, ymin_lb, xmax_lb, ymax_lb, target_w, target_h
                )
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")

def main():
    csv_path = "/home/team11/data/train/localization.csv"  
    images_base_dir = "/home/team11/data/train/images"
    output_base_dir = "/home/team11/dev/MediSense/loc/upute/processed_dataset"
    
    process_annotations(
        csv_path=csv_path,
        images_base_dir=images_base_dir,
        output_base_dir=output_base_dir,
        target_size=(640, 640)
    )
    
    # Create a dataset YAML file for YOLOv8 training
    dataset_yaml_content = textwrap.dedent(f"""\
    train: {os.path.join(output_base_dir, 'train', 'images')}
    val: {os.path.join(output_base_dir, 'val', 'images')}
    test: {os.path.join(output_base_dir, 'test', 'images')}
    nc: 6
    names: ["Mass", "Suspicious_Calcification", "Focal_Asymmetry", "Architectural_Distortion", "Suspicious_Lymph_Node", "Other"]
    """)
    
    dataset_yaml_path = "dataset.yaml"
    with open(dataset_yaml_path, "w") as f:
        f.write(dataset_yaml_content)
    
    # Load and train YOLOv8
    model = YOLO("yolov8n.pt")
    model.train(
        data=dataset_yaml_path,
        epochs=100,
        imgsz=640,
        batch=4,
        seed=42,
        lr0=0.005
    )

if __name__ == "__main__":
    main()