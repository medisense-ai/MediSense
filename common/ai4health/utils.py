import os
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import models
import datasets

import pandas as pd
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from skimage.measure import find_contours
from skimage.io import imread
from numpy import ndarray
from collections import Counter

DATA_DIR = '/home/data/train'

def get_image_contour(image: ndarray) -> tuple[int, int, int, int]:
    # Convert tensor to numpy array and then to OpenCV format
    contours = find_contours(image, level=60)
    # If contours are found
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=lambda x: x.shape[0])

        # Get bounding box coordinates
        min_row, min_col = largest_contour.min(axis=0)
        max_row, max_col = largest_contour.max(axis=0)
        return min_col, min_row, max_col, max_row
    else:
        height, width = image.shape[0], image.shape[1]
        return 0, 0, width, height
    

def process_case(case_id: str, data: Dict[str, ndarray]):
    local_heights = []
    local_widths = []
    local_wb_xmins = []
    local_wb_ymins = []
    local_wb_xmaxs = []
    local_wb_ymaxs = []

    local_heights.append(data.shape[0])
    local_widths.append(data.shape[1])

    x, y, w, h = get_image_contour(data)
    local_wb_xmins.append(x)
    local_wb_ymins.append(y)
    local_wb_xmaxs.append(x + w)
    local_wb_ymaxs.append(y + h)
    return local_heights, local_widths, local_wb_xmins, local_wb_ymins, local_wb_xmaxs, local_wb_ymaxs, case_id


def get_images_dim_and_contour(dfl: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    heights = []
    widths = []
    wb_xmins = []
    wb_ymins = []
    wb_xmaxs = []
    wb_ymaxs = []
    i = 0

    image_info = Parallel(n_jobs=4)(
        delayed(process_case)(case_id, data) for case_id, data in dataset)

    for res in image_info:
        local_heights, local_widths, local_wb_xmins, local_wb_ymins, local_wb_xmaxs, local_wb_ymaxs, cid = res
        heights.extend(local_heights)
        widths.extend(local_widths)
        wb_xmins.extend(local_wb_xmins)
        wb_ymins.extend(local_wb_ymins)
        wb_xmaxs.extend(local_wb_xmaxs)
        wb_ymaxs.extend(local_wb_ymaxs)
        i += len(local_heights)
    # Find duplicate values in wb_xmins and their indices
    
    dfl['wb_xmin'] = wb_xmins
    dfl['wb_ymin'] = wb_ymins
    dfl['wb_xmax'] = wb_xmaxs
    dfl['wb_ymax'] = wb_ymaxs
    dfl['height'] = heights
    dfl['width'] = widths
    return dfl

def get_xmin_ymin(row):
    if row['laterality'] == 'L' and row['view'] == 'MLO':
        xmin, ymin = row['wb_xmax'] / 2, 2 * row['wb_ymax'] / 3 
    elif row['laterality'] == 'L' and row['view'] == 'CC':
        xmin, ymin = 2 * row['wb_xmax'] / 3, row['wb_ymax'] / 3
    elif row['laterality'] == 'R' and row['view'] == 'MLO':
        xmin, ymin = row['width'] - row['wb_xmax'] / 2, 2 * row['wb_ymax'] / 3
    elif row['laterality'] == 'R' and row['view'] == 'CC':
        xmin, ymin = 2 * row['wb_xmax'] / 3, row['wb_ymax'] / 3 # width and height 
    return xmin, ymin