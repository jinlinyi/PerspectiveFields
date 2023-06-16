# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import json
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np
"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_cities360_json"]


def load_cities360_json(json_file, img_root):
    """
    Load a json file with mp3d's instances annotation format.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See DATASETS.md)
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    with open(json_file, 'r') as f:
        summary = json.load(f)
    for idx in range(len(summary['data'])):
        summary['data'][idx]['file_name'] = os.path.join(img_root, summary['data'][idx]['file_name'])
        summary['data'][idx]['dataset'] = 'cities360'
    logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
    return summary['data']
    
def load_cities360_distort_json(json_file, img_root):
    with open(json_file, 'r') as f:
        summary = json.load(f)
    for idx in range(len(summary['data'])):
        summary['data'][idx]['file_name'] = os.path.join(img_root, summary['data'][idx]['file_name'])
        summary['data'][idx]['dataset'] = 'cities360_distort'
        summary['data'][idx]['latitude_file_name'] = os.path.join(img_root, summary['data'][idx]['latitude_file_name'])
        summary['data'][idx]['gravity_file_name'] = os.path.join(img_root, summary['data'][idx]['gravity_file_name'])
        
    logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
    return summary['data']