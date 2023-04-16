import logging
import os
import json
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np
import csv

logger = logging.getLogger(__name__)

__all__ = ["load_gsv_json"]


def load_gsv_json(json_file, img_root):
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
    if ".csv" in json_file:
        summary = {'data':[]}
        with open(json_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                dp = {}
                dp['file_name'] = os.path.join(img_root, row[0])
                dp['pitch'] = np.float32(row[3])
                dp['roll'] = np.float32(row[4])
                dp['width'] = 640
                dp['height'] = 640
                focal_length = np.float32(row[5])
                dp['vfov'] = np.degrees(2 * np.arctan(dp['height'] / 2 / focal_length))
                dp['dataset'] = 'gsv'
                dp['list_hvps'] = [[np.float32(row[6]),np.float32(row[7]), 1.0],
                        [np.float32(row[8]),np.float32(row[9]), 1.0]]
                summary['data'].append(dp)
        logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
        return summary['data']
    elif ".json" in json_file:
        with open(json_file, 'r') as f:
            summary = json.load(f)
        for idx in range(len(summary['data'])):
            summary['data'][idx]['file_name'] = os.path.join(img_root, summary['data'][idx]['file_name'])
            summary['data'][idx]['dataset'] = 'gsv_crop'
            summary['data'][idx]['mask_on'] = False
            if 'latitude_file_name' in summary['data'][idx].keys():
                summary['data'][idx]['latitude_file_name'] = os.path.join(img_root, summary['data'][idx]['latitude_file_name'])
            if 'gravity_file_name' in summary['data'][idx].keys():
                summary['data'][idx]['gravity_file_name'] = os.path.join(img_root, summary['data'][idx]['gravity_file_name'])
        logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
        return summary['data']
    # if not 'train' in os.path.basename(json_file):
    #     np.random.seed(2021)
    #     sample_data = np.random.choice(summary['data'], 300, replace=False)
    #     return sample_data
    # else:
    #     return summary['data']