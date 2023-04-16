import logging
import os
import json
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["load_objectron_json"]


def load_objectron_json(json_file, img_root, mask_on, dataset_type):
    with open(json_file, 'r') as f:
        summary = json.load(f)
    for idx in range(len(summary['data'])):
        if mask_on: 
            summary['data'][idx]['file_name'] = os.path.join(img_root, summary['data'][idx]['file_name'].replace('.png', '_masked.png'))
        else:
            summary['data'][idx]['file_name'] = os.path.join(img_root, summary['data'][idx]['file_name'])
        summary['data'][idx]['gravity_file_name'] = os.path.join(img_root, summary['data'][idx]['gravity_file_name'])
        summary['data'][idx]['latitude_file_name'] = os.path.join(img_root, summary['data'][idx]['latitude_file_name'])

        summary['data'][idx]['dataset'] = dataset_type
        summary['data'][idx]['mask_on'] = mask_on
    logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
    return summary['data']