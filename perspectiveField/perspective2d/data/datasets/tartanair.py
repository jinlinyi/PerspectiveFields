import logging
import os
import json
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np
logger = logging.getLogger(__name__)

__all__ = ["load_tartanair_json"]


def load_tartanair_json(json_file, img_root):
    with open(json_file, 'r') as f:
        summary = json.load(f)
    if len(img_root) == 0:
        img_root = summary['info']['root']
    for idx in range(len(summary['data'])):
        tmp_file_name = os.path.join(img_root, summary['data'][idx]['file_name'])
        if not os.path.exists(tmp_file_name):
            tmp_file_name = os.path.join(img_root, summary['data'][idx]['file_name'].replace('/', '-'))
        assert os.path.exists(tmp_file_name), tmp_file_name + ' not found!'
        summary['data'][idx]['file_name'] = tmp_file_name
        summary['data'][idx]['dataset'] = summary['info']['dataset']
        summary['data'][idx]['mask_on'] = False
        if 'latitude_file_name' in summary['data'][idx].keys():
            summary['data'][idx]['latitude_file_name'] = os.path.join(img_root, summary['data'][idx]['latitude_file_name'])
        if 'gravity_file_name' in summary['data'][idx].keys():
            summary['data'][idx]['gravity_file_name'] = os.path.join(img_root, summary['data'][idx]['gravity_file_name'])
    logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
    return summary['data']