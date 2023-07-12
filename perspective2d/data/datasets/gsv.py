import csv
import json
import logging
import os

import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

__all__ = ["load_gsv_json"]


def load_gsv_json(json_file, img_root):
    if ".csv" in json_file:
        summary = {"data": []}
        with open(json_file) as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                dp = {}
                dp["file_name"] = os.path.join(img_root, row[0])
                dp["pitch"] = np.float32(row[3])
                dp["roll"] = np.float32(row[4])
                dp["width"] = 640
                dp["height"] = 640
                focal_length = np.float32(row[5])
                dp["vfov"] = np.degrees(2 * np.arctan(dp["height"] / 2 / focal_length))
                dp["dataset"] = "gsv"
                # list_hvps not used
                # dp["list_hvps"] = [
                #     [np.float32(row[6]), np.float32(row[7]), 1.0],
                #     [np.float32(row[8]), np.float32(row[9]), 1.0],
                # ]
                summary["data"].append(dp)
        logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
        return summary["data"]
    elif ".json" in json_file:
        with open(json_file) as f:
            summary = json.load(f)
        for idx in range(len(summary["data"])):
            summary["data"][idx]["file_name"] = os.path.join(
                img_root, summary["data"][idx]["file_name"]
            )
            summary["data"][idx]["dataset"] = "gsv_crop"
            summary["data"][idx]["mask_on"] = False
            if "latitude_file_name" in summary["data"][idx].keys():
                summary["data"][idx]["latitude_file_name"] = os.path.join(
                    img_root, summary["data"][idx]["latitude_file_name"]
                )
            if "gravity_file_name" in summary["data"][idx].keys():
                summary["data"][idx]["gravity_file_name"] = os.path.join(
                    img_root, summary["data"][idx]["gravity_file_name"]
                )
        logger.info(f"{os.path.basename(json_file)}: {len(summary['data'])}")
        return summary["data"]
