# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.events import get_event_storage
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table, setup_logger
from torch.nn import functional as F

from perspective2d.utils import draw_prediction_distribution


class ParamEvaluator:
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        cfg,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if not self._logger.isEnabledFor(logging.INFO):
            setup_logger(name=__name__)
        self._cpu_device = torch.device("cpu")
        self.predicted_targets = cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS

        self.loss_type = cfg.MODEL.PARAM_DECODER.LOSS_TYPE

    def process(self, input, output):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        ret = {}
        for key in self.predicted_targets:
            ret[key + "_err"] = (
                torch.abs(output["pred_" + key] - input[key])
                .to(self._cpu_device)
                .numpy()
            )
            ret["gt_" + key] = input[key]
            ret["pred_" + key] = output["pred_" + key].to(self._cpu_device).numpy()
        ret["dataset"] = input["file_name"].split("/")[2]
        return ret

    def evaluate(self, predictions):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        figs = {}
        res = {}
        for key in self.predicted_targets:
            res[f"mean_{key}_err"] = np.average([e[key + "_err"] for e in predictions])
            res[f"med_{key}_err"] = np.median([e[key + "_err"] for e in predictions])
            figs[key] = draw_prediction_distribution(
                np.array([e[f"pred_{key}"] for e in predictions]).flatten(),
                np.array([e[f"gt_{key}"] for e in predictions]).flatten(),
            )
        self._logger.info("parameters: \n" + create_small_table(res))
        results = {"parameters": res}
        storage = get_event_storage()
        for key in figs.keys():
            storage.put_image(
                predictions[0]["dataset"] + "/" + key,
                torch.tensor(figs[key].transpose(2, 0, 1) / 255),
            )
        return results
