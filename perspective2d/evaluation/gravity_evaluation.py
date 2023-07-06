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
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table, setup_logger
from torch.nn import functional as F


class GravityEvaluator:
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

        if cfg.MODEL.META_ARCHITECTURE == "PerspectiveNet":
            self._num_classes = cfg.MODEL.FPN_GRAVITY_HEAD.NUM_CLASSES
            self._ignore_label = cfg.MODEL.FPN_GRAVITY_HEAD.IGNORE_VALUE
        elif cfg.MODEL.META_ARCHITECTURE == "PersFormer":
            self.loss_type = cfg.MODEL.GRAVITY_DECODER.LOSS_TYPE
            self._num_classes = cfg.MODEL.GRAVITY_DECODER.NUM_CLASSES
            self._ignore_label = cfg.MODEL.GRAVITY_DECODER.IGNORE_VALUE
        else:
            raise NotImplementedError

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
        if "pred_gravity_original" in output.keys():
            pred = output["pred_gravity_original"].flatten(start_dim=1).unsqueeze(0)
            gt = (
                input["gt_gravity_original"]
                .flatten(start_dim=1)
                .unsqueeze(0)
                .to(pred.device)
            )
            cos = F.cosine_similarity(pred, gt).clamp(-1, 1)
            angle_err = torch.acos(cos) / np.pi * 180
            if "mask_on" in input.keys() and input["mask_on"]:
                mask = mask_util.decode(input["mask"]).astype(bool)
            else:
                mask = np.ones((input["height"], input["width"]), dtype=bool)
            mask = torch.tensor(mask).flatten().unsqueeze(0)
            ret["gravity_angle_err_mean"] = torch.mean(
                angle_err.to(self._cpu_device)[mask]
            ).numpy()
            ret["gravity_angle_err_med"] = torch.median(
                angle_err.to(self._cpu_device)[mask]
            ).numpy()
        if "pred_gravity" in output.keys():
            if self.loss_type == "regression":
                pred = output["pred_gravity"]
                gt = input["gt_gravity"].to(pred.device)
                loss = F.l1_loss(
                    pred.unsqueeze(0),
                    gt.unsqueeze(0),
                    reduction="mean",
                )
            elif self.loss_type == "classification":
                pred = output["pred_gravity"]
                gt = input["gt_gravity"].to(pred.device)
                pred = pred[:, : gt.shape[0], : gt.shape[1]]
                loss = F.cross_entropy(
                    pred.unsqueeze(0),
                    gt.unsqueeze(0),
                    reduction="mean",
                    ignore_index=self._ignore_label,
                )
            else:
                raise NotImplementedError
            ret["gravity_loss"] = loss.to(self._cpu_device).numpy()
        return ret

    def evaluate(self, predictions):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        res = {}
        res["meanAngle"] = np.average(
            [e["gravity_angle_err_mean"] for e in predictions]
        )
        # res["medianAngle"] = np.average([e['gravity_angle_err_med'] for e in predictions])
        res["medianAngle"] = np.median(
            [e["gravity_angle_err_mean"] for e in predictions]
        )
        res["gravity_Loss"] = np.average([e["gravity_loss"] for e in predictions])
        self._logger.info("gravity: \n" + create_small_table(res))
        results = {"gravity": res}
        return results
