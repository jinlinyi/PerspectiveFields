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


class LatitudeEvaluator:
    """
    Evaluate latitude maps.
    """

    def __init__(
        self,
        cfg,
    ):
        self._logger = logging.getLogger(__name__)

        if not self._logger.isEnabledFor(logging.INFO):
            setup_logger(name=__name__)
        self._cpu_device = torch.device("cpu")

        if cfg.MODEL.META_ARCHITECTURE == "PerspectiveNet":
            self.loss_type = "classification"
            self.ignore_value = cfg.MODEL.FPN_LATITUDE_HEAD.IGNORE_VALUE
        elif cfg.MODEL.META_ARCHITECTURE == "PersFormer":
            self.loss_type = cfg.MODEL.LATITUDE_DECODER.LOSS_TYPE
            self.ignore_value = cfg.MODEL.LATITUDE_DECODER.IGNORE_VALUE
        else:
            raise NotImplementedError

    def process(self, input, output):
        ret = {}
        if "pred_latitude_original" in output.keys():
            pred = output["pred_latitude"]
            gt = input["gt_latitude"].to(pred.device)
            # pred = pred[:, :gt.shape[0], :gt.shape[1]]
            if self.loss_type == "regression":
                loss = F.l1_loss(
                    pred,
                    gt,
                    reduction="mean",
                )
            elif self.loss_type == "classification":
                loss = F.cross_entropy(
                    pred.unsqueeze(0),
                    gt.unsqueeze(0),
                    reduction="mean",
                    ignore_index=self.ignore_value,
                )
            else:
                raise NotImplementedError
            if input["gt_latitude_original_mode"] == "rad":
                gt_lati_ori = torch.rad2deg(input["gt_latitude_original"]).to(
                    output["pred_latitude_original"].device
                )
            else:
                gt_lati_ori = input["gt_latitude_original"].to(
                    output["pred_latitude_original"].device
                )
            if output["pred_latitude_original_mode"] == "rad":
                pred_lati_ori = torch.rad2deg(output["pred_latitude_original"])
            else:
                pred_lati_ori = output["pred_latitude_original"]
            if "mask_on" in input.keys() and input["mask_on"]:
                mask = mask_util.decode(input["mask"]).astype(bool)
            else:
                mask = np.ones((input["height"], input["width"]), dtype=bool)
            mask = torch.tensor(mask)
            ret["latitude_err_mean"] = torch.mean(
                torch.abs(pred_lati_ori - gt_lati_ori).to(self._cpu_device)[mask]
            ).numpy()
            # ret['latitude_err_median'] = torch.median(torch.abs(pred_lati_ori - gt_lati_ori).to(self._cpu_device)[mask]).numpy()

            ret["latitude_loss"] = loss.to(self._cpu_device).numpy()
        return ret

    def evaluate(self, predictions):
        res = {}
        res["latitude_Loss"] = np.average([e["latitude_loss"] for e in predictions])
        res["latitude_err_mean"] = np.average(
            [e["latitude_err_mean"] for e in predictions]
        )
        res["latitude_err_median"] = np.median(
            [e["latitude_err_mean"] for e in predictions]
        )
        self._logger.info("latitude: \n" + create_small_table(res))
        results = {"latitude": res}
        return results
