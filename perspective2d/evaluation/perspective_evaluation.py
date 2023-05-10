# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
from torch.nn import functional as F
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.logger import setup_logger, create_small_table

from .gravity_evaluation import GravityEvaluator
from .latitude_evaluation import LatitudeEvaluator
from .param_evaluation import ParamEvaluator

class PerspectiveEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        cfg,
        distributed=True,
        output_dir=None,
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

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self.evaluators = []
        self.gravity_on     = cfg.MODEL.GRAVITY_ON
        self.latitude_on    = cfg.MODEL.LATITUDE_ON
        self.param_on       = cfg.MODEL.RECOVER_RPF | cfg.MODEL.RECOVER_PP
        if self.gravity_on and not 'persformer_heads' in cfg.MODEL.FREEZE and not cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN:
            self.evaluators.append(GravityEvaluator(cfg))
        if self.latitude_on and not 'persformer_heads' in cfg.MODEL.FREEZE and not cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN:
            self.evaluators.append(LatitudeEvaluator(cfg))
        if self.param_on:
            self.evaluators.append(ParamEvaluator(cfg))



    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            ret = {}
            for evaluator in self.evaluators:
                ret.update(evaluator.process(input, output))
            self._predictions.append(ret)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        
        else:
            predictions = self._predictions

        results = {}
        for evaluator in self.evaluators:
            results.update(evaluator.evaluate(predictions))
        return results