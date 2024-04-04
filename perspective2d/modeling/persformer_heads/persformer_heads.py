import inspect
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...utils.config import configurable
from .gravity_head import build_gravity_decoder
from .latitude_head import build_latitude_decoder


class StandardPersformerHeads(torch.nn.Module):
    @configurable
    def __init__(
        self,
        gravity_head: Optional[nn.Module] = None,
        latitude_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.gravity_on = gravity_head is not None
        self.latitude_on = latitude_head is not None
        if self.gravity_on:
            self.gravity_head = gravity_head
        if self.latitude_on:
            self.latitude_head = latitude_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        if inspect.ismethod(cls._init_gravity_head):
            ret.update(cls._init_gravity_head(cfg, input_shape))
        if inspect.ismethod(cls._init_latitude_head):
            ret.update(cls._init_latitude_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_gravity_head(cls, cfg, input_shape):
        if not cfg.MODEL.GRAVITY_ON:
            return {}
        ret = {}
        ret["gravity_head"] = build_gravity_decoder(cfg, input_shape)
        return ret

    @classmethod
    def _init_latitude_head(cls, cfg, input_shape):
        if not cfg.MODEL.LATITUDE_ON:
            return {}
        ret = {}
        ret["latitude_head"] = build_latitude_decoder(cfg, input_shape)
        return ret

    def forward(
        self,
        features,
        targets=None,
    ):
        losses = {}
        prediction = {}
        if self.gravity_on:
            prediction["pred_gravity"], loss_gravity = self.gravity_head(
                features, targets["gt_gravity"]
            )
            losses.update(loss_gravity)
        if self.latitude_on:
            prediction["pred_latitude"], loss_latitude = self.latitude_head(
                features, targets["gt_latitude"]
            )
            losses.update(loss_latitude)
        return losses, prediction

    def inference(self, features):
        results = {}
        if self.gravity_on:
            x = self.gravity_head.inference(features)
            results["pred_gravity"] = x
        if self.latitude_on:
            x = self.latitude_head.inference(features)
            results["pred_latitude"] = x
        return results

    def postprocess(self, results, batched_inputs, images):
        processed_results = []
        if self.gravity_on:
            processed_gravity = self.gravity_head.postprocess(
                results["pred_gravity"], batched_inputs, images
            )
        else:
            processed_gravity = [{} for _ in batched_inputs]

        if self.latitude_on:
            processed_latitude = self.latitude_head.postprocess(
                results["pred_latitude"], batched_inputs, images
            )
        else:
            processed_latitude = [{} for _ in batched_inputs]

        for p_g, p_l in zip(processed_gravity, processed_latitude):
            processed_results.append({**p_g, **p_l})
        return processed_results

    def visualize(self, img, feature, target):
        with torch.no_grad():
            results = self.inference(feature)
        vis_dict = {}
        if self.gravity_on:
            # Score maps
            vis_dict[f"gravity-score-map"] = self.visualize_scoremap(
                results["pred_gravity"]
            )

            gt = target["gt_gravity"]
            pred = results["pred_gravity"][0]
            vis_dict.update(self.gravity_head.visualize(img, pred, gt))
        if self.latitude_on:
            gt = target["gt_latitude"]
            pred = results["pred_latitude"][0]
            vis_dict.update(self.latitude_head.visualize(img, pred, gt))

        return vis_dict

    @staticmethod
    def visualize_scoremap(pred):
        softmax = torch.softmax(pred, dim=1)
        score_maps = []
        for c in np.arange(0, softmax.size(1), 1):
            score_maps.append(softmax[0, c, :, :].repeat(3, 1, 1).cpu())
        score_maps = torch.cat((score_maps), 1)
        score_maps = F.interpolate(
            score_maps.unsqueeze(0),
            size=(score_maps.size(1) // 4, score_maps.size(2) // 4),
            mode="bilinear",
            align_corners=False,
        )[0]
        return score_maps


def build_persformer_heads(cfg, input_shape):
    persformer_name = cfg.MODEL.PERSFORMER_HEADS.NAME
    if persformer_name == "StandardPersformerHeads":
        return StandardPersformerHeads(cfg, input_shape)
    # Add more conditions here for other decoders
    else:
        raise ValueError(f"Unknown arch name: {persformer_name}")
