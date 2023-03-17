import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.utils.events import get_event_storage

from ..persformer_heads import build_persformer_heads
from ..backbone import build_mit_backbone, LowLevelEncoder
from ..param_network import build_param_net

__all__ = ["PersFormer"]



@META_ARCH_REGISTRY.register()
class PersFormer(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        ll_enc: nn.Module,
        persformer_heads: nn.Module,
        param_net: nn.Module, 
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        vis_period: int,
        freeze, 
        debug_on,
        cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.ll_enc = ll_enc
        self.persformer_heads = persformer_heads
        self.param_net = param_net
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.vis_period = vis_period
        self.freeze = freeze
        self.debug_on = debug_on
        self.cfg = cfg
        for layers in self.freeze:
            layer = layers.split(".")
            final = self
            for l in layer:
                final = getattr(final, l)
            for params in final.parameters():
                params.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN:
            backbone = None
            ll_enc = None
            persformer_heads = None
        else:
            backbone = build_backbone(cfg)
            ll_enc = LowLevelEncoder()
            persformer_heads = build_persformer_heads(cfg, backbone.output_shape())
        param_net = build_param_net(cfg) if cfg.MODEL.RECOVER_RPF or cfg.MODEL.RECOVER_PP else None
        return {
            "backbone": backbone,
            "ll_enc": ll_enc,
            "persformer_heads": persformer_heads,
            "param_net": param_net,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
            "freeze": cfg.MODEL.FREEZE,
            "debug_on": cfg.DEBUG_ON,
            "cfg": cfg, 
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        """
        if not self.cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            hl_features = self.backbone(images.tensor)
            ll_features = self.ll_enc(images.tensor)
            features = {
                'hl': hl_features, # features from backbone
                'll': ll_features, # low level features
            }

            targets_dict = {}
            if "gt_gravity" in batched_inputs[0]:
                targets = [x["gt_gravity"].to(self.device) for x in batched_inputs]
                targets = ImageList.from_tensors(
                    targets, self.backbone.size_divisibility, self.persformer_heads.gravity_head.ignore_value
                ).tensor
                targets_dict['gt_gravity'] = targets

            if "gt_latitude" in batched_inputs[0]:
                targets = [x["gt_latitude"].to(self.device) for x in batched_inputs]
                targets = ImageList.from_tensors(
                    targets, self.backbone.size_divisibility, self.persformer_heads.latitude_head.ignore_value
                ).tensor
                targets_dict['gt_latitude'] = targets
        else:
            targets_dict = {}
            targets = [x["gt_gravity"].to(self.device) for x in batched_inputs]
            targets = torch.cat(targets)
            targets_dict['gt_gravity'] = targets
            targets = [x["gt_latitude"].to(self.device) for x in batched_inputs]
            targets = torch.cat(targets)
            targets_dict['gt_latitude'] = targets

        
        if self.training:
            if not self.cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN:
                losses, predictions = self.persformer_heads(features, targets_dict)
            else:
                losses = {}
                images_g = torch.cat([x['gt_gravity'][None] for x in batched_inputs])
                images_l = torch.cat([x['gt_latitude'][None] for x in batched_inputs])
                predictions = torch.cat((images_g, images_l), dim=1).to(self.device)

            if self.param_net is not None:
                losses_param = self.param_net(predictions, batched_inputs)
                losses.update(losses_param)

            if not self.cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN:
                if self.vis_period > 0:
                    storage = get_event_storage()
                    if storage.iter % self.vis_period == 0:
                        self.visualize(images, features, targets_dict, predictions, batched_inputs, storage)
            return losses
            
        if not self.cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN:
            results = self.persformer_heads.inference(features)
            processed_results = self.persformer_heads.postprocess(results, batched_inputs, images)
        else:
            images_g = torch.cat([x['gt_gravity'][None] for x in batched_inputs])
            images_l = torch.cat([x['gt_latitude'][None] for x in batched_inputs])
            results = torch.cat((images_g, images_l), dim=1).to(self.device)
            processed_results = [{}]

        if self.param_net is not None:
            param = self.param_net(results, batched_inputs)
            processed_results[0].update(param)
        return processed_results


    def visualize(self, images, features, targets_dict, predictions, batched_inputs, storage, img_idx = 0):
        vis_dict_total = {}
        for img_idx in range(min(5, len(images))):
            feature_vis = {}
            for key in features.keys():
                if type(features[key]) is list:
                    feature_vis[key] = [ft[img_idx:img_idx+1] for ft in features[key]]
                else:
                    feature_vis[key] = features[key][img_idx:img_idx+1]

            
            image_vis = ((images.tensor[img_idx]*self.pixel_std+ self.pixel_mean)[[2,1,0],:,:]).cpu()
            target_vis = {}
            for key in targets_dict:
                target_vis[key] = targets_dict[key][img_idx]

            vis_dict = self.persformer_heads.visualize(image_vis, feature_vis, target_vis)
            if self.debug_on:
                predictions_vis = {}
                for key in predictions:
                    predictions_vis[key] = predictions[key][img_idx:img_idx+1]
                vis_dict.update(self.param_net.visualize(predictions_vis, batched_inputs[img_idx:img_idx+1]))
 
            # Horizontal stack
            for key in vis_dict.keys():
                if key in vis_dict_total:
                    vis_dict_total[key] = torch.cat((vis_dict_total[key], vis_dict[key]), 2)
                else:
                    vis_dict_total[key] = vis_dict[key]

        for key in vis_dict_total.keys():
            storage.put_image(key, vis_dict_total[key])


