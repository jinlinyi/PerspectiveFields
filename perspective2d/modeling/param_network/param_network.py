import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...utils import general_vfov_to_focal
from ..backbone import ConvNeXt


def build_param_net(cfg):
    name = cfg.MODEL.PARAM_DECODER.NAME
    if name == "ParamNet":
        return ParamNet(cfg)
    elif name == "ParamNetConvNextRegress":
        return ParamNetConvNextRegress(cfg)
    # Add more conditions here for other decoders
    else:
        raise ValueError(f"Unknown paramnet name: {name}")


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.detach().numpy()
    else:
        return np.array(x)


class ParamNet(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.MODEL.PARAM_DECODER.LOSS_TYPE == "regression":
            num_classes = 5
        self.backbone = ConvNeXt(num_classes=num_classes)
        self.loss_weight = cfg.MODEL.PARAM_DECODER.LOSS_WEIGHT

    def forward(self, predictions, batched_inputs=None):
        images = torch.cat(
            (predictions["pred_gravity"], predictions["pred_latitude"]), dim=1
        )

        x = self.backbone(images)
        # x[:,:2] = torch.clip(x[:,:2], -1, 1)
        if not self.training:
            if self.cfg.MODEL.RECOVER_PP:
                param = {
                    "pred_roll": x[:, 0] * 90.0,
                    "pred_pitch": x[:, 1] * 90.0,
                    "pred_rel_focal": x[:, 2],
                    "pred_rel_pp": x[:, 3:],
                }
            else:
                param = {
                    "pred_roll": x[:, 0] * 90.0,
                    "pred_pitch": x[:, 1] * 90.0,
                    "pred_vfov": x[:, 2] * 90.0,
                    "pred_rel_focal": 1 / 2 / torch.tan(x[:, 2]),
                }

            return param

        targets_dict = {}
        if self.cfg.MODEL.RECOVER_RPF:
            if self.cfg.MODEL.RECOVER_PP:
                # roll [-90, 90]->(-1,1), pitch [-90, 90]->(-1,1), focal/img_h
                targets = torch.FloatTensor(
                    [
                        np.array([x["roll"] / 90.0, x["pitch"] / 90.0, x["rel_focal"]])
                        for x in batched_inputs
                    ]
                )
            else:
                # roll [-90, 90]->(-1,1), pitch [-90, 90]->(-1,1), vfov -> vfov / 90
                targets = torch.FloatTensor(
                    [
                        np.array(
                            [x["roll"] / 90.0, x["pitch"] / 90.0, x["vfov"] / 90.0]
                        )
                        for x in batched_inputs
                    ]
                )
            targets_dict["rpf"] = targets.to(images.device)
        else:
            targets_dict["rpf"] = torch.zeros((len(x), 3)).to(images.device)
        if self.cfg.MODEL.RECOVER_PP:
            targets = torch.FloatTensor([x["rel_pp"] for x in batched_inputs])
            targets_dict["rel_pp"] = targets.to(images.device)
        else:
            targets_dict["rel_pp"] = torch.zeros((len(x), 2)).to(images.device)
        losses = self.losses(x, targets_dict)
        return losses

    def losses(self, pred, gt):
        losses = {}
        if self.cfg.MODEL.PARAM_DECODER.LOSS_TYPE == "regression":
            mask = torch.ones_like(pred).to(pred.device)
            if not self.cfg.MODEL.RECOVER_PP:
                mask[:, 3:] = 0.0
            if not self.cfg.MODEL.RECOVER_RPF:
                mask[:, :3] = 0.0
            gt = torch.cat([gt["rpf"], gt["rel_pp"]], dim=1)
            if self.cfg.MODEL.RECOVER_PP:
                loss_itemized = (
                    F.mse_loss(pred, gt, reduction="none") * mask * self.loss_weight
                )
                losses["param/roll-loss"] = loss_itemized[:, 0].mean()
                losses["param/pitch-loss"] = loss_itemized[:, 1].mean()
                losses["param/focal-loss"] = loss_itemized[:, 2].mean()
                losses["param/cx-loss"] = loss_itemized[:, 3].mean()
                losses["param/cy-loss"] = loss_itemized[:, 4].mean()
                # losses['param-l2-loss'] = (F.mse_loss(pred, gt, reduction='none') * mask).mean()
            else:
                losses["param-l1-loss"] = (
                    F.l1_loss(pred, gt, reduction="none") * mask
                ).mean() * self.loss_weight

        else:
            raise NotImplementedError
        return losses

    def visualize(self, predictions, batched_inputs):
        with torch.no_grad():
            images = torch.cat(
                (predictions["pred_gravity"], predictions["pred_latitude"]), dim=1
            )

            x = self.backbone(images)
            assert self.cfg.MODEL.RECOVER_PP
            param = {
                "pred_roll": x[:, 0] * 90.0,
                "pred_pitch": x[:, 1] * 90.0,
                "pred_rel_focal": x[:, 2],
                "pred_rel_pp": x[:, 3:],
            }
        vis_dict = {}
        _, h, w = batched_inputs[0]["image"].shape
        pp_gt = batched_inputs[0]["rel_pp"] * h + np.array([w, h]) / 2
        vis_img = cv2.circle(
            batched_inputs[0]["image"].cpu().numpy().transpose(1, 2, 0).copy(),
            pp_gt.astype(int),
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )
        pp_pred = param["pred_rel_pp"].cpu().numpy() * h + np.array([w, h]) / 2
        vis_img = cv2.circle(
            vis_img, pp_pred[0].astype(int), radius=8, color=(0, 255, 0), thickness=-1
        )

        original = cv2.resize(
            batched_inputs[0]["img_center_original"], (vis_img.shape[:2])
        )

        original = original[:, :, ::-1] / 255
        original = torch.tensor(original.transpose(2, 0, 1))
        vis_img = vis_img[:, :, ::-1] / 255
        vis_img = torch.tensor(vis_img.transpose(2, 0, 1))
        cat = torch.cat((original, vis_img), 1)
        return {"principal point": cat}


class ParamNetConvNextRegress(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.MODEL.PARAM_DECODER.LOSS_TYPE == "regression":
            num_classes = len(cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS)
        self.backbone = ConvNeXt(num_classes=num_classes)
        self.loss_weight = cfg.MODEL.PARAM_DECODER.LOSS_WEIGHT
        self.input_size = cfg.MODEL.PARAM_DECODER.INPUT_SIZE
        self.factors = {
            "roll": 90.0,
            "pitch": 90.0,
            "vfov": 90.0,
            "rel_focal": 1.0,
            "rel_cx": 1.0,
            "rel_cy": 1.0,
            "general_vfov": 90.0,
        }

    def forward(self, predictions, batched_inputs=None):
        images = torch.cat(
            (predictions["pred_gravity"], predictions["pred_latitude"]), dim=1
        )
        images = F.interpolate(images, (self.input_size, self.input_size))

        x = self.backbone(images)
        # x[:,:2] = torch.clip(x[:,:2], -1, 1)
        if not self.training:
            param = {}
            for idx, key in enumerate(self.cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS):
                param["pred_" + key] = x[:, idx] * self.factors[key]

            # make output contain everything
            if "pred_rel_cx" not in param and "pred_rel_cy" not in param:
                param["pred_rel_cx"] = param["pred_rel_cy"] = torch.FloatTensor([0])
            if "pred_general_vfov" not in param:
                param["pred_general_vfov"] = param["pred_vfov"]
            if "pred_rel_focal" not in param:
                param["pred_rel_focal"] = torch.FloatTensor(
                    general_vfov_to_focal(
                        to_numpy(param["pred_rel_cx"]),
                        to_numpy(param["pred_rel_cy"]),
                        1,
                        to_numpy(param["pred_general_vfov"]),
                        degree=True,
                    )
                )
            return param

        targets = []
        for batched_input in batched_inputs:
            target = []
            for key in self.cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS:
                target.append(batched_input[key] / self.factors[key])
            targets.append(target)
        targets = torch.FloatTensor(targets).to(images.device)
        losses = self.losses(x, targets)
        return losses

    def losses(self, pred, gt):
        losses = {}
        if self.cfg.MODEL.PARAM_DECODER.LOSS_TYPE == "regression":
            loss_itemized = F.mse_loss(pred, gt, reduction="none") * self.loss_weight
            for idx, key in enumerate(self.cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS):
                losses[f"param/{key}-loss"] = loss_itemized[:, idx].mean()
        else:
            raise NotImplementedError
        return losses

    def visualize(self, predictions, batched_inputs):
        with torch.no_grad():
            images = torch.cat(
                (predictions["pred_gravity"], predictions["pred_latitude"]), dim=1
            )
            images = F.interpolate(images, (self.input_size, self.input_size))

            x = self.backbone(images)
        assert "rel_cx" in self.cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS
        assert "rel_cy" in self.cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS
        predictions["pred_rel_pp"] = torch.cat(
            [
                predictions["pred_rel_cx"].view(-1, 1),
                predictions["pred_rel_cy"].view(-1, 1),
            ],
            dim=-1,
        )
        vis_dict = {}
        _, h, w = batched_inputs[0]["image"].shape
        pp_gt = batched_inputs[0]["rel_pp"] * h + np.array([w, h]) / 2
        vis_img = cv2.circle(
            batched_inputs[0]["image"].cpu().numpy().transpose(1, 2, 0).copy(),
            pp_gt.astype(int),
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )
        pp_pred = predictions["pred_rel_pp"].cpu().numpy() * h + np.array([w, h]) / 2
        vis_img = cv2.circle(
            vis_img, pp_pred[0].astype(int), radius=8, color=(0, 255, 0), thickness=-1
        )

        original = cv2.resize(
            batched_inputs[0]["img_center_original"], (vis_img.shape[:2])
        )

        original = original[:, :, ::-1] / 255
        original = torch.tensor(original.transpose(2, 0, 1))
        vis_img = vis_img[:, :, ::-1] / 255
        vis_img = torch.tensor(vis_img.transpose(2, 0, 1))
        cat = torch.cat((original, vis_img), 1)
        return {"principal point": cat}
