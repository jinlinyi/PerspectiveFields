from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .config import get_perspective2d_cfg_defaults
from .modeling.backbone import build_backbone
from .modeling.param_network import build_param_net
from .modeling.persformer_heads import build_persformer_heads


class ResizeTransform:
    """
    Resize the image to a target size.
    """

    def __init__(self, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        if interp is None:
            interp = Image.BILINEAR
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, img, interp=None):
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)
        return ret


class LowLevelEncoder(nn.Module):
    def __init__(self, feat_dim=64, in_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, feat_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


model_zoo = {
    "Paramnet-360Cities-edina-centered": {
        "weights": "https://huggingface.co/spaces/jinlinyi/PerspectiveFields/resolve/main/models/paramnet_360cities_edina_rpf.pth",
        "config_file": "paramnet_360cities_edina_rpf.yaml",
        "param": True,
        "description": "Trained on 360cities and EDINA dataset. Assumes centered principal point. Predicts roll, pitch and fov.",
    },
    "Paramnet-360Cities-edina-uncentered": {
        "weights": "https://huggingface.co/spaces/jinlinyi/PerspectiveFields/resolve/main/models/paramnet_360cities_edina_rpfpp.pth",
        "config_file": "paramnet_360cities_edina_rpfpp.yaml",
        "param": True,
        "description": "Trained on 360cities and EDINA dataset. Predicts roll, pitch, fov and principal point.",
    },
    "PersNet-360Cities": {
        "weights": "https://huggingface.co/spaces/jinlinyi/PerspectiveFields/resolve/main/models/cvpr2023.pth",
        "config_file": "cvpr2023.yaml",
        "param": False,
        "description": "Trained on 360cities. Predicts perspective fields.",
    },
    "PersNet_Paramnet-GSV-uncentered": {
        "weights": "https://huggingface.co/spaces/jinlinyi/PerspectiveFields/resolve/main/models/paramnet_gsv_rpfpp.pth",
        "config_file": "paramnet_gsv_rpfpp.yaml",
        "param": True,
        "description": "Trained on GSV. Predicts roll, pitch, fov and principal point.",
    },
    # trained on GSV dataset, predicts Perspective Fields + camera parameters (roll, pitch, fov), assuming centered principal point
    "PersNet_Paramnet-GSV-centered": {
        "weights": "https://huggingface.co/spaces/jinlinyi/PerspectiveFields/resolve/main/models/paramnet_gsv_rpf.pth",
        "config_file": "paramnet_gsv_rpf.yaml",
        "param": True,
        "description": "Trained on GSV. Assumes centered principal point. Predicts roll, pitch and fov.",
    },
}


class PerspectiveFields(nn.Module):
    def __init__(self, version="Paramnet-360Cities-edina-centered"):
        super().__init__()
        default_conf = get_perspective2d_cfg_defaults()
        # To get the path
        with resources.path(
            "perspective2d.config", model_zoo[version]["config_file"]
        ) as config_path:
            default_conf.merge_from_file(str(config_path))
        # default_conf.merge_from_file(model_zoo[version]['config_file'])
        default_conf.freeze()
        self.version = version
        self.param_on = model_zoo[version]["param"]
        self.cfg = cfg = default_conf
        self.backbone = build_backbone(cfg)
        self.ll_enc = LowLevelEncoder()
        self.persformer_heads = build_persformer_heads(
            cfg, self.backbone.output_shape()
        )
        self.param_net = (
            build_param_net(cfg)
            if cfg.MODEL.RECOVER_RPF or cfg.MODEL.RECOVER_PP
            else None
        )
        self.register_buffer(
            "pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False
        )
        self.vis_period = cfg.VIS_PERIOD
        self.freeze = cfg.MODEL.FREEZE
        self.debug_on = cfg.DEBUG_ON
        self.input_format = cfg.INPUT.FORMAT
        self.aug = ResizeTransform(cfg.DATALOADER.RESIZE[0], cfg.DATALOADER.RESIZE[1])
        for layers in self.freeze:
            layer = layers.split(".")
            final = self
            for l in layer:
                final = getattr(final, l)
            for params in final.parameters():
                params.requires_grad = False
        self._init_weights()

    @property
    def device(self):
        return self.pixel_mean.device

    @staticmethod
    def versions():
        for key in model_zoo:
            print(f"{key}")
            print(f"   - {model_zoo[key]['description']}")

    def version(self):
        return self.version

    def _init_weights(self):
        state_dict = None
        if self.version in model_zoo:
            state_dict = torch.hub.load_state_dict_from_url(
                model_zoo[self.version]["weights"], 
                map_location=torch.device('cpu'),
            )
            self.load_state_dict(state_dict, strict=False)
        elif self.cfg.MODEL.WEIGHTS is not None:
            path = Path(__file__).parent
            path = path / "weights/{}.pth".format(self.cfg.MODEL.WEIGHTS)
            state_dict = torch.load(str(path), map_location="cpu")

        if state_dict:
            status = self.load_state_dict(state_dict["model"], strict=False)

    @torch.no_grad()
    def inference(self, img_bgr):
        original_image = img_bgr.copy()
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        predictions = self.forward([inputs])[0]
        return predictions

    def forward(self, batched_inputs) -> dict:
        """
        Forward pass of the PerspectiveFields model.

        Args:
            batched_inputs (list): A list of dictionaries containing the input data.

        Returns:
            dict: A dictionary containing the computed losses or processed results.

        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = torch.stack(images)
        hl_features = self.backbone(images)
        ll_features = self.ll_enc(images)
        features = {
            "hl": hl_features,  # features from backbone
            "ll": ll_features,  # low level features
        }

        targets_dict = {}
        if "gt_gravity" in batched_inputs[0]:
            targets = [x["gt_gravity"].to(self.device) for x in batched_inputs]
            targets = torch.stack(targets)
            targets_dict["gt_gravity"] = targets

        if "gt_latitude" in batched_inputs[0]:
            targets = [x["gt_latitude"].to(self.device) for x in batched_inputs]
            targets = torch.stack(targets)
            targets_dict["gt_latitude"] = targets

        results = self.persformer_heads.inference(features)
        processed_results = self.persformer_heads.postprocess(
            results, batched_inputs, images
        )

        if self.param_net is not None:
            param = self.param_net(results, batched_inputs)
            if "pred_general_vfov" not in param.keys():
                param["pred_general_vfov"] = param["pred_vfov"]
            if "pred_rel_cx" not in param.keys():
                param["pred_rel_cx"] = torch.zeros_like(param["pred_vfov"])
            if "pred_rel_cy" not in param.keys():
                param["pred_rel_cy"] = torch.zeros_like(param["pred_vfov"])
            processed_results[0].update(param)
        return processed_results
