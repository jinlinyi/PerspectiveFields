import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...utils import decode_bin, draw_up_field, pf_postprocess
from ...utils.config import configurable
from . import BaseDecodeHead
from .decode_head import MLP, FeatureFusionBlock
from .loss_fns import msgil_norm_loss


def build_gravity_decoder(cfg, input_shape):
    decoder_name = cfg.MODEL.GRAVITY_DECODER.NAME
    if decoder_name == "GravityDecoder":
        return GravityDecoder(cfg, input_shape)
    # Add more conditions here for other decoders
    else:
        raise ValueError(f"Unknown decoder name: {decoder_name}")


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class GravityDecoder(BaseDecodeHead):
    @configurable
    def __init__(self, feature_strides, loss_weight, **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.common_stride = 1
        self.loss_weight = loss_weight

        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        decoder_params = kwargs["decoder_params"]
        embedding_dim = decoder_params["embed_dim"]
        self.num_classes = kwargs["num_classes"]
        self.ignore_value = kwargs["ignore_value"]
        self.loss_type = kwargs["loss_type"]
        self.image_size = kwargs["image_size"]
        if self.loss_type == "regression":
            self.num_classes = 2

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_c4_proc = torch.nn.Conv2d(
            embedding_dim,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.linear_c3_proc = torch.nn.Conv2d(
            embedding_dim,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.linear_c2_proc = torch.nn.Conv2d(
            embedding_dim,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.linear_c1_proc = torch.nn.Conv2d(
            embedding_dim,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.fusion1 = FeatureFusionBlock(256)
        self.fusion2 = FeatureFusionBlock(256)
        self.fusion3 = FeatureFusionBlock(256)
        self.fusion4 = FeatureFusionBlock(256, unit2only=True)

        self.conv_fuse_conv0 = ConvModule(
            in_channels=256 + 64,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )

        self.conv_fuse_conv1 = ConvModule(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.linear_pred_gravity = nn.Conv2d(32, self.num_classes, kernel_size=1)

        # weight_init.c2_msra_fill(self.linear_pred_gravity)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_channels": [64, 128, 320, 512],
            "in_index": [0, 1, 2, 3],
            "feature_strides": [4, 8, 16, 32],
            "channels": 128,
            "dropout_ratio": 0.1,
            "num_classes": cfg.MODEL.GRAVITY_DECODER.NUM_CLASSES,
            "ignore_value": cfg.MODEL.GRAVITY_DECODER.IGNORE_VALUE,
            "norm_cfg": dict(type="SyncBN", requires_grad=True),
            "align_corners": False,
            "decoder_params": dict(embed_dim=768),
            "loss_weight": cfg.MODEL.GRAVITY_DECODER.LOSS_WEIGHT,
            "loss_type": cfg.MODEL.GRAVITY_DECODER.LOSS_TYPE,
            "image_size": cfg.DATALOADER.RESIZE,
        }

    def layers(self, features):
        x = self._transform_inputs(features["hl"])  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = (
            self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        )
        _c4 = self.linear_c4_proc(_c4)
        _c4 = self.fusion4(_c4)

        _c3 = (
            self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        )
        _c3 = self.linear_c3_proc(_c3)
        _c3 = self.fusion3(_c4, _c3)

        _c2 = (
            self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        )
        _c2 = self.linear_c2_proc(_c2)
        _c2 = self.fusion2(_c3, _c2)

        _c1 = (
            self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        )
        _c1 = self.linear_c1_proc(_c1)
        _c1 = self.fusion1(_c2, _c1)

        x = torch.cat([_c1, features["ll"]], dim=1)
        x = self.conv_fuse_conv0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_fuse_conv1(x)

        x = self.linear_pred_gravity(x)
        return x

    def forward(self, features, targets=None):
        x = self.layers(features)
        if self.loss_type == "regression":
            x = F.normalize(x, dim=1)
        if self.training:
            return x, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def inference(self, features):
        x = self.layers(features)
        if self.loss_type == "regression":
            x = F.normalize(x, dim=1)
        x = F.interpolate(
            x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        return x

    def losses(self, predictions, targets):
        predictions = (
            predictions.float()
        )  # https://github.com/pytorch/pytorch/issues/48163

        if self.loss_type == "regression":
            losses = {}
            mask = (torch.norm(targets, dim=1) > 1e-5).unsqueeze(1)
            mask_tiled = torch.tile(mask, (1, 2, 1, 1))
            losses["gravity-msg-normal-loss"] = (
                0.1
                * msgil_norm_loss(predictions, targets, mask_tiled)
                * self.loss_weight
            )
            losses["gravity-l2-loss"] = (
                torch.sum((predictions - targets) ** 2, dim=1, keepdim=True)[
                    mask
                ].mean()
                * self.loss_weight
            )
            for k in losses.keys():
                if torch.isnan(losses[k]):
                    import pdb

                    pdb.set_trace()
        elif self.loss_type == "classification":
            loss = F.cross_entropy(
                predictions, targets, reduction="mean", ignore_index=self.ignore_value
            )
            if torch.isnan(loss):
                import pdb

                pdb.set_trace()
            losses = {"loss_gravity": loss * self.loss_weight}
        else:
            raise NotImplementedError
        return losses

    def postprocess(self, results, batched_inputs, images):
        processed_results = []
        for result, input_per_image in zip(results, batched_inputs):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            if self.loss_type == "regression":
                vec = result
            elif self.loss_type == "classification":
                vec = decode_bin(result.argmax(dim=0), self.num_classes)
            else:
                raise NotImplementedError
            scale = (
                torch.tensor(
                    [[width / self.image_size[1]], [height / self.image_size[0]]]
                )
                .unsqueeze(-1)
                .to(vec.device)
            )
            vec_original = vec * scale
            vec_original = pf_postprocess(vec_original, self.image_size, height, width)
            vec_original = F.normalize(vec_original, dim=0)
            processed_results.append(
                {"pred_gravity": result, "pred_gravity_original": vec_original}
            )
        return processed_results

    def visualize(self, img, pred, gt):
        if self.loss_type == "regression":
            # Pred map
            pred = (
                draw_up_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1],
                    vector_field=pred.cpu(),
                    color=(0, 1, 0),
                )[:, :, ::-1]
                / 255.0
            )
            gt = (
                draw_up_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1],
                    vector_field=gt.cpu(),
                    color=(1, 0, 0),
                )[:, :, ::-1]
                / 255.0
            )
        elif self.loss_type == "classification":
            # Pred map
            pred = pred.argmax(dim=0)
            pred = (
                draw_up_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1],
                    vector_field=decode_bin(pred.cpu(), self.num_classes),
                    color=(0, 1, 0),
                )[:, :, ::-1]
                / 255.0
            )
            gt = (
                draw_up_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1],
                    vector_field=decode_bin(gt.cpu(), self.num_classes),
                    color=(1, 0, 0),
                )[:, :, ::-1]
                / 255.0
            )
        else:
            raise NotImplementedError
        img = img.cpu() / 255
        pred = torch.tensor(pred.transpose(2, 0, 1))
        gt = torch.tensor(gt.transpose(2, 0, 1))
        cat = torch.cat((img, pred, gt), 1)
        return {"gravity-pred-gt": cat}
