import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...utils import decode_bin_latitude, draw_latitude_field, pf_postprocess
from ...utils.config import configurable
from . import BaseDecodeHead
from .decode_head import MLP, FeatureFusionBlock
from .loss_fns import msgil_norm_loss


def build_latitude_decoder(cfg, input_shape):
    decoder_name = cfg.MODEL.LATITUDE_DECODER.NAME
    if decoder_name == "LatitudeDecoder":
        return LatitudeDecoder(cfg, input_shape)
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


class LatitudeDecoder(BaseDecodeHead):
    @configurable
    def __init__(self, feature_strides, loss_weight, **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.common_stride = 1
        self.loss_weight = loss_weight
        self.loss_type = kwargs["loss_type"]
        self.num_classes = kwargs["num_classes"]
        self.ignore_value = kwargs["ignore_value"]
        self.image_size = kwargs["image_size"]
        if self.loss_type == "regression":
            self.num_classes == 1

        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        decoder_params = kwargs["decoder_params"]
        embedding_dim = decoder_params["embed_dim"]

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

        self.linear_pred_latitude = nn.Conv2d(32, self.num_classes, kernel_size=1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_channels": [64, 128, 320, 512],
            "in_index": [0, 1, 2, 3],
            "feature_strides": [4, 8, 16, 32],
            "channels": 128,
            "dropout_ratio": 0.1,
            "norm_cfg": dict(type="SyncBN", requires_grad=True),
            "align_corners": False,
            "decoder_params": dict(embed_dim=768),
            "loss_weight": cfg.MODEL.LATITUDE_DECODER.LOSS_WEIGHT,
            "loss_type": cfg.MODEL.LATITUDE_DECODER.LOSS_TYPE,
            "num_classes": cfg.MODEL.LATITUDE_DECODER.NUM_CLASSES,
            "ignore_value": cfg.MODEL.LATITUDE_DECODER.IGNORE_VALUE,
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

        x = self.linear_pred_latitude(x)
        return x

    def forward(self, features, targets=None):
        x = self.layers(features)
        if self.loss_type == "regression":
            x = torch.clamp(x, -1, 1)
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
            x = torch.clamp(x, -1, 1)
        return x

    def postprocess(self, results, batched_inputs, images):
        processed_results = []
        for result, input_per_image in zip(results, batched_inputs):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            if self.loss_type == "regression":
                latimap = pf_postprocess(result, self.image_size, height, width)[0]
                latimap = torch.asin(latimap)
                latimap = torch.rad2deg(latimap)
            elif self.loss_type == "classification":
                latimap_bin = result.argmax(dim=0)
                latimap = decode_bin_latitude(latimap_bin, self.num_classes).unsqueeze(
                    0
                )
                latimap = pf_postprocess(latimap, self.image_size, height, width)[0]
            else:
                raise NotImplementedError
            processed_results.append(
                {
                    "pred_latitude": result,
                    "pred_latitude_original": latimap,
                    "pred_latitude_original_mode": "deg",
                }
            )
        return processed_results

    def losses(self, predictions, targets):
        predictions = (
            predictions.float()
        )  # https://github.com/pytorch/pytorch/issues/48163
        if self.loss_type == "regression":
            # loss = F.mse_loss(
            #     predictions, targets, reduction="mean",
            # )
            losses = {}
            mask = torch.ones(predictions.shape).to(bool)
            losses["latitude-msg-normal-loss"] = (
                0.1 * msgil_norm_loss(predictions, targets, mask) * self.loss_weight
            )
            losses["latitude-l2-loss"] = (
                F.mse_loss(predictions, targets, reduction="none")[mask].mean()
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
            losses = {"loss_latitude": loss * self.loss_weight}
            if torch.isnan(loss):
                import pdb

                pdb.set_trace()
        else:
            raise NotImplementedError
        return losses

    def visualize(self, img, pred, gt):
        if self.loss_type == "regression":
            # Pred map
            pred = (
                draw_latitude_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8),
                    latimap=np.arcsin(pred.squeeze(0).cpu().numpy()),
                )
                / 255.0
            )
            gt = (
                draw_latitude_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8),
                    latimap=np.arcsin(gt.squeeze(0).cpu().numpy()),
                )
                / 255.0
            )
        elif self.loss_type == "classification":
            pred = pred.argmax(dim=0)
            pred = (
                draw_latitude_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8),
                    # binmap=pred.cpu()
                    latimap=np.radians(
                        decode_bin_latitude(pred.cpu(), self.num_classes)
                    ),
                )
                / 255.0
            )
            gt = (
                draw_latitude_field(
                    img_rgb=img.numpy().transpose(1, 2, 0).astype(np.uint8),
                    # binmap=gt.cpu()
                    latimap=np.radians(decode_bin_latitude(gt.cpu(), self.num_classes)),
                )
                / 255.0
            )
        else:
            raise NotImplementedError
        img = img.cpu() / 255
        pred = torch.tensor(pred.transpose(2, 0, 1))
        gt = torch.tensor(gt.transpose(2, 0, 1))
        cat = torch.cat((img, pred, gt), 1)
        return {"latitude-pred-gt": cat}
