import h5py
import numpy as np
import torch

from perspective2d.utils import encode_bin_latitude
from perspective2d.utils.panocam import PanoCam


def read_hdf5(path):
    with h5py.File(path, "r") as f:
        return f["dataset"][:]


class LatitudeTransform:
    def __init__(self, cfg, is_train=True):
        """
        Args:
            cfg (CfgNode): model configurations
            is_train (bool, optional): Defaults to True.
        """
        self.is_train = is_train
        if cfg.MODEL.META_ARCHITECTURE == "PerspectiveNet":
            self.loss_type = "classification"
            self.num_classes = cfg.MODEL.FPN_LATITUDE_HEAD.NUM_CLASSES
        elif cfg.MODEL.META_ARCHITECTURE == "PersFormer":
            self.loss_type = cfg.MODEL.LATITUDE_DECODER.LOSS_TYPE
            self.num_classes = cfg.MODEL.LATITUDE_DECODER.NUM_CLASSES
        elif cfg.MODEL.META_ARCHITECTURE == "ParamNetStandalone":
            self.loss_type = cfg.MODEL.LATITUDE_DECODER.LOSS_TYPE
            self.num_class = cfg.MODEL.LATITUDE_DECODER.NUM_CLASSES
        else:
            raise NotImplementedError

    def get_input_label(self, dataset_dict):
        """Retrieve ground truth latitude map

        Args:
            dataset_dict (dict): dict corresponding to one input example

        Returns:
            latimap (np.ndarray): ground truth latitude map
            gt_latitude_original_mode (str): latimap mode, either "deg" or "rad"
        """
        if dataset_dict["dataset"] in [
            "cities360",
            "rgbdpano",
            "sun360",
            "tartanair",
            "stanford2d3d",
            "objectron",
            "gsv",
            "edina",
        ]:
            latimap = PanoCam.get_lat(
                vfov=np.radians(dataset_dict["vfov"]),
                im_w=dataset_dict["width"],
                im_h=dataset_dict["height"],
                elevation=np.radians(dataset_dict["pitch"]),
                roll=np.radians(dataset_dict["roll"]),
            )
            latimap = latimap.astype("float32")
        elif dataset_dict["dataset"] in [
            "hypersim",
            "sun360_warp",
            "sun360_crop",
            "sun360_uncrop",
            "tartanair_warp",
            "tartanair_crop",
            "stanford2d3d_warp",
            "stanford2d3d_crop",
            "objectron_crop",
            "objectron_crop_mask",
            "gsv_crop",
            "cities360_distort",
            "edina_crop",
        ]:
            latimap = read_hdf5(dataset_dict["latitude_file_name"]).astype("float32")
            assert np.allclose(
                latimap.shape, [dataset_dict["height"], dataset_dict["width"]]
            )
        elif dataset_dict["dataset"] in ["coco-pseudo"]:
            latimap = read_hdf5(dataset_dict["latitude_file_name"]).astype("float32")
        else:
            raise NotImplementedError

        gt_latitude_original_mode = "deg"
        return latimap, gt_latitude_original_mode

    def to_tensor(self, latitude):
        """Convert latitude map to tensor and encode bins if using classification

        Args:
            latitude (np.ndarray): latitude map

        Returns:
            torch.Tensor: latitude map converted to tensor and binned if using classification
        """
        latitude = torch.as_tensor(latitude.astype("float32"))
        if self.loss_type == "classification":
            latitude = encode_bin_latitude(latitude, self.num_classes).long()
        elif self.loss_type == "regression":
            # [-1, 1]
            latitude = torch.sin(torch.deg2rad(latitude.unsqueeze(0)))
        else:
            raise NotImplementedError
        return latitude
