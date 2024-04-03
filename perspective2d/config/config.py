from yacs.config import CfgNode as CN


def get_perspective2d_cfg_defaults():
    """
    PerspectiveNet and ParamNet configs.
    """
    cfg = CN()
    cfg.VIS_PERIOD = 100
    cfg.INPUT = CN()
    cfg.INPUT.ONLINE_CROP = False
    cfg.INPUT.FORMAT = "BGR"
    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = []
    cfg.DATASETS.TEST = []

    cfg.DATALOADER = CN()
    cfg.DATALOADER.AUGMENTATION = False
    cfg.DATALOADER.AUGMENTATION_TYPE = "geometry"
    cfg.DATALOADER.RESIZE = [320, 320]  # Height, Width
    cfg.DATALOADER.AUGMENTATION_FUN = "default"
    cfg.DATALOADER.NO_GEOMETRY_AUG = False  # requested by R3 cvpr2023

    cfg.MODEL = CN()
    cfg.MODEL.GRAVITY_ON = False
    cfg.MODEL.LATITUDE_ON = False
    cfg.MODEL.RECOVER_RPF = False
    cfg.MODEL.RECOVER_PP = False

    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = "mitb3"

    cfg.MODEL.PERSFORMER_HEADS = CN()
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.PERSFORMER_HEADS.NAME = "StandardPersformerHeads"
    cfg.MODEL.LATITUDE_DECODER = CN()
    cfg.MODEL.LATITUDE_DECODER.NAME = "LatitudeDecoder"
    cfg.MODEL.LATITUDE_DECODER.LOSS_WEIGHT = 1.0
    cfg.MODEL.LATITUDE_DECODER.LOSS_TYPE = "regression"
    cfg.MODEL.LATITUDE_DECODER.NUM_CLASSES = 1
    cfg.MODEL.LATITUDE_DECODER.IGNORE_VALUE = -1
    cfg.MODEL.GRAVITY_DECODER = CN()
    cfg.MODEL.GRAVITY_DECODER.NAME = "GravityDecoder"
    cfg.MODEL.GRAVITY_DECODER.LOSS_WEIGHT = 1.0
    cfg.MODEL.GRAVITY_DECODER.LOSS_TYPE = "classification"
    cfg.MODEL.GRAVITY_DECODER.NUM_CLASSES = 73
    cfg.MODEL.GRAVITY_DECODER.IGNORE_VALUE = 72
    cfg.MODEL.HEIGHT_DECODER = CN()
    cfg.MODEL.HEIGHT_DECODER.NAME = "HeightDecoder"
    cfg.MODEL.HEIGHT_DECODER.LOSS_WEIGHT = 1.0

    cfg.MODEL.PARAM_DECODER = CN()
    cfg.MODEL.PARAM_DECODER.NAME = "ParamNet"
    cfg.MODEL.PARAM_DECODER.LOSS_TYPE = "regression"
    cfg.MODEL.PARAM_DECODER.LOSS_WEIGHT = 1.0
    cfg.MODEL.PARAM_DECODER.PREDICT_PARAMS = [
        "roll",
        "pitch",
        "rel_focal",
        "rel_cx",
        "rel_cy",
    ]
    cfg.MODEL.PARAM_DECODER.SYNTHETIC_PRETRAIN = False
    cfg.MODEL.PARAM_DECODER.INPUT_SIZE = 320
    cfg.MODEL.PARAM_DECODER.DEBUG_LAT = False
    cfg.MODEL.PARAM_DECODER.DEBUG_UP = False

    cfg.MODEL.FREEZE = []
    cfg.DEBUG_ON = False
    cfg.OVERFIT_ON = False

    """
    The configs below are not used.
    """
    cfg.MODEL.CENTER_ON = False
    cfg.MODEL.HEIGHT_ON = False
    cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    cfg.MODEL.FPN_HEADS = CN()
    cfg.MODEL.FPN_HEADS.NAME = "StandardFPNHeads"
    # Gravity

    cfg.MODEL.FPN_GRAVITY_HEAD = CN()
    cfg.MODEL.FPN_GRAVITY_HEAD.NAME = "GravityFPNHead"
    cfg.MODEL.FPN_GRAVITY_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    # Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
    # the correposnding pixel.
    cfg.MODEL.FPN_GRAVITY_HEAD.IGNORE_VALUE = 360
    # Number of classes in the semantic segmentation head
    cfg.MODEL.FPN_GRAVITY_HEAD.NUM_CLASSES = 361
    # Number of channels in the 3x3 convs inside semantic-FPN heads.
    cfg.MODEL.FPN_GRAVITY_HEAD.CONVS_DIM = 128
    # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
    cfg.MODEL.FPN_GRAVITY_HEAD.COMMON_STRIDE = 4
    # Normalization method for the convolution layers. Options: "" (no norm), "GN".
    cfg.MODEL.FPN_GRAVITY_HEAD.NORM = "GN"
    cfg.MODEL.FPN_GRAVITY_HEAD.LOSS_WEIGHT = 1.0

    # Latitude

    cfg.MODEL.FPN_LATITUDE_HEAD = CN()
    cfg.MODEL.FPN_LATITUDE_HEAD.NAME = "LatitudeFPNHead"
    cfg.MODEL.FPN_LATITUDE_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    # Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
    # the correposnding pixel.
    cfg.MODEL.FPN_LATITUDE_HEAD.IGNORE_VALUE = -1
    # Number of classes in the semantic segmentation head
    cfg.MODEL.FPN_LATITUDE_HEAD.NUM_CLASSES = 9
    # Number of channels in the 3x3 convs inside semantic-FPN heads.
    cfg.MODEL.FPN_LATITUDE_HEAD.CONVS_DIM = 128
    # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
    cfg.MODEL.FPN_LATITUDE_HEAD.COMMON_STRIDE = 4
    # Normalization method for the convolution layers. Options: "" (no norm), "GN".
    cfg.MODEL.FPN_LATITUDE_HEAD.NORM = "GN"
    cfg.MODEL.FPN_LATITUDE_HEAD.LOSS_WEIGHT = 1.0
    # Center

    cfg.MODEL.FPN_CENTER_HEAD = CN()
    cfg.MODEL.FPN_CENTER_HEAD.NAME = "CenterFPNHead"
    cfg.MODEL.FPN_CENTER_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    # Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
    # the correposnding pixel.
    cfg.MODEL.FPN_CENTER_HEAD.IGNORE_VALUE = 360
    # Number of classes in the semantic segmentation head
    cfg.MODEL.FPN_CENTER_HEAD.NUM_CLASSES = 30
    # Number of channels in the 3x3 convs inside semantic-FPN heads.
    cfg.MODEL.FPN_CENTER_HEAD.CONVS_DIM = 128
    # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
    cfg.MODEL.FPN_CENTER_HEAD.COMMON_STRIDE = 4
    # Normalization method for the convolution layers. Options: "" (no norm), "GN".
    cfg.MODEL.FPN_CENTER_HEAD.NORM = "GN"
    cfg.MODEL.FPN_CENTER_HEAD.LOSS_WEIGHT = 1.0

    ############################################################

    return cfg
