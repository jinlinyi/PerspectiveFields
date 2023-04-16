import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from perspective2d.data.datasets import (
    load_gsv_json,
)


SPLITS_GSV = {
    "gsv_train": ("./datasets/google_street_view_191210/manhattan", "./datasets/gsv_train_20210313.csv"), 
    "gsv_test": ("./datasets/google_street_view_191210/manhattan", "./datasets/gsv_test_20210313.csv"), 
    "gsv_val": ("./datasets/google_street_view_191210/manhattan", "./datasets/gsv_val_20210313.csv"), 
    "gsv_test_crop_uniform": ("./datasets/gsv_test_crop_uniform", "./datasets/gsv_test_crop_uniform.json"), 
}



def register_gsv(dataset_name, json_file, img_root):
    DatasetCatalog.register(
        dataset_name, lambda: load_gsv_json(json_file, img_root)
    )
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=img_root, evaluator_type="perspective",
        ignore_label=-1,
    )


for key, (img_root, anno_file) in SPLITS_GSV.items():
    register_gsv(key, anno_file, img_root)
    