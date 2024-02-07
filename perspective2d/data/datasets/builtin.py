import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from perspective2d.data.datasets import (
    load_cities360_json,
    load_edina_json,
    load_gsv_json,
    load_tartanair_json,
    load_stanford2d3d_json,
)

SPLITS_STANFORD2D3D = {
    "stanford2d3d_test": ("./datasets/stanford2d3d-processed/test", "./datasets/stanford2d3d-processed/test.json"),
    "stanford2d3d_test_crop": ("./datasets/stanford2d3d-processed/stanford2d3d_crop", "./datasets/stanford2d3d-processed/stanford2d3d_test_crop.json"),
    "stanford2d3d_test_warp": ("./datasets/stanford2d3d-processed/stanford2d3d_warp", "./datasets/stanford2d3d-processed/stanford2d3d_test_warp.json")
}

SPLITS_TARTANAIR = {
    "tartanair_test": ("./datasets/tartanair-processed/test", "./datasets/tartanair-processed/test.json"),
    "tartanair_test_crop": ("./datasets/tartanair-processed/tartanair_crop", "./datasets/tartanair-processed/tartanair_test_crop.json"),
    "tartanair_test_warp": ("./datasets/tartanair-processed/tartanair_warp", "./datasets/tartanair-processed/tartanair_test_warp.json"),
}

SPLITS_GSV = {
    "gsv_train": (
        "./datasets/gsv/google_street_view_191210/manhattan",
        "./datasets/gsv/gsv_train_20210313.csv",
    ),
    "gsv_test": (
        "./datasets/gsv/google_street_view_191210/manhattan",
        "./datasets/gsv/gsv_test_20210313.csv",
    ),
    "gsv_val": (
        "./datasets/gsv/google_street_view_191210/manhattan",
        "./datasets/gsv/gsv_val_20210313.csv",
    ),
    "gsv_test_crop_uniform": (
        "./datasets/gsv/gsv_test_crop_uniform",
        "./datasets/gsv/gsv_test_crop_uniform.json",
    ),
}

SPLITS_EDINA = {
    "edina_train": (
        "./datasets/edina/edina_train",
        "./datasets/edina/edina_train.json",
    ),
    "edina_test": ("./datasets/edina/edina_test", "./datasets/edina/edina_test.json"),
    "edina_test_crop_uniform": (
        "./datasets/edina/edina_test_crop_uniform",
        "./datasets/edina/edina_test_crop_uniform.json",
    ),
    "edina_test_crop_vfov": (
        "./datasets/edina/edina_test_crop_vfov",
        "./datasets/edina/edina_test_crop_vfov.json",
    ),
}

SPLITS_CITIES360 = {
    "cities360_train": (
        "./datasets/cities360/cities360_json_v3/train",
        "./datasets/cities360/cities360_json_v3/train.json",
    ),
    "cities360_test": (
        "./datasets/cities360/cities360_json_v3/test",
        "./datasets/cities360/cities360_json_v3/test.json",
    ),
}


def register_gsv(dataset_name, json_file, img_root):
    DatasetCatalog.register(dataset_name, lambda: load_gsv_json(json_file, img_root))
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file,
        image_root=img_root,
        evaluator_type="perspective",
        ignore_label=-1,
    )


def register_edina(dataset_name, json_file, img_root):
    DatasetCatalog.register(dataset_name, lambda: load_edina_json(json_file, img_root))
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file,
        image_root=img_root,
        evaluator_type="perspective",
        ignore_label=-1,
    )


def register_cities360(dataset_name, json_file, img_root="datasets"):
    DatasetCatalog.register(
        dataset_name, lambda: load_cities360_json(json_file, img_root)
    )
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file,
        image_root=img_root,
        evaluator_type="perspective",
        ignore_label=-1,
    )


def register_tartanair(dataset_name, json_file, img_root="datasets"):
    DatasetCatalog.register(
        dataset_name, lambda: load_tartanair_json(json_file, img_root)
    )
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=img_root, evaluator_type="perspective",
        ignore_label=-1,
    )

def register_stanford2d3d(dataset_name, json_file, img_root="datasets"):
    DatasetCatalog.register(
        dataset_name, lambda: load_stanford2d3d_json(json_file, img_root)
    )
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=img_root, evaluator_type="perspective",
        ignore_label=-1,
    )

for key, (img_root, anno_file) in SPLITS_GSV.items():
    register_gsv(key, anno_file, img_root)

for key, (img_root, anno_file) in SPLITS_EDINA.items():
    register_edina(key, anno_file, img_root)

for key, (img_root, anno_file) in SPLITS_CITIES360.items():
    register_cities360(key, anno_file, img_root)


for key, (img_root, anno_file) in SPLITS_STANFORD2D3D.items():
    register_stanford2d3d(key, anno_file, img_root)

for key, (img_root, anno_file) in SPLITS_TARTANAIR.items():
    register_tartanair(key, anno_file, img_root)

