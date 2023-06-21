# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import json
from tqdm import tqdm
import torch
import multiprocessing
import math
from multiprocessing import Pool, Process, Queue
import copy
import pickle
import pycocotools.mask as mask_util
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.data.common import MapDataset

from perspective2d.utils.predictor import VisualizationDemo
import perspective2d.modeling  # noqa
from perspective2d.config import get_perspective2d_cfg_defaults
from perspective2d.data import PerspectiveMapper

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import transforms as T

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    cfgs = []
    configs = args.config_file.split('#')
    weights_id = args.opts.index('MODEL.WEIGHTS') + 1
    weights = args.opts[weights_id].split('#')
    for i, conf in enumerate(configs):
        if len(conf) != 0:
            tmp_opts = copy.deepcopy(args.opts)
            tmp_opts[weights_id] = weights[i]
            cfg = get_cfg()
            get_perspective2d_cfg_defaults(cfg)
            cfg.merge_from_file(conf)
            cfg.merge_from_list(tmp_opts)
            cfg.freeze()
            cfgs.append(cfg)
    return cfgs


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=-1,
        help="max number of items for inference",
    )
    parser.add_argument(
        "--expname",
        type=str,
        help="ours name",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def dict2str(info):
    text = ""
    for key in info.keys():
        text += f"{key}: {np.array2string(info[key], precision=2, floatmode='fixed')}\n"
    return text


def save_vis(demo, img, pred, gt, idx, info, output_folder):
    pred_vis = demo.draw(
        image=img,
        latimap=pred['pred_latitude_original'],
        gravity=pred['pred_gravity_original'],
        latimap_format=pred['pred_latitude_original_mode'],
    )


    gt_vis = demo.draw(
        image=img,
        latimap=gt['gt_latitude_original'],
        gravity=gt['gt_gravity_original'],
        latimap_format=gt['gt_latitude_original_mode'],
    )
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, str(idx).zfill(4) + '_2input.jpg'), img)
    pred_vis.save(os.path.join(output_folder, str(idx).zfill(4) + '_1pred'))
    gt_vis.save(os.path.join(output_folder, str(idx).zfill(4) + '_0gt'))


def get_r_p_from_rotation(gravity_world, R_world_from_cam):
    """
    Convert Hypersim camera rotation matrix to roll and pitch
    """
    gravity_world = gravity_world / np.linalg.norm(gravity_world)
    gravity_world = gravity_world.astype(float)
    rot = R_world_from_cam.T
    # The convention here is that the camera's positive x axis points right, the positive y
    # axis points up, and the positive z axis points away from where the camera is looking.
    up_n = rot@gravity_world
    # Uprightnet: Z upward, Y right, X backward
    up_n = np.array([up_n[2], up_n[0], up_n[1]])
    pitch = - math.asin(up_n[0])

    sin_roll = up_n[1]/math.cos(pitch)

    roll = math.asin(sin_roll)
    roll = -roll
    return roll, pitch


def vec_angle_err(pred, gt):
    dot = torch.clamp(torch.sum(pred * gt, dim=0), -1, 1)
    angle = torch.acos(dot)
    return angle

def eval_by_idx(cfg, demo, dataset, idx):
    img_path = dataset[idx]['file_name']
    img = read_image(img_path, format=cfg.INPUT.FORMAT)
    predictions = demo.run_on_image(img)
    if 'mask_on' in dataset[idx].keys() and dataset[idx]['mask_on']:
        mask = mask_util.decode(dataset[idx]['mask']).astype(bool)
    else:
        mask = np.ones(img.shape[:2], dtype=bool)
    mat_up_err_rad = vec_angle_err(predictions['pred_gravity_original'], dataset[idx]['gt_gravity_original'].to(predictions['pred_gravity_original'].device))
    mat_up_err_deg = torch.rad2deg(mat_up_err_rad).detach().cpu().numpy()
    perc_up_err_less_5 = ((mat_up_err_deg[mask] <= 5).sum() / mask.sum()).item()
    avg_up_err_deg = mat_up_err_deg[mask].mean()
    med_up_err_deg = np.median(mat_up_err_deg[mask])
    if predictions['pred_latitude_original_mode'] == 'deg' and dataset[idx]['gt_latitude_original_mode'] == 'deg':  
        mat_lati_err_deg = torch.abs(predictions['pred_latitude_original'].detach().cpu() - dataset[idx]['gt_latitude_original'])
        perc_lati_err_less_5 = ((mat_lati_err_deg[mask] <= 5).sum() / mask.sum()).item()
        avg_lati_err_deg = mat_lati_err_deg.numpy()[mask].mean()
        med_lati_err_deg = np.median(mat_lati_err_deg.numpy()[mask])
    else:
        import pdb;pdb.set_trace()
        raise NotImplementedError
    rtn = {
        'pred_gravity_original': predictions['pred_gravity_original'].cpu().numpy(),
        'pred_latitude_original': predictions['pred_latitude_original'].cpu().numpy(),
        'pred_latitude_original_mode': predictions['pred_latitude_original_mode'],
        'avg_up_err_deg': avg_up_err_deg,
        'med_up_err_deg': med_up_err_deg,
        'avg_lati_err_deg': avg_lati_err_deg,
        'med_lati_err_deg': med_lati_err_deg,
        'perc_up_err_less_5': perc_up_err_less_5, 
        'perc_lati_err_less_5': perc_lati_err_less_5, 
    }
    # save_vis(demo, img, predictions, dataset[idx], f"{str(math.floor(avg_up_err_deg)).zfill(3)}_" + str(idx), rtn, output_folder='/Pool1/users/jinlinyi/public_html_shared/densefields/v06_ours_objectron_crop_mask')
    return rtn


def eval_by_list(cfg, demo, dataset, idx_list, return_dict, multiprocessing=False):
    if multiprocessing:
        loop = idx_list
    else:
        loop = tqdm(idx_list)
    for idx in loop:
        rtn = eval_by_idx(cfg, demo, dataset, idx)
        return_dict[idx] = rtn
        # if not multiprocessing:
        #     loop.set_description('Optimizing (loss %.4f)' % rtn['loss'])


def multiprocess_by_list(cfg, demo, dataset, idx_list, num_process):
    max_iter = len(idx_list)
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    per_thread = int(np.ceil(max_iter / num_process))
    split_by_thread = [
        idx_list[i * per_thread : (i + 1) * per_thread] for i in range(num_process)
    ]
    for i in range(num_process):
        p = Process(
            target=eval_by_list, args=(cfg, demo, dataset, split_by_thread[i], return_dict, True)
        )
        p.start()
        jobs.append(p)

    prev = 0
    with tqdm(total=max_iter) as pbar:
        while True:
            time.sleep(0.1)
            curr = len(return_dict.keys())
            pbar.update(curr - prev)
            prev = curr
            if curr == max_iter:
                break

    for job in jobs:
        job.join()

    return return_dict


class dataset_list(MapDataset):
    def __init__(self, cfg_list, dataset):
        dataloaders = []
        for cfg in cfg_list:
            dataloader = build_detection_test_loader(
                cfg, dataset, mapper=PerspectiveMapper(cfg, False, dataset_names=(dataset,))
            )
            dataloaders.append(dataloader.dataset)
        l = len(dataloaders[0])
        for dataloader in dataloaders:
            assert l == len(dataloader)
        self.dataloaders = dataloaders
        
    
    def __getitem__(self, idx):
        rtn = {}
        for dataloader in self.dataloaders:
            item = dataloader[idx]
            for key in item.keys():
                if key in rtn.keys():
                    # assert rtn[key] == item[key]
                    continue
                else:
                    rtn[key] = item[key]
        return rtn

    def __len__(self):
        return len(self.dataloaders[0])



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg_list = setup_cfg(args)

    demo = VisualizationDemo(cfg_list=cfg_list)

    cfg = cfg_list[0]
    dataloader = dataset_list(cfg_list, args.dataset)
    return_dict = {}
    
    np.random.seed(2022)
    idx_list = np.arange(len(dataloader))

    eval_by_list(cfg, demo, dataloader, idx_list, return_dict)
    
    up_errs, lati_errs = [], []
    for idx in return_dict.keys():
        up_errs.append(return_dict[idx]['avg_up_err_deg'])
        lati_errs.append(return_dict[idx]['avg_lati_err_deg'])
    up_errs = np.array(up_errs)
    lati_errs = np.array(lati_errs)
    percent_up = np.sum(np.array(up_errs)<5)/len(up_errs)*100
    percent_lati = np.sum(np.array(lati_errs)<5)/len(lati_errs)*100
    print("up_errs mean: {:.2f}".format(np.average(up_errs)))
    print("up_errs median: {:.2f}".format(np.median(up_errs)))
    print("up_errs %<5: {:.2f}".format(percent_up))
    print("lati_errs mean: {:.2f}".format(np.average(lati_errs)))
    print("lati_errs median: {:.2f}".format(np.median(lati_errs)))
    print("lati_errs %<5: {:.2f}".format(percent_lati))
    print("{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        np.average(up_errs), np.median(up_errs), percent_up,
        np.average(lati_errs), np.median(lati_errs), percent_lati,
    ))
    
    with open(os.path.join(args.output, f'{args.expname}-{args.dataset}.pickle'), 'wb') as f:
        pickle.dump(return_dict, f)
    