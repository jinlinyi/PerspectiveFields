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
import tqdm
import torch
import copy
import pickle
import json
import imageio
from PIL import Image, ImageDraw

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from perspective2d.utils.predictor import VisualizationDemo
import perspective2d.modeling  # noqa
from perspective2d.config import get_perspective2d_cfg_defaults
from perspective2d.utils import draw_vector_field, draw_latitude_field, draw_up_field, draw_from_r_p_f_cx_cy
from perspective2d.utils.panocam  import PanoCam
from perspective2d.utils.visualizer import VisualizerPerspective

# constants
WINDOW_NAME = "Perspective Fields"



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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--optimize", action="store_true", help="whether optimize for the camera param.")
    parser.add_argument("--net-init", action="store_true", help="whether use network init during optimization.")
    parser.add_argument("--noncenter", action="store_true", help="Whether assume centered principal point during optimization.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
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
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



def resize_img(img, min_size, max_size):
    im_h, im_w, _ = img.shape
    im_h_new = min(max(im_h, min_size[0]), max_size[0])
    fraction_h = im_h_new / im_h
    # try keeping aspect ratio
    im_w = fraction_h * im_w
    im_w_new = min(max(im_w, min_size[1]), max_size[1])
    return cv2.resize(img, (int(im_w_new), int(im_h_new)))



def resize_fix_aspect_ratio(img, field, target_width=None, target_height=None):
    height = img.shape[0]
    width = img.shape[1]
    if target_height is None:
        factor = target_width / width
    elif target_width is None:
        factor = target_height / height
    else:
        factor = max(target_width / width, target_height / height)
    if factor == target_width / width:
        target_height = int(height * factor)
    else:
        target_width = int(width * factor)
        
    img = cv2.resize(img, (target_width, target_height))
    for key in field:
        if key not in ['up', 'lati']:
            continue
        tmp = field[key].numpy()
        transpose = len(tmp.shape) == 3
        if transpose:
            tmp = tmp.transpose(1,2,0)
        tmp = cv2.resize(tmp, (target_width, target_height))
        if transpose:
            tmp = tmp.transpose(2,0,1)
        field[key] = torch.tensor(tmp)
    return img, field

def crop_to_nearest_factor_of_16(image, field):
    """
    Crops an image to the nearest factor of 16.
    
    Args:
        image: A PIL Image object.
    
    Returns:
        A new PIL Image object cropped to the nearest factor of 16.
    """
    height, width, _ = image.shape
    
    # Calculate the new width and height
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16
    
    # Crop the image using the new width and height
    image = image[:new_height, :new_width, :]
    
    for key in field:
        if key not in ['up', 'lati']:
            continue
        if len(field[key]) == 3:
            field[key] = field[key][:new_height, :new_width, :]
        else:
            field[key] = field[key][:new_height, :new_width]
    return image, field


def get_rgba_image(vis):
    """
    Returns:
        ndarray:
            the visualized image of shape (H, W, 3) (RGB) in uint8 type.
            The shape is scaled w.r.t the input image using the given `scale` argument.
    """
    canvas = vis.canvas
    s, (width, height) = canvas.print_to_buffer()

    buffer = np.frombuffer(s, dtype="uint8")

    img_rgba = buffer.reshape(height, width, 4)
    return img_rgba


def save_gif(demo, img, pred, output_folder):
    field = {
        'up': pred['pred_gravity_original'].cpu().detach(),
        'lati': pred['pred_latitude_original'].cpu().detach(),
    }
    field_np = {
        'up': field['up'].numpy(),
        'lati': field['lati'].numpy(),
    }
    img, field = resize_fix_aspect_ratio(img, field, target_height=480)
    img, field = crop_to_nearest_factor_of_16(img, field)

    up = get_rgba_image(draw_up_field(
        np.zeros((*img.shape[:2], 4)),
        field['up'],
        (0,1,0),
        return_img=False,
    ))
    lat = get_rgba_image(draw_latitude_field(
        np.zeros((*img.shape[:2], 4)), 
        latimap=torch.deg2rad(field['lati']),
        alpha_contourf=0.4, 
        alpha_contour=0.9,
        return_img=False
    ))
    alpha_lat = np.array(lat)[:,:,3:] / 255
    pred_vis = img[:,:,::-1] * (1-alpha_lat) + lat[:,:,:3][:, :, ::-1] * alpha_lat
    alpha_up = np.array(up)[:,:,3:] / 255
    pred_vis = pred_vis * (1-alpha_up) + up[:,:,:3][:, :, ::-1] * alpha_up

    bottom = img[:,:,::-1]
    up = pred_vis
    
    num_frames = 150

    frames = [bottom]
    alpha = 1
    for frame_id in range(num_frames):
        i = 1 - (frame_id + 1) / num_frames
        rendered = copy.deepcopy(bottom)
        # rendered[:, :int(i * bottom.shape[1]), :] = (1-alpha) * bottom[:, :int(i * bottom.shape[1]), :] + alpha * up[:, :int(i * bottom.shape[1]), :]
        rendered[int(i * bottom.shape[0]):, :, :] = (1-alpha) * bottom[int(i * bottom.shape[0]):, :, :] + alpha * up[int(i * bottom.shape[0]):, :, :]
        
        frames.append(rendered)

    # Save the list of frames as an animated GIF
    writer = imageio.get_writer(os.path.join(output_folder, 'swiping.mp4'), fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_vis(demo, img, pred, output_folder):
    field = {
        'up': pred['pred_gravity_original'].cpu().detach(),
        'lati': pred['pred_latitude_original'].cpu().detach(),
    }
    field_np = {
        'up': field['up'].numpy(),
        'lati': field['lati'].numpy(),
    }
    img, field = resize_fix_aspect_ratio(img, field, 640)
    # pred_vis = draw_up_field(np.ones_like(img)*255, field['up'], color=(0,1,0))
    # pred_vis.fig.savefig(os.path.join(output_folder, 'perspective_pred.pdf'), transparent=True)
    # pred_vis.fig.savefig(os.path.join(output_folder, 'perspective_pred.svg'), transparent=True)
    pred_vis = demo.draw(
        image=img,
        latimap=field['lati'],
        gravity=field['up'],
        latimap_format=pred['pred_latitude_original_mode'],
    )
    pred_vis.save(os.path.join(output_folder, 'perspective_pred'))

    if 'opt_param' in pred.keys():
        optimized_vis = draw_from_r_p_f_cx_cy(
            img[:,:,::-1], 
            pred['opt_param']['pred_roll'], 
            pred['opt_param']['pred_pitch'], 
            pred['opt_param']['pred_general_vfov'], 
            pred['opt_param']['pred_rel_cx'], 
            pred['opt_param']['pred_rel_cy'],
            'deg',
            up_color=(0,1,0),
        )
        cv2.imwrite(os.path.join(output_folder, 'opt_param.jpg'), optimized_vis[:,:,::-1])
        # write as json file
        summary = pred['opt_param']
        with open(os.path.join(output_folder, 'opt_param.json'), 'w') as f:
            json.dump(summary, f)



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))



    if args.input:
        if len(args.input) == 1:
            if os.path.isdir(args.input[0]):
                args.input = glob.glob(os.path.join(args.input[0], '*.*g'))
            else:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
        if args.max != -1:
            # args.input = args.input[:args.max]
            args.input = np.random.choice(args.input, args.max, replace=False)
        cfg_list = setup_cfg(args)
        demo = VisualizationDemo(cfg_list=cfg_list)
        for path in tqdm.tqdm(args.input, disable=not args.output):
            output_f = os.path.join(args.output, os.path.basename(path).split('.')[0])
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions = demo.run_on_image(img)
            if args.optimize:
                predictions['opt_param'] = demo.opt_rpfpp(predictions, "cpu", net_init=args.net_init, pp_on=args.noncenter)

            if args.output:
                os.makedirs(output_f, exist_ok=True)
                save_vis(demo, img, predictions, output_folder=output_f)
                # save_gif(demo, img, predictions, output_folder=output_f)
                