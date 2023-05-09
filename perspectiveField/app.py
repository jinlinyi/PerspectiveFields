import gradio as gr
import cv2
import copy
import torch
from PIL import Image, ImageDraw
from glob import glob
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from perspective2d.utils.predictor import VisualizationDemo
import perspective2d.modeling  # noqa
from perspective2d.config import get_perspective2d_cfg_defaults
from perspective2d.utils import draw_from_r_p_f_cx_cy




title = "Perspective Fields Demo"
description = """
<Perspective Fields width=200px>
"""

article = "Article Name"

def setup_cfg(args):
    cfgs = []
    configs = args['config_file'].split('#')
    weights_id = args['opts'].index('MODEL.WEIGHTS') + 1
    weights = args['opts'][weights_id].split('#')
    for i, conf in enumerate(configs):
        if len(conf) != 0:
            tmp_opts = copy.deepcopy(args['opts'])
            tmp_opts[weights_id] = weights[i]
            cfg = get_cfg()
            get_perspective2d_cfg_defaults(cfg)
            cfg.merge_from_file(conf)
            cfg.merge_from_list(tmp_opts)
            cfg.freeze()
            cfgs.append(cfg)
    return cfgs

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


def inference(img, model_type):
    perspective_cfg_list = setup_cfg(model_zoo[model_type])
    demo = VisualizationDemo(cfg_list=perspective_cfg_list)

    # img = read_image(image_path, format="BGR")
    img = img[..., ::-1] # rgb->bgr
    pred = demo.run_on_image(img)
    field = {
        'up': pred['pred_gravity_original'].cpu().detach(),
        'lati': pred['pred_latitude_original'].cpu().detach(),
    }
    img, field = resize_fix_aspect_ratio(img, field, 640)
    if not model_zoo[model_type]['param']:
        pred_vis = demo.draw(
            image=img,
            latimap=field['lati'],
            gravity=field['up'],
            latimap_format=pred['pred_latitude_original_mode'],
        ).get_image()
        param = "Not Implemented"
    else:
        if 'pred_general_vfov' not in pred.keys():
            pred['pred_general_vfov'] = pred['pred_vfov']
        if 'pred_rel_cx' not in pred.keys():
            pred['pred_rel_cx'] = torch.FloatTensor([0])
        if 'pred_rel_cy' not in pred.keys():
            pred['pred_rel_cy'] = torch.FloatTensor([0])
            
        r_p_f_rad = np.radians(
            [
                pred['pred_roll'].cpu().item(),
                pred['pred_pitch'].cpu().item(),
                pred['pred_general_vfov'].cpu().item(),
            ]
        )
        cx_cy = [
            pred['pred_rel_cx'].cpu().item(),
            pred['pred_rel_cy'].cpu().item(),
        ]
        param = f"roll {pred['pred_roll'].cpu().item() :.2f}\npitch {pred['pred_pitch'].cpu().item() :.2f}\nfov {pred['pred_general_vfov'].cpu().item() :.2f}\n"
        param += f"principal point {pred['pred_rel_cx'].cpu().item() :.2f} {pred['pred_rel_cy'].cpu().item() :.2f}"
        pred_vis = draw_from_r_p_f_cx_cy(
            img[:,:,::-1], 
            *r_p_f_rad,
            *cx_cy,
            'rad',
            up_color=(0,1,0),
        )
    return Image.fromarray(pred_vis), param

examples = []
for img_name in glob('../assets/imgs/*.*g'):
    examples.append([img_name])

model_map = {
    "": 'cvpr2023_360',
}

model_zoo = {
    'PersNet-360Cities': {
        'weights': ['https://www.dropbox.com/s/czqrepqe7x70b7y/cvpr2023.pth'],
        'opts': ['MODEL.WEIGHTS', 'models/cvpr2023.pth'],
        'config_file': 'models/cvpr2023.yaml',
        'param': False,
    },
    'PersNet_Paramnet-GSV-uncentered': {
        'weights': ['https://www.dropbox.com/s/ufdadxigewakzlz/paramnet_gsv_rpfpp.pth'],
        'opts': ['MODEL.WEIGHTS', 'models/paramnet_gsv_rpfpp.pth'],
        'config_file': 'models/paramnet_gsv_rpfpp.yaml',
        'param': True,
    },
    # trained on GSV dataset, predicts Perspective Fields + camera parameters (roll, pitch, fov), assuming centered principal point
    'PersNet_Paramnet-GSV-centered': {
        'weights': ['https://www.dropbox.com/s/g6xwbgnkggapyeu/paramnet_gsv_rpf.pth'],
        'opts': ['MODEL.WEIGHTS', 'models/paramnet_gsv_rpf.pth'],
        'config_file': 'models/paramnet_gsv_rpf.yaml',
        'param': True,
    },
}

gr.Interface(
    fn=inference,
    inputs=["image", gr.Radio(list(model_zoo.keys()), value=list(sorted(model_zoo.keys()))[0], label="Model", info="Which model?"),],
    outputs=[gr.Image(label='Perspective Fields'), gr.Textbox(label='Pred Camera Parameters')],
    title=title,
    description=description,
    article=article,
    examples=examples,
).launch(share=True)