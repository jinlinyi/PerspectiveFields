import cv2
import torch
import math
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from detectron2.utils.colormap import colormap
from perspective2d.utils.visualizer import VisualizerPerspective
import albumentations as A
from .panocam import PanoCam

from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.optimize 


def general_vfov(d_cx, d_cy, h, focal, degree):
    p_sqr = focal ** 2 + d_cx ** 2 + (d_cy + 0.5 * h) ** 2
    q_sqr = focal ** 2 + d_cx ** 2 + (d_cy - 0.5 * h) ** 2
    cos_FoV = (p_sqr + q_sqr - h ** 2) / 2 / np.sqrt(p_sqr) / np.sqrt(q_sqr)
    FoV_rad = np.arccos(cos_FoV)
    if degree:
        return np.degrees(FoV_rad)
    else: 
        return FoV_rad


def general_vfov_to_focal(rel_cx, rel_cy, h, gvfov, degree):
    def fun(focal, *args):
        h, d_cx, d_cy, target_cos_FoV = args
        
        p_sqr = (focal/h) ** 2 + d_cx ** 2 + (d_cy + 0.5) ** 2
        q_sqr = (focal/h) ** 2 + d_cx ** 2 + (d_cy - 0.5) ** 2
        cos_FoV = (p_sqr + q_sqr - 1) / 2 / np.sqrt(p_sqr) / np.sqrt(q_sqr)
        return cos_FoV - target_cos_FoV
    if degree:
        gvfov = np.radians(gvfov)
    focal = scipy.optimize.fsolve(fun, 1.5, args=(h, rel_cx, rel_cy, np.cos(gvfov)))[0]
    focal = np.abs(focal)
    return focal
        

def encode_bin(vector_field, num_bin):
    """
    vector_field [2, h, w], which channel 0 cos(theta) and 1 sin(theta)
    return bin index [1, h, w]
    """
    angle = (torch.atan2(vector_field[1,:,:], vector_field[0,:,:])/np.pi*180 + 180) % 360 # [0,360)
    angle_bin = torch.round(torch.div(angle, (360 / (num_bin - 1)))).long()
    angle_bin[angle_bin == num_bin-1] = 0
    invalid = (vector_field==0).sum(0)==vector_field.size(0)
    angle_bin[invalid] = num_bin - 1
    return angle_bin.type(torch.LongTensor)


def decode_bin(angle_bin, num_bin):
    angle = (angle_bin * (360 / (num_bin - 1)) - 180) / 180 * np.pi
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    vector_field = torch.stack((cos, sin), dim=0)
    invalid = angle_bin == num_bin - 1
    vector_field[:, invalid] = 0
    return vector_field


def draw_vector_field(vector_field):
    zero = torch.zeros((1, vector_field.size(1), vector_field.size(2)))
    normal = torch.cat((vector_field,zero),0)
    normal = (normal + 1.0) / 2.0 * 255.0
    return normal.long()


def encode_bin_latitude(latimap, num_classes):
    """
    latitude map is [-90, 90]
    """
    boundaries = torch.arange(-90, 90, 180 / num_classes)[1:]
    binmap = torch.bucketize(latimap, boundaries)
    # bin_size = 180 / (num_classes - 1)
    # pos = torch.ceil(latimap / bin_size)
    # neg = torch.floor(latimap / bin_size)
    # binmap = torch.zeros(latimap.shape)
    # binmap[latimap > 0] = pos[latimap > 0]
    # binmap[latimap < 0] = neg[latimap < 0]
    # binmap = binmap + (num_classes + 1) / 2 - 1
    return binmap.type(torch.LongTensor)
    

def decode_bin_latitude(binmap, num_classes):
    """
    decode bin to latitude map
    """
    bin_size = 180 / num_classes
    bin_centers = torch.arange(-90, 90, bin_size) + bin_size / 2
    bin_centers = bin_centers.to(binmap.device)
    latimap = bin_centers[binmap]
    # bin_size = 180 / (num_classes - 1)
    # latimap = (binmap - ((num_classes + 1) / 2 - 1)) * bin_size
    # latimap = latimap - bin_size / 2 * torch.sign(latimap)
    return latimap


def draw_perspective_fields(img_rgb, up, latimap, color=None, density=10, arrow_inv_len=20, return_img=True):
    """
    latitude in radians
    """
    visualizer = VisualizerPerspective(img_rgb[:,:,::-1].copy())
    vis_output = visualizer.draw_lati(latimap)
    if torch.is_tensor(up):
        up = up.numpy().transpose(1,2,0)
    im_h, im_w, _ = img_rgb.shape
    x, y = np.meshgrid(
        np.arange(0, im_w, im_w//density), 
        np.arange(0, im_h, im_h//density)
    )
    x, y = x.ravel(), y.ravel()
    start = np.stack((x,y))
    arrow_len = np.sqrt(im_w ** 2 + im_h ** 2) // arrow_inv_len
    end = (up[y, x, :] * arrow_len)
    if color is None:
        color = (0,1,0)
    vis_output = visualizer.draw_arrow(x, y, end[:, 0], -end[:, 1], color=color)
    if return_img:
        return vis_output.get_image()[:, :, ::-1]
    else:
        return vis_output


def draw_up_field(img_rgb, vector_field, color=None, density=10, arrow_inv_len=20, return_img=True):
    """
    return format: RGB
    """
    if torch.is_tensor(vector_field):
        vector_field = vector_field.numpy().transpose(1,2,0)
    visualizer = VisualizerPerspective(img_rgb[:,:,::-1].copy())
    im_h, im_w, _ = img_rgb.shape
    x, y = np.meshgrid(
        # np.arange(0, im_w, im_w//20), 
        # np.arange(0, im_h, im_h//20)
        np.arange(0, im_w, im_w//density), 
        np.arange(0, im_h, im_h//density)
    )
    x, y = x.ravel(), y.ravel()
    start = np.stack((x,y))
    arrow_len = np.sqrt(im_w ** 2 + im_h ** 2) // arrow_inv_len
    end = (vector_field[y, x, :] * arrow_len)
#     end = (vector_field[:, y, x] * 30).numpy()
    vis_output = visualizer.draw_arrow(x, y, end[:, 0], -end[:, 1], color=color)
    if return_img:
        return vis_output.get_image()[:, :, ::-1]
    else:
        return vis_output
    

def draw_from_r_p_f(img, roll, pitch, vfov, mode, up_color=None, 
    alpha_contourf=0.4, alpha_contour=0.9, 
    draw_up=True, draw_lat=True, lati_alpha=0.5):
    # lati_alpha is deprecated
    im_h, im_w, _ = img.shape
    if mode == 'deg':
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        vfov = np.radians(vfov)
    elif mode == 'rad':
        pass
    else:
        raise "Bad argument"
    lati_deg = PanoCam.get_lat(
        vfov=vfov,
        im_w=im_w,
        im_h=im_h,
        elevation=pitch,
        roll=roll,
    )
    up = PanoCam.get_up(
        vfov=vfov,
        im_w=im_w,
        im_h=im_h,
        elevation=pitch,
        roll=roll,
    )
    # up[lati_deg > 89] = 0
    # up[lati_deg < -89] = 0

    if draw_lat:
        img = draw_latitude_field(
            img, 
            np.radians(lati_deg), 
            alpha_contourf=alpha_contourf, 
            alpha_contour=alpha_contour,
        )    
    if draw_up:
        img = draw_up_field(img, up, color=up_color)
    return img


def draw_from_r_p_f_cx_cy(
    img, 
    roll, pitch, vfov, rel_cx, rel_cy, mode, 
    up_color=None, 
    alpha_contourf=0.4, alpha_contour=0.9, 
    draw_up=True, draw_lat=True):
    im_h, im_w, _ = img.shape
    if mode == 'deg':
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        vfov = np.radians(vfov)
    elif mode == 'rad':
        pass
    else:
        raise "Bad argument"
    rel_focal = general_vfov_to_focal(rel_cx, rel_cy, 1, vfov, False)
    lati_deg = PanoCam.get_lat_general(
        focal_rel=rel_focal,
        im_w=im_w,
        im_h=im_h,
        elevation=pitch,
        roll=roll,
        cx_rel=rel_cx,
        cy_rel=rel_cy,
    )
    up = PanoCam.get_up_general(
        focal_rel=rel_focal,
        im_w=im_w,
        im_h=im_h,
        elevation=pitch,
        roll=roll,
        cx_rel=rel_cx,
        cy_rel=rel_cy,
    )
    # up[lati_deg > 89] = 0
    # up[lati_deg < -89] = 0

    if draw_lat:
        img = draw_latitude_field(
            img, 
            np.radians(lati_deg), 
            alpha_contourf=alpha_contourf, 
            alpha_contour=alpha_contour,
        )    
    if draw_up:
        img = draw_up_field(img, up, color=up_color)
    return img



def draw_latitude_field(
    img_rgb, 
    latimap=None,
    binmap=None,
    alpha_contourf=0.4, 
    alpha_contour=0.9,
    return_img=True
    ):
    """
    latitude in radians
    """
    visualizer = VisualizerPerspective(img_rgb[:,:,::-1].copy())
    vis_output = visualizer.draw_lati(latimap, alpha_contourf, alpha_contour)
    if return_img:
        return vis_output.get_image()[:, :, ::-1]
    else:
        return vis_output




def draw_horizon_line(img, horizon, color, thickness=3):
    im_h, im_w, _ = img.shape
    output = img.copy()
    cv2.line(output, 
        (0, int(horizon[0] * im_h)), 
        (im_w, int(horizon[1] * im_h)), 
        color, thickness)
    return output


def draw_prediction_distribution(pred, gt):
    fig = plt.figure()
    plt.hexbin(gt, pred)
    plt.xlabel("gt")
    plt.ylabel("pred")
    plt.xlim(min(min(gt), min(pred)), max(max(gt), max(pred)))
    plt.ylim(min(min(gt), min(pred)), max(max(gt), max(pred)))
    plt.gca().set_aspect('equal', adjustable='box')
    canvas = FigureCanvasAgg(fig)

    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype="uint8")

    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb
