import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .panocam import PanoCam
from .visualizer import VisualizerPerspective


def general_vfov(d_cx, d_cy, h, focal, degree):
    """
    Calculate the general vertical field of view (gvfov) given the camera intrinsic parameters.

    The general vertical field of view (gvfov) is a concept employed to define the field of view (FoV) for images that may be cropped or have an off-center principal point.

    The gfov is defined as follows:
        Consider the camera's pinhole as 'O'. Let 'M1' and 'M2' represent the midpoints of the top and bottom edges of the image, respectively.
        The gfov is defined as the angle subtended by the lines OM1 and OM2 at 'O'.

    This function can handle parameters given in two ways:
    1. Relative to the image height: In this case, h should be 1, and d_cx, d_cy, and focal should be normalized by the image height.
    2. Absolute pixel values: In this case, h should be the image height in pixels, and d_cx, d_cy, and focal should be provided in pixels.

    Args:
        d_cx (float): Horizontal offset of the principal point (cx) from the image center.
        d_cy (float): Vertical offset of the principal point (cy) from the image center.
        h (float): Image height, either relative (1) or in absolute pixel values.
        focal (float): Focal length of the camera, either relative to the image height or in absolute pixel values.
        degree (bool): Indicator for the FoV return unit. If True, FoV is returned in degrees. If False, it's returned in radians.

    Returns:
        float: General vertical field of view (FoV), computed based on the provided parameters and returned in either degrees or radians, depending on the 'degree' parameter.
    """
    p_sqr = focal**2 + d_cx**2 + (d_cy + 0.5 * h) ** 2
    q_sqr = focal**2 + d_cx**2 + (d_cy - 0.5 * h) ** 2
    cos_FoV = (p_sqr + q_sqr - h**2) / 2 / np.sqrt(p_sqr) / np.sqrt(q_sqr)
    FoV_rad = np.arccos(cos_FoV)
    if degree:
        return np.degrees(FoV_rad)
    else:
        return FoV_rad


def general_vfov_to_focal(rel_cx, rel_cy, h, gvfov, degree):
    """
    Converts a given general vertical field of view (gvfov) to the equivalent focal length.

    The general vertical field of view (gvfov) is a concept employed to define the field of view (FoV) for images that may be cropped or have an off-center principal point.

    The gfov is defined as follows:
        Consider the camera's pinhole as 'O'. Let 'M1' and 'M2' represent the midpoints of the top and bottom edges of the image, respectively.
        The gfov is defined as the angle subtended by the lines OM1 and OM2 at 'O'.

    This function accepts parameters in either relative terms or absolute pixel values:
    1. Relative to the image height: In this case, h should be 1, and d_cx, d_cy should be normalized by the image height.
    2. Absolute pixel values: In this case, h should be the image height in pixels, and d_cx, d_cy should be provided in pixels.

    Args:
        rel_cx (float): Horizontal offset of the principal point (cx) from the image center.
                        It's in absolute terms if h is set to image height, else it's relative (cx coordinate / image width - 0.5).
        rel_cy (float): Vertical offset of the principal point (cy) from the image center.
                        It's in absolute terms if h is set to image height, else it's relative (cy coordinate / image height - 0.5).
        h (float): Image height, either in relative terms (set as 1) or as absolute pixel values.
        gvfov (float): General vertical field of view. It's in degrees if degree is set to True, else it's in radians.
        degree (bool): Indicator for the gvfov unit. If True, gvfov is assumed to be in degrees. If False, it's in radians.

    Returns:
        float: Focal length, derived from the input gvfov and the principal point offsets (rel_cx, rel_cy).
               It is relative to the image height if h is set to 1, else it's an absolute value (in pixels).
    """

    def fun(focal, *args):
        h, d_cx, d_cy, target_cos_FoV = args

        p_sqr = (focal / h) ** 2 + d_cx**2 + (d_cy + 0.5) ** 2
        q_sqr = (focal / h) ** 2 + d_cx**2 + (d_cy - 0.5) ** 2
        cos_FoV = (p_sqr + q_sqr - 1) / 2 / np.sqrt(p_sqr) / np.sqrt(q_sqr)
        return cos_FoV - target_cos_FoV
    if degree:
        gvfov = np.radians(gvfov)
    if type(rel_cx) != np.ndarray:
        # if input is float
        focal = scipy.optimize.fsolve(fun, 1.5, args=(h, rel_cx, rel_cy, np.cos(gvfov)))[0]
    else:
        # if input is numpy array
        focal = scipy.optimize.fsolve(fun, np.ones(len(rel_cx)) * 1.5, args=(h, rel_cx, rel_cy, np.cos(gvfov)))
    focal = np.abs(focal)
    return focal


def encode_bin(vector_field, num_bin):
    """encode vector field into classification bins

    Args:
        vector_field (np.ndarray): gravity field of shape (2, h, w), with channel 0 cos(theta) and 1 sin(theta)
        num_bin (int): number of classification bins

    Returns:
        np.ndarray: encoded bin indices of shape (1, h, w)
    """
    angle = (
        torch.atan2(vector_field[1, :, :], vector_field[0, :, :]) / np.pi * 180 + 180
    ) % 360  # [0,360)
    angle_bin = torch.round(torch.div(angle, (360 / (num_bin - 1)))).long()
    angle_bin[angle_bin == num_bin - 1] = 0
    invalid = (vector_field == 0).sum(0) == vector_field.size(0)
    angle_bin[invalid] = num_bin - 1
    return angle_bin.type(torch.LongTensor)


def decode_bin(angle_bin, num_bin):
    """decode classification bins into vector field

    Args:
        angle_bin (np.ndarray): bin indices of shape (1, h, 1)
        num_bin (int): number of classification bins

    Returns:
        np.ndarray: decoded vector field of shape (2, h, w)
    """
    angle = (angle_bin * (360 / (num_bin - 1)) - 180) / 180 * np.pi
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    vector_field = torch.stack((cos, sin), dim=0)
    invalid = angle_bin == num_bin - 1
    vector_field[:, invalid] = 0
    return vector_field


def encode_bin_latitude(latimap, num_classes):
    """encode latitude map into classification bins

    Args:
        latimap (np.ndarray): latitude map of shape (h, w) with values in [-90, 90]
        num_classes (int): number of classes

    Returns:
        np.ndarray: encoded latitude bin indices
    """
    boundaries = torch.arange(-90, 90, 180 / num_classes)[1:]
    binmap = torch.bucketize(latimap, boundaries)
    return binmap.type(torch.LongTensor)


def decode_bin_latitude(binmap, num_classes):
    """decode classification bins to latitude map

    Args:
        binmap (np.ndarray): encoded classification bins
        num_classes (int): number of classes

    Returns:
        np.ndarray: latitude map of shape (h, w)
    """
    bin_size = 180 / num_classes
    bin_centers = torch.arange(-90, 90, bin_size) + bin_size / 2
    bin_centers = bin_centers.to(binmap.device)
    latimap = bin_centers[binmap]
    return latimap


def draw_perspective_fields(
    img_rgb, up, latimap, color=None, density=10, arrow_inv_len=20, return_img=True
):
    """draw perspective field on top of input image

    Args:
        img_rgb (np.ndarray): input image
        up (np.ndarray): gravity field (h, w, 2)
        latimap (np.ndarray): latitude map (h, w) (radians)
        color ((float, float, float), optional): RGB color for up vectors. [0, 1]
                                                 Defaults to None.
        density (int, optional): Value to control density of up vectors.
                                 Each row has (width // density) vectors.
                                 Each column has (height // density) vectors.
                                 Defaults to 10.
        arrow_inv_len (int, optional): Value to control vector length
                                       Vector length set to (image plane diagonal // arrow_inv_len).
                                       Defaults to 20.
        return_img (bool, optional): bool to control if to return np array or VisImage

    Returns:
        image blended with perspective fields.
    """
    visualizer = VisualizerPerspective(img_rgb.copy())
    vis_output = visualizer.draw_lati(latimap)
    if torch.is_tensor(up):
        up = up.numpy().transpose(1, 2, 0)
    im_h, im_w, _ = img_rgb.shape
    x, y = np.meshgrid(
        np.arange(0, im_w, im_w // density), np.arange(0, im_h, im_h // density)
    )
    x, y = x.ravel(), y.ravel()
    start = np.stack((x, y))
    arrow_len = np.sqrt(im_w**2 + im_h**2) // arrow_inv_len
    end = up[y, x, :] * arrow_len
    if color is None:
        color = (0, 1, 0)
    vis_output = visualizer.draw_arrow(x, y, end[:, 0], -end[:, 1], color=color)
    if return_img:
        return vis_output.get_image()
    else:
        return vis_output


def draw_up_field(
    img_rgb, vector_field, color=None, density=10, arrow_inv_len=20, return_img=True
):
    """draw vector field on top of rgb image

    Args:
        img_rgb (np.ndarray): input rgb image
        vector_field (np.ndarray): gravity field of shape (h, w, 2)
        color ((float, float, float), optional): RGB color for up vectors. [0, 1]
                                                 Defaults to None.
        density (int, optional): Value to control density of up vectors.
                                 Each row has (width // density) vectors.
                                 Each column has (height // density) vectors.
                                 Defaults to 10.
        arrow_inv_len (int, optional): Value to control vector length
                                       Vector length set to (image plane diagonal // arrow_inv_len).
                                       Defaults to 20.
        return_img (bool, optional): bool to control if to return np array or VisImage

    Returns:
        image blended with up vectors
    """
    if torch.is_tensor(vector_field):
        vector_field = vector_field.numpy().transpose(1, 2, 0)
    visualizer = VisualizerPerspective(img_rgb.copy())
    im_h, im_w, _ = img_rgb.shape
    x, y = np.meshgrid(
        # np.arange(0, im_w, im_w//20),
        # np.arange(0, im_h, im_h//20)
        np.arange(0, im_w, im_w // density),
        np.arange(0, im_h, im_h // density),
    )
    x, y = x.ravel(), y.ravel()
    start = np.stack((x, y))
    arrow_len = np.sqrt(im_w**2 + im_h**2) // arrow_inv_len
    end = vector_field[y, x, :] * arrow_len
    #     end = (vector_field[:, y, x] * 30).numpy()
    vis_output = visualizer.draw_arrow(x, y, end[:, 0], -end[:, 1], color=color)
    if return_img:
        return vis_output.get_image()
    else:
        return vis_output


def draw_from_r_p_f(
    img,
    roll,
    pitch,
    vfov,
    mode,
    up_color=None,
    alpha_contourf=0.4,
    alpha_contour=0.9,
    draw_up=True,
    draw_lat=True,
    lati_alpha=0.5,
):
    """Draw latitude map and gravity field on top of input image.
       Generate latitude map and gravity field from camera parameters

    Args:
        img (np.ndarray): input rgb image
        roll (float): rotation of camera about the world frame z-axis
        pitch (float): rotation of camera about the world frame x-axis
        vfov (float): vertical field of view
        mode (str): specifies the mode of input parameters. "deg" or "rad"
        up_color ((float, float, float), optional): RGB value of up vectors. [0, 1]. Defaults to None.
        alpha_contourf (float, optional): value to control transparency of contour fill. Defaults to 0.4.
        alpha_contour (float, optional): value to control transparency of contour lines. Defaults to 0.9.
        draw_up (bool, optional): bool to specify if up vectors should be drawn. Defaults to True.
        draw_lat (bool, optional): bool to specify if latitude map should be drawn. Defaults to True.

    Returns:
        np.ndarray: img with up vectors drawn on (if draw_up == True)
                    and latitude map drawn on (if draw_lat == True)
    """
    # lati_alpha is deprecated
    im_h, im_w, _ = img.shape
    if mode == "deg":
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        vfov = np.radians(vfov)
    elif mode == "rad":
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
    roll,
    pitch,
    vfov,
    rel_cx,
    rel_cy,
    mode,
    up_color=None,
    alpha_contourf=0.4,
    alpha_contour=0.9,
    draw_up=True,
    draw_lat=True,
):
    """Draw latitude map and gravity field on top of input image.
       Generate latitude map and gravity field from camera parameters

    Args:
        img (np.ndarray): input image (RGB)
        roll (float): rotation of camera about the world frame z-axis
        pitch (float): rotation of camera about the world frame x-axis
        vfov (float): vertical field of view
        rel_cx (float): relative cx location (pixel location / image width - 0.5)
        rel_cy (float): relative cy location (pixel location / image height - 0.5)
        mode (str): specifies the mode of input parameters. "deg" or "radians"
        up_color ((float, float, float), optional): RGB value of up vectors. [0, 1]. Defaults to None.
        alpha_contourf (float, optional): value to control transparency of contour fill. Defaults to 0.4.
        alpha_contour (float, optional): value to control transparency of contour lines. Defaults to 0.9.
        draw_up (bool, optional): bool to specify if up vectors should be drawn. Defaults to True.
        draw_lat (bool, optional): bool to specify if latitude map should be drawn. Defaults to True.

    Returns:
        np.ndarray: rgb img with up vectors drawn on (if draw_up == True)
                    and latitude map drawn on (if draw_lat == True)

    """
    im_h, im_w, _ = img.shape
    if mode == "deg":
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        vfov = np.radians(vfov)
    elif mode == "rad":
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
    return_img=True,
):
    """draw latitude field on top of rgb image

    Args:
        img_rgb (np.ndarray): input rgb image
        latimap (np.ndarray, optional): latitude map in radians. Defaults to None.
        binmap: deprecated.
        alpha_contourf (float, optional): value to control transparency of contour fill. Defaults to 0.4.
        alpha_contour (float, optional): value to control transparenct of contour lines. Defaults to 0.9.
        return_img (bool, optional): bool to control if to return np array or VisImage

    Returns:
        np array or VisImage depending on return_img
    """
    visualizer = VisualizerPerspective(img_rgb.copy())
    vis_output = visualizer.draw_lati(latimap, alpha_contourf, alpha_contour)
    if return_img:
        return vis_output.get_image()
    else:
        return vis_output


def draw_horizon_line(img, horizon, color, thickness=3):
    """draw horizon line on image

    Args:
        img (np.ndarray): input image
        horizon (float, float): fraction of image left/right border intersection with respect to image height
        color (float, float, float): RGB color value for line. [0, 1]
        thickness (int, optional): line thickness in pixels. Defaults to 3.

    Returns:
        np.ndarray: image with horizon line drawn on it
    """
    im_h, im_w, _ = img.shape
    output = img.copy()
    cv2.line(
        output,
        (0, int(horizon[0] * im_h)),
        (im_w, int(horizon[1] * im_h)),
        color,
        thickness,
    )
    return output


def draw_prediction_distribution(pred, gt):
    """create 2D histogram of ground truth camera parameters vs. ParamNet predictions

    Args:
        pred (np.ndarray): ParamNet predictions
        gt (np.ndarray): ground truth parameters

    Returns:
        np.ndarray: 2D histogram
    """
    fig = plt.figure()
    plt.hexbin(gt, pred)
    plt.xlabel("gt")
    plt.ylabel("pred")
    plt.xlim(min(min(gt), min(pred)), max(max(gt), max(pred)))
    plt.ylim(min(min(gt), min(pred)), max(max(gt), max(pred)))
    plt.gca().set_aspect("equal", adjustable="box")
    canvas = FigureCanvasAgg(fig)

    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype="uint8")

    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb


def pf_postprocess(result, img_size, output_height, output_width):
    """
    Reference https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/postprocessing.py#L77C1-L100C18
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result
