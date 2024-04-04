import cv2
import torch
import os
import numpy as np
from perspective2d import PerspectiveFields
from perspective2d.utils import draw_perspective_fields, draw_from_r_p_f_cx_cy



def log_results(img_rgb, pred, output_folder, param_on):
    """
    Save perspective field prediction visualizations.

    Args:
        img_rgb (np.ndarray): The input image in RGB format.
        pred (dict): The model predictions.
        output_folder (str): The path to save the visualizations to.
        param_on (bool): A flag indicating whether to include parameter predictions.

    Returns:
        None
    """
    def resize_fix_aspect_ratio(img, field, target_width=None, target_height=None):
        """
        Resize image and perspective field to target width or height while maintaining aspect ratio.
        """
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
            if key not in ["up", "lati"]:
                continue
            tmp = field[key].numpy()
            transpose = len(tmp.shape) == 3
            if transpose:
                tmp = tmp.transpose(1, 2, 0)
            tmp = cv2.resize(tmp, (target_width, target_height))
            if transpose:
                tmp = tmp.transpose(2, 0, 1)
            field[key] = torch.tensor(tmp)
        return img, field

    os.makedirs(output_folder, exist_ok=True)
    field = {
        "up": pred["pred_gravity_original"].cpu().detach(),
        "lati": pred["pred_latitude_original"].cpu().detach(),
    }
    img_rgb, field = resize_fix_aspect_ratio(img_rgb, field, 640)
    pred_vis = draw_perspective_fields(
        img_rgb, field["up"], torch.deg2rad(field["lati"]), color=(0,1,0), return_img=False
    )
    pred_vis.save(os.path.join(output_folder, "perspective_pred"))
    
    if not param_on:
        return

    # Draw perspective field from ParamNet predictions
    param_vis = draw_from_r_p_f_cx_cy(
        img_rgb,
        pred["pred_roll"].item(),
        pred["pred_pitch"].item(),
        pred["pred_general_vfov"].item(),
        pred["pred_rel_cx"].item(),
        pred["pred_rel_cy"].item(),
        "deg",
        up_color=(0, 1, 0),
    ).astype(np.uint8)

    param_vis = cv2.cvtColor(param_vis, cv2.COLOR_RGB2BGR)
    pred_roll = f"roll: {pred['pred_roll'].item() :.2f}"
    pred_pitch = f"pitch: {pred['pred_pitch'].item() :.2f}"
    pred_vfov = f"vfov: {pred['pred_general_vfov'].item() :.2f}"
    pred_cx = f"cx: {pred['pred_rel_cx'].item() :.2f}"
    pred_cy = f"cy: {pred['pred_rel_cy'].item() :.2f}"

    print(pred_roll)
    print(pred_pitch)
    print(pred_vfov)
    print(pred_cx)
    print(pred_cy)
    # Write parameter predictions on the visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    param_vis = cv2.putText(
        param_vis,
        pred_roll,
        (int(param_vis.shape[1] * 0.6) - 2, int(param_vis.shape[0] * 0.1)),
        font,
        font_scale,
        (0, 0, 255),
        2,
    )
    param_vis = cv2.putText(
        param_vis,
        pred_pitch,
        (int(param_vis.shape[1] * 0.6) - 2, int(param_vis.shape[0] * 0.1) + 25),
        font,
        font_scale,
        (0, 0, 255),
        2,
    )
    param_vis = cv2.putText(
        param_vis,
        pred_vfov,
        (int(param_vis.shape[1] * 0.6) - 2, int(param_vis.shape[0] * 0.1) + 50),
        font,
        font_scale,
        (0, 0, 255),
        2,
    )
    param_vis = cv2.putText(
        param_vis,
        pred_cx,
        (int(param_vis.shape[1] * 0.6) - 2, int(param_vis.shape[0] * 0.1) + 75),
        font,
        font_scale,
        (0, 0, 255),
        2,
    )
    param_vis = cv2.putText(
        param_vis,
        pred_cy,
        (int(param_vis.shape[1] * 0.6) - 2, int(param_vis.shape[0] * 0.1) + 100),
        font,
        font_scale,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(os.path.join(output_folder, "param_pred.png"), param_vis)


PerspectiveFields.versions()

version = 'Paramnet-360Cities-edina-centered'
# version = 'Paramnet-360Cities-edina-uncentered'
# version = 'PersNet_Paramnet-GSV-centered'
# version = 'PersNet_Paramnet-GSV-uncentered'
# version = 'PersNet-360Cities'
pf_model = PerspectiveFields(version).eval().cuda()
img_bgr = cv2.imread('assets/imgs/cityscape.jpg')
predictions = pf_model.inference(img_bgr=img_bgr)

log_results(img_bgr[..., ::-1], predictions, output_folder="debug", param_on=pf_model.param_on)

print("\nexpected output: ")
print("""roll: 4.54
pitch: 48.88
vfov: 52.82
cx: 0.00
cy: 0.00""")