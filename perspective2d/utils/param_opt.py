import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from perspective2d.utils import general_vfov, general_vfov_to_focal


def get_latitude(R_world_from_cam, fltFocal, intWidth, intHeight, cx_rel, cy_rel):
    """get latitude map from camera rotation matrix and relative principal point location

    Args:
        R_world_from_cam (np.ndarray): camera rotation matrix
        fltFocal (float): focal length
        intWidth (int): image width
        intHeight (int): image height
        cx_rel (float): relative cx location (pixel location / image width - 0.5)
        cy_rel (float): relative cy location (pixel location / image height - 0.5)

    Returns:
        np.ndarray: latitude map of shape (1, intHeight, intWidth)
    """
    cx = (cx_rel + 0.5) * intWidth
    cy = (cy_rel + 0.5) * intHeight
    X = (
        np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
        .reshape(1, intWidth)
        .repeat(intHeight, 0)
        .astype(np.float32)
    )
    Y = (
        np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)
        .reshape(intHeight, 1)
        .repeat(intWidth, 1)
        .astype(np.float32)
    )
    f = np.ones_like(X)
    X = torch.from_numpy(X).to(R_world_from_cam.device) - (cx - (intWidth / 2))
    Y = torch.from_numpy(Y).to(R_world_from_cam.device) - (cy - (intHeight / 2))
    f = torch.from_numpy(f).to(R_world_from_cam.device)
    f = torch.mul(f, fltFocal)
    p_im = torch.stack((X, Y, f)).type(torch.FloatTensor).to(R_world_from_cam.device)
    p_world = (R_world_from_cam @ p_im.view(3, -1)).view(p_im.shape)
    l = -torch.arctan(
        p_world[1, :, :] / torch.sqrt(p_world[0, :, :] ** 2 + p_world[2, :, :] ** 2)
    )
    l = l[None, :, :]
    return l


def get_abs_vvp(R_world_from_cam, fltFocal, cx, cy):
    """get absolute vertical vanishing point from camera rotation matrix and principal point location

    Args:
        R_world_from_cam (np.ndarray): camera rotation matrix
        fltFocal (float): focal length
        cx (float): x-coordinate of principal point in image frame
        cy (float): y-coordinate of principal point in image frame

    Returns:
        tuple: absolute vertical vanishing point
        int: sign of camera pitch with respect to the world frame, {-1, 1}
    """
    gravity_world = torch.from_numpy(np.array([0, -1, 0]).astype(np.float32)).to(
        R_world_from_cam.device
    )
    gravity_cam = R_world_from_cam.transpose(0, 1) @ gravity_world
    # in case divide by 0
    if gravity_cam[2] == 0:
        gravity_cam[2] = gravity_cam[2] - 1e-6
    vvp_abs = ((fltFocal * gravity_cam) / gravity_cam[2:3])[:2]
    vvp_abs = vvp_abs + torch.tensor([cx, cy]).to(R_world_from_cam.device)

    sign_value = (
        R_world_from_cam
        @ torch.FloatTensor([0, 0, 1])
        .T.type(torch.FloatTensor)
        .to(R_world_from_cam.device)
    )[1]
    if sign_value == 0:
        elevation_sign = torch.sign(sign_value - 1e-6)
    else:
        elevation_sign = -torch.sign(sign_value)
    return vvp_abs, elevation_sign


def get_up(R_world_from_cam, fltFocal, intWidth, intHeight, cx_rel, cy_rel):
    """get gravity field from camera rotation matrix and relative principal point location

    Args:
        R_world_from_cam (np.ndarray): camera rotation matrix
        fltFocal (float): focal length
        intWidth (int): image width
        intHeight (int): image height
        cx_rel (float): relative cx location (pixel location / image width - 0.5)
        cy_rel (float): relative cy location (pixel location / image height - 0.5)

    Returns:
        torch.Tensor: gravity field of shape (2, intHeight, intWidth)
    """
    X = (
        np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
        .reshape(1, intWidth)
        .repeat(intHeight, 0)
        .astype(np.float32)
        + 0.5 * intWidth
    )
    Y = (
        np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)
        .reshape(intHeight, 1)
        .repeat(intWidth, 1)
        .astype(np.float32)
        + 0.5 * intHeight
    )
    xy_cam = np.stack([X, Y], axis=2)
    xy_cam = torch.from_numpy(xy_cam).to(R_world_from_cam.device)
    cx = (cx_rel + 0.5) * intWidth
    cy = (cy_rel + 0.5) * intHeight
    vvp_abs, elevation_sign = get_abs_vvp(R_world_from_cam, fltFocal, cx, cy)
    up = (
        torch.ones((intHeight, intWidth, 2)).to(R_world_from_cam.device) * vvp_abs
        - xy_cam
    ) * elevation_sign
    up = F.normalize(up, dim=2)
    up = up.permute(2, 0, 1)
    return up


def create_R_cam_from_world(
    roll,
    pitch,
):
    """create camera rotation matric from camera roll and pitch

    Args:
        roll (float): rotation of camera about the world frame z-axis (radians)
        pitch (float): rotation of camera about the worls frame x-axis (radians)

    Returns:
        np.ndarray: rotation R_z @ R_x @ R_y
    """
    # calculate rotation about the x-axis
    R_x = torch.stack(
        (
            torch.tensor([1.0, 0.0, 0.0]).to(roll.device),
            torch.cat(
                (
                    torch.tensor([0.0]).to(roll.device),
                    pitch.cos().view(1),
                    pitch.sin().view(1),
                )
            ),
            torch.cat(
                (
                    torch.tensor([0.0]).to(roll.device),
                    -pitch.sin().view(1),
                    pitch.cos().view(1),
                )
            ),
        ),
        dim=0,
    )
    R_z = torch.stack(
        (
            torch.cat(
                (
                    roll.cos().view(1),
                    roll.sin().view(1),
                    torch.tensor([0.0]).to(roll.device),
                )
            ),
            torch.cat(
                (
                    -roll.sin().view(1),
                    roll.cos().view(1),
                    torch.tensor([0.0]).to(roll.device),
                )
            ),
            torch.tensor([0.0, 0.0, 1.0]).to(roll.device),
        ),
        dim=0,
    )

    M = R_z @ R_x
    return M


class Model(nn.Module):
    def __init__(
        self,
        up_ref,
        lati_ref,
        device,
        init_params,
        pp_on,
        mask=None,
    ):
        """Initialize optimization model

        Args:
            up_ref (torch.Tensor): gravity field to optimize for. shape (2, h, w)
            lati_ref (np.ndarray): latitude map to optimize for. shape (1, h, w)
            device (str): device to place tensors on
            init_params (dict): dict of ParamNet predictions if model is to be initialized with these predictions.
                                None if using random initalization.
            pp_on (bool): indicates if testing on cropped or uncropped data
            mask (np.ndarray, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.up_ref = up_ref
        self.lati_ref = lati_ref[None, :, :]

        self.intHeight = intHeight = lati_ref.shape[0]
        self.intWidth = intWidth = lati_ref.shape[1]
        if mask is not None:
            self.mask = torch.tensor(mask[None, :, :]).to(device)
        else:
            self.mask = None

        # Create an optimizable parameter for the x, y, z position of the camera.
        if init_params is not None:
            init_r = torch.deg2rad(torch.tensor(init_params["roll"])).to(device)
            init_p = torch.deg2rad(torch.tensor(init_params["pitch"])).to(device)
            init_f = torch.tensor(init_params["focal"]).to(device)
            init_cx = torch.tensor(init_params["cx"]).to(device)
            init_cy = torch.tensor(init_params["cy"]).to(device)
        else:
            init_r = torch.tensor(
                -np.arctan2(
                    self.up_ref[0, int(intHeight / 2), int(intWidth / 2)],
                    -self.up_ref[1, int(intHeight / 2), int(intWidth / 2)],
                )
            ).to(device)
            init_p = torch.tensor(
                self.lati_ref[0, int(intHeight / 2), int(intWidth / 2)]
            ).to(device)
            init_vfov = np.abs(
                self.lati_ref[0, 0, int(intWidth / 2)]
                - self.lati_ref[0, int(intHeight - 1), int(intWidth / 2)]
            )
            init_f = torch.tensor(general_vfov_to_focal(0.0, 0.0, 1, init_vfov, False))
            init_cx = torch.tensor(0.0).to(device)
            init_cy = torch.tensor(0.0).to(device)

        self.cy = nn.Parameter(init_cy, requires_grad=pp_on)
        self.cx = nn.Parameter(init_cx, requires_grad=pp_on)
        self.roll = nn.Parameter(init_r, requires_grad=True)
        self.pitch = nn.Parameter(init_p, requires_grad=True)
        self.focal_rel = nn.Parameter(init_f, requires_grad=True)

    def forward(self, device):
        """Forward pass of optimization. Convert camera params to gravity field and latitude map.
           Compute loss with respect to self.up_ref and self.lati_ref

        Args:
            device (str): device to store tensors on

        Returns:
            torch.Tensor: total loss between gravity field generated by predicted camera params
                          and self.up_ref and self.lati_ref.
        """
        R_cam_from_world = create_R_cam_from_world(self.roll, self.pitch)
        R_world_from_cam = R_cam_from_world.transpose(0, 1)
        fltFocal = self.focal_rel * self.intHeight
        latimap = get_latitude(
            R_world_from_cam, fltFocal, self.intWidth, self.intHeight, self.cx, self.cy
        )
        up = get_up(
            R_world_from_cam, fltFocal, self.intWidth, self.intHeight, self.cx, self.cy
        )

        eps = 1e-7

        up_loss = torch.acos(
            torch.clip(
                torch.sum(up * torch.tensor(self.up_ref).to(device), dim=0),
                min=-1 + eps,
                max=1 - eps,
            )
        )
        lat_loss = torch.nn.functional.l1_loss(
            latimap, torch.tensor(self.lati_ref).to(device), reduction="none"
        )
        total_loss = (0.5 * up_loss) + (0.5 * lat_loss)
        if self.mask is not None:
            total_loss = torch.mean(total_loss[self.mask])
        else:
            total_loss = torch.mean(total_loss)

        return total_loss


def predict_rpfpp(up, latimap, tolerance, device, init_params, pp_on=None, mask=None):
    """Optimize camera params (roll, pitch, vfov, and optionally principal point)
       Camera params are cnverted to gravity field and latitude map in the forward pass.
       Loss is calculated with respect to up and latimap arguments and parameters are updated.
       torch.optim.Adam is the optimizer used with a learning rate of 0.0001

    Args:
        up (torch.Tensor): gravity field to optimize for. shape (2, h, w)
        latimap (np.ndarray): latitude map to optimize for. shape (1, h, w)
        tolerance (float): optimization exits if loss < tolerance
        device (str): device to store tensors on
        init_params (dict): dict of ParamNet predictions if model is to be initialized with these predictions.
                                None if using random initalization.
        pp_on (bool): indicates if testing on cropped or uncropped data
        mask (np.ndarray, optional): Can be used to mask out parts of the loss to ignore in the optimization.
                                     Defaults to None.

    Returns:
        dict: dict of optimized camera parameter predictions (angles in degrees)
    """
    model = Model(up, latimap, device, init_params, pp_on, mask).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    loop = range(1000)
    loss_prev = None
    for i in loop:
        optimizer.zero_grad()

        loss = model(device)
        loss.backward()

        optimizer.step()

        if loss.item() <= tolerance:
            print("EXITING OPTIMIZATION")
            print("Loss < ", tolerance)
            break
        if loss_prev is None:
            loss_prev = loss.item()
        else:
            if np.abs(loss.item() - loss_prev) < 1e-9:
                print("EXITING OPTIMIZATION")
                print("Loss - Loss Prev < ", 1e-9)
                break
            else:
                loss_prev = loss.item()

    pred_roll = model.roll.cpu().detach().item()
    pred_pitch = model.pitch.cpu().detach().item()
    pred_focal_rel = model.focal_rel.cpu().detach().item()
    pred_cx = model.cx.cpu().detach().item()
    pred_cy = model.cy.cpu().detach().item()

    del model
    return_dict = {
        "pred_roll": np.degrees(pred_roll),
        "pred_pitch": np.degrees(pred_pitch),
        "pred_rel_focal": pred_focal_rel,
        "pred_general_vfov": general_vfov(pred_cx, pred_cy, 1, pred_focal_rel, True),
        "pred_rel_cx": pred_cx,
        "pred_rel_cy": pred_cy,
    }
    return return_dict
