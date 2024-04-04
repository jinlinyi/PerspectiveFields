import cv2
import equilib
import imageio
import numpy as np
from equilib import equi2pers
from sklearn.preprocessing import normalize

assert equilib.__version__ == "0.3.0"
from typing import Union

import torch
from equilib import grid_sample
from numpy.lib.scimath import sqrt as csqrt
from PIL import Image
from torchvision import transforms


def diskradius(xi, f):  # compute the disk radius when the image is catadioptric
    return np.sqrt(-(f * f) / (1 - xi * xi))


def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    r"""Create rotation matrix from extrinsic parameters
    Args:
        roll (float): camera rotation about camera frame z-axis
        pitch (float): camera rotation about camera frame x-axis
        yaw (float): camera rotation about camera frame y-axis

    Returns:
        np.ndarray: rotation R_z @ R_x @ R_y
    """
    # calculate rotation about the x-axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), np.sin(pitch)],
            [0.0, -np.sin(pitch), np.cos(pitch)],
        ]
    )
    # calculate rotation about the y-axis
    R_y = np.array(
        [
            [np.cos(yaw), 0.0, -np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    # calculate rotation about the z-axis
    R_z = np.array(
        [
            [np.cos(roll), np.sin(roll), 0.0],
            [-np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return R_z @ R_x @ R_y


def minfocal(u0, v0, xi, xref=1, yref=1):
    """compute the minimum focal for the image to be catadioptric given xi"""
    fmin = np.sqrt(
        -(1 - xi * xi) * ((xref - u0) * (xref - u0) + (yref - v0) * (yref - v0))
    )

    return fmin * 1.0001


def deg2rad(deg):
    """convert degrees to radians"""
    return deg * np.pi / 180


def preprocess(
    img: Union[np.ndarray, Image.Image],
    is_cv2: bool = False,
) -> torch.Tensor:
    """Convert img to tensor"""
    if isinstance(img, np.ndarray) and is_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img, Image.Image):
        # Sometimes images are RGBA
        img = img.convert("RGB")

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    img = to_tensor(img)
    assert len(img.shape) == 3, "input must be dim=3"
    assert img.shape[0] == 3, "input must be HWC"
    return img


def postprocess(
    img: torch.Tensor,
    to_cv2: bool = False,
) -> Union[np.ndarray, Image.Image]:
    """Convert img from tensor to image format"""
    if to_cv2:
        img = np.asarray(img.to("cpu").numpy() * 255, dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        to_PIL = transforms.Compose(
            [
                transforms.ToPILImage(),
            ]
        )
        img = img.to("cpu")
        img = to_PIL(img)
        return img


class PanoCam:
    def __init__(self, pano_path, device="cpu"):
        """initialize PanoCam model

        Args:
            pano_path (str): path to panorama image
            device (str, optional): type of device used to execute instructions. Defaults to "cpu".
        """
        self.pano_path = pano_path
        self.device = device

    def get_image(
        self,
        vfov=85,
        im_w=640,
        im_h=480,
        azimuth=0,
        elevation=30,
        roll=0,
        ar=4.0 / 3.0,
        img_format="RGB",
    ):
        """
        Crop a perspective image from an equirectangular image with specified camera parameters
        camera frame: x right, y down, z out.
        image frame: u right, v down, origin at top left

        Args:
            vfov (float): vertical field of view of cropped image (degrees)
            im_w (int): width of cropped image
            im_h (int): height of cropped image
            azimuth (float): camera rotation about camera frame y-axis of cropped image (degrees)
            elevation (float): camera rotation about camera frame x-axis of cropped image (degrees)
            roll (float): camera rotation about camera frame z-axis of cropped image (degrees)
            ar (float): aspect ratio of cropped image
            img_format (str): format to return image

        Returns:
            crop (np.ndarray): Cropped perspective image
            horizon (float, float): fraction of image left/right border intersection with respect to image height
            vvp (float, float, float): Vertical Vanishing Point, which is the vanishing point for the vertical lines. absvvp is in pixels, vvp is normalized by the image size.
        """
        equi_img = Image.open(self.pano_path)
        equi_img = preprocess(equi_img).to(self.device)
        fov_x = float(
            2 * np.arctan(np.tan(vfov * np.pi / 180.0 / 2) * ar) * 180 / np.pi
        )

        # Switch to https://github.com/haruishi43/equilib#coordinate-system
        rot = {
            "roll": float(roll / 180 * np.pi),
            "pitch": -float(elevation / 180 * np.pi),  # rotate vertical
            "yaw": -float(azimuth / 180 * np.pi),  # rotate horizontal
        }
        # Run equi2pers
        crop = equi2pers(
            equi=equi_img,
            rot=rot,
            w_pers=im_w,
            h_pers=im_h,
            fov_x=fov_x,
            skew=0.0,
            sampling_method="default",
            mode="bilinear",
        )
        crop = postprocess(crop, to_cv2=img_format == "BGR")

        horizon = self.getRelativeHorizonLineFromAngles(
            elevation / 180 * np.pi, roll / 180 * np.pi, vfov / 180 * np.pi, im_h, im_w
        )
        vvp = self.getRelativeVVP(
            elevation / 180 * np.pi, roll / 180 * np.pi, vfov / 180 * np.pi, im_h, im_w
        )
        return crop, horizon, vvp

    @staticmethod
    def crop_equi(equi_img, vfov, im_w, im_h, azimuth, elevation, roll, ar, mode):
        """
        Crop a perspective image from an equirectangular image with specified camera parameters
        camera frame: x right, y down, z out.
        image frame: u right, v down, origin at top left

        Args:
            equi_img (np.ndarray): equirectangular image
            vfov (float): vertical field of view of cropped image (degrees)
            im_w (int): width of cropped image
            im_h (int): height of cropped image
            azimuth (float): camera rotation about camera frame y-axis of cropped image (degrees)
            elevation (float): camera rotation about camera frame x-axis of cropped image (degrees)
            roll (float): camera rotation about camera frame z-axis of cropped image (degrees)
            ar (float): aspect ratio od cropped image
            mode (str): sampling mode for grid sample
        Returns:
            crop (np.ndarray): Cropped perspective image
        """
        fov_x = float(
            2 * np.arctan(np.tan(vfov * np.pi / 180.0 / 2) * ar) * 180 / np.pi
        )

        # Switch to https://github.com/haruishi43/equilib#coordinate-system
        rot = {
            "roll": float(roll / 180 * np.pi),
            "pitch": -float(elevation / 180 * np.pi),  # rotate vertical
            "yaw": -float(azimuth / 180 * np.pi),  # rotate horizontal
        }
        # Preprocess
        if len(equi_img.shape) == 3:
            equi_img_processed = equi_img.transpose(2, 0, 1)
        else:
            equi_img_processed = equi_img[None, :, :]
        equi_img_processed = torch.FloatTensor(equi_img_processed)

        # Run equi2pers
        crop = equi2pers(
            equi=equi_img_processed,
            rot=rot,
            w_pers=im_w,
            h_pers=im_h,
            fov_x=fov_x,
            skew=0.0,
            sampling_method="default",
            mode=mode,
        )
        if len(crop.shape) > 2:
            crop = np.asarray(crop.to("cpu").numpy(), dtype=equi_img.dtype)
            crop = np.transpose(crop, (1, 2, 0))
        else:
            crop = np.asarray(crop.to("cpu").numpy(), dtype=equi_img.dtype)
        return crop

    @staticmethod
    def getGravityField(im_h, im_w, absvvp):
        """
        Retrieve gravity field from absolute vertical vanishing point

        Args:
            im_h (int): image height
            im_w (int): image width
            absvvp ([float, float, float]): Absolute vertical vanishing point in image frame (top left corner as 0)

        Returns:
            np.ndarray: gravity field of shape (im_h, im_w, 2)
        """
        assert not np.isinf(absvvp).any()
        # arrow
        gridx, gridy = np.meshgrid(
            np.arange(0, im_w),
            np.arange(0, im_h),
        )
        start = np.stack((gridx.reshape(-1), gridy.reshape(-1))).T
        arrow = normalize(absvvp[:2] - start) * absvvp[2]
        arrow_map = arrow.reshape(im_h, im_w, 2)
        return arrow_map

    @staticmethod
    def getAbsVVP(im_h, im_w, horizon, vvp):
        """get absolute vertical vanishing point from horizon line and relative vertical vanishing point

        Args:
            im_h (int): image height
            im_w (int): image width
            horizon ([float, float]): fraction of image left/right border intersection with respect to image height
            vvp ([float, float, {-1, 1}]): relative vertical vanishing point, defined as vertical vanishing point divided by image height
        Returns:
            vvp_abs ([float, float, float]): absolute vertical vanishing point in image frame (top left corner as 0),
            vvp_abs[2] in {-1, 1} depending on if it is south or north pole, or if the up vectors are pointing towards (+1) or away (-1) from it.
        """
        if not np.isinf(vvp).any():
            # VVP
            vvp_abs = np.array([vvp[0] * im_w, vvp[1] * im_h])
            return np.array([vvp_abs[0], vvp_abs[1], vvp[2]])
        else:
            # approximate
            vvp_abs = (
                1e8
                * normalize(np.array([[im_h * (horizon[1] - horizon[0]), -im_w]]))[0]
            )
            return np.array(
                [vvp_abs[0] + 0.5 * im_w - 0.5, vvp_abs[1] + 0.5 * im_h - 0.5, 1]
            )

    @staticmethod
    def getRelativeVVP(elevation, roll, vfov, im_h, im_w):
        """Relative vertical vanishing point in image frame (top left corner as 0)
           Defined as vertical vanishing point divided by image height

        Args:
            elevation (float): camera rotation about camera frame x-axis (radians)
            roll (float): camera rotation about camera frame z-axis (radians)
            vfov (float): vertical field of view (radians)
            im_h (int): image height
            im_w (int): image width

        Returns:
            vvp[0] (float): x coordinate of vertical vanishing point, divided by image height.
            vvp[1] (float): y coordinate of vertical vanishing point, divided by image height.
            vvp[2] {-1, 1}: whether the up vectors are pointing towards (+1) or away (-1) from the vertical vanishing point.

        """
        if elevation == 0:
            return (
                np.inf,
                np.inf,
            )
        vx = (
            0.5
            - 0.5 / im_w
            - 0.5 * np.sin(roll) / np.tan(elevation) / np.tan(vfov / 2) * im_h / im_w
        )
        vy = (
            0.5 - 0.5 / im_h - 0.5 * np.cos(roll) / np.tan(elevation) / np.tan(vfov / 2)
        )
        return vx, vy, np.sign(elevation)

    @staticmethod
    def getRelativeHorizonLineFromAngles(elevation, roll, vfov, im_h, im_w):
        """Get relative horizon line from camera parameters

        Args:
            elevation (float): camera rotation about camera frame x-axis (radians)
            roll (float): camera rotation about camera frame z-axis (radians)
            vfov (float): vertical field of view (radians)
            im_h (int): image height
            im_w (int): image width

        Returns:
            (float, float): in image frame, fraction of image left/right border intersection with respect to image height
        """
        midpoint = PanoCam.getMidpointFromAngle(elevation, roll, vfov)
        dh = PanoCam.getDeltaHeightFromRoll(roll, im_h, im_w)
        return midpoint - dh, midpoint + dh

    @staticmethod
    def getMidpointFromAngle(elevation, roll, vfov):
        """get midpoint of the horizon line from roll, pitch, and vertical field of view

        Args:
            elevation (float): camera rotation about camera frame x-axis (radians)
            roll (float): camera rotation about camera frame z-axis (radians)
            vfov (float): vertical field of view (radians)

        Returns:
            float: location of the midpoint of the horizon line with respect to the image height
        """
        if elevation == np.pi / 2 or elevation == -np.pi / 2:
            return np.inf * np.sign(elevation)
        return 0.5 + 0.5 * np.tan(elevation) / np.cos(roll) / np.tan(vfov / 2)

    @staticmethod
    def getDeltaHeightFromRoll(roll, im_h, im_w):
        """
        Args:
            roll (float): camera rotation about camera frame z-axis (radians)
            im_h (int): image height
            im_w (int): image width

        Returns:
            float: the height distance of horizon from the midpoint at image left/right border intersection.
        """
        if roll == np.pi / 2 or roll == -np.pi / 2:
            return np.inf * np.sign(roll)
        return -im_w / im_h * np.tan(roll) / 2

    @staticmethod
    def get_lat(vfov, im_w, im_h, elevation, roll):
        """get latitude map from camera parameters

        Args:
            vfov (float): vertical field of view (radians)
            im_w (int): image width
            im_h (int): image height
            elevation (float): camera rotation about camera frame x-axis (radians)
            roll (float): camera rotation aboout camera frame z-axis (radians)

        Returns:
            np.ndarray: latitude map of shape (im_h, im_w) in degrees
        """
        focal_length = im_h / 2 / np.tan(vfov / 2)

        # Uniform sampling on the plane
        dy = np.linspace(-im_h / 2, im_h / 2, im_h)
        dx = np.linspace(-im_w / 2, im_w / 2, im_w)
        x, y = np.meshgrid(dx, dy)

        x, y = x.ravel() / focal_length, y.ravel() / focal_length
        focal_length = 1
        x_world = x * np.cos(roll) - y * np.sin(roll)
        y_world = (
            x * np.cos(elevation) * np.sin(roll)
            + y * np.cos(elevation) * np.cos(roll)
            - focal_length * np.sin(elevation)
        )
        z_world = (
            x * np.sin(elevation) * np.sin(roll)
            + y * np.sin(elevation) * np.cos(roll)
            + focal_length * np.cos(elevation)
        )
        l = -np.arctan2(y_world, np.sqrt(x_world**2 + z_world**2)) / np.pi * 180

        return l.reshape(im_h, im_w)

    @staticmethod
    def get_up(vfov, im_w, im_h, elevation, roll):
        """get gravity field from camera parameters

        Args:
            vfov (float): vertical field of view (radians)
            im_w (int): image width
            im_h (int): image height
            elevation (float): camera rotation about camera frame x-axis (radians)
            roll (float): camera rotation rotation aboout camera frame z-axis (radians)

        Returns:
            np.ndarray: gravity field of shape (im_h, im_w, 2)
        """
        horizon = PanoCam.getRelativeHorizonLineFromAngles(
            elevation=elevation, roll=roll, vfov=vfov, im_h=im_h, im_w=im_w
        )
        vvp = PanoCam.getRelativeVVP(
            elevation=elevation, roll=roll, vfov=vfov, im_h=im_h, im_w=im_w
        )
        absvvp = PanoCam.getAbsVVP(im_h=im_h, im_w=im_w, horizon=horizon, vvp=vvp)

        gridx, gridy = np.meshgrid(np.arange(0, im_w), np.arange(0, im_h))
        start = np.stack((gridx.reshape(-1), gridy.reshape(-1))).T
        arrow = normalize(absvvp[:2] - start) * absvvp[2]
        gt_up = arrow.reshape(im_h, im_w, 2)
        return gt_up

    @staticmethod
    def get_up_general(focal_rel, im_w, im_h, elevation, roll, cx_rel, cy_rel):
        """get gravity field from camera parameters.
           no assumptions about centered principal point.

        Args:
            focal_rel (float): relative focal length, defined as focal length divided by image height
            im_w (int): image width
            im_h (int): image height
            elevation (float): camera rotation about camera frame x-axis (radians)
            roll (float): rotation aboout z-axis (radians)
            cx_rel (float): relative cx location (pixel coordinate / image width - 0.5)
            cy_rel (float): relative cy location (pixel coordinate / image height - 0.5)

        Returns:
            np.ndarray: gravity field of shape (im_h, im_w, 2)
        """
        cx = (cx_rel + 0.5) * im_w
        cy = (cy_rel + 0.5) * im_h
        X = (
            np.linspace((-0.5 * im_w) + 0.5, (0.5 * im_w) - 0.5, im_w)
            .reshape(1, im_w)
            .repeat(im_h, 0)
            .astype(np.float32)
            + 0.5 * im_w
        )
        Y = (
            np.linspace((-0.5 * im_h) + 0.5, (0.5 * im_h) - 0.5, im_h)
            .reshape(im_h, 1)
            .repeat(im_w, 1)
            .astype(np.float32)
            + 0.5 * im_h
        )
        xy_cam = np.stack([X, Y], axis=2)
        focal_length = focal_rel * im_h

        if elevation == 0:
            up_vecs = np.ones(xy_cam.shape) * np.array(
                [[-np.sin(roll)], [-np.cos(roll)]]
            ).reshape((1, 2))
        else:
            vvp = np.array(
                [
                    [
                        (np.sin(roll) * np.cos(elevation) * focal_length)
                        / -np.sin(elevation)
                        + (cx)
                    ],
                    [
                        (np.cos(roll) * np.cos(elevation) * focal_length)
                        / -np.sin(elevation)
                        + (cy)
                    ],
                ]
            ).reshape((1, 2))
            up_vecs = vvp - xy_cam
            up_vecs = up_vecs * np.sign(elevation)

        up_vecs_norm = np.linalg.norm(up_vecs, axis=2)[:, :, None]
        up_vecs = up_vecs / up_vecs_norm
        return up_vecs

    @staticmethod
    def get_lat_general(focal_rel, im_w, im_h, elevation, roll, cx_rel, cy_rel):
        """get latitude map from camera parameters.
           no assumptions about centered principal point.

        Args:
            focal_rel (float): relative focal length, defined as focal length divided by image height
            im_w (int): image width
            im_h (int): image height
            elevation (float): camera rotation about camera frame x-axis (radians)
            roll (float): rotation aboout z-axis (radians)
            cx_rel (float): relative cx location (pixel coordinate / image width - 0.5)
            cy_rel (float): relative cy location (pixel coordinate / image height - 0.5)

        Returns:
            np.ndarray: latitude map of shape (im_h, im_w) in degrees
        """
        # Uniform sampling on the plane
        focal_length = focal_rel * im_h
        cx = (cx_rel + 0.5) * im_w
        cy = (cy_rel + 0.5) * im_h
        dy = np.linspace(
            (-im_h / 2) - (cy - (im_h / 2)), (im_h / 2) - (cy - (im_h / 2)), im_h
        )
        dx = np.linspace(
            (-im_w / 2) - (cx - (im_w / 2)), (im_w / 2) - (cx - (im_w / 2)), im_w
        )
        x, y = np.meshgrid(dx, dy)

        x, y = (x.ravel() / focal_length), (y.ravel() / focal_length)
        focal_length = 1
        x_world = x * np.cos(roll) - y * np.sin(roll)
        y_world = (
            x * np.cos(elevation) * np.sin(roll)
            + y * np.cos(elevation) * np.cos(roll)
            - focal_length * np.sin(elevation)
        )
        z_world = (
            x * np.sin(elevation) * np.sin(roll)
            + y * np.sin(elevation) * np.cos(roll)
            + focal_length * np.cos(elevation)
        )
        l = -np.arctan2(y_world, np.sqrt(x_world**2 + z_world**2)) / np.pi * 180

        return l.reshape(im_h, im_w)

    @staticmethod
    def crop_distortion(image360_path, f, xi, H, W, az, el, roll):
        """
        Reference: https://github.com/dompm/spherical-distortion-dataset/blob/main/spherical_distortion/spherical_distortion.py
        Crop distorted image with specified camera parameters

        Args:
            image360_path (str): path to image which to crop from
            f (float): focal_length of cropped image
            xi:
            H (int): height of cropped image
            W (int): width of cropped image
            az: camera rotation about camera frame y-axis of cropped image (degrees)
            el: camera rotation about camera frame x-axis of cropped image (degrees)
            roll: camera rotation about camera frame z-axis of cropped image (degrees)
        Returns:
            im (np.ndarray): cropped, distorted image
        """

        u0 = W / 2.0
        v0 = H / 2.0

        grid_x, grid_y = np.meshgrid(list(range(W)), list(range(H)))

        if isinstance(image360_path, str):
            image360 = imageio.imread(image360_path)  # .astype('float32') / 255.
        else:
            image360 = image360_path.copy()

        ImPano_W = np.shape(image360)[1]
        ImPano_H = np.shape(image360)[0]
        x_ref = 1
        y_ref = 1

        fmin = minfocal(
            u0, v0, xi, x_ref, y_ref
        )  # compute minimal focal length for the image to ve catadioptric with given xi

        # 1. Projection on the camera plane

        X_Cam = np.divide(grid_x - u0, f)
        Y_Cam = -np.divide(grid_y - v0, f)

        # 2. Projection on the sphere

        AuxVal = np.multiply(X_Cam, X_Cam) + np.multiply(Y_Cam, Y_Cam)

        alpha_cam = np.real(xi + csqrt(1 + np.multiply((1 - xi * xi), AuxVal)))

        alpha_div = AuxVal + 1

        alpha_cam_div = np.divide(alpha_cam, alpha_div)

        X_Sph = np.multiply(X_Cam, alpha_cam_div)
        Y_Sph = np.multiply(Y_Cam, alpha_cam_div)
        Z_Sph = alpha_cam_div - xi

        # 3. Rotation of the sphere
        coords = np.vstack((X_Sph.ravel(), Y_Sph.ravel(), Z_Sph.ravel()))
        rot_el = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                np.cos(deg2rad(el)),
                -np.sin(deg2rad(el)),
                0.0,
                np.sin(deg2rad(el)),
                np.cos(deg2rad(el)),
            ]
        ).reshape((3, 3))
        rot_az = np.array(
            [
                np.cos(deg2rad(az)),
                0.0,
                np.sin(deg2rad(az)),
                0.0,
                1.0,
                0.0,
                -np.sin(deg2rad(az)),
                0.0,
                np.cos(deg2rad(az)),
            ]
        ).reshape((3, 3))
        rot_roll = np.array(
            [
                np.cos(deg2rad(roll)),
                -np.sin(deg2rad(roll)),
                0.0,
                np.sin(deg2rad(roll)),
                np.cos(deg2rad(roll)),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ).reshape((3, 3))
        sph = rot_roll.T.dot(rot_el.dot(coords))
        sph = rot_az.dot(sph)

        sph = sph.reshape((3, H, W)).transpose((1, 2, 0))
        X_Sph, Y_Sph, Z_Sph = sph[:, :, 0], sph[:, :, 1], sph[:, :, 2]

        # 4. cart 2 sph
        ntheta = np.arctan2(X_Sph, Z_Sph)
        nphi = np.arctan2(Y_Sph, np.sqrt(Z_Sph**2 + X_Sph**2))

        pi = np.pi

        # 5. Sphere to pano
        min_theta = -pi
        max_theta = pi
        min_phi = -pi / 2.0
        max_phi = pi / 2.0

        min_x = 0
        max_x = ImPano_W - 1.0
        min_y = 0
        max_y = ImPano_H - 1.0

        ## for x
        a = (max_theta - min_theta) / (max_x - min_x)
        b = max_theta - a * max_x  # from y=ax+b %% -a;
        nx = (1.0 / a) * (ntheta - b)

        ## for y
        a = (min_phi - max_phi) / (max_y - min_y)
        b = max_phi - a * min_y  # from y=ax+b %% -a;
        ny = (1.0 / a) * (nphi - b)
        lat = nphi.copy()
        xy_map = np.stack((nx, ny)).transpose(1, 2, 0)

        # 6. Final step interpolation and mapping
        # im = np.array(my_interpol.interp2linear(image360, nx, ny), dtype=np.uint8)
        im = grid_sample.numpy_grid_sample.default(
            image360.transpose(2, 0, 1), np.stack((ny, nx))
        ).transpose(1, 2, 0)
        if (
            f < fmin
        ):  # if it is a catadioptric image, apply mask and a disk in the middle
            r = diskradius(xi, f)
            DIM = im.shape
            ci = (np.round(DIM[0] / 2), np.round(DIM[1] / 2))
            xx, yy = np.meshgrid(
                list(range(DIM[0])) - ci[0], list(range(DIM[1])) - ci[1]
            )
            mask = np.double((np.multiply(xx, xx) + np.multiply(yy, yy)) < r * r)
            mask_3channel = np.stack([mask, mask, mask], axis=-1).transpose((1, 0, 2))
            im = np.array(np.multiply(im, mask_3channel), dtype=np.uint8)

        col = nphi[:, W // 2]
        zero_crossings_rows = np.where(np.diff(np.sign(col)))[0]
        if len(zero_crossings_rows) >= 2:
            print("WARNING | Number of zero crossings:", len(zero_crossings_rows))
            zero_crossings_rows = [zero_crossings_rows[0]]

        if len(zero_crossings_rows) == 0:
            offset = np.nan
        else:
            assert col[zero_crossings_rows[0]] >= 0
            assert col[zero_crossings_rows[0] + 1] <= 0
            dy = col[zero_crossings_rows[0] + 1] - col[zero_crossings_rows[0]]
            offset = zero_crossings_rows[0] - col[zero_crossings_rows[0]] / dy
            assert col[zero_crossings_rows[0]] / dy <= 1.0
        # Reproject [nx, ny+epsilon] back
        epsilon = 1e-5
        end_vector_x = nx.copy()
        end_vector_y = ny.copy() - epsilon
        # -5. pano to Sphere
        a = (max_theta - min_theta) / (max_x - min_x)
        b = max_theta - a * max_x  # from y=ax+b %% -a;
        ntheta_end = end_vector_x * a + b
        ## for y
        a = (min_phi - max_phi) / (max_y - min_y)
        b = max_phi - a * min_y
        nphi_end = end_vector_y * a + b
        # -4. sph 2 cart
        Y_Sph = np.sin(nphi)
        X_Sph = np.cos(nphi_end) * np.sin(ntheta_end)
        Z_Sph = np.cos(nphi_end) * np.cos(ntheta_end)
        # -3. Reverse Rotation of the sphere
        coords = np.vstack((X_Sph.ravel(), Y_Sph.ravel(), Z_Sph.ravel()))
        sph = rot_el.T.dot(rot_roll.dot(rot_az.T.dot(coords)))
        sph = sph.reshape((3, H, W)).transpose((1, 2, 0))
        X_Sph, Y_Sph, Z_Sph = sph[:, :, 0], sph[:, :, 1], sph[:, :, 2]

        # -1. Projection on the image plane

        X_Cam = X_Sph * f / (xi * csqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2) + Z_Sph) + u0
        Y_Cam = -Y_Sph * f / (xi * csqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2) + Z_Sph) + v0
        up = np.stack((X_Cam - grid_x, Y_Cam - grid_y)).transpose(1, 2, 0)
        up = normalize(up.reshape(-1, 2)).reshape(up.shape)

        return im, ntheta, nphi, offset, up, lat, xy_map


def draw_vanishing_opencv(
    img, horizon, vvp, pad=(1, 1), elevation=0, roll=0, azimuth=0, vfov=30
):
    if img.dtype == "uint8":
        img = img.astype(float) / 255
    im_h, im_w, im_c = img.shape
    canvas = np.ones((im_h * (pad[0] * 2 + 1), im_w * (pad[1] * 2 + 1), im_c))
    offset_h = pad[0] * im_h
    offset_w = pad[1] * im_w
    canvas[offset_h : offset_h + im_h, offset_w : offset_w + im_w, :] = img

    # Horizon
    if not np.isinf(horizon).any():
        cv2.line(
            canvas,
            (int(offset_w), int(offset_h + horizon[0] * im_h)),
            (int(offset_w + im_w), int(offset_h + horizon[1] * im_h)),
            (1, 0, 0),
            3,
        )

    if not np.isinf(vvp).any():
        # VVP
        vvp_abs = np.array([vvp[0] * im_w + offset_w, vvp[1] * im_h + offset_h])
        cv2.circle(canvas, (int(vvp_abs[0]), int(vvp_abs[1])), 5, (1, 0, 0), -1)

    # arrow
    gridx, gridy = np.meshgrid(
        np.arange(offset_w, offset_w + im_w + 20, 20),
        np.arange(offset_h, offset_h + im_h + 20, 20),
    )

    start = np.stack((gridx.reshape(-1), gridy.reshape(-1))).T

    if not np.isinf(vvp).any():
        arrow = normalize(vvp_abs - start) * vvp[2] * 30
    else:
        arrow = normalize(np.array([[im_h * (horizon[1] - horizon[0]), -im_w]])) * 30
    end = start + arrow

    start = start.astype(int)
    end = end.astype(int)
    for i in range(len(start)):
        cv2.arrowedLine(canvas, start[i], end[i], (0, 1, 0), thickness=1, tipLength=0.1)

    canvas = (255 * canvas).astype(np.uint8)
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGR)
    # cv2.imwrite(os.path.join(save_path, prefix+'.png'), canvas)
    return canvas


def blend_color(img, color, alpha=0.2):
    if img.dtype == "uint8":
        foreground = img[:, :, :3]
    else:
        foreground = img[:, :, :3] * 255

    if color.dtype == "uint8":
        background = color[:, :, :3]
    else:
        background = color[:, :, :3] * 255

    alpha = np.ones_like(foreground) * alpha
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)

    outImage = outImage.astype(np.uint8)
    return outImage
