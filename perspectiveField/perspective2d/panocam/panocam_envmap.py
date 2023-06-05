import numpy as np
from imageio import imread, imsave
from envmap import EnvironmentMap, rotation_matrix
import cv2, os
import matplotlib.pyplot as plt
import io
import imageio
import random
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import normalize
from utils import create_rotation_matrix
import matplotlib.cm as cm


class PanoCam():
    def __init__(self, pano_path):
        self.pano_path = pano_path
        self.env = EnvironmentMap(pano_path, 'latlong')
        

    def get_image(self, vfov=85, im_w=640, im_h=480, azimuth=0, elevation=30, roll=0, ar=4./3., img_format='RGB'):
        """
        camera frame: x right, y down, z out.
        image frame: u right, v down, origin at top left
        """
        dcm = rotation_matrix(azimuth=azimuth/180*np.pi, # yaw
                            elevation=-elevation/180*np.pi, # pitch
                            roll=roll/180*np.pi)
        crop = self.env.project(vfov=vfov, # degrees
                        rotation_matrix=dcm,
                        ar=ar,
                        resolution=(im_w, im_h),
                        projection="perspective",
                        mode="normal")
        if img_format=='RGB':
            pass
        elif img_format == 'BGR':
            crop = crop[:,:,::-1]
        else:
            raise NotImplementedError
        horizon = self.getRelativeHorizonLineFromAngles(elevation/180*np.pi, roll/180*np.pi, vfov/180*np.pi, im_h, im_w)
        vvp = self.getRelativeVVP(elevation/180*np.pi, roll/180*np.pi, vfov/180*np.pi, im_h, im_w)
        return crop, horizon, vvp

    @staticmethod
    def get_latitude(vfov=85, im_w=640, im_h=480, azimuth=0, elevation=30, roll=0, colormap=None):
        focal_length = im_h / 2 / np.tan(vfov*np.pi/180./2)

        # Uniform sampling on the plane
        dy = np.linspace(-im_h / 2, im_h / 2, im_h)
        dx = np.linspace(-im_w / 2, im_w / 2, im_w)
        x, y = np.meshgrid(dx, dy)

        x, y = x.ravel(), y.ravel()
        f = np.ones_like(x) * focal_length
        p_im = np.stack((x,y,f))
        # theta = np.arctan(np.sqrt((x**2+y**2)) / focal_length)
        rotation_m = create_rotation_matrix(
            roll=roll/180*np.pi,
            pitch=elevation/180*np.pi,
            yaw=azimuth/180*np.pi,
        )

        p_world = np.linalg.inv(rotation_m) @ p_im
        l = - np.arctan(p_world[1,:] / np.sqrt(p_world[0,:]**2 + p_world[2,:]**2)) * 180 / np.pi
        if colormap is None:
            return l.reshape(im_h, im_w)
        else:
            cmap = getattr(cm, colormap)
            return cmap((l / 90 + 1) / 2).reshape(im_h, im_w, 4)

    @staticmethod
    def getGravityField(crop, horizon, vvp):
        im_h, im_w, im_c = crop.shape
        assert(im_c == 3)
        if not np.isinf(vvp).any():
            # VVP
            vvp_abs = np.array([vvp[0] * im_w, vvp[1] * im_h])
        # arrow
        gridx, gridy = np.meshgrid(
            np.arange(0, im_w, ), 
            np.arange(0, im_h, )
        )
        if not np.isinf(vvp).any():
            start = np.stack((gridx.reshape(-1), gridy.reshape(-1))).T
            arrow = normalize(vvp_abs - start) * vvp[2]
            arrow_map = arrow.reshape(im_h, im_w, 2)
        else:
            arrow = normalize(np.array([[im_h*(horizon[1]-horizon[0]), -im_w]]))
            arrow_map = np.ones((im_h, im_w, 1)) * arrow
        return arrow_map
        

    @staticmethod
    def getRelativeVVP(elevation, roll, vfov, im_h, im_w):
        """
        elevation: x
        roll: z
        output: vertical vanishing point, divided by image height
        """
        if elevation == 0:
            return np.inf, np.inf
        vx =  0.5 - 0.5 * np.sin(roll) / np.tan(elevation) / np.tan(vfov/2) * im_h / im_w
        vy =  0.5 - 0.5 * np.cos(roll) / np.tan(elevation) / np.tan(vfov/2)
        return vx, vy, np.sign(elevation)


    @staticmethod
    def getRelativeHorizonLineFromAngles(elevation, roll, vfov, im_h, im_w):
        """ 
        elevation: x
        roll: z
        output: in image frame, fraction of image left/right border intersection with respect to image height
        """
        midpoint = PanoCam.getMidpointFromAngle(elevation, roll, vfov)
        dh = PanoCam.getDeltaHeightFromRoll(elevation, roll, im_h, im_w)
        return midpoint - dh, midpoint + dh


    @staticmethod
    def getMidpointFromAngle(elevation, roll, vfov):
        if elevation == np.pi/2 or elevation == -np.pi/2:
            return np.inf * np.sign(elevation)
        return 0.5 + 0.5 * np.tan(elevation) / np.cos(roll) / np.tan(vfov/2)


    @staticmethod
    def getDeltaHeightFromRoll(elevation, roll, im_h, im_w):
        "The height distance of horizon from the midpoint at image left/right border intersection."
        if roll == np.pi/2 or roll == -np.pi/2:
            return np.inf * np.sign(roll)
        return - im_w / im_h * np.tan(roll) / 2


def draw_vanishing_opencv(img, horizon, vvp, pad=(1,1), elevation=0, roll=0, azimuth=0, vfov=30):
    im_h, im_w, im_c = img.shape
    canvas = np.ones((im_h*(pad[0]*2+1), im_w*(pad[1]*2+1), im_c))
    offset_h = pad[0] * im_h
    offset_w = pad[1] * im_w
    canvas[offset_h:offset_h+im_h, offset_w:offset_w+im_w, :] = img

    # Horizon
    if not np.isinf(horizon).any():
        cv2.line(canvas, 
            (int(offset_w), int(offset_h + horizon[0] * im_h)), 
            (int(offset_w + im_w), int(offset_h + horizon[1] * im_h)), 
            (1,0,0), 3)

    if not np.isinf(vvp).any():
        # VVP
        vvp_abs = np.array([vvp[0] * im_w + offset_w, vvp[1] * im_h + offset_h])
        cv2.circle(canvas, 
            (int(vvp_abs[0]), int(vvp_abs[1])),
            5, (1,0,0), -1)

    # arrow
    gridx, gridy = np.meshgrid(
        np.arange(offset_w, offset_w+im_w+20, 20), 
        np.arange(offset_h, offset_h+im_h+20, 20)
    )

    start = np.stack((gridx.reshape(-1), gridy.reshape(-1))).T

    if not np.isinf(vvp).any():
        arrow = normalize(vvp_abs - start) * vvp[2] * 30
    else:
        arrow = normalize(np.array([[im_h*(horizon[1]-horizon[0]), -im_w]])) * 30
    end = start + arrow

    start = start.astype(int)
    end = end.astype(int)
    for i in range(len(start)):
        cv2.arrowedLine(canvas, start[i], end[i], 
                        (0,1,0), thickness=1, tipLength = 0.1)

    canvas = (255*canvas).astype(np.uint8)
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGR)
    # cv2.imwrite(os.path.join(save_path, prefix+'.png'), canvas)
    return canvas


def blend_color(img, color, alpha=0.2):
    if img.dtype == 'uint8':
        foreground = img[:, :, :3]
    else:
        foreground = img[:, :, :3] * 255

    if color.dtype == 'uint8':
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
    


def random_plot():
    save_path = '/home/code-base/user_space/public_html/360cities/e05_vanishing_random'
    os.makedirs(save_path, exist_ok=True)
    pano_paths = list(sorted(glob('data/10000x5000/*.jpg')))
    pano_paths = np.random.choice(pano_paths, 10)
    for pano_path in tqdm(pano_paths):
        cam = PanoCam(pano_path)
        img_id = os.path.basename(pano_path).split('.')[0]

        rand = random.randint(0,3)
        
        if rand % 4 == 0:
            # change r
            # r = random.randint(-45, 45)
            p = random.randint(-90, 90)
            y = random.randint(-180, 180)
            fov = random.randint(30, 120)
            writer = imageio.get_writer(os.path.join(save_path, 'r_' + img_id + f'_p{p}_y{y}f{fov}.mp4'), fps=10)
            for r in np.arange(-45,50,5):
                crop, horizon, vvp = cam.get_image(elevation=p, roll=r, azimuth=y, vfov=fov)
                im = draw_vanishing_opencv(crop.copy(), horizon, vvp, elevation=p, roll=r, azimuth=y, vfov=fov)
                writer.append_data(im)
            writer.close()
        elif rand % 4 == 1:
            # change p
            r = random.randint(-45, 45)
            # p = random.randint(-90, 90)
            y = random.randint(-180, 180)
            fov = random.randint(30, 120)
            writer = imageio.get_writer(os.path.join(save_path, 'p_' + img_id + f'_r{r}y{y}f{fov}.mp4'), fps=10)
            for p in np.arange(-90,95,5):
                crop, horizon, vvp = cam.get_image(elevation=p, roll=r, azimuth=y, vfov=fov)
                im = draw_vanishing_opencv(crop.copy(), horizon, vvp, elevation=p, roll=r, azimuth=y, vfov=fov)
                writer.append_data(im)
            writer.close()
        elif rand % 4 == 2:
            # change y
            r = random.randint(-45, 45)
            p = random.randint(-90, 90)
            # y = random.randint(-180, 180)
            fov = random.randint(30, 120)
            writer = imageio.get_writer(os.path.join(save_path, 'y_' + img_id + f'_r{r}p{p}f{fov}.mp4'), fps=10)
            for y in np.arange(-180, 180, 30):
                crop, horizon, vvp = cam.get_image(elevation=p, roll=r, azimuth=y, vfov=fov)
                im = draw_vanishing_opencv(crop.copy(), horizon, vvp, elevation=p, roll=r, azimuth=y, vfov=fov)
                writer.append_data(im)
            writer.close()
        elif rand % 4 == 3:
            # change f
            r = random.randint(-45, 45)
            p = random.randint(-90, 90)
            y = random.randint(-180, 180)
            # fov = random.randint(30, 120)
            writer = imageio.get_writer(os.path.join(save_path, 'f_' + img_id + f'_r{r}p{p}y{y}.mp4'), fps=10)
            for fov in np.arange(30, 130, 10):
                crop, horizon, vvp = cam.get_image(elevation=p, roll=r, azimuth=y, vfov=fov)
                im = draw_vanishing_opencv(crop.copy(), horizon, vvp, elevation=p, roll=r, azimuth=y, vfov=fov)
                writer.append_data(im)
            writer.close()


def pano_grid():
    save_path = './debug2envmap'
    os.makedirs(save_path, exist_ok=True)
    # pano_path = 'example/pano_grid.png'
    pano_path = 'example/pano_grid2.jpg'
    cam = PanoCam(pano_path)
    img_id = os.path.basename(pano_path).split('.')[0]
    im_w=640
    im_h=640
    ar = im_w / im_h
    vfov = 60
    for p in np.arange(-20,20.5,0.5):
        writer = imageio.get_writer(os.path.join(save_path, '_' + img_id + f'_p{p}.mp4'), fps=10)
        for r in np.arange(-45,50,5):
            latimap = cam.get_latitude(vfov=vfov, im_w=im_w, im_h=im_h, elevation=p, roll=r, colormap='tab20b')
            crop, horizon, vvp = cam.get_image(vfov=vfov, im_w=im_w, im_h=im_h, elevation=p, roll=r, ar=ar, img_format='BGR')
            # im = draw_vanishing_opencv(crop.copy(), horizon, vvp)
            im = blend_color(crop.copy(), latimap)
            # cv2.imwrite("debug.png", im[:,:,::-1])
            # cv2.imwrite("latitude.png", crop[:,:,::-1])
            # import pdb;pdb.set_trace()
            writer.append_data(im)
        writer.close()
        break
    for r in np.arange(-30,40,2):
        writer = imageio.get_writer(os.path.join(save_path, '_' + img_id + f'_r{r}.mp4'), fps=10)
        for p in np.arange(-90,95,5):
            latimap = cam.get_latitude(vfov=vfov, im_w=im_w, im_h=im_h, elevation=p, roll=r, colormap='tab20b')
            crop, horizon, vvp = cam.get_image(vfov=vfov, im_w=im_w, im_h=im_h, elevation=p, roll=r, ar=ar, img_format='BGR')
            # im = draw_vanishing_opencv(crop.copy(), horizon, vvp)
            im = blend_color(crop.copy(), latimap)
            # cv2.imwrite("debug.png", im[:,:,::-1])
            # cv2.imwrite("latitude.png", crop[:,:,::-1])
            # import pdb;pdb.set_trace()
            writer.append_data(im)
        writer.close()
        break    


def debugCoord():
    """
    Test coordinate system of the world frame.
    Result:
    Left hand frame with x left, y down, z out
    """
    save_path = '/home/code-base/user_space/public_html/360cities/e00_calibrate_camera'
    os.makedirs(save_path, exist_ok=True)
    pano_path = 'data/10000x5000/59511.jpg'
    env = EnvironmentMap(pano_path, 'latlong')
    img_id = os.path.basename(pano_path).split('.')[0]

    vfov=85;im_w=640;im_h=480

    # azimuth
    writer = imageio.get_writer(os.path.join(save_path, img_id + f'_azimuth.mp4'), fps=10)
    for azimuth in np.arange(-180,180,10):
        dcm = rotation_matrix(azimuth=azimuth/180*np.pi, # yaw
                            elevation=0, # pitch
                            roll=0)
        crop = env.project(vfov=vfov, # degrees
                        rotation_matrix=dcm,
                        ar=4./3.,
                        resolution=(im_w, im_h),
                        projection="perspective",
                        mode="normal")
        writer.append_data(crop)
    writer.close()

    # roll
    writer = imageio.get_writer(os.path.join(save_path, img_id + f'_roll.mp4'), fps=10)
    for roll in np.arange(-90,90,10):
        dcm = rotation_matrix(azimuth=0, # yaw
                            elevation=0, # pitch
                            roll=roll/180*np.pi)
        crop = env.project(vfov=vfov, # degrees
                        rotation_matrix=dcm,
                        ar=4./3.,
                        resolution=(im_w, im_h),
                        projection="perspective",
                        mode="normal")
        writer.append_data(crop)
    writer.close()

    # pitch
    writer = imageio.get_writer(os.path.join(save_path, img_id + f'_elevation.mp4'), fps=10)
    for elevation in np.arange(-90,90,10):
        dcm = rotation_matrix(azimuth=0, # yaw
                            elevation=elevation/180*np.pi, # pitch
                            roll=0)
        crop = env.project(vfov=vfov, # degrees
                        rotation_matrix=dcm,
                        ar=4./3.,
                        resolution=(im_w, im_h),
                        projection="perspective",
                        mode="normal")
        writer.append_data(crop)
    writer.close()


if __name__=='__main__':
    # debugCoord()
    pano_grid()
    # random_plot()