import numpy as np


def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    r"""Create Rotation Matrix
    params:
    - x: x-axis rotation float
    - y: y-axis rotation float
    - z: z-axis rotation float
    return:
    - rotation R_z @ R_x @ R_y
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