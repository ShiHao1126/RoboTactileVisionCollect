import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def pose_matrix_to_quaternion(pose_matrix: np.ndarray) -> np.ndarray:
    position = pose_matrix[:3, 3]
    rotation_matrix = pose_matrix[:3, :3]
    rot = R.from_matrix(rotation_matrix)
    quaternion = rot.as_quat()  # [x, y, z, w]
    return np.array([position[0], position[1], position[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2]])  # [w, x, y, z]

def mat2xyzquat(pose_4x4:np.ndarray=np.eye(4)) -> Tuple[np.ndarray, np.ndarray]:
    xyz = pose_4x4[:3, 3]
    quat = R.from_matrix(pose_4x4[:3, :3]).as_quat()
    return (xyz, quat)

def delta_xyz(xyz1:np.ndarray, xyz2:np.ndarray) -> float:
    delta = xyz1 - xyz2
    return np.linalg.norm(delta)

def delta_quat(quat1:np.ndarray, quat2:np.ndarray) -> float:
    delta = R.from_quat(quat1).inv() * R.from_quat(quat2) # represented in quat1
    delta_value = delta.magnitude()
    return delta_value

def quat2eulerZYX(quat, degree=False):
    """
    Convert quaternion to Euler angles with ZYX axis rotations.

    Parameters
    ----------
    quat : float list
        Quaternion input in [w,x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.

    Returns
    ----------
    float list
        Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    """

    # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
    # NOTE: scipy uses [x,y,z,w] order to represent quaternion
    eulerZYX = (
        R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        .as_euler("xyz", degrees=degree)
        .tolist()
    )

    return eulerZYX