"""
This module provides functionality for handling 3D transformations, especially for sensors such as
Lidar, Camera, IMU, and GNSS. It includes classes and functions to create, combine, and invert transformations,
as well as to extract parameters like translation and rotation.

Classes:
    Transformation: Represents a 3D transformation consisting of translation and rotation, providing methods
                    to combine and invert transformations.

Functions:
    get_transformation: Creates a Transformation object for a given sensor (Camera, Lidar, IMU, GNSS).
    transform_points_to_origin: Transforms LiDAR points to the origin of the associated agent.
"""
from typing import Union, Tuple
from aeifdataset.data import Lidar, Camera, IMU, GNSS, Dynamics, CameraInformation, LidarInformation, GNSSInformation, \
    IMUInformation, DynamicsInformation
from scipy.spatial.transform import Rotation as R
import numpy as np


class Transformation:
    """Class representing a 3D transformation consisting of translation and rotation.

    Attributes:
        at (str): The origin frame of the transformation.
        to (str): The destination frame of the transformation.
        transformation_mtx (np.array): The 4x4 transformation matrix combining rotation and translation.
    """

    def __init__(self, at: str, to: str, transformation_mtx: np.array):
        """Initialize the Transformation object.

        Args:
            at (str): The origin frame of the transformation.
            to (str): The destination frame of the transformation.
            transformation_mtx (np.array): The 4x4 transformation matrix.
        """
        self.at = at
        self.to = to
        self.mtx = transformation_mtx

    @classmethod
    def from_xyz_and_rpy(cls, at: str, to: str, xyz: np.array, rpy: np.array):
        """Create a Transformation object from translation (xyz) and rotation (rpy).

        Args:
            at (str): The origin frame of the transformation.
            to (str): The destination frame of the transformation.
            xyz (np.array): Translation vector [x, y, z].
            rpy (np.array): Rotation vector [roll, pitch, yaw] in radians.

        Returns:
            Transformation: A new Transformation object.
        """
        # Erstelle die 4x4 Transformationsmatrix aus xyz und rpy
        rotation_matrix = R.from_euler('xyz', rpy, degrees=False).as_matrix()
        transformation_mtx = np.identity(4)
        transformation_mtx[:3, :3] = rotation_matrix
        transformation_mtx[:3, 3] = xyz
        return cls(at, to, transformation_mtx)

    @property
    def translation(self) -> np.array:
        """Get or set the translation vector (x, y, z)."""
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    @property
    def rotation(self) -> np.array:
        """Get or set the rotation vector (roll, pitch, yaw) in radians."""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    @property
    def mtx(self) -> np.array:
        """Get or set the 4x4 transformation matrix."""
        return self._transformation_mtx

    @mtx.setter
    def mtx(self, value):
        if value.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")
        self._transformation_mtx = value.copy()
        self._extract_translation_and_rotation()

    def _update_transformation_matrix(self):
        """Update the 4x4 transformation matrix based on current translation and rotation."""
        rotation = R.from_euler('xyz', self._rotation, degrees=False)
        rotation_matrix = rotation.as_matrix()
        self._transformation_mtx = np.identity(4)
        self._transformation_mtx[:3, :3] = rotation_matrix
        self._transformation_mtx[:3, 3] = self._translation

    def _extract_translation_and_rotation(self):
        """Extract translation and rotation from the transformation matrix."""
        self._translation = self._transformation_mtx[:3, 3]
        rotation_matrix = self._transformation_mtx[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        self._rotation = rotation.as_euler('xyz', degrees=False)

    def combine_transformation(self, transformation_to):
        """Combine this transformation with another transformation.

        Args:
            transformation_to (Transformation): The transformation to combine with.

        Returns:
            Transformation: The new combined transformation.
        """
        new_transformation_mtx = np.dot(transformation_to.mtx, self.mtx)
        return Transformation(self.at, transformation_to.to, transformation_mtx=new_transformation_mtx)

    def invert_transformation(self):
        """Invert this transformation.

        Returns:
            Transformation: The inverted transformation.
        """
        inverse_mtx = np.linalg.inv(self.mtx)
        return Transformation(self.to, self.at, transformation_mtx=inverse_mtx)

    def __repr__(self):
        """Return a string representation of the Transformation object."""
        translation_str = ', '.join(f"{coord:.3f}" for coord in self.translation)
        rotation_str = ', '.join(f"{angle:.3f}" for angle in self.rotation)
        return (f"Transformation at {self.at} to {self.to},\n"
                f"  translation=[{translation_str}],\n"
                f"  rotation=[{rotation_str}]\n")


def get_transformation(sensor_info: Union[
    Camera, Lidar, IMU, GNSS, CameraInformation, LidarInformation, IMUInformation, GNSSInformation]) -> Transformation:
    """Create a Transformation object for a given sensor or its corresponding information object.

    Args:
        sensor_info (Union[Camera, Lidar, IMU, GNSS, CameraInformation, LidarInformation, IMUInformation, GNSSInformation]):
            Either a sensor object (Camera, Lidar, IMU, GNSS) or directly the sensor's information object.

    Returns:
        Transformation: The transformation object for the given sensor or sensor information.

    Raises:
        ValueError: If Dynamics or DynamicsInformation is passed, as they are not supported.
    """
    if hasattr(sensor_info, 'info'):
        sensor_info = sensor_info.info

    if isinstance(sensor_info, (Dynamics, DynamicsInformation)):
        raise ValueError(
            "Dynamics and DynamicsInformation are not supported for this function yet. \
             Create your own Transformation object off the correct sensor until implemented.")

    if 'view' in getattr(sensor_info, 'name', ''):
        sensor_to = 'lidar_upper_platform/os_sensor'
    else:
        sensor_to = 'lidar_top/os_sensor'

    if isinstance(sensor_info, CameraInformation):
        sensor_at = f'cam_{sensor_info.name}'
    elif isinstance(sensor_info, LidarInformation):
        if 'view' in getattr(sensor_info, 'name', ''):
            sensor_at = f'lidar_{sensor_info.name}'
        else:
            sensor_at = f'lidar_{sensor_info.name}/os_sensor'
    else:
        sensor_at = 'ins'

    tf = Transformation(sensor_at, sensor_to, sensor_info.extrinsic)
    return tf


def transform_points_to_origin(data: Union[Lidar, Tuple[np.ndarray, LidarInformation]]) -> np.ndarray:
    """Transforms LiDAR points to the origin of the associated agent.

    This function takes either a LiDAR sensor object or a tuple containing LiDAR points
    and corresponding LiDAR information. It applies the transformation matrix of the
    associated agent (vehicle or tower) to convert the points into the coordinate frame
    of that agent. For vehicles, this is represented by the top LiDAR, and for towers,
    by the upper platform LiDAR.

    Args:
        data (Union[Lidar, Tuple[np.ndarray, LidarInformation]]): Either a LiDAR sensor object or a tuple containing
            a NumPy array of LiDAR points and LidarInformation.

    Returns:
        np.ndarray: A NumPy array containing the transformed 3D points in the associated agent's coordinate frame.
    """
    if isinstance(data, Lidar):
        points = data.points.points
        points = np.stack((points['x'], points['y'], points['z'], np.ones((points['x'].shape[0]))))
        lidar_info = data.info
    else:
        points, lidar_info = data
        points = np.stack((points[:, 0], points[:, 1], points[:, 2], np.ones((points.shape[0]))))

    # Get the transformation matrix and apply it
    trans = get_transformation(lidar_info)
    transformed_points = trans.mtx @ points

    return transformed_points.T[:, :3]
