"""
This module provides classes and functions for handling 3D transformations, specifically for sensors like
Lidar, Camera, IMU, and GNSS. It includes utilities to create transformations, combine and invert them,
and extract transformation parameters such as translation and rotation.

Classes:
    Transformation: Represents a 3D transformation with translation and rotation, providing methods to
                    combine and invert transformations.

Functions:
    get_transformation: Creates a Transformation object for a given sensor (Camera, Lidar, IMU, GNSS).
"""
from typing import Union
from aeifdataset.data import Lidar, Camera, IMU, GNSS
from scipy.spatial.transform import Rotation as R
import numpy as np


class Transformation:
    """Class representing a 3D transformation consisting of translation and rotation.

    This class provides utilities to manage transformations between different coordinate frames,
    including combining and inverting transformations.

    Attributes:
        at (str): The origin frame of the transformation.
        to (str): The destination frame of the transformation.
        translation (np.array): The translation vector (x, y, z).
        rotation (np.array): The rotation vector (roll, pitch, yaw) in radians.
        transformation_mtx (np.array): The 4x4 transformation matrix combining rotation and translation.
    """

    def __init__(self, at, to, x, y, z, roll, pitch, yaw):
        """Initialize the Transformation object.

        Args:
            at (str): The origin frame of the transformation.
            to (str): The destination frame of the transformation.
            x (float): X component of the translation vector.
            y (float): Y component of the translation vector.
            z (float): Z component of the translation vector.
            roll (float): Roll component of the rotation in radians.
            pitch (float): Pitch component of the rotation in radians.
            yaw (float): Yaw component of the rotation in radians.
        """
        self._at = at
        self._to = to
        self._translation = np.array([x, y, z], dtype=float)
        self._rotation = np.array([roll, pitch, yaw], dtype=float)
        self._update_transformation_matrix()

    @property
    def at(self):
        """str: The origin frame of the transformation."""
        return self._at

    @at.setter
    def at(self, value):
        self._at = value

    @property
    def to(self):
        """str: The destination frame of the transformation."""
        return self._to

    @to.setter
    def to(self, value):
        self._to = value

    @property
    def translation(self):
        """np.array: The translation vector (x, y, z)."""
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    @property
    def rotation(self):
        """np.array: The rotation vector (roll, pitch, yaw) in radians."""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    def _update_transformation_matrix(self):
        """Update the 4x4 transformation matrix based on the current translation and rotation."""
        rotation = R.from_euler('xyz', self._rotation, degrees=False)
        rotation_matrix = rotation.as_matrix()
        self.transformation_mtx = np.identity(4)
        self.transformation_mtx[:3, :3] = rotation_matrix
        self.transformation_mtx[:3, 3] = self._translation

    def combine_transformation(self, transformation_to):
        """Combine this transformation with another transformation.

        Args:
            transformation_to (Transformation): The transformation to combine with.

        Returns:
            Transformation: The new combined transformation.
        """
        second_transformation_mtx = transformation_to.transformation_mtx
        new_transformation_mtx = np.dot(second_transformation_mtx, self.transformation_mtx)

        translation_vector, euler_angles = Transformation.extract_translation_and_euler_from_matrix(
            new_transformation_mtx)
        x, y, z = translation_vector
        roll, pitch, yaw = euler_angles

        new_transformation = Transformation(self.at, transformation_to.to, x, y, z, roll, pitch, yaw)

        return new_transformation

    def invert_transformation(self):
        """Invert this transformation.

        Returns:
            Transformation: The inverted transformation.
        """
        inverse_transformation_matrix = np.linalg.inv(self.transformation_mtx)

        translation_vector, euler_angles = Transformation.extract_translation_and_euler_from_matrix(
            inverse_transformation_matrix)
        x, y, z = translation_vector
        roll, pitch, yaw = euler_angles

        inverse_transformation = Transformation(self.to, self.at, x, y, z, roll, pitch, yaw)

        return inverse_transformation

    @staticmethod
    def extract_translation_and_euler_from_matrix(mtx):
        """Extract translation vector and Euler angles from a 4x4 transformation matrix.

        Args:
            mtx (np.array): The 4x4 transformation matrix.

        Returns:
            tuple: A tuple containing the translation vector and Euler angles in radians.
        """
        # Extract the translation vector
        translation_vector = mtx[:3, 3]

        # Extract the rotation matrix and convert to Euler angles (radians)
        rotation_matrix = mtx[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        euler_angles_rad = rotation.as_euler('xyz', degrees=False)

        return translation_vector, euler_angles_rad

    def __repr__(self):
        """Return a string representation of the Transformation object."""
        translation_str = ', '.join(f"{coord:.3f}" for coord in self.translation)
        rotation_str = ', '.join(f"{angle:.3f}" for angle in self.rotation)
        return (f"Transformation at {self._at} to {self._to},\n"
                f"  translation=[{translation_str}],\n"
                f"  rotation=[{rotation_str}]\n")


def get_transformation(sensor: Union[Camera, Lidar, IMU, GNSS]) -> Transformation:
    """Create a Transformation object for a given sensor.

    Args:
        sensor (Union[Camera, Lidar, IMU, GNSS]): The sensor for which to create the transformation.

    Returns:
        Transformation: The transformation object for the given sensor.

    Raises:
        AssertionError: If the sensor is not a Camera, Lidar, IMU, or GNSS object.
    """
    # Assert that sensor is of the correct type
    assert isinstance(sensor, (Camera, Lidar, IMU, GNSS)), "sensor must be a Camera, Lidar, IMU, or GNSS object"
    if 'view' in getattr(sensor.info, 'name', ''):
        sensor_to = 'lidar_upper_platform/os_sensor'
    else:
        sensor_to = 'lidar_top/os_sensor'

    if isinstance(sensor, Camera):
        sensor_at = f'cam_{sensor.info.name}'
    elif isinstance(sensor, Lidar):
        if 'view' in getattr(sensor.info, 'name', ''):
            sensor_at = f'lidar_{sensor.info.name}'
        else:
            sensor_at = f'lidar_{sensor.info.name}/os_sensor'
    else:
        sensor_at = 'ins'

    x, y, z = sensor.info.extrinsic.xyz
    roll, pitch, yaw = sensor.info.extrinsic.rpy

    tf = Transformation(sensor_at, sensor_to, x, y, z, roll, pitch, yaw)
    return tf
