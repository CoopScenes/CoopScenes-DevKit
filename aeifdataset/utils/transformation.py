from aeifdataset.data import Lidar, Camera, IMU, GNSS
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union


class Transformation:
    def __init__(self, at, to, x, y, z, roll, pitch, yaw):
        self._at = at
        self._to = to
        self._translation = np.array([x, y, z], dtype=float)
        self._rotation = np.array([roll, pitch, yaw], dtype=float)
        self._update_transformation_matrix()

    @property
    def at(self):
        return self._at

    @at.setter
    def at(self, value):
        self._at = value

    @property
    def to(self):
        return self._to

    @to.setter
    def to(self, value):
        self._to = value

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    def _update_transformation_matrix(self):
        rotation = R.from_euler('xyz', self._rotation, degrees=False)
        rotation_matrix = rotation.as_matrix()
        self.transformation_mtx = np.identity(4)
        self.transformation_mtx[:3, :3] = rotation_matrix
        self.transformation_mtx[:3, 3] = self._translation

    def combine_transformation(self, transformation_to):
        second_transformation_mtx = transformation_to.transformation_mtx
        new_transformation_mtx = np.dot(second_transformation_mtx, self.transformation_mtx)

        translation_vector, euler_angles = Transformation.extract_translation_and_euler_from_matrix(
            new_transformation_mtx)
        x, y, z = translation_vector
        roll, pitch, yaw = euler_angles

        new_transformation = Transformation(self.at, transformation_to.to, x, y, z, roll, pitch, yaw)

        return new_transformation

    def invert_transformation(self):
        inverse_transformation_matrix = np.linalg.inv(self.transformation_mtx)

        translation_vector, euler_angles = Transformation.extract_translation_and_euler_from_matrix(
            inverse_transformation_matrix)
        x, y, z = translation_vector
        roll, pitch, yaw = euler_angles

        inverse_transformation = Transformation(self.to, self.at, x, y, z, roll, pitch, yaw)

        return inverse_transformation

    @staticmethod
    def extract_translation_and_euler_from_matrix(mtx):
        # Extract the translation vector
        translation_vector = mtx[:3, 3]

        # Extract the rotation matrix and convert to Euler angles (radians)
        rotation_matrix = mtx[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        euler_angles_rad = rotation.as_euler('xyz', degrees=False)

        return translation_vector, euler_angles_rad

    def __repr__(self):
        translation_str = ', '.join(f"{coord:.3f}" for coord in self.translation)
        rotation_str = ', '.join(f"{angle:.3f}" for angle in self.rotation)
        return (f"Transformation at {self._at} to {self._to},\n"
                f"  translation=[{translation_str}],\n"
                f"  rotation=[{rotation_str}]\n")


def get_transformation(sensor: Union[Camera, Lidar, IMU, GNSS]) -> Transformation:
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
