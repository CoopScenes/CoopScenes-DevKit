"""
This module provides functionality for handling 3D transformations and processing LiDAR data, especially for sensors such as
Lidar, Camera, IMU, and GNSS. It includes classes and functions to create, combine, and invert transformations,
as well as to extract parameters like translation and rotation. Additionally, the module provides methods for
deskewing LiDAR point clouds to account for motion distortion.

Classes:
    Transformation: Represents a 3D transformation consisting of translation and rotation, providing methods
        to combine and invert transformations.

Functions:
    get_transformation: Creates a Transformation object for a given sensor (Camera, Lidar, IMU, GNSS).
    transform_points_to_origin: Transforms LiDAR points to the origin of the associated agent or global coordinate system.
    get_deskewed_points: Deskews LiDAR points by compensating for motion distortion using transformation matrices.
"""
from typing import Union, Tuple, Optional

from coopscenes import Points
from coopscenes.data import Lidar, Camera, IMU, GNSS, Dynamics, CameraInformation, LidarInformation, GNSSInformation, \
    IMUInformation, DynamicsInformation, VehicleInformation, Vehicle
from scipy.spatial.transform import Rotation as R
import numpy as np
from kiss_icp.preprocess import get_preprocessor
from kiss_icp.config import KISSConfig


class Transformation:
    """Class representing a 3D transformation consisting of translation and rotation.

    Attributes:
        at (str): The origin frame of the transformation.
        to (str): The destination frame of the transformation.
        mtx (np.ndarray): The 4x4 transformation matrix combining rotation and translation.
    """

    def __init__(self, at: str, to: str, transformation_mtx: np.ndarray):
        """Initializes the Transformation object.

        Args:
            at (str): The origin frame of the transformation.
            to (str): The destination frame of the transformation.
            transformation_mtx (np.ndarray): The 4x4 transformation matrix.
        """
        self.at = at
        self.to = to
        self.mtx = transformation_mtx

    @classmethod
    def from_xyz_and_rpy(cls, at: str, to: str, xyz: np.ndarray, rpy: np.ndarray):
        """Creates a Transformation object from translation (xyz) and rotation (rpy).

        Args:
            at (str): The origin frame of the transformation.
            to (str): The destination frame of the transformation.
            xyz (np.ndarray): Translation vector [x, y, z].
            rpy (np.ndarray): Rotation vector [roll, pitch, yaw] in radians.

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

    def combine_transformation(self, transformation_to: 'Transformation') -> 'Transformation':
        """Combines this transformation with another transformation.

        Args:
            transformation_to (Transformation): The transformation to combine with.

        Returns:
            Transformation: The new combined transformation.
        """
        new_transformation_mtx = np.dot(transformation_to.mtx, self.mtx)
        return Transformation(self.at, transformation_to.to, transformation_mtx=new_transformation_mtx)

    def invert_transformation(self) -> 'Transformation':
        """Inverts this transformation.

        Returns:
            Transformation: The inverted transformation.
        """
        inverse_mtx = np.linalg.inv(self.mtx)
        return Transformation(self.to, self.at, transformation_mtx=inverse_mtx)

    def __repr__(self):
        """Returns a string representation of the Transformation object.

        Returns:
            str: String representation of the transformation.
        """
        translation_str = ', '.join(f"{coord:.3f}" for coord in self.translation)
        rotation_str = ', '.join(f"{angle:.3f}" for angle in self.rotation)
        return (f"Transformation at {self.at} to {self.to},\n"
                f"  translation=[{translation_str}],\n"
                f"  rotation=[{rotation_str}]\n")


def get_transformation(sensor_info: Union[
    Camera, Lidar, IMU, GNSS, Vehicle, CameraInformation, LidarInformation, IMUInformation, GNSSInformation, VehicleInformation]) -> Transformation:
    """Creates a Transformation object for a given sensor or its corresponding information object.

    Args:
        sensor_info (Union[Camera, Lidar, IMU, GNSS, Vehicle, CameraInformation, LidarInformation, IMUInformation, GNSSInformation, VehicleInformation]):
            Either a sensor object (Camera, Lidar, IMU, GNSS, Vehicle) or directly the sensor's information object.

    Returns:
        Transformation: The transformation object for the given sensor or sensor information.

    Raises:
        ValueError: If Dynamics or DynamicsInformation is passed, as they are not supported.
    """
    if hasattr(sensor_info, 'info'):
        sensor_info = sensor_info.info

    if isinstance(sensor_info, (Dynamics, DynamicsInformation)):
        raise ValueError(
            "Dynamics and DynamicsInformation are not supported for this function yet. "
            "Create your own Transformation object off the correct sensor until implemented.")

    if any(key in getattr(sensor_info, 'name', '').lower() for key in
           ['left', 'right', 'top']) or 'microstrain' in getattr(sensor_info, 'model_name', '').lower():
        sensor_to = 'lidar_top'
    else:
        sensor_to = 'lidar_upper_platform'

    if isinstance(sensor_info, CameraInformation):
        sensor_at = f'cam_{sensor_info.name}'
    elif isinstance(sensor_info, LidarInformation):
        sensor_at = f'lidar_{sensor_info.name}'
    elif isinstance(sensor_info, VehicleInformation):
        sensor_at = 'lidar_top'
    else:
        sensor_at = 'ins'

    tf = Transformation(sensor_at, sensor_to, sensor_info.extrinsic)
    return tf


def transform_points_to_origin(data: Union[Lidar, Tuple[np.ndarray, LidarInformation]],
                               vehicle_info: Optional[VehicleInformation] = None) -> np.ndarray:
    """Transforms LiDAR points to the origin of the associated agent or global coordinate system.

    Args:
        data (Union[Lidar, Tuple[np.ndarray, LidarInformation]]): Either a LiDAR sensor object or a tuple containing
            a NumPy array of LiDAR points and LidarInformation.
        vehicle_info (Optional[VehicleInformation]): Vehicle information for global transformation. Default is None.

    Returns:
        np.ndarray: Transformed 3D points in the agent's or global coordinate frame.
    """
    if isinstance(data, Lidar):
        points = data.points.points
        points_homogeneous = np.vstack((points['x'], points['y'], points['z'], np.ones(points['x'].shape[0])))
        lidar_info = data.info
    else:
        points, lidar_info = data
        points_homogeneous = np.vstack((points[:, 0], points[:, 1], points[:, 2], np.ones(points.shape[0])))

    trans = get_transformation(lidar_info)

    if any(key in getattr(lidar_info, 'name', '').lower() for key in ['left', 'right', 'top']):
        if vehicle_info is not None and vehicle_info.extrinsic is not None:
            trans = trans.combine_transformation(get_transformation(vehicle_info))
        else:
            print('Missing vehicle information. Resulting points are in local agent coordinate system.')

    transformed_points = trans.mtx @ points_homogeneous

    return transformed_points[:3].T


def get_deskewed_points(data: Union[Lidar, Tuple[Points, LidarInformation]]) -> Points:
    """Applies motion compensation to deskew LiDAR points.

    This function processes LiDAR points to correct for motion distortion caused by sensor movement during data capture.
    It uses transformation matrices provided in the LiDAR information (`motion_transform`) to deskew the point cloud.

    Args:
        data (Union[Lidar, Tuple[Points, LidarInformation]]): The LiDAR data to be deskewed. This can be either:
            - A `Lidar` object containing raw points and associated metadata.
            - A tuple of `Points` and `LidarInformation`.

    Returns:
        Points: The deskewed LiDAR point cloud as a structured array with additional fields like intensity, timestamp,
            reflectivity, ring, and ambient light data.

    Notes:
        - If the `motion_transform` attribute in `LidarInformation` is `None`, the function returns the original points
          without deskewing.
        - Deskewing is performed using a preprocessor from the KISS-ICP library.
        - The timestamps are normalized to account for motion during data acquisition.
    """
    if isinstance(data, Lidar):
        points = data._points_raw
        lidar_info = data.info
    else:
        points, lidar_info = data
    if hasattr(lidar_info, 'motion_transform'):
        if lidar_info.motion_transform is not None:
            k_config = KISSConfig()
            k_config.data.max_range = 1000
            k_config.data.min_range = 0
            k_config.data.deskew = True
            k_config.registration.max_num_threads = 0
            preprocessor = get_preprocessor(k_config)

            points_ts = points['t']

            timestamps = (points_ts - points_ts.min()) / (points_ts.max() - points_ts.min())
            points_xyz = np.stack((points['x'], points['y'], points['z']), axis=-1)

            points_deskewed = preprocessor.preprocess(points_xyz, 1 - timestamps, lidar_info.motion_transform)
            fields = ['intensity', 't', 'reflectivity', 'ring', 'ambient']
            points_additional = np.stack([points[field] for field in fields], axis=-1)
            combined_data = np.hstack((points_deskewed, points_additional))
            deskewed_points_structured = np.array([tuple(row) for row in combined_data],
                                                  dtype=np.dtype(LidarInformation._os_dtype_structure()))
            return Points(deskewed_points_structured, points.timestamp)
    return Points(points.points, points.timestamp)
