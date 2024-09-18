"""
This module defines classes representing metadata for various sensors and components in an
autonomous vehicle system. These classes encapsulate detailed information about the vehicle,
sensor configuration, and their poses within a coordinate system.

Classes:
    Pose: Represents the position and orientation of a sensor in 3D space.
    TransformationMtx: Represents a transformation matrix with separate rotation and translation components.
    ROI: Represents a Region of Interest (ROI) within an image or sensor frame.
    VehicleInformation: Encapsulates metadata about the vehicle, including model name and pose.
    TowerInformation: Encapsulates metadata about a sensor tower, including model name and pose.
    DynamicsInformation: Holds information about the source of velocity and heading data for vehicle dynamics.
    IMUInformation: Represents metadata about an IMU sensor, including its model name and extrinsic pose.
    GNSSInformation: Represents metadata about a GNSS sensor, including its model name and extrinsic pose.
    CameraInformation: Provides detailed metadata for a camera sensor, including intrinsic and extrinsic parameters.
    LidarInformation: Provides detailed metadata for a Lidar sensor, including intrinsic and extrinsic parameters,
                      varying based on whether it is a Blickfeld or Ouster sensor.

Each class in this module is designed to store and manage detailed metadata, which is crucial for
sensor calibration, sensor fusion, and the overall understanding of sensor placement and characteristics
in autonomous vehicle systems.
"""
from typing import Tuple, Optional, Dict, List
import numpy as np
from numpy import dtype

from aeifdataset.miscellaneous import ReprFormaterMixin


class Pose(ReprFormaterMixin):
    """Class representing the position and rotation of a sensor in 3D space.

    This class describes the position (xyz) and rotation (rpy) of a sensor
    relative to the reference coordinate system.

    Attributes:
        xyz (Optional[np.array]): The position in the reference coordinate system (x, y, z).
        rpy (Optional[np.array]): The rotation in roll, pitch, and yaw (r, p, y).
    """

    def __init__(self, xyz: Optional[np.array] = None, rpy: Optional[np.array] = None):
        """Initialize a Pose object with position and rotation data.

        Args:
            xyz (Optional[np.array]): The position in the reference coordinate system (x, y, z).
            rpy (Optional[np.array]): The rotation in roll, pitch, and yaw (r, p, y).
        """
        self.xyz = xyz
        self.rpy = rpy

    def __repr__(self):
        """Return a string representation of the Pose object with xyz and rpy."""
        return (
            f"Pose(\n"
            f"    xyz={self._format_array(self.xyz)},\n"
            f"    rpy={self._format_array(self.rpy)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class TransformationMtx(ReprFormaterMixin):
    """Class representing a transformation matrix with rotation and translation components.

    This class describes a transformation matrix that separates the rotation
    and translation components for transforming between coordinate systems.

    Attributes:
        rotation (Optional[np.array]): The rotation matrix (3x3).
        translation (Optional[np.array]): The translation vector (x, y, z).
    """

    def __init__(self, rotation: Optional[np.array] = None, translation: Optional[np.array] = None):
        """Initialize a TransformationMtx object with rotation and translation data.

        Args:
            rotation (Optional[np.array]): The rotation matrix (3x3).
            translation (Optional[np.array]): The translation vector (x, y, z).
        """
        self.rotation = rotation
        self.translation = translation

    def __repr__(self):
        """Return a string representation of the TransformationMtx object with rotation and translation."""
        return (
            f"TransformationMtx(\n"
            f"    rotation=\n{self._format_array(self.rotation)},\n"
            f"    translation={self._format_array(self.translation)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class ROI:
    """Class representing a Region of Interest (ROI).

    This class defines a region within an image or sensor frame, represented by
    an offset and dimensions (width, height).

    Attributes:
        x_offset (Optional[int]): The x-coordinate of the top-left corner of the ROI.
        y_offset (Optional[int]): The y-coordinate of the top-left corner of the ROI.
        width (Optional[int]): The width of the ROI.
        height (Optional[int]): The height of the ROI.
    """

    def __init__(self, x_off: Optional[int] = None, y_off: Optional[int] = None,
                 width: Optional[int] = None, height: Optional[int] = None):
        """Initialize an ROI object with offset and dimensions.

        Args:
            x_off (Optional[int]): The x-coordinate of the top-left corner of the ROI.
            y_off (Optional[int]): The y-coordinate of the top-left corner of the ROI.
            width (Optional[int]): The width of the ROI.
            height (Optional[int]): The height of the ROI.
        """
        self.x_offset = x_off
        self.y_offset = y_off
        self.width = width
        self.height = height

    def __iter__(self):
        return iter((self.x_offset, self.y_offset, self.width, self.height))

    def __repr__(self):
        """Return a string representation of the ROI object with offsets and dimensions."""
        return (
            f"ROI(\n"
            f"    x_offset={self.x_offset},\n"
            f"    y_offset={self.y_offset},\n"
            f"    width={self.width},\n"
            f"    height={self.height}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class VehicleInformation(ReprFormaterMixin):
    """Class representing metadata about a vehicle.

    Attributes:
        model_name (Optional[str]): The model name of the vehicle.
        extrinsic (Optional[Pose]): The extrinsic pose of the TOP Lidar relative to the UPPER_PLATFORM Lidar.
    """

    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        """Initialize a VehicleInformation object.

        Args:
            model_name (Optional[str]): The model name of the vehicle.
            extrinsic (Optional[Pose]): The extrinsic pose of the TOP Lidar relative to the UPPER_PLATFORM Lidar.
        """
        self.model_name = model_name
        self.extrinsic = extrinsic

    def __repr__(self):
        """Return a string representation of the VehicleInformation object."""
        return (
            f"VehicleInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    extrinsic={self._format_object(self.extrinsic)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class TowerInformation(ReprFormaterMixin):
    """Class representing metadata about a sensor tower.

    Attributes:
        model_name (Optional[str]): The model name of the tower.
        extrinsic (Optional[Pose]): The extrinsic pose of the UPPER_PLATFORM Lidar relative to the TOP Lidar.
    """

    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        """Initialize a TowerInformation object.

        Args:
            model_name (Optional[str]): The model name of the tower.
            extrinsic (Optional[Pose]): The extrinsic pose of the UPPER_PLATFORM Lidar relative to the TOP Lidar.
        """
        self.model_name = model_name
        self.extrinsic = extrinsic

    def __repr__(self):
        """Return a string representation of the TowerInformation object."""
        return (
            f"TowerInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    extrinsic={self._format_object(self.extrinsic)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class IMUInformation(ReprFormaterMixin):
    """Class representing metadata about an IMU sensor.

    Attributes:
        model_name (Optional[str]): The model name of the IMU sensor.
        extrinsic (Optional[Pose]): The extrinsic pose of the IMU sensor relative to the TOP Lidar for the vehicle.
    """

    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        """Initialize an IMUInformation object.

        Args:
            model_name (Optional[str]): The model name of the IMU sensor.
            extrinsic (Optional[Pose]): The extrinsic pose of the IMU sensor relative to the TOP Lidar for the vehicle.
        """
        self.model_name = model_name
        self.extrinsic = extrinsic

    def __repr__(self):
        """Return a string representation of the IMUInformation object."""
        return (
            f"IMUInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    extrinsic={self._format_object(self.extrinsic)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class GNSSInformation(ReprFormaterMixin):
    """Class representing metadata about a GNSS sensor.

    Attributes:
        model_name (Optional[str]): The model name of the GNSS sensor.
        extrinsic (Optional[Pose]): The extrinsic pose of the GNSS sensor relative to the TOP Lidar for the vehicle.
    """

    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        """Initialize a GNSSInformation object.

        Args:
            model_name (Optional[str]): The model name of the GNSS sensor.
            extrinsic (Optional[Pose]): The extrinsic pose of the GNSS sensor relative to the TOP Lidar for the vehicle.
        """
        self.model_name = model_name
        self.extrinsic = extrinsic

    def __repr__(self):
        """Return a string representation of the GNSSInformation object."""
        return (
            f"GNSSInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    extrinsic={self._format_object(self.extrinsic)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class DynamicsInformation:
    """Class representing metadata about the dynamics of a vehicle.

    Attributes:
        velocity_source (Optional[str]): The source of velocity data (e.g., GNSS, IMU).
        heading_source (Optional[str]): The source of heading data.
    """

    def __init__(self, velocity_source: Optional[str] = None, heading_source: Optional[str] = None):
        """Initialize a DynamicsInformation object.

        Args:
            velocity_source (Optional[str]): The source of velocity data.
            heading_source (Optional[str]): The source of heading data.
        """
        self.velocity_source = velocity_source
        self.heading_source = heading_source

    def __repr__(self):
        """Return a string representation of the DynamicsInformation object."""
        return (
            f"DynamicsInformation(\n"
            f"    velocity_source={self.velocity_source},\n"
            f"    heading_source={self.heading_source}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class CameraInformation(ReprFormaterMixin):
    """Class representing metadata about a camera sensor.

    Attributes:
        name (str): The name of the camera.
        model_name (Optional[str]): The model name of the camera.
        shape (Optional[Tuple[int, int]]): The resolution of the camera (width, height).
        distortion_type (Optional[str]): The type of lens distortion (e.g., radial, tangential).
        camera_mtx (Optional[np.array]): The intrinsic camera matrix.
        distortion_mtx (Optional[np.array]): The distortion matrix.
        rectification_mtx (Optional[np.array]): The rectification matrix.
        projection_mtx (Optional[np.array]): The projection matrix.
        region_of_interest (Optional[ROI]): The region of interest within the camera's field of view.
        camera_type (Optional[str]): The type of camera (e.g., monocular, stereo).
        focal_length (Optional[int]): The focal length of the camera in mm.
        aperture (Optional[int]): The aperture size of the camera in mm.
        exposure_time (Optional[int]): The exposure time of the camera in microseconds.
        extrinsic (Optional[Pose]): The extrinsic pose of the Camera sensor relative to the TOP Lidar for the vehicle or the UPPER_PLATFORM Lidar for the tower.
        stereo_transform (Optional[TransformationMtx]): The transformation matrix from STEREO_LEFT camera to STEREO_RIGHT camera.
    """

    def __init__(self, name: str, model_name: Optional[str] = None, shape: Optional[Tuple[int, int]] = None,
                 distortion_type: Optional[str] = None, camera_mtx: Optional[np.array] = None,
                 distortion_mtx: Optional[np.array] = None, rectification_mtx: Optional[np.array] = None,
                 projection_mtx: Optional[np.array] = None, region_of_interest: Optional[ROI] = None,
                 camera_type: Optional[str] = None, focal_length: Optional[int] = None,
                 aperture: Optional[int] = None, exposure_time: Optional[int] = None,
                 extrinsic: Optional[Pose] = None, stereo_transform: Optional[TransformationMtx] = None):
        """Initialize a CameraInformation object.

        Args:
            name (str): The name of the camera.
            model_name (Optional[str]): The model name of the camera.
            shape (Optional[Tuple[int, int]]): The resolution of the camera (width, height).
            distortion_type (Optional[str]): The type of lens distortion.
            camera_mtx (Optional[np.array]): The intrinsic camera matrix.
            distortion_mtx (Optional[np.array]): The distortion matrix.
            rectification_mtx (Optional[np.array]): The rectification matrix.
            projection_mtx (Optional[np.array]): The projection matrix.
            region_of_interest (Optional[ROI]): The region of interest within the camera's field of view.
            camera_type (Optional[str]): The type of camera.
            focal_length (Optional[int]): The focal length of the camera.
            aperture (Optional[int]): The aperture size of the camera.
            exposure_time (Optional[int]): The exposure time of the camera.
            extrinsic (Optional[Pose]): The extrinsic pose of the Camera sensor relative to the TOP Lidar for the vehicle or the UPPER_PLATFORM Lidar for the tower.
            stereo_transform (Optional[TransformationMtx]): The transformation matrix from STEREO_LEFT camera to STEREO_RIGHT camera.
        """
        self.name = name
        self.model_name = model_name
        self.shape = shape
        self.distortion_type = distortion_type
        self.camera_mtx = camera_mtx
        self.distortion_mtx = distortion_mtx
        self.rectification_mtx = rectification_mtx
        self.projection_mtx = projection_mtx
        self.region_of_interest = region_of_interest
        self.camera_type = camera_type
        self.focal_length = focal_length
        self.aperture = aperture
        self.exposure_time = exposure_time
        self.extrinsic = extrinsic
        self.stereo_transform = stereo_transform

    def __repr__(self):
        """Return a string representation of the CameraInformation object with key attributes."""
        return (
            f"CameraInformation(\n"
            f"    name={self.name},\n"
            f"    model_name={self.model_name or 'N/A'},\n"
            f"    camera_mtx=\n    {self._format_array(self.camera_mtx, indent=4)},\n"
            f"    distortion_mtx=\n    {self._format_array(self.distortion_mtx, indent=4)},\n"
            f"    extrinsic={self._format_object(self.extrinsic)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()

    def to_dict(self) -> Dict[str, str]:
        """Convert the CameraInformation object into a dictionary.

        Returns:
            Dict[str, str]: The dictionary representation of the CameraInformation object.
        """
        info_dict = vars(self).copy()
        for key, value in info_dict.items():
            if isinstance(value, np.ndarray):
                info_dict[key] = str(value.tolist())
            elif isinstance(value, (ROI, Pose, TransformationMtx)):
                info_dict[key] = str(value)
            elif isinstance(value, tuple):
                info_dict[key] = ', '.join(map(str, value))
            elif isinstance(value, int):
                info_dict[key] = str(value)
            elif isinstance(value, float):
                info_dict[key] = str(value)
            elif value is None:
                info_dict[key] = "N/A"
        return info_dict


class LidarInformation(ReprFormaterMixin):
    """Class representing metadata about a Lidar sensor.

    Attributes:
        name (str): The name of the Lidar sensor.
        model_name (Optional[str]): The model name of the Lidar sensor.
        extrinsic (Optional[Pose]): The extrinsic pose of the Lidar sensor relative to the TOP Lidar for the vehicle or the UPPER_PLATFORM Lidar for the tower.
        vertical_fov (Optional[float]): The vertical field of view of the Lidar (for Blickfeld sensors).
        horizontal_fov (Optional[float]): The horizontal field of view of the Lidar (for Blickfeld sensors).
        beam_altitude_angles (Optional[np.array]): Beam altitude angles (for Ouster sensors).
        beam_azimuth_angles (Optional[np.array]): Beam azimuth angles (for Ouster sensors).
        lidar_origin_to_beam_origin_mm (Optional[np.array]): Distance from the Lidar origin to the beam origin in mm (for Ouster sensors).
        horizontal_scanlines (Optional[int]): The number of horizontal scanlines (for Ouster sensors).
        vertical_scanlines (Optional[int]): The number of vertical scanlines (for Ouster sensors).
        phase_lock_offset (Optional[int]): The phase lock offset (for Ouster sensors).
        lidar_to_sensor_transform (Optional[np.array]): Transformation matrix from the Lidar frame to the sensor frame (for Ouster sensors).
        horizontal_angle_spacing (Optional[float]): The horizontal angle spacing of the Lidar (for Blickfeld sensors).
        frame_mode (Optional[str]): The frame mode of the Lidar (for Blickfeld sensors).
        scan_pattern (Optional[str]): The scan pattern of the Lidar (for Blickfeld sensors).
        dtype (np.dtype): Data type structure of the Lidar point cloud data.
    """

    def __init__(self, name: str, model_name: Optional[str] = None, beam_altitude_angles: Optional[np.array] = None,
                 beam_azimuth_angles: Optional[np.array] = None,
                 lidar_origin_to_beam_origin_mm: Optional[np.array] = None,
                 horizontal_scanlines: Optional[int] = None, vertical_scanlines: Optional[int] = None,
                 phase_lock_offset: Optional[int] = None, lidar_to_sensor_transform: Optional[np.array] = None,
                 extrinsic: Optional[Pose] = None, vertical_fov: Optional[float] = None,
                 horizontal_fov: Optional[float] = None, horizontal_angle_spacing: Optional[float] = None,
                 frame_mode: Optional[str] = None, scan_pattern: Optional[str] = None):
        """Initialize a LidarInformation object.

        Args:
            name (str): The name of the Lidar sensor.
            model_name (Optional[str]): The model name of the Lidar sensor.
            beam_altitude_angles (Optional[np.array]): Beam altitude angles (for Ouster sensors).
            beam_azimuth_angles (Optional[np.array]): Beam azimuth angles (for Ouster sensors).
            lidar_origin_to_beam_origin_mm (Optional[np.array]): Distance from the Lidar origin to the beam origin in mm (for Ouster sensors).
            horizontal_scanlines (Optional[int]): The number of horizontal scanlines (for Ouster sensors).
            vertical_scanlines (Optional[int]): The number of vertical scanlines (for Ouster sensors).
            phase_lock_offset (Optional[int]): The phase lock offset (for Ouster sensors).
            lidar_to_sensor_transform (Optional[np.array]): Transformation matrix from the Lidar frame to the sensor frame (for Ouster sensors).
            extrinsic (Optional[Pose]): The extrinsic pose of the Lidar sensor relative to the TOP Lidar for the vehicle or the UPPER_PLATFORM Lidar for the tower.
            vertical_fov (Optional[float]): The vertical field of view of the Lidar (for Blickfeld sensors).
            horizontal_fov (Optional[float]): The horizontal field of view of the Lidar (for Blickfeld sensors).
            horizontal_angle_spacing (Optional[float]): The horizontal angle spacing of the Lidar (for Blickfeld sensors).
            frame_mode (Optional[str]): The frame mode of the Lidar (for Blickfeld sensors).
            scan_pattern (Optional[str]): The scan pattern of the Lidar (for Blickfeld sensors).
            dtype (np.dtype): Data type structure of the Lidar point cloud data.
        """
        self.name = name
        self.model_name = model_name
        self.extrinsic = extrinsic

        # Initialize attributes based on sensor type
        if 'view' in name.lower():
            self._initialize_blickfeld(vertical_fov, horizontal_fov, horizontal_angle_spacing, frame_mode, scan_pattern)
        else:
            self._initialize_ouster(beam_altitude_angles, beam_azimuth_angles, lidar_origin_to_beam_origin_mm,
                                    horizontal_scanlines, vertical_scanlines, phase_lock_offset,
                                    lidar_to_sensor_transform)

    def __repr__(self):
        """Return a string representation of the LidarInformation object with key attributes."""
        return (
            f"LidarInformation(\n"
            f"    name={self.name},\n"
            f"    model_name={self.model_name or 'N/A'},\n"
            f"    extrinsic={self._format_object(self.extrinsic)},\n"
            f"    dtype=[{', '.join(self.dtype.names)}]\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()

    def _initialize_blickfeld(self, vertical_fov: Optional[float], horizontal_fov: Optional[float],
                              horizontal_angle_spacing: Optional[float], frame_mode: Optional[str],
                              scan_pattern: Optional[str]):
        """Initialize attributes specific to Blickfeld Lidar sensors."""
        self.vertical_fov = vertical_fov
        self.horizontal_fov = horizontal_fov
        self.horizontal_angle_spacing = horizontal_angle_spacing
        self.frame_mode = frame_mode
        self.scan_pattern = scan_pattern
        self.dtype = np.dtype(self._blickfeld_dtype_structure())

    def _initialize_ouster(self, beam_altitude_angles: Optional[np.array], beam_azimuth_angles: Optional[np.array],
                           lidar_origin_to_beam_origin_mm: Optional[np.array], horizontal_scanlines: Optional[int],
                           vertical_scanlines: Optional[int], phase_lock_offset: Optional[int],
                           lidar_to_sensor_transform: Optional[np.array]):
        """Initialize attributes specific to Ouster Lidar sensors."""
        self.beam_altitude_angles = beam_altitude_angles
        self.beam_azimuth_angles = beam_azimuth_angles
        self.lidar_origin_to_beam_origin_mm = lidar_origin_to_beam_origin_mm
        self.horizontal_scanlines = horizontal_scanlines
        self.vertical_scanlines = vertical_scanlines
        self.phase_lock_offset = phase_lock_offset
        self.lidar_to_sensor_transform = lidar_to_sensor_transform
        self.dtype = np.dtype(self._os_dtype_structure())

    @staticmethod
    def _os_dtype_structure() -> Dict[str, List]:
        """Return the dtype structure for 'OS' (Ouster) Lidar models."""
        return {
            'names': [
                'x', 'y', 'z', 'intensity', 't',
                'reflectivity', 'ring', 'ambient', 'range'
            ],
            'formats': [
                '<f4', '<f4', '<f4', '<f4', '<u4',
                '<u2', '<u2', '<u2', '<u4'
            ],
            'offsets': [0, 4, 8, 16, 20, 24, 26, 28, 32],
            'itemsize': 48
        }

    @staticmethod
    def _blickfeld_dtype_structure() -> Dict[str, List]:
        """Return the dtype structure for 'Blickfeld' Lidar models."""
        return {
            'names': ['x', 'y', 'z', 'range', 'intensity', 'point_id', 'point_time_offset'],
            'formats': ['<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<u4'],
            'offsets': [0, 4, 8, 12, 16, 20, 24],
            'itemsize': 28  # The total size (in bytes) of the structure
        }
