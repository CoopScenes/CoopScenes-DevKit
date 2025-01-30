"""
This module defines classes representing metadata for various sensors and components in an
autonomous vehicle system. These classes encapsulate detailed information about the vehicle,
sensor configuration, and their poses within a coordinate system.

Classes:
    ROI: Represents a Region of Interest (ROI) within an image or sensor frame.
    VehicleInformation: Encapsulates metadata about the vehicle, including model name, height and extrinsic pose.
    TowerInformation: Encapsulates metadata about a sensor tower, including model name and height.
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

from coopscenes.miscellaneous import ReprFormaterMixin


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
    """Represents metadata about the vehicle.

    For the vehicle, the TOP Lidar always represents the origin for the transformations.
    This means all extrinsic poses are relative to the TOP Lidar.

    Attributes:
        model_name (Optional[str]): The model name of the vehicle.
        extrinsic (Optional[np.ndarray]): Extrinsic calibration parameters as a 4x4 transformation matrix.
        height (Optional[np.ndarray]): Height of the TOP Lidar above ground in meters.
    """

    def __init__(self, model_name: Optional[str] = None,
                 extrinsic: Optional[np.ndarray] = None,
                 height: Optional[np.ndarray] = None):
        """Initializes a VehicleInformation object.

        Args:
            model_name: The model name of the vehicle.
            extrinsic: 4x4 transformation matrix for the vehicle's pose relative to the TOP Lidar.
            height: Height of the TOP Lidar above ground as a 4x4 transformation matrix.
        """
        self.model_name = model_name
        self.extrinsic = extrinsic
        self.height = height

    def __repr__(self):
        """Return a string representation of the VehicleInformation object."""
        return (
            f"VehicleInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    extrinsic=\n{self._format_array(self.extrinsic, indent=8)},\n"
            f"    height={self._format_height(self.height)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class TowerInformation(ReprFormaterMixin):
    """Represents metadata about the sensor tower.

    For the tower, the UPPER_PLATFORM Lidar always represents the origin for the transformations.
    This means all extrinsic poses are relative to the UPPER_PLATFORM Lidar.

    Attributes:
        model_name (Optional[str]): The model name of the tower.
        height (Optional[np.ndarray]): The height of the UPPER_PLATFORM Lidar above the ground as a 4x4 transformation matrix.
    """

    def __init__(self, model_name: Optional[str] = None,
                 height: Optional[np.ndarray] = None):
        """Initializes a TowerInformation object.

        Args:
            model_name (Optional[str]): The model name of the tower.
            height (Optional[np.ndarray]): The height of the UPPER_PLATFORM Lidar above the ground as a 4x4 transformation matrix.
        """
        self.model_name = model_name
        self.height = height

    def __repr__(self):
        """Returns a string representation of the TowerInformation object."""
        return (
            f"TowerInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    height={self._format_height(self.height)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class IMUInformation(ReprFormaterMixin):
    """Class representing metadata about an IMU sensor.

    Attributes:
        model_name (Optional[str]): The model name of the IMU sensor.
        extrinsic (Optional[np.ndarray]): The extrinsic pose of the IMU sensor as a 4x4 transformation matrix relative to the TOP Lidar.
    """

    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[np.ndarray] = None):
        """Initialize an IMUInformation object.

        Args:
            model_name (Optional[str]): The model name of the IMU sensor.
            extrinsic (Optional[np.ndarray]): The extrinsic pose of the IMU sensor as a 4x4 transformation matrix relative to the TOP Lidar.
        """
        self.model_name = model_name
        self.extrinsic = extrinsic

    def __repr__(self):
        """Return a string representation of the IMUInformation object."""
        return (
            f"IMUInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    extrinsic=\n{self._format_array(self.extrinsic, indent=8)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class GNSSInformation(ReprFormaterMixin):
    """Class representing metadata about a GNSS sensor.

    Attributes:
        model_name (Optional[str]): The model name of the GNSS sensor.
        extrinsic (Optional[np.ndarray]): The extrinsic pose of the GNSS sensor as a 4x4 transformation matrix relative to the TOP Lidar.
    """

    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[np.ndarray] = None):
        """Initialize a GNSSInformation object.

        Args:
            model_name (Optional[str]): The model name of the GNSS sensor.
            extrinsic (Optional[np.ndarray]): The extrinsic pose of the GNSS sensor as a 4x4 transformation matrix relative to the TOP Lidar.
        """
        self.model_name = model_name
        self.extrinsic = extrinsic

    def __repr__(self):
        """Return a string representation of the GNSSInformation object."""
        return (
            f"GNSSInformation(\n"
            f"    model_name={self.model_name},\n"
            f"    extrinsic=\n{self._format_array(self.extrinsic, indent=8)}\n"
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
        camera_mtx (Optional[np.ndarray]): The intrinsic camera matrix.
        distortion_mtx (Optional[np.ndarray]): The distortion matrix.
        rectification_mtx (Optional[np.ndarray]): The rectification matrix.
        projection_mtx (Optional[np.ndarray]): The projection matrix.
        region_of_interest (Optional[ROI]): The region of interest within the camera's field of view.
        camera_type (Optional[str]): The type of camera (e.g., monocular, stereo).
        focal_length (Optional[int]): The focal length of the camera in mm.
        aperture (Optional[int]): The aperture size of the camera in mm.
        exposure_time (Optional[int]): The exposure time of the camera in microseconds.
        extrinsic (Optional[np.ndarray]): The extrinsic pose of the Camera sensor as a 4x4 transformation matrix.
        stereo_transform (Optional[np.ndarray]): The extrinsic pose of the STEREO_LEFT camera relative to STEREO_RIGHT camera as a 4x4 transformation matrix.
    """

    def __init__(self, name: str, model_name: Optional[str] = None, shape: Optional[Tuple[int, int]] = None,
                 distortion_type: Optional[str] = None, camera_mtx: Optional[np.ndarray] = None,
                 distortion_mtx: Optional[np.ndarray] = None, rectification_mtx: Optional[np.ndarray] = None,
                 projection_mtx: Optional[np.ndarray] = None, region_of_interest: Optional[ROI] = None,
                 camera_type: Optional[str] = None, focal_length: Optional[int] = None,
                 aperture: Optional[int] = None, exposure_time: Optional[int] = None,
                 extrinsic: Optional[np.ndarray] = None, stereo_transform: Optional[np.ndarray] = None):
        """Initialize a CameraInformation object.

        Args:
            name (str): The name of the camera.
            model_name (Optional[str]): The model name of the camera.
            shape (Optional[Tuple[int, int]]): The resolution of the camera (width, height).
            distortion_type (Optional[str]): The type of lens distortion.
            camera_mtx (Optional[np.ndarray]): The intrinsic camera matrix.
            distortion_mtx (Optional[np.ndarray]): The distortion matrix.
            rectification_mtx (Optional[np.ndarray]): The rectification matrix.
            projection_mtx (Optional[np.ndarray]): The projection matrix.
            region_of_interest (Optional[ROI]): The region of interest within the camera's field of view.
            camera_type (Optional[str]): The type of camera.
            focal_length (Optional[int]): The focal length of the camera.
            aperture (Optional[int]): The aperture size of the camera.
            exposure_time (Optional[int]): The exposure time of the camera.
            extrinsic (Optional[np.ndarray]): The extrinsic pose of the Camera sensor as a 4x4 transformation matrix.
            stereo_transform (Optional[np.ndarray]): The extrinsic pose of the STEREO_LEFT camera relative to STEREO_RIGHT camera as a 4x4 transformation matrix.
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
            f"    camera_mtx=\n{self._format_array(self.camera_mtx, indent=8)},\n"
            f"    distortion_mtx={self._format_array(self.distortion_mtx)},\n"
            f"    extrinsic=\n{self._format_array(self.extrinsic, indent=8)}\n"
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
            elif isinstance(value, (ROI)):
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
    """Represents metadata about a Lidar sensor.

    Attributes:
        name (str): The name of the Lidar sensor.
        model_name (Optional[str]): The model name of the Lidar sensor.
        extrinsic (Optional[np.ndarray]): The extrinsic pose of the Lidar sensor as a 4x4 transformation matrix.
        vertical_fov (Optional[float]): The vertical field of view of the Lidar (for Blickfeld sensors).
        horizontal_fov (Optional[float]): The horizontal field of view of the Lidar (for Blickfeld sensors).
        beam_altitude_angles (Optional[np.ndarray]): Beam altitude angles (for Ouster sensors).
        beam_azimuth_angles (Optional[np.ndarray]): Beam azimuth angles (for Ouster sensors).
        lidar_origin_to_beam_origin_mm (Optional[np.ndarray]): Distance from the Lidar origin to the beam
            origin in mm (for Ouster sensors).
        horizontal_scanlines (Optional[int]): The number of horizontal scanlines (for Ouster sensors).
        vertical_scanlines (Optional[int]): The number of vertical scanlines (for Ouster sensors).
        phase_lock_offset (Optional[int]): The phase lock offset (for Ouster sensors).
        lidar_to_sensor_transform (Optional[np.ndarray]): Transformation matrix from the Lidar frame
            to the sensor frame (for Ouster sensors).
        horizontal_angle_spacing (Optional[float]): The horizontal angle spacing of the Lidar (for Blickfeld sensors).
        frame_mode (Optional[str]): The frame mode of the Lidar (for Blickfeld sensors).
        scan_pattern (Optional[str]): The scan pattern of the Lidar (for Blickfeld sensors).
        motion_transform (Optional[np.ndarray]): Transformation matrix for motion compensation (for Ouster sensor).
        dtype (np.dtype): Data type structure of the Lidar point cloud data.
    """

    def __init__(self, name: str, model_name: Optional[str] = None, beam_altitude_angles: Optional[np.ndarray] = None,
                 beam_azimuth_angles: Optional[np.ndarray] = None,
                 lidar_origin_to_beam_origin_mm: Optional[np.ndarray] = None,
                 horizontal_scanlines: Optional[int] = None, vertical_scanlines: Optional[int] = None,
                 phase_lock_offset: Optional[int] = None, lidar_to_sensor_transform: Optional[np.ndarray] = None,
                 extrinsic: Optional[np.ndarray] = None, vertical_fov: Optional[float] = None,
                 horizontal_fov: Optional[float] = None, horizontal_angle_spacing: Optional[float] = None,
                 frame_mode: Optional[str] = None, scan_pattern: Optional[str] = None,
                 motion_transform: Optional[np.ndarray] = None):
        """Initialize a LidarInformation object.

        Args:
            name (str): The name of the Lidar sensor.
            model_name (Optional[str]): The model name of the Lidar sensor.
            beam_altitude_angles (Optional[np.ndarray]): Beam altitude angles (for Ouster sensors).
            beam_azimuth_angles (Optional[np.ndarray]): Beam azimuth angles (for Ouster sensors).
            lidar_origin_to_beam_origin_mm (Optional[np.ndarray]): Distance from the Lidar origin to the beam origin in mm (for Ouster sensors).
            horizontal_scanlines (Optional[int]): The number of horizontal scanlines (for Ouster sensors).
            vertical_scanlines (Optional[int]): The number of vertical scanlines (for Ouster sensors).
            phase_lock_offset (Optional[int]): The phase lock offset (for Ouster sensors).
            lidar_to_sensor_transform (Optional[np.ndarray]): Transformation matrix from the Lidar frame to the sensor frame (for Ouster sensors).
            extrinsic (Optional[np.ndarray]): The extrinsic pose of the Lidar sensor as a 4x4 transformation matrix.
            vertical_fov (Optional[float]): The vertical field of view of the Lidar (for Blickfeld sensors).
            horizontal_fov (Optional[float]): The horizontal field of view of the Lidar (for Blickfeld sensors).
            horizontal_angle_spacing (Optional[float]): The horizontal angle spacing of the Lidar (for Blickfeld sensors).
            frame_mode (Optional[str]): The frame mode of the Lidar (for Blickfeld sensors).
            scan_pattern (Optional[str]): The scan pattern of the Lidar (for Blickfeld sensors).
            motion_transform (Optional[np.ndarray]): Transformation matrix for motion compensation (for Ouster sensor).
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
                                    lidar_to_sensor_transform, motion_transform)

    def __repr__(self):
        """Return a string representation of the LidarInformation object with key attributes."""
        return (
            f"LidarInformation(\n"
            f"    name={self.name},\n"
            f"    model_name={self.model_name or 'N/A'},\n"
            f"    extrinsic=\n{self._format_array(self.extrinsic, indent=8)},\n"
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

    def _initialize_ouster(self, beam_altitude_angles: Optional[np.ndarray], beam_azimuth_angles: Optional[np.ndarray],
                           lidar_origin_to_beam_origin_mm: Optional[np.ndarray], horizontal_scanlines: Optional[int],
                           vertical_scanlines: Optional[int], phase_lock_offset: Optional[int],
                           lidar_to_sensor_transform: Optional[np.ndarray],
                           motion_transform: Optional[np.ndarray] = None):
        """Initialize attributes specific to Ouster Lidar sensors."""
        self.beam_altitude_angles = beam_altitude_angles
        self.beam_azimuth_angles = beam_azimuth_angles
        self.lidar_origin_to_beam_origin_mm = lidar_origin_to_beam_origin_mm
        self.horizontal_scanlines = horizontal_scanlines
        self.vertical_scanlines = vertical_scanlines
        self.phase_lock_offset = phase_lock_offset
        self.lidar_to_sensor_transform = lidar_to_sensor_transform
        self.dtype = np.dtype(self._os_dtype_structure())
        self.motion_transform = motion_transform

    @staticmethod
    def _os_dtype_structure() -> Dict[str, List]:
        """Return the dtype structure for 'OS' (Ouster) Lidar models."""
        return {
            'names': ['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'ambient'],
            'formats': ['<f4', '<f4', '<f4', '<f4', '<u4', '<u2', '<u2', '<u2']
        }

    @staticmethod
    def _blickfeld_dtype_structure() -> Dict[str, List]:
        """Return the dtype structure for 'Blickfeld' Lidar models."""
        return {
            'names': ['x', 'y', 'z', 'intensity', 'point_time_offset'],
            'formats': ['<f4', '<f4', '<f4', '<u4', '<u4']
        }
