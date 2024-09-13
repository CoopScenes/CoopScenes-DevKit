from typing import Tuple, Optional, Dict
import numpy as np


class Pose:
    """
    Describes the position of a sensor in terms of its position and rotation relative to
    the reference coordinate system (Top_LiDAR).
    """

    def __init__(self, xyz: Optional[np.array] = None, rpy: Optional[np.array] = None):
        self.xyz = xyz
        self.rpy = rpy

    def __str__(self):
        return f"Pose(xyz={self.xyz}, rpy={self.rpy})"


class TransformationMtx:
    """
    Represents a transformation matrix with separate rotation and translation components.
    """

    def __init__(self, rotation: Optional[np.array] = None, translation: Optional[np.array] = None):
        self.rotation = rotation
        self.translation = translation

    def __str__(self):
        return f"TransformationMtx(rotation={self.rotation.tolist()}, translation={self.translation.tolist()})"


class ROI:
    """
    Represents a Region of Interest (ROI) defined by its offset and dimensions.
    """

    def __init__(self, x_off: Optional[int] = None, y_off: Optional[int] = None,
                 width: Optional[int] = None, height: Optional[int] = None):
        self.x_offset = x_off
        self.y_offset = y_off
        self.width = width
        self.height = height

    def __iter__(self):
        return iter((self.x_offset, self.y_offset, self.width, self.height))

    def __str__(self):
        return f"ROI(x_offset={self.x_offset}, y_offset={self.y_offset}, width={self.width}, height={self.height})"


class VehicleInformation:
    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        self.model_name = model_name
        self.extrinsic = extrinsic


class TowerInformation:
    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        self.model_name = model_name
        self.extrinsic = extrinsic


class DynamicsInformation:
    def __init__(self, velocity_source: Optional[str] = None, heading_source: Optional[str] = None):
        self.velocity_source = velocity_source
        self.heading_source = heading_source


class IMUInformation:
    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        self.model_name = model_name
        self.extrinsic = extrinsic


class GNSSInformation:
    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        self.model_name = model_name
        self.extrinsic = extrinsic


class CameraInformation:
    def __init__(self, name: str, model_name: Optional[str] = None, shape: Optional[Tuple[int, int]] = None,
                 distortion_type: Optional[str] = None, camera_mtx: Optional[np.array] = None,
                 distortion_mtx: Optional[np.array] = None, rectification_mtx: Optional[np.array] = None,
                 projection_mtx: Optional[np.array] = None, region_of_interest: Optional[ROI] = None,
                 camera_type: Optional[str] = None, focal_length: Optional[int] = None,
                 aperture: Optional[int] = None, exposure_time: Optional[int] = None,
                 extrinsic: Optional[Pose] = None, stereo_transform: Optional[TransformationMtx] = None):
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

    def to_dict(self) -> Dict[str, str]:
        """Konvertiert das Objekt in ein Dictionary."""
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


class LidarInformation:
    """
    Represents the information related to a Lidar sensor. The structure and attributes vary depending on whether
    the Lidar is a Blickfeld ('view' in name) or an Ouster sensor.
    """

    def __init__(self, name: str, model_name: Optional[str] = None, beam_altitude_angles: Optional[np.array] = None,
                 beam_azimuth_angles: Optional[np.array] = None,
                 lidar_origin_to_beam_origin_mm: Optional[np.array] = None,
                 horizontal_scanlines: Optional[int] = None, vertical_scanlines: Optional[int] = None,
                 phase_lock_offset: Optional[int] = None, lidar_to_sensor_transform: Optional[np.array] = None,
                 extrinsic: Optional[Pose] = None, vertical_fov: Optional[float] = None,
                 horizontal_fov: Optional[float] = None, horizontal_angle_spacing: Optional[float] = None,
                 frame_mode: Optional[str] = None, scan_pattern: Optional[str] = None):
        self.name = name
        self.model_name = model_name
        self.extrinsic = extrinsic

        # Set attributes and dtype based on sensor type
        if 'view' in name.lower():
            self._initialize_blickfeld(vertical_fov, horizontal_fov, horizontal_angle_spacing, frame_mode, scan_pattern)
        else:
            self._initialize_ouster(beam_altitude_angles, beam_azimuth_angles, lidar_origin_to_beam_origin_mm,
                                    horizontal_scanlines, vertical_scanlines, phase_lock_offset,
                                    lidar_to_sensor_transform)

    def _initialize_blickfeld(self, vertical_fov: Optional[float], horizontal_fov: Optional[float],
                              horizontal_angle_spacing: Optional[float], frame_mode: Optional[str],
                              scan_pattern: Optional[str]):
        """Initialize attributes specific to Blickfeld sensors."""
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
        """Initialize attributes specific to Ouster sensors."""
        self.beam_altitude_angles = beam_altitude_angles
        self.beam_azimuth_angles = beam_azimuth_angles
        self.lidar_origin_to_beam_origin_mm = lidar_origin_to_beam_origin_mm
        self.horizontal_scanlines = horizontal_scanlines
        self.vertical_scanlines = vertical_scanlines
        self.phase_lock_offset = phase_lock_offset
        self.lidar_to_sensor_transform = lidar_to_sensor_transform
        self.dtype = np.dtype(self._os_dtype_structure())

    def _os_dtype_structure(self) -> dict:
        """Return the dtype structure for 'OS' (Ouster) models."""
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

    def _blickfeld_dtype_structure(self) -> dict:
        """Return the dtype structure for 'Blickfeld' models."""
        return {
            'names': ['x', 'y', 'z', 'range', 'intensity', 'point_id', 'point_time_offset'],
            'formats': ['<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<u4'],
            'offsets': [0, 4, 8, 12, 16, 20, 24],
            'itemsize': 28  # The total size (in bytes) of the structure
        }
