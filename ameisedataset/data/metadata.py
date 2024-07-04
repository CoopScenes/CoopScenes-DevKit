from typing import Tuple, Optional
import numpy as np


class Pose:
    """
    Describes the position of a sensor in terms of its position and rotation relative to
    the reference coordinate system (Top_LiDAR).
    Attributes:
        xyz (Optional[np.array]): A 1x3 array representing the position of the sensor in the
                                  reference coordinate system (Top_LiDAR).
        rpy (Optional[np.array]): A 1x3 array representing the roll, pitch, and yaw angles of the sensor,
                                  describing its rotation in itself.
    """

    def __init__(self, xyz: Optional[np.array] = None, rpy: Optional[np.array] = None):
        """
        Initializes the Pose with default position and rotation as None.
        """
        self.xyz = xyz
        self.rpy = rpy


class TransformationMtx:
    """
    Represents a transformation matrix with separate rotation and translation components.
    Attributes:
        rotation (np.array): A 3x3 matrix representing the rotation component of the transformation.
        translation (np.array): A 1x3 matrix representing the translation component of the transformation.
    """

    def __init__(self, rotation: Optional[np.array] = None, translation: Optional[np.array] = None):
        """
        Initializes the TransformationMtx with zero rotation and translation matrices.
        """
        self.rotation = rotation
        self.translation = translation


class ROI:
    """
    Represents a Region of Interest (ROI) defined by its offset and dimensions.
    Attributes:
        x_offset (Optional[int]): The horizontal offset of the ROI.
        y_offset (Optional[int]): The vertical offset of the ROI.
        width (Optional[int]): The width of the ROI.
        height (Optional[int]): The height of the ROI.
    """

    def __init__(self, x_off: Optional[int] = None, y_off: Optional[int] = None, width: Optional[int] = None,
                 height: Optional[int] = None):
        """
        Initializes the ROI with the provided offset and dimensions.
        Parameters:
            x_off (Optional[int], optional): Horizontal offset. Defaults to None.
            y_off (Optional[int], optional): Vertical offset. Defaults to None.
            width (Optional[int], optional): Width of the ROI. Defaults to None.
            height (Optional[int], optional): Height of the ROI. Defaults to None.
        """
        self.x_offset = x_off
        self.y_offset = y_off
        self.width = width
        self.height = height

    def __iter__(self):
        """
        Allows iteration over the ROI attributes in the order: x_offset, y_offset, width, height.
        Returns:
            iterator: An iterator over the ROI attributes.
        """
        return iter((self.x_offset, self.y_offset, self.width, self.height))


class DynamicsInformation:
    def __init__(self, velocity_source: Optional[str] = None):
        self.velocity_source = velocity_source


class IMUInformation:
    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        self.model_name = model_name
        self.extrinsic = extrinsic


class GNSSInformation:
    def __init__(self, model_name: Optional[str] = None, extrinsic: Optional[Pose] = None):
        self.model_name = model_name
        self.extrinsic = extrinsic


class CameraInformation:

    def __init__(self,
                 name: str,
                 model_name: Optional[str] = None,
                 shape: Optional[Tuple[int, int]] = None,
                 distortion_type: Optional[str] = None,
                 camera_mtx: Optional[np.array] = None,
                 distortion_mtx: Optional[np.array] = None,
                 rectification_mtx: Optional[np.array] = None,
                 projection_mtx: Optional[np.array] = None,
                 region_of_interest: Optional[ROI] = None,
                 camera_type: Optional[str] = None,
                 focal_length: Optional[int] = None,
                 aperture: Optional[int] = None,
                 exposure_time: Optional[int] = None,
                 extrinsic: Optional[Pose] = None,
                 stereo_transform: Optional[TransformationMtx] = None):
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


class LidarInformation:
    ouster_datatype_structure = {
        'names': [
            'x',  # x-coordinate of the point
            'y',  # y-coordinate of the point
            'z',  # z-coordinate of the point
            'intensity',  # Intensity of the point
            't',  # Time after the frame timestamp in ns
            'reflectivity',  # Reflectivity of the point
            'ring',  # Ring number (for multi-beam LiDARs)
            'ambient',  # Ambient light intensity
            'range'  # Distance from the LiDAR sensor to the measured point (hypotenuse) in mm.
        ],
        'formats': ['<f4', '<f4', '<f4', '<f4', '<u4', '<u2', '<u2', '<u2', '<u4'],
        'offsets': [0, 4, 8, 16, 20, 24, 26, 28, 32],
        'itemsize': 48
    }

    blickfeld_datatype_structure = {
        'names': [
            'x',  # x-coordinate of the point
            'y',  # y-coordinate of the point
            'z',  # z-coordinate of the point
            'intensity',  # Intensity of the point
            'point_id',
        ],
        'formats': ['<f4', '<f4', '<f4', '<u4', '<u4'],
        'offsets': [0, 4, 8, 12, 16],
        'itemsize': 20
    }

    def __init__(self,
                 name: str,
                 model_name: Optional[str] = None,
                 beam_altitude_angles: Optional[np.array] = None,
                 beam_azimuth_angles: Optional[np.array] = None,
                 lidar_origin_to_beam_origin_mm: Optional[np.array] = None,
                 columns_per_frame: Optional[int] = None,
                 pixels_per_column: Optional[int] = None,
                 phase_lock_offset: Optional[int] = None,
                 lidar_to_sensor_transform: Optional[np.array] = None,
                 extrinsic: Optional[Pose] = None):

        self._model_name = None
        self.name = name
        self.model_name = model_name  # This will trigger the setter and update dtype
        self.beam_altitude_angles = beam_altitude_angles
        self.beam_azimuth_angles = beam_azimuth_angles
        self.lidar_origin_to_beam_origin_mm = lidar_origin_to_beam_origin_mm
        self.columns_per_frame = columns_per_frame
        self.pixels_per_column = pixels_per_column
        self.phase_lock_offset = phase_lock_offset
        self.lidar_to_sensor_transform = lidar_to_sensor_transform
        self.extrinsic = extrinsic

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @model_name.setter
    def model_name(self, value: Optional[str]):
        self._model_name = value
        if value is not None:
            if 'OS' in value:
                dtype_structure = LidarInformation.ouster_datatype_structure
            elif 'Blickfeld' in value:
                dtype_structure = LidarInformation.blickfeld_datatype_structure
            else:
                dtype_structure = None
            self.dtype = np.dtype(dtype_structure) if dtype_structure else None
        else:
            self.dtype = None
