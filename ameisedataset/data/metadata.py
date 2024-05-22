import json
from typing import Tuple
import numpy as np


class DynamicsInformation:
    def __init__(self, velocity_source: str = 'Microstrain 3DM_GQ7'):
        self.velocity_source: str = velocity_source


class IMUInformation:
    def __init__(self, name: str = 'Microstrain 3DM_GQ7'):
        self.name: str = name


class GNSSInformation:
    def __init__(self, name: str = 'C099-F9P'):
        self.name: str = name


class CameraInformation:
    """ Represents detailed information about a camera.
    Attributes:
        name (str): Name of the camera.
        shape (Tuple[int, int]): Dimensions (width, height) of the camera's image.
        camera_mtx: Camera matrix.
        distortion_mtx: Distortion matrix.
        rectification_mtx: Rectification matrix.
        projection_mtx: Projection matrix.
        region_of_interest: Region of interest in the camera's view.
        camera_type (str): Type of the camera.
        focal_length (int): Focal length of the camera in millimeters.
        aperture (int): Aperture size of the camera.
        exposure_time (int): Exposure time of the camera in milliseconds.
    """
    def __init__(self, name: str = ''):
        self.name: str = name
        self.shape: Tuple[int, int] = (0, 0)
        self.distortion_type: str = ''
        self.camera_mtx: np.array = np.array([])
        self.distortion_mtx: np.array = np.array([])
        self.rectification_mtx: np.array = np.array([])
        self.projection_mtx: np.array = np.array([])
        self.region_of_interest: ROI = ROI()
        self.camera_type: str = ''
        self.focal_length: int = 0
        self.aperture: int = 0
        self.exposure_time: int = 0
        self.extrinsic: Pose = Pose()   # Transformation to Top_Lidar
        self.stereo_transform: TransformationMtx = TransformationMtx()


class LidarInformation:
    """ Represents detailed information about a LiDAR sensor.
    Attributes:
        name (str): Name of the LiDAR sensor.
        dtype: Data type of the LiDAR points.
        beam_altitude_angles: Altitude angles of the LiDAR beams.
        beam_azimuth_angles: Azimuth angles of the LiDAR beams.
        lidar_origin_to_beam_origin_mm: Distance from the LiDAR origin to the origin of the beams.
        columns_per_frame: Number of columns in each LiDAR frame.
        pixels_per_column: Number of pixels in each LiDAR column.
        phase_lock_offset: Phase lock offset of the LiDAR sensor.
        lidar_to_sensor_transform: Transformation matrix from the LiDAR to the sensor.
        type: Product line or type of the LiDAR sensor.
    """
    ouster_datatype_structure = {
        'names': [
            'x',             # x-coordinate of the point
            'y',             # y-coordinate of the point
            'z',             # z-coordinate of the point
            'intensity',     # Intensity of the point
            't',             # Time after the frame timestamp in ns
            'reflectivity',  # Reflectivity of the point
            'ring',          # Ring number (for multi-beam LiDARs)
            'ambient',       # Ambient light intensity
            'range'          # Distance from the LiDAR sensor to the measured point (hypotenuse) in mm.
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

    def __init__(self, name: str = "", dtype: str = "ouster"):
        """ Initialize a LidarInformation instance with a given name.
        Args:
            name (str): Name of the LiDAR sensor.
        """
        self.name: str = name
        if dtype == "ouster":
            dtype = LidarInformation.ouster_datatype_structure
        elif dtype == "blickfeld":
            dtype = LidarInformation.blickfeld_datatype_structure
        else:
            raise Exception("Unrecognized dtype!")
        self.dtype = np.dtype(dtype)
        self.beam_altitude_angles = None
        self.beam_azimuth_angles = None
        self.lidar_origin_to_beam_origin_mm = None
        self.columns_per_frame = None
        self.pixels_per_column = None
        self.phase_lock_offset = None
        self.lidar_to_sensor_transform = None
        self.type = None
        self.extrinsic: Pose = Pose()

    def add_from_json_lidar_info(self, laser_info_obj):
        """ Populate the Ouster LidarInformation attributes from a ROS (Roboter Operating System) ouster std_string_msg
        LiDAR info object.
        Args:
            laser_info_obj: json LiDAR info object.
        """
        data_dict = json.loads(laser_info_obj.data)
        self.beam_altitude_angles = data_dict["beam_intrinsics"]["beam_altitude_angles"]
        self.beam_azimuth_angles = data_dict["beam_intrinsics"]["beam_azimuth_angles"]
        self.lidar_origin_to_beam_origin_mm = data_dict["beam_intrinsics"]["lidar_origin_to_beam_origin_mm"]
        self.columns_per_frame = data_dict["lidar_data_format"]["columns_per_frame"]
        self.pixels_per_column = data_dict["lidar_data_format"]["pixels_per_column"]
        self.phase_lock_offset = data_dict["config_params"]["phase_lock_offset"]
        self.lidar_to_sensor_transform = data_dict["lidar_intrinsics"]["lidar_to_sensor_transform"]
        self.type = data_dict["sensor_info"]["prod_line"]


class Pose:
    """
    Describes the position of a sensor in terms of its position and rotation relative to
    the reference coordinate system (Top_LiDAR).
    Attributes:
        xyz (np.array): A 1x3 array representing the position of the sensor in the
                        reference coordinate system (Top_LiDAR).
        rpy (np.array): A 1x3 array representing the roll, pitch, and yaw angles of the sensor,
                        describing its rotation in itself.
    """
    def __init__(self):
        """
        Initializes the Pose with default position (0, 0, 0) and rotation (0, 0, 0).
        """
        self.xyz: np.array = np.array([0, 0, 0])
        self.rpy: np.array = np.array([0, 0, 0])


class TransformationMtx:
    """
    Represents a transformation matrix with separate rotation and translation components.
    Attributes:
        rotation (np.array): A 3x3 matrix representing the rotation component of the transformation.
        translation (np.array): A 1x3 matrix representing the translation component of the transformation.
    """
    def __init__(self):
        """
        Initializes the TransformationMtx with zero rotation and translation matrices.
        """
        self.rotation: np.array = np.zeros((3, 3))
        self.translation: np.array = np.zeros((1, 3))


class ROI:
    """
    Represents a Region of Interest (ROI) defined by its offset and dimensions.
    Attributes:
        x_offset (int): The horizontal offset of the ROI.
        y_offset (int): The vertical offset of the ROI.
        width (int): The width of the ROI.
        height (int): The height of the ROI.
    """
    def __init__(self, x_off=0, y_off=0, width=0, height=0):
        """
        Initializes the ROI with the provided offset and dimensions.
        Defaults to an ROI at the origin with zero width and height.
        Parameters:
            x_off (int, optional): Horizontal offset. Defaults to 0.
            y_off (int, optional): Vertical offset. Defaults to 0.
            width (int, optional): Width of the ROI. Defaults to 0.
            height (int, optional): Height of the ROI. Defaults to 0.
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
