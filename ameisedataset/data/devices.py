import dill
import json
from typing import List, Tuple, Optional
import numpy as np
from decimal import Decimal

from ameisedataset.data import Camera, Lidar, IMU, GNSS


class SensorInformation:
    """
    Represents a collection of metadata information about a dataset.
    Attributes:
        filename (str): Name of the dataset file.
        SHA256 (str): SHA256 checksum of the dataset.
        cameras (List[CameraInformation]): List of camera information associated with the dataset.
        lidar (List[LidarInformation]): List of lidar information associated with the dataset.
    """
    def __init__(self, filename: str = ""):
        """
        Initializes the Infos object with the provided dataset filename.
        Sets default values for SHA256, cameras, and lidar attributes.
        Parameters:
            filename (str, optional): Name of the dataset file. Defaults to an empty string.
        """
        self.filename: str = filename
        self.SHA256: str = ""
        #TODO: Implement version of ad
        self.version: float = 0.0
        self.cameras: List[CameraInformation] = [CameraInformation()] * NUM_CAMERAS
        self.lidar: List[LidarInformation] = [LidarInformation()] * NUM_LIDAR
        self.gnss: GNSSInformation = GNSSInformation()

    def get_info_lists(self) -> Tuple[List[int], List[int], bool]:
        """
        Retrieves indices of cameras and lidars based on specific conditions.
        Returns:
            Tuple[List[int], List[int]]:
                - First list contains indices of cameras with non-zero shape.
                - Second list contains indices of lidars with defined dtype.
        """
        camera_indices = [idx for idx, item in enumerate(self.cameras) if item.shape[0] != 0]
        lidar_indices = [idx for idx, item in enumerate(self.lidar) if item.name != '']
        gnss_available = True if self.gnss.name != '' else False
        return camera_indices, lidar_indices, gnss_available

    def to_bytes(self):
        available_cam_info, available_lidar_info, available_gnss_info = self.get_info_lists()
        cam_info_to_write = [self.cameras[idx] for idx in available_cam_info]
        lidar_info_to_write = [self.lidar[idx] for idx in available_lidar_info]
        gnss_info_to_write = []
        if available_gnss_info:
            gnss_info_to_write.append(self.gnss)

        chunk_info = []
        sensor_info_array = cam_info_to_write + lidar_info_to_write + gnss_info_to_write

        for sensor in sensor_info_array:
            chunk_info.append(sensor.name)

        chunk_info_bytes = dill.dumps(chunk_info)
        chunk_info_len = len(chunk_info_bytes).to_bytes(4, 'big')
        info_bytes = dill.dumps(self)
        info_len = len(info_bytes).to_bytes(4, 'big')

        combined_info = chunk_info_len + chunk_info_bytes + info_len + info_bytes

        combined_info_len = len(combined_info).to_bytes(4, 'big')
        combined_data_checksum = compute_checksum(combined_info)

        return combined_info_len + combined_data_checksum + combined_info

    @classmethod
    def from_bytes(cls, data):
        chunk_info_len = int.from_bytes(data[:INT_LENGTH], 'big')
        chunk_info_bytes = data[INT_LENGTH:INT_LENGTH + chunk_info_len]
        chunk_info = dill.loads(chunk_info_bytes)

        offset = INT_LENGTH + chunk_info_len
        info_len = int.from_bytes(data[offset:offset + INT_LENGTH], 'big')
        offset += INT_LENGTH
        info_bytes = data[offset:offset + info_len]
        info = dill.loads(info_bytes)

        return chunk_info, info


class VisionSensorsVeh:
    def __init__(self):
        self.BACK_LEFT: Camera = Camera()
        self.FRONT_LEFT: Camera = Camera()
        self.STEREO_LEFT: Camera = Camera()
        self.STEREO_RIGHT: Camera = Camera()
        self.FRONT_RIGHT: Camera = Camera()
        self.BACK_RIGHT: Camera = Camera()
        self.REAR: Optional[Camera] = None    # not implemented


class LaserSensorsVeh:
    def __init__(self):
        self.LEFT: Lidar = Lidar()
        self.TOP: Lidar = Lidar()
        self.RIGHT: Lidar = Lidar()
        self.REAR: Optional[Lidar] = None     # not implemented

class VisionSensorsTow:
    def __init__(self):
        self.VIEW_1: Camera = Camera()
        self.VIEW_2: Camera = Camera()


class LaserSensorsTow:
    def __init__(self):
        self.VIEW_1: Lidar = Lidar()
        self.VIEW_2: Lidar = Lidar()
        self.TOP: Lidar = Lidar()


class Tower:
    def __init__(self):
        self.cameras: VisionSensorsTow = VisionSensorsTow()
        self.lidars: LaserSensorsTow = LaserSensorsTow()
        self.GNSS: GNSS = GNSS()


class Vehicle:
    def __init__(self):
        self.cameras: VisionSensorsVeh = VisionSensorsVeh()
        self.lidars: LaserSensorsVeh = LaserSensorsVeh()
        self.IMU: IMU = IMU()
        self.GNSS: GNSS = GNSS()
