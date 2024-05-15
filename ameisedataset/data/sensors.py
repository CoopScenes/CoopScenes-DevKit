import numpy as np
from typing import List, Optional
from ameisedataset.miscellaneous import serialize, deserialize, obj_to_bytes, obj_from_bytes, read_data_block
from ameisedataset.data import Image, Points, Motion, Position, CameraInformation, LidarInformation, GNSSInformation, IMUInformation


class Camera:
    def __init__(self):
        self.image: Optional[Image] = None
        self.info: Optional[CameraInformation] = None

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.image, attr):
            return getattr(self.image, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return serialize(self.image) + obj_to_bytes(self.info)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Camera':
        instance = cls()
        setattr(instance, 'image', deserialize(data, Image)[0])
        info_bytes, _ = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        return instance


class Lidar:
    def __init__(self):
        self.points: Optional[Points] = None
        self.info: Optional[LidarInformation] = None

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.points, attr):
            return getattr(self.points, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return serialize(self.points) + obj_to_bytes(self.info)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Lidar':
        instance = cls()
        setattr(instance, 'points', deserialize(data, Points)[0])
        info_bytes, _ = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        return instance


class IMU:
    def __init__(self, name=''):
        self.motion: List[Motion] = []
        self.info: IMUInformation = IMUInformation(name=name)

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return serialize(self.motion) + obj_to_bytes(self.info)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'IMU':
        instance = cls()
        motion_bytes, data = read_data_block(data)
        info_bytes, _ = read_data_block(data)
        setattr(instance, 'motion', obj_from_bytes(motion_bytes))
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        return instance


class GNSS:
    def __init__(self, name=''):
        self.position: List[Position] = []
        self.info: GNSSInformation = GNSSInformation(name=name)

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return serialize(self.position) + obj_to_bytes(self.info)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'GNSS':
        instance = cls()
        position_bytes, data = read_data_block(data)
        info_bytes, _ = read_data_block(data)
        setattr(instance, 'position', obj_from_bytes(position_bytes))
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        return instance
