import numpy as np
from typing import List, Optional
from ameisedataset.miscellaneous import serialize, deserialize, obj_to_bytes, obj_from_bytes, read_data_block
from ameisedataset.data import Image, Points, Motion, Position, CameraInformation, LidarInformation, GNSSInformation, \
    IMUInformation, Velocity, DynamicsInformation


class Camera:
    def __init__(self, info: Optional[CameraInformation] = None, image: Optional[Image] = None):
        self.info = info
        self.image = image

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.image, attr):
            return getattr(self.image, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return obj_to_bytes(self.info) + serialize(self.image)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Camera':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        image, _ = deserialize(data, Image, instance.info.shape)
        setattr(instance, 'image', image)
        return instance


class Lidar:
    def __init__(self, info: Optional[LidarInformation] = None, points: Optional[Points] = None):
        self.info = info
        self.points = points

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.points, attr):
            return getattr(self.points, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return obj_to_bytes(self.info) + serialize(self.points)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Lidar':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        points, _ = deserialize(data, Points, instance.info.dtype)
        setattr(instance, 'points', points)
        return instance


class IMU:
    def __init__(self, info: Optional[IMUInformation] = None):
        self.info = info
        self.motion: List[Motion] = []

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return obj_to_bytes(self.info) + obj_to_bytes(self.motion)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'IMU':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        motion_bytes, _ = read_data_block(data)
        setattr(instance, 'motion', obj_from_bytes(motion_bytes))
        return instance


class Dynamics:
    def __init__(self, info: Optional[DynamicsInformation] = None):
        self.info = info
        self.velocity: List[Velocity] = []

    def to_bytes(self) -> bytes:
        return obj_to_bytes(self.info) + obj_to_bytes(self.velocity)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Dynamics':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        velocity_bytes, _ = read_data_block(data)
        setattr(instance, 'velocity', obj_from_bytes(velocity_bytes))
        return instance


class GNSS:
    def __init__(self, info: Optional[GNSSInformation] = None):
        self.info = info
        self.position: List[Position] = []

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return obj_to_bytes(self.info) + obj_to_bytes(self.position)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'GNSS':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        position_bytes, _ = read_data_block(data)
        setattr(instance, 'position', obj_from_bytes(position_bytes))
        return instance
