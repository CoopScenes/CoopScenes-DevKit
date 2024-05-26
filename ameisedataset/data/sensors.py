import numpy as np
from typing import List, Optional
from ameisedataset.miscellaneous import serialize, deserialize, obj_to_bytes, obj_from_bytes, read_data_block
from ameisedataset.data import Image, Points, Motion, Position, CameraInformation, LidarInformation, GNSSInformation, \
    IMUInformation, Velocity, DynamicsInformation


class Camera:
    def __init__(self, image: Optional[Image] = None, info: Optional[CameraInformation] = None):
        self.image = image
        self.info = info

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
    def __init__(self, points: Optional[Points] = None, info: Optional[LidarInformation] = None):
        self.points = points
        self.info = info

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
    def __init__(self, motion: Optional[List[Motion]] = None, info: Optional[IMUInformation] = None):
        self.motion = motion
        self.info = info

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


class Dynamics:
    def __init__(self, velocity: Optional[List[Velocity]] = None, info: Optional[DynamicsInformation] = None):
        self.velocity = velocity
        self.info = info

    def to_bytes(self) -> bytes:
        return serialize(self.velocity) + obj_to_bytes(self.info)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Dynamics':
        instance = cls()
        velocity_bytes, data = read_data_block(data)
        info_bytes, _ = read_data_block(data)
        setattr(instance, 'velocity', obj_from_bytes(velocity_bytes))
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        return instance


class GNSS:
    def __init__(self, motion: Optional[List[Position]] = None, info: Optional[GNSSInformation] = None):
        self.position = motion
        self.info = info

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
