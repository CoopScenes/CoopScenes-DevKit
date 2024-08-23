import numpy as np
from typing import List, Optional
from aeifdataset.miscellaneous import serialize, deserialize, obj_to_bytes, obj_from_bytes, read_data_block
from aeifdataset.data import Image, Points, Motion, Position, CameraInformation, LidarInformation, GNSSInformation, \
    IMUInformation, Velocity, DynamicsInformation, Heading

from PIL import Image as PilImage


class Camera:
    def __init__(self, info: Optional[CameraInformation] = None, image: Optional[Image] = None):
        self.info = info
        self._image_raw = image

    @property
    def image(self) -> PilImage:
        from aeifdataset.utils import get_rect_img
        if self._image_raw is not None:
            return get_rect_img(self._image_raw, self.info)
        raise AttributeError("Image is not set.")

    def __getattr__(self, attr):
        if self._image_raw is not None and hasattr(self._image_raw, attr):
            return getattr(self._image_raw, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        return obj_to_bytes(self.info) + serialize(self._image_raw)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Camera':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        image, _ = deserialize(data, Image, instance.info.shape)
        setattr(instance, '_image_raw', image)
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
        self.heading: List[Heading] = []

    def to_bytes(self) -> bytes:
        return obj_to_bytes(self.info) + obj_to_bytes(self.velocity) + obj_to_bytes(self.heading)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Dynamics':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        velocity_bytes, data = read_data_block(data)
        setattr(instance, 'velocity', obj_from_bytes(velocity_bytes))
        heading_bytes, _ = read_data_block(data)
        setattr(instance, 'heading', obj_from_bytes(heading_bytes))
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
