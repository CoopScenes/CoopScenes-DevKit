"""
This module defines classes representing different sensors used in autonomous vehicle systems,
including cameras, lidar, IMU, and GNSS. Each class encapsulates the sensor's metadata and
associated data, such as images, point clouds, motion data, and positional information.

Classes:
    Camera: Represents a camera sensor, managing metadata and raw image data.
    Lidar: Represents a lidar sensor, handling point cloud data and sensor metadata.
    IMU: Represents an Inertial Measurement Unit, managing motion data and metadata.
    Dynamics: Represents vehicle dynamics, including velocity and heading.
    GNSS: Represents a GNSS sensor, handling metadata and position data.

Each class provides methods for serializing and deserializing the sensor data to and from bytes,
enabling easy storage and transmission of sensor information. They also offer utility functions
for accessing and manipulating the data, such as image rectification for cameras and dynamic
attribute access for lidar and IMU data.
"""
from typing import List, Optional
from aeifdataset.miscellaneous import serialize, deserialize, obj_to_bytes, obj_from_bytes, read_data_block
from aeifdataset.data import Image, Points, Motion, Position, CameraInformation, LidarInformation, GNSSInformation, \
    IMUInformation, Velocity, DynamicsInformation, Heading
from PIL import Image as PilImage
import numpy as np


class Camera:
    """Class representing a Camera sensor.

    The Camera class handles the camera information and raw image data. It provides
    properties for accessing rectified images and supports serialization.

    Attributes:
        info (Optional[CameraInformation]): Metadata about the camera.
        _image_raw (Optional[Image]): The raw image data.
    """

    def __init__(self, info: Optional[CameraInformation] = None, image: Optional[Image] = None):
        """Initialize a Camera object with camera information and raw image data.

        Args:
            info (Optional[CameraInformation]): Camera metadata.
            image (Optional[Image]): The raw image data.
        """
        self.info = info
        self._image_raw = image

    @property
    def image(self) -> PilImage:
        """Get the rectified image from the raw data.

        Returns:
            PilImage: The rectified image as a PIL image object.

        Raises:
            AttributeError: If the raw image data is not set.
        """
        from aeifdataset.utils import get_rect_img
        if self._image_raw is not None:
            return get_rect_img(self)
        raise AttributeError("Image is not set.")

    def __getattr__(self, attr):
        """Handle dynamic access to raw image attributes."""
        if self._image_raw is not None and hasattr(self._image_raw, attr):
            return getattr(self.image, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        """Serialize the camera data to bytes.

        Returns:
            bytes: Serialized byte representation of the camera's information and image.
        """
        return obj_to_bytes(self.info) + serialize(self._image_raw)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Camera':
        """Deserialize bytes to create a Camera object.

        Args:
            data (bytes): The serialized byte data to deserialize.

        Returns:
            Camera: The deserialized Camera object.
        """
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        image, _ = deserialize(data, Image, instance.info.shape)
        setattr(instance, '_image_raw', image)
        return instance


class Lidar:
    """Class representing a Lidar sensor.

    The Lidar class handles the lidar information and point cloud data. It provides
    dynamic access to point cloud attributes and supports serialization.

    Attributes:
        info (Optional[LidarInformation]): Metadata about the lidar.
        points (Optional[Points]): The point cloud data.
    """

    def __init__(self, info: Optional[LidarInformation] = None, points: Optional[Points] = None):
        """Initialize a Lidar object with lidar information and point cloud data.

        Args:
            info (Optional[LidarInformation]): Lidar metadata.
            points (Optional[Points]): The point cloud data.
        """
        self.info = info
        self.points = points

    def __getattr__(self, attr):
        """Handle dynamic access to point cloud attributes."""
        if self.points is not None and hasattr(self.points, attr):
            return getattr(self.points, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, index: int) -> np.array:
        """Enable subscriptable access to the points array.

        Args:
            index (int): Index to access the points array.

        Returns:
            np.array: The indexed points data.
        """
        if self.points is not None and self.points.points is not None:
            return self.points.points[index]
        raise IndexError(f"'{type(self).__name__}' object has no points data to index.")

    def to_bytes(self) -> bytes:
        """Serialize the lidar data to bytes.

        Returns:
            bytes: Serialized byte representation of the lidar's information and point cloud.
        """
        return obj_to_bytes(self.info) + serialize(self.points)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Lidar':
        """Deserialize bytes to create a Lidar object.

        Args:
            data (bytes): The serialized byte data to deserialize.

        Returns:
            Lidar: The deserialized Lidar object.
        """
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        points, _ = deserialize(data, Points, instance.info.dtype)
        setattr(instance, 'points', points)
        return instance


class IMU:
    """Class representing an IMU sensor.

    The IMU class handles IMU metadata and motion data. It provides dynamic
    access to motion attributes and supports serialization.

    Attributes:
        info (Optional[IMUInformation]): Metadata about the IMU.
        motion (List[Motion]): The motion data.
    """

    def __init__(self, info: Optional[IMUInformation] = None):
        """Initialize an IMU object with metadata and motion data.

        Args:
            info (Optional[IMUInformation]): IMU metadata.
        """
        self.info = info
        self.motion: List[Motion] = []

    def __getattr__(self, attr) -> np.array:
        """Handle dynamic access to motion attributes."""
        if hasattr(self.motion, attr):
            return getattr(self.motion, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, index: int) -> Motion:
        """Enable subscriptable access to the motion list."""
        return self.motion[index]

    def to_bytes(self) -> bytes:
        """Serialize the IMU data to bytes.

        Returns:
            bytes: Serialized byte representation of the IMU's information and motion data.
        """
        return obj_to_bytes(self.info) + obj_to_bytes(self.motion)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'IMU':
        """Deserialize bytes to create an IMU object.

        Args:
            data (bytes): The serialized byte data to deserialize.

        Returns:
            IMU: The deserialized IMU object.
        """
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        motion_bytes, _ = read_data_block(data)
        setattr(instance, 'motion', obj_from_bytes(motion_bytes))
        return instance


class Dynamics:
    """Class representing vehicle dynamics.

    The Dynamics class handles dynamic state information, including velocity
    and heading, and supports serialization.

    Attributes:
        info (Optional[DynamicsInformation]): Metadata about the dynamics.
        velocity (List[Velocity]): The velocity data.
        heading (List[Heading]): The heading data.
    """

    def __init__(self, info: Optional[DynamicsInformation] = None):
        """Initialize a Dynamics object with metadata, velocity, and heading data.

        Args:
            info (Optional[DynamicsInformation]): Dynamics metadata.
        """
        self.info = info
        self.velocity: List[Velocity] = []
        self.heading: List[Heading] = []

    def to_bytes(self) -> bytes:
        """Serialize the dynamics data to bytes.

        Returns:
            bytes: Serialized byte representation of the dynamics information, velocity, and heading data.
        """
        return obj_to_bytes(self.info) + obj_to_bytes(self.velocity) + obj_to_bytes(self.heading)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Dynamics':
        """Deserialize bytes to create a Dynamics object.

        Args:
            data (bytes): The serialized byte data to deserialize.

        Returns:
            Dynamics: The deserialized Dynamics object.
        """
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        velocity_bytes, data = read_data_block(data)
        setattr(instance, 'velocity', obj_from_bytes(velocity_bytes))
        heading_bytes, _ = read_data_block(data)
        setattr(instance, 'heading', obj_from_bytes(heading_bytes))
        return instance


class GNSS:
    """Class representing a GNSS sensor.

    The GNSS class handles GNSS metadata and position data. It provides dynamic
    access to position attributes and supports serialization.

    Attributes:
        info (Optional[GNSSInformation]): Metadata about the GNSS.
        position (List[Position]): The position data.
    """

    def __init__(self, info: Optional[GNSSInformation] = None):
        """Initialize a GNSS object with metadata and position data.

        Args:
            info (Optional[GNSSInformation]): GNSS metadata.
        """
        self.info = info
        self.position: List[Position] = []

    def __getattr__(self, attr) -> np.array:
        """Handle dynamic access to position attributes."""
        if hasattr(self.position, attr):
            return getattr(self.position, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, index: int) -> Position:
        """Enable subscriptable access to the position list."""
        return self.position[index]

    def to_bytes(self) -> bytes:
        """Serialize the GNSS data to bytes.

        Returns:
            bytes: Serialized byte representation of the GNSS information and position data.
        """
        return obj_to_bytes(self.info) + obj_to_bytes(self.position)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'GNSS':
        """Deserialize bytes to create a GNSS object.

        Args:
            data (bytes): The serialized byte data to deserialize.

        Returns:
            GNSS: The deserialized GNSS object.
        """
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        position_bytes, _ = read_data_block(data)
        setattr(instance, 'position', obj_from_bytes(position_bytes))
        return instance
