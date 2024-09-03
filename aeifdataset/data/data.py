import numpy as np
from typing import Tuple, Optional, Dict
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from PIL import Image as PilImage

from aeifdataset.miscellaneous import read_data_block, unix_to_utc


class Velocity:
    def __init__(self, timestamp: Optional[Decimal] = None,
                 linear_velocity: Optional[np.array] = None,
                 angular_velocity: Optional[np.array] = None,
                 covariance: Optional[np.array] = None):
        self.timestamp: Optional[Decimal] = timestamp
        self.linear_velocity: Optional[np.array] = linear_velocity
        self.angular_velocity: Optional[np.array] = angular_velocity
        self.covariance: Optional[np.array] = covariance


class Heading:
    def __init__(self,
                 timestamp: Optional[Decimal] = None,
                 orientation: Optional[np.array] = None,
                 covariance: Optional[np.array] = None):
        self.timestamp: Optional[Decimal] = timestamp
        self.orientation: Optional[np.array] = orientation
        self.covariance: Optional[np.array] = covariance


class Motion:
    def __init__(self,
                 timestamp: Optional[Decimal] = None,
                 orientation: Optional[np.array] = None,
                 orientation_covariance: Optional[np.array] = None,
                 angular_velocity: Optional[np.array] = None,
                 angular_velocity_covariance: Optional[np.array] = None,
                 linear_acceleration: Optional[np.array] = None,
                 linear_acceleration_covariance: Optional[np.array] = None):
        self.timestamp: Optional[Decimal] = timestamp
        self.orientation: Optional[np.array] = orientation
        self.orientation_covariance: Optional[np.array] = orientation_covariance
        self.angular_velocity: Optional[np.array] = angular_velocity
        self.angular_velocity_covariance: Optional[np.array] = angular_velocity_covariance
        self.linear_acceleration: Optional[np.array] = linear_acceleration
        self.linear_acceleration_covariance: Optional[np.array] = linear_acceleration_covariance


class Position:
    """
    Represents the position information including timestamp, latitude, longitude, altitude,
    status, and covariance data.
    """

    def __init__(self, timestamp: Optional[Decimal] = None, status: Optional[str] = None,
                 services: Optional[Dict[str, Optional[bool]]] = None, latitude: Optional[Decimal] = None,
                 longitude: Optional[Decimal] = None, altitude: Optional[Decimal] = None,
                 covariance: Optional[np.array] = None, covariance_type: Optional[str] = None):
        self.timestamp: Optional[Decimal] = timestamp
        self.status: Optional[str] = status
        self.services: Optional[Dict[str, Optional[bool]]] = self.init_services(services)
        self.latitude: Optional[Decimal] = latitude
        self.longitude: Optional[Decimal] = longitude
        self.altitude: Optional[Decimal] = altitude
        self.covariance: Optional[np.array] = covariance
        self.covariance_type: Optional[str] = covariance_type

    def __iter__(self):
        return iter((self.latitude, self.longitude, self.timestamp))

    @staticmethod
    def init_services(services):
        """
        Initializes the services dictionary with default values for GPS, Glonass, Galileo, and Baidou.
        """
        default_services = {'GPS': None, 'Glonass': None, 'Galileo': None, 'Baidou': None}
        if services is None:
            return default_services
        for key in default_services:
            services.setdefault(key, default_services[key])
        return services


class Image:
    """
    Represents an image along with its metadata.
    Attributes:
        timestamp (Optional[Decimal]): Timestamp of the image as UNIX time, can be None.
        image (Optional[PilImage]): The actual image data, can be None.
    """

    def __init__(self, image: PilImage = None, timestamp: Optional[Decimal] = None):
        """
        Initializes the Image object with the provided image data and timestamp.
        Parameters:
            image (Optional[PilImage]): The actual image data. Defaults to None.
            timestamp (Optional[Decimal]): Timestamp of the image as UNIX time. Defaults to None.
        """
        self.image = image
        self.timestamp = timestamp

    def __getattr__(self, attr) -> PilImage:
        """
        Enables direct access to attributes of the `image` object.
        Parameters:
            attr (str): Name of the attribute to access.
        Returns:
            PilImage: Attribute value if it exists in the `image` object.
        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self.image, attr):
            return getattr(self.image, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        encoded_img = self.image.tobytes()
        encoded_ts = str(self.timestamp).encode('utf-8')
        img_len = len(encoded_img).to_bytes(4, 'big')
        ts_len = len(encoded_ts).to_bytes(4, 'big')
        image_bytes = img_len + encoded_img + ts_len + encoded_ts
        return image_bytes

    @classmethod
    def from_bytes(cls, data: bytes, shape: Tuple[int, int]):
        img_bytes, data = read_data_block(data)
        ts_bytes, _ = read_data_block(data)
        img_instance = cls()
        img_instance.timestamp = Decimal(ts_bytes.decode('utf-8'))
        img_instance.image = PilImage.frombytes("RGB", shape, img_bytes)
        return img_instance

    def get_timestamp(self, precision='ns', timezone_offset_hours=2):
        return unix_to_utc(self.timestamp, precision=precision, timezone_offset_hours=timezone_offset_hours)


class Points:
    """
    Represents a collection of points with an associated timestamp.
    Attributes:
        points (np.array): Array containing the points.
        timestamp (Decimal): Timestamp associated with the points.
    """

    def __init__(self, points: Optional[np.array] = None, timestamp: Optional[Decimal] = None):
        """
        Initializes the Points object with the provided points and timestamp.
        Parameters:
            points (np.array, optional): Array containing the points. Defaults to an empty array.
            timestamp (Decimal, optional): Timestamp associated with the points. Defaults to '0'.
        """
        self.points: np.array = points
        self.timestamp: Decimal = timestamp

    def __getattr__(self, attr) -> np.array:
        """
        Enables direct access to attributes of the `points` object.
        Parameters:
            attr (str): Name of the attribute to access.
        Returns:
            np.array: Attribute value if it exists.
        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self.points, attr):
            return getattr(self.points, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        encoded_pts = self.points.tobytes()
        encoded_ts = str(self.timestamp).encode('utf-8')
        pts_len = len(encoded_pts).to_bytes(4, 'big')
        ts_len = len(encoded_ts).to_bytes(4, 'big')
        laser_bytes = pts_len + encoded_pts + ts_len + encoded_ts
        return laser_bytes

    @classmethod
    def from_bytes(cls, data: bytes, dtype: np.dtype):
        """
        Creates a Points instance from byte data.
        Parameters:
            data_bytes (bytes): Byte data representing the points.
            ts_data (bytes): Byte data representing the timestamp.
            dtype (np.dtype): Data type of the points.
        Returns:
            Points: A Points instance initialized with the provided data.
        """
        pts_bytes, data = read_data_block(data)
        ts_bytes, _ = read_data_block(data)
        pts_instance = cls()
        pts_instance.timestamp = Decimal(ts_bytes.decode('utf-8'))
        pts_instance.points = np.frombuffer(pts_bytes, dtype=dtype)
        return pts_instance

    def get_timestamp(self, precision='ns', timezone_offset_hours=2):
        return unix_to_utc(self.timestamp, precision=precision, timezone_offset_hours=timezone_offset_hours)
