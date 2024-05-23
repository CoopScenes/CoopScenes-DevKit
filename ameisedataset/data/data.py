import numpy as np

from enum import Enum
from typing import Tuple, Optional, Dict
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from PIL import Image as PilImage


def _convert_unix_to_utc(unix_timestamp_ns: Decimal, utc_offset_hours: int = 2) -> str:
    """
    Convert a Unix timestamp (in nanoseconds as Decimal) to a human-readable UTC string with a timezone offset.
    This function also displays milliseconds, microseconds, and nanoseconds.
    Parameters:
    - unix_timestamp_ns: Unix timestamp in nanoseconds as a Decimal.
    - offset_hours: UTC timezone offset in hours.
    Returns:
    - Human-readable UTC string with the given timezone offset and extended precision.
    """
    # Convert the Decimal to integer for calculations
    unix_timestamp_ns = int(unix_timestamp_ns)
    # Extract the whole seconds and the fractional part
    timestamp_s, fraction_ns = divmod(unix_timestamp_ns, int(1e9))
    milliseconds, remainder_ns = divmod(fraction_ns, int(1e6))
    microseconds, nanoseconds = divmod(remainder_ns, int(1e3))
    # Convert to datetime object and apply the offset
    dt = datetime.fromtimestamp(timestamp_s, timezone.utc) + timedelta(hours=utc_offset_hours)
    # Create the formatted string with extended precision
    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
    extended_precision = f".{milliseconds:03}{microseconds:03}{nanoseconds:03}"
    return formatted_time + extended_precision


class Velocity:
    def __init__(self, timestamp: Optional[Decimal] = None,
                 linear_velocity: Optional[np.array] = None,
                 angular_velocity: Optional[np.array] = None,
                 covariance: Optional[np.array] = None):
        self.timestamp: Optional[Decimal] = timestamp
        self.linear_velocity: Optional[np.array] = linear_velocity
        self.angular_velocity: Optional[np.array] = angular_velocity
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
    class NavSatFixStatus(Enum):
        NO_FIX = -1
        FIX = 0
        SBAS_FIX = 1
        GBAS_FIX = 2

    class CovarianceType(Enum):
        UNKNOWN = 0
        APPROXIMATED = 1
        DIAGONAL_KNOWN = 2
        KNOWN = 3

    def __init__(self, timestamp: Optional[Decimal] = None, status: Optional[NavSatFixStatus] = None,
                 services: Optional[Dict[str, Optional[bool]]] = None, latitude: Optional[Decimal] = None,
                 longitude: Optional[Decimal] = None, altitude: Optional[Decimal] = None,
                 covariance: Optional[np.array] = None, covariance_type: Optional[CovarianceType] = None):
        self.timestamp: Optional[Decimal] = timestamp
        self.status = status
        self.services: Optional[Dict[str, Optional[bool]]] = self.init_services(services)
        self.latitude: Optional[Decimal] = latitude
        self.longitude: Optional[Decimal] = longitude
        self.altitude: Optional[Decimal] = altitude
        self.covariance: Optional[np.array] = covariance
        self.covariance_type = covariance_type

    def __iter__(self):
        return iter((self.latitude, self.longitude, self.timestamp))

    @staticmethod
    def init_services(services):
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
    def from_bytes(cls, data_bytes: bytes, ts_data: bytes, shape: Tuple[int, int]):
        """
        Creates an Image instance from byte data.
        Args:
            data_bytes (bytes): Byte data of the image.
            ts_data (bytes): Serialized timestamp data associated with the image.
            shape (Tuple[int, int]): Dimensions of the image as (width, height).
        Returns:
            Image: An instance of the Image class populated with the provided data.
        """
        img_instance = cls()
        img_instance.timestamp = Decimal(ts_data.decode('utf-8'))
        img_instance.image = PilImage.frombytes("RGB", shape, data_bytes)
        return img_instance

    def get_timestamp(self, utc=2):
        """
        Retrieves the UTC timestamp of the points.
        Args:
            utc (int, optional): Timezone offset in hours. Default is 2.
        Returns:
            str: The UTC timestamp of the points.
        """
        return _convert_unix_to_utc(self.timestamp, utc_offset_hours=utc)


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
    def from_bytes(cls, data_bytes: bytes, ts_data: bytes, dtype: np.dtype):
        """
        Creates a Points instance from byte data.
        Parameters:
            data_bytes (bytes): Byte data representing the points.
            ts_data (bytes): Byte data representing the timestamp.
            dtype (np.dtype): Data type of the points.
        Returns:
            Points: A Points instance initialized with the provided data.
        """
        img_instance = cls()
        img_instance.timestamp = Decimal(ts_data.decode('utf-8'))
        img_instance.points = np.frombuffer(data_bytes, dtype=dtype)
        return img_instance

    def get_timestamp(self, utc=2):
        """
        Retrieves the UTC timestamp of the points.
        Args:
            utc (int, optional): Timezone offset in hours. Default is 2.
        Returns:
            str: The UTC timestamp of the points.
        """
        return _convert_unix_to_utc(self.timestamp, utc_offset_hours=utc)
