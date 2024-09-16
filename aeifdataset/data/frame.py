"""
This module defines the `Frame` class, which represents a single frame of data containing
both vehicle and tower sensor information. The `Frame` class manages metadata such as the
frame ID, timestamp, and version, along with the associated sensor data.

Classes:
    Frame: Represents a frame of data, including methods for serialization, deserialization,
           data integrity verification via checksums, and completeness checks.

Functions:
    to_bytes: Serializes the `Frame` object to a byte stream, including a checksum for data integrity.
    from_bytes: Deserializes a byte stream to create a `Frame` object, verifying the checksum.
    is_complete: Checks if all sensors in the `Frame` are filled.
    get_timestamp: Converts the frame's timestamp to a formatted UTC string with specified precision.
"""
from decimal import Decimal
from aeifdataset.miscellaneous import obj_to_bytes, obj_from_bytes, read_data_block, unix_to_utc, compute_checksum, \
    ChecksumError
from aeifdataset.data import Tower, Vehicle, VisionSensorsVeh, VisionSensorsTow, LaserSensorsVeh, LaserSensorsTow
from aeifdataset.miscellaneous.helper import read_checksum


class Frame:
    """Class representing a frame of data containing vehicle and tower sensor information.

    This class handles the storage and serialization of a frame's metadata,
    including its ID, timestamp, and version, as well as vehicle and tower sensor data.

    Attributes:
        frame_id (int): Unique identifier for the frame.
        timestamp (Decimal): Timestamp associated with the frame.
        vehicle (Vehicle): Vehicle sensor data associated with the frame.
        tower (Tower): Tower sensor data associated with the frame.
        version (float): Version of the frame format.
    """

    def __init__(self, frame_id: int, timestamp: Decimal, version: float):
        """Initialize the Frame object with the provided frame ID, timestamp, and version.

        Args:
            frame_id (int): Unique identifier for the frame.
            timestamp (Decimal): Timestamp associated with the frame.
            version (float): Version of the frame format.
        """
        self.frame_id: int = frame_id
        self.timestamp: Decimal = timestamp
        self.vehicle: Vehicle = Vehicle()
        self.tower: Tower = Tower()
        self.version: float = version

    def to_bytes(self) -> bytes:
        """Serialize the Frame object, including metadata, vehicle, and tower data, to bytes.

        The method computes a checksum for the entire frame data and appends it at the beginning
        of the serialized data for data integrity verification.

        Returns:
            bytes: The serialized byte representation of the Frame object, including the checksum.
        """
        # Serialize metadata, vehicle, and tower information
        meta_bytes = obj_to_bytes([self.frame_id, self.timestamp, self.version])
        veh_bytes = self.vehicle.to_bytes()
        tow_bytes = self.tower.to_bytes()

        # Combine all serialized components
        frame_bytes = meta_bytes + veh_bytes + tow_bytes
        # Add checksum for data integrity
        safe_frame_bytes = compute_checksum(frame_bytes) + frame_bytes
        return safe_frame_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "Frame":
        """Deserialize bytes to create a Frame object, verifying the checksum for data integrity.

        Args:
            data (bytes): The serialized byte data to deserialize.

        Returns:
            Frame: The deserialized Frame object.

        Raises:
            ChecksumError: If the checksum of the data does not match, indicating possible corruption.
        """
        # Extract and verify checksum
        frame_checksum, data = read_checksum(data)
        if compute_checksum(data) != frame_checksum:
            raise ChecksumError("Checksum mismatch. Data might be corrupted!")

        # Deserialize metadata, vehicle, and tower blocks
        meta_bytes, data = read_data_block(data)
        vehicle_bytes, data = read_data_block(data)
        tower_bytes, _ = read_data_block(data)

        # Extract metadata and initialize Frame object
        meta_data = obj_from_bytes(meta_bytes)
        frame = cls(frame_id=meta_data[0], timestamp=meta_data[1], version=meta_data[2])

        # Deserialize vehicle and tower data
        frame.vehicle = Vehicle.from_bytes(vehicle_bytes)
        frame.tower = Tower.from_bytes(tower_bytes)
        return frame

    def is_complete(self):
        """Check if all sensor fields in the frame are filled.

        Returns:
            list or bool: Returns True if all fields are filled, otherwise returns a list of unfilled fields.
        """
        unfilled_fields = []
        for agent in [self.vehicle, self.tower]:
            for attr_name, attr_value in agent.__dict__.items():
                if isinstance(attr_value, (VisionSensorsVeh, VisionSensorsTow)):
                    for sub_attr_name, sub_attr_value in attr_value.__dict__.items():
                        if sub_attr_value is None:
                            unfilled_fields.append(f'CAMERA_{sub_attr_name}')
                if isinstance(attr_value, (LaserSensorsVeh, LaserSensorsTow)):
                    for sub_attr_name, sub_attr_value in attr_value.__dict__.items():
                        if sub_attr_value is None:
                            unfilled_fields.append(f'LIDAR_{sub_attr_name}')
                elif attr_value is None:
                    unfilled_fields.append(attr_value)
        return True if not unfilled_fields else unfilled_fields

    def get_timestamp(self, precision='s', timezone_offset_hours=2) -> str:
        """Get the timestamp of the frame in UTC, formatted to the desired precision.

        Args:
            precision (str): Precision of the timestamp ('s' for seconds, 'ns' for nanoseconds). Defaults to 's'.
            timezone_offset_hours (int): The timezone offset in hours for local time adjustment. Defaults to 2.

        Returns:
            str: The formatted timestamp string in UTC.
        """
        return unix_to_utc(self.timestamp, precision=precision, timezone_offset_hours=timezone_offset_hours)
