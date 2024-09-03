from decimal import Decimal
from aeifdataset.miscellaneous import obj_to_bytes, obj_from_bytes, read_data_block, unix_to_utc
from aeifdataset.data import Tower, Vehicle, VisionSensorsVeh, VisionSensorsTow, LaserSensorsVeh, LaserSensorsTow


class Frame:
    def __init__(self, frame_id: int, timestamp: Decimal, version: float):
        """
        Initializes the Frame object with the provided frame ID and timestamp.
        Sets default values for cameras and lidar attributes.
        Parameters:
            frame_id (int): Unique identifier for the frame.
            timestamp (Decimal): Timestamp associated with the frame.
        """
        self.frame_id: int = frame_id
        self.timestamp: Decimal = timestamp
        self.vehicle: Vehicle = Vehicle()
        self.tower: Tower = Tower()
        self.version: float = version

    def to_bytes(self) -> bytes:
        # dump meta info like timestamp
        meta_bytes = obj_to_bytes([self.frame_id, self.timestamp, self.version])
        # vehicle bytes
        veh_bytes = self.vehicle.to_bytes()
        # tower bytes
        tow_bytes = self.tower.to_bytes()
        # always dumped in same pattern: len + content
        frame_bytes = meta_bytes + veh_bytes + tow_bytes
        return frame_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "Frame":
        # read block (len + content) by block and divide into fixed sequence: meta, vehicle, tower
        meta_bytes, data = read_data_block(data)
        vehicle_bytes, data = read_data_block(data)
        tower_bytes, data = read_data_block(data)
        # load meta
        meta_data = obj_from_bytes(meta_bytes)
        # create object with metadata
        frame = cls(frame_id=meta_data[0], timestamp=meta_data[1], version=meta_data[2])
        # vehicle content
        frame.vehicle = Vehicle.from_bytes(vehicle_bytes)
        # tower content
        frame.tower = Tower.from_bytes(tower_bytes)
        return frame

    def is_complete(self):
        """ Checks if all fields of the frame object are filled and returns a list of unfilled fields.
        Returns:
            list: List of strings representing the unfilled fields.
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
                    # check for len(obj.attr)
                    unfilled_fields.append(attr_value)
        return True if not unfilled_fields else unfilled_fields

    def get_timestamp(self, precision='s', timezone_offset_hours=2):
        return unix_to_utc(self.timestamp, precision=precision, timezone_offset_hours=timezone_offset_hours)
