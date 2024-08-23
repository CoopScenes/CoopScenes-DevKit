from typing import Optional
from aeifdataset.data import Camera, Lidar, IMU, GNSS, Dynamics, VehicleInformation, TowerInformation
from aeifdataset.miscellaneous import serialize, deserialize, INT_LENGTH, obj_to_bytes, obj_from_bytes, read_data_block


class VisionSensorsVeh:
    def __init__(self):
        self.BACK_LEFT: Optional[Camera] = None
        self.FRONT_LEFT: Optional[Camera] = None
        self.STEREO_LEFT: Optional[Camera] = None
        self.STEREO_RIGHT: Optional[Camera] = None
        self.FRONT_RIGHT: Optional[Camera] = None
        self.BACK_RIGHT: Optional[Camera] = None
        self.REAR: Optional[Camera] = None

    def to_bytes(self):
        return b''.join(serialize(camera) for camera in [
            self.BACK_LEFT,
            self.FRONT_LEFT,
            self.STEREO_LEFT,
            self.STEREO_RIGHT,
            self.FRONT_RIGHT,
            self.BACK_RIGHT,
            self.REAR
        ])

    @classmethod
    def from_bytes(cls, data) -> 'VisionSensorsVeh':
        instance = cls()
        for attr in ['BACK_LEFT', 'FRONT_LEFT', 'STEREO_LEFT', 'STEREO_RIGHT', 'FRONT_RIGHT', 'BACK_RIGHT', 'REAR']:
            camera, data = deserialize(data, Camera)
            setattr(instance, attr, camera)
        return instance, data


class LaserSensorsVeh:
    def __init__(self):
        self.LEFT: Optional[Lidar] = None
        self.TOP: Optional[Lidar] = None
        self.RIGHT: Optional[Lidar] = None
        self.REAR: Optional[Lidar] = None

    def to_bytes(self):
        return b''.join(serialize(lidar) for lidar in [
            self.LEFT,
            self.TOP,
            self.RIGHT,
            self.REAR
        ])

    @classmethod
    def from_bytes(cls, data) -> 'LaserSensorsVeh':
        instance = cls()
        for attr in ['LEFT', 'TOP', 'RIGHT', 'REAR']:
            lidar, data = deserialize(data, Lidar)
            setattr(instance, attr, lidar)
        return instance, data


class VisionSensorsTow:
    def __init__(self):
        self.VIEW_1: Optional[Camera] = None
        self.VIEW_2: Optional[Camera] = None

    def to_bytes(self):
        return b''.join(serialize(camera) for camera in [
            self.VIEW_1,
            self.VIEW_2
        ])

    @classmethod
    def from_bytes(cls, data) -> 'VisionSensorsTow':
        instance = cls()
        for attr in ['VIEW_1', 'VIEW_2']:
            camera, data = deserialize(data, Camera)
            setattr(instance, attr, camera)
        return instance, data


class LaserSensorsTow:
    def __init__(self):
        self.VIEW_1: Optional[Lidar] = None
        self.VIEW_2: Optional[Lidar] = None
        self.UPPER_PLATFORM: Optional[Lidar] = None

    def to_bytes(self):
        return b''.join(serialize(lidar) for lidar in [
            self.VIEW_1,
            self.VIEW_2,
            self.UPPER_PLATFORM
        ])

    @classmethod
    def from_bytes(cls, data) -> 'LaserSensorsTow':
        instance = cls()
        for attr in ['VIEW_1', 'VIEW_2', 'UPPER_PLATFORM']:
            lidar, data = deserialize(data, Lidar)
            setattr(instance, attr, lidar)
        return instance, data


class Tower:
    def __init__(self, info: Optional[TowerInformation] = None):
        self.info = info
        self.cameras: VisionSensorsTow = VisionSensorsTow()
        self.lidars: LaserSensorsTow = LaserSensorsTow()
        self.GNSS: Optional[GNSS] = GNSS()

    def to_bytes(self):
        tower_bytes = obj_to_bytes(self.info) + self.cameras.to_bytes() + self.lidars.to_bytes() + serialize(self.GNSS)
        return len(tower_bytes).to_bytes(INT_LENGTH, 'big') + tower_bytes

    @classmethod
    def from_bytes(cls, data) -> 'Tower':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        instance.cameras, data = VisionSensorsTow.from_bytes(data)
        instance.lidars, data = LaserSensorsTow.from_bytes(data)
        instance.GNSS, _ = deserialize(data, GNSS)
        return instance


class Vehicle:
    def __init__(self, info: Optional[VehicleInformation] = None):
        self.info = info
        self.cameras: VisionSensorsVeh = VisionSensorsVeh()
        self.lidars: LaserSensorsVeh = LaserSensorsVeh()
        self.IMU: IMU = IMU()
        self.GNSS: GNSS = GNSS()
        self.DYNAMICS: Dynamics = Dynamics()

    def to_bytes(self):
        vehicle_bytes = obj_to_bytes(self.info) + self.cameras.to_bytes() + self.lidars.to_bytes() + serialize(
            self.IMU) + serialize(
            self.GNSS) + serialize(self.DYNAMICS)
        return len(vehicle_bytes).to_bytes(INT_LENGTH, 'big') + vehicle_bytes

    @classmethod
    def from_bytes(cls, data) -> 'Vehicle':
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        instance.cameras, data = VisionSensorsVeh.from_bytes(data)
        instance.lidars, data = LaserSensorsVeh.from_bytes(data)
        instance.IMU, data = deserialize(data, IMU)
        instance.GNSS, data = deserialize(data, GNSS)
        instance.DYNAMICS, _ = deserialize(data, Dynamics)
        return instance
