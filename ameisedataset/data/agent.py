from typing import Optional
from ameisedataset.data import Camera, Lidar, IMU, GNSS, Dynamics
from ameisedataset.miscellaneous import serialize, deserialize, obj_to_bytes, obj_from_bytes


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
            setattr(instance, attr, deserialize(data, Camera)[0])
        return instance


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
            setattr(instance, attr, deserialize(data, Lidar)[0])
        return instance


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
            setattr(instance, attr, deserialize(data, Camera)[0])
        return instance


class LaserSensorsTow:
    def __init__(self):
        self.VIEW_1: Optional[Lidar] = None
        self.VIEW_2: Optional[Lidar] = None
        self.TOP: Optional[Lidar] = None

    def to_bytes(self):
        return b''.join(serialize(lidar) for lidar in [
            self.VIEW_1,
            self.VIEW_2,
            self.TOP
        ])

    @classmethod
    def from_bytes(cls, data) -> 'LaserSensorsTow':
        instance = cls()
        for attr in ['VIEW_1', 'VIEW_2', 'TOP']:
            setattr(instance, attr, deserialize(data, Lidar)[0])
        return instance


class Tower:
    def __init__(self):
        self.cameras: VisionSensorsTow = VisionSensorsTow()
        self.lidars: LaserSensorsTow = LaserSensorsTow()
        self.GNSS: Optional[GNSS] = GNSS(name="C099-F9P")

    def to_bytes(self):
        return self.cameras.to_bytes() + self.lidars.to_bytes() + serialize(self.GNSS)

    @classmethod
    def from_bytes(cls, data) -> 'Tower':
        instance = cls()
        instance.cameras = VisionSensorsTow.from_bytes(data)
        instance.lidars = LaserSensorsTow.from_bytes(data)
        instance.GNSS = deserialize(data, GNSS)[0]
        return instance


class Vehicle:
    def __init__(self):
        self.cameras: VisionSensorsVeh = VisionSensorsVeh()
        self.lidars: LaserSensorsVeh = LaserSensorsVeh()
        self.IMU: IMU = IMU(name="Microstrain 3DM_GQ7")
        self.GNSS: GNSS = GNSS(name="Microstrain 3DM_GQ7")
        self.dynamics: Dynamics = Dynamics(name="Microstrain 3DM_GQ7")

    def to_bytes(self):
        return self.cameras.to_bytes() + self.lidars.to_bytes() + serialize(self.IMU) + serialize(self.GNSS) + obj_to_bytes(self.odometry)

    @classmethod
    def from_bytes(cls, data) -> 'Vehicle':
        instance = cls()
        instance.cameras = VisionSensorsVeh.from_bytes(data)
        instance.lidars = LaserSensorsVeh.from_bytes(data)
        instance.IMU, data = deserialize(data, IMU)
        instance.GNSS, data = deserialize(data, GNSS)
        instance.odometry = obj_from_bytes(data)
        return instance
