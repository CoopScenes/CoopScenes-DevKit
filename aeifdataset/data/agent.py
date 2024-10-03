"""
This module defines classes representing vehicles and sensor towers, grouping various
sensor types such as cameras and lidars into cohesive structures. These classes facilitate
the management, serialization, and deserialization of sensor data for autonomous vehicles
and sensor towers.

Classes:
    VisionSensorsVeh: Groups vision (camera) sensors mounted on a vehicle.
    LaserSensorsVeh: Groups laser (lidar) sensors mounted on a vehicle.
    VisionSensorsTow: Groups vision (camera) sensors mounted on a sensor tower.
    LaserSensorsTow: Groups laser (lidar) sensors mounted on a sensor tower.
    Tower: Represents a sensor tower with grouped sensors and relevant information.
    Vehicle: Represents a vehicle with grouped sensors and relevant information.

Each class provides methods to serialize and deserialize the sensor data, enabling the
transfer and storage of complex sensor setups in a compact binary format.
"""
from typing import Optional
from aeifdataset.data import Camera, Lidar, IMU, GNSS, Dynamics, VehicleInformation, TowerInformation
from aeifdataset.miscellaneous import serialize, deserialize, INT_LENGTH, obj_to_bytes, obj_from_bytes, read_data_block


class VisionSensorsVeh:
    """Class representing a grouping of vision sensors for a vehicle.

    This class organizes the camera sensors mounted on a vehicle for easier access,
    e.g., `frame.vehicle.cameras.BACK_LEFT`.

    Attributes:
        BACK_LEFT (Optional[Camera]): Camera on the back-left side of the vehicle.
        FRONT_LEFT (Optional[Camera]): Camera on the front-left side of the vehicle.
        STEREO_LEFT (Optional[Camera]): Left stereo camera of the vehicle.
        STEREO_RIGHT (Optional[Camera]): Right stereo camera of the vehicle.
        FRONT_RIGHT (Optional[Camera]): Camera on the front-right side of the vehicle.
        BACK_RIGHT (Optional[Camera]): Camera on the back-right side of the vehicle.
        REAR (Optional[Camera]): Rear camera of the vehicle.
    """

    _CAMERA_NAMES = ['BACK_LEFT', 'FRONT_LEFT', 'STEREO_LEFT', 'STEREO_RIGHT', 'FRONT_RIGHT', 'BACK_RIGHT', 'REAR']

    def __init__(self):
        """Initialize the VisionSensorsVeh object with all cameras set to None."""
        self.BACK_LEFT: Optional[Camera] = None
        self.FRONT_LEFT: Optional[Camera] = None
        self.STEREO_LEFT: Optional[Camera] = None
        self.STEREO_RIGHT: Optional[Camera] = None
        self.FRONT_RIGHT: Optional[Camera] = None
        self.BACK_RIGHT: Optional[Camera] = None
        self.REAR: Optional[Camera] = None

    def __iter__(self):
        """Make the object iteratable over its cameras."""
        for name in self._CAMERA_NAMES:
            camera = getattr(self, name)
            if camera is not None:
                yield name, camera

    def __len__(self):
        """Return the number of non-None cameras."""
        return sum(1 for name in self._CAMERA_NAMES if getattr(self, name) is not None)

    def to_bytes(self) -> bytes:
        """Serialize all vision sensors to bytes.

        This method serializes all camera sensor objects of the vehicle into bytes using
        the `serialize` function. Each camera's serialized byte representation is preceded
        by a length prefix. If a sensor is `None`, a default placeholder is serialized.

        Returns:
            bytes: The serialized byte representation of all vision sensors, each prefixed by its length.
        """
        return b''.join(serialize(getattr(self, name)) for name in self._CAMERA_NAMES)

    @classmethod
    def from_bytes(cls, data) -> 'VisionSensorsVeh':
        """Deserialize bytes to create a VisionSensorsVeh object.

        This method deserializes a byte stream into a `VisionSensorsVeh` object,
        reconstructing each camera sensor by calling `deserialize`. The byte stream
        contains each camera's serialized data prefixed by its length.

        Args:
            data (bytes): The byte data to deserialize.

        Returns:
            VisionSensorsVeh: The deserialized VisionSensorsVeh object, with camera sensors re-initialized.
        """
        instance = cls()
        for name in cls._CAMERA_NAMES:
            camera, data = deserialize(data, Camera)
            setattr(instance, name, camera)
        return instance, data


class LaserSensorsVeh:
    """Class representing a grouping of laser sensors for a vehicle.

    Attributes:
        LEFT (Optional[Lidar]): Lidar sensor on the left side of the vehicle.
        TOP (Optional[Lidar]): Lidar sensor on the top of the vehicle.
        RIGHT (Optional[Lidar]): Lidar sensor on the right side of the vehicle.
        REAR (Optional[Lidar]): Lidar sensor at the rear of the vehicle.
    """

    _LIDAR_NAMES = ['LEFT', 'TOP', 'RIGHT', 'REAR']

    def __init__(self):
        """Initialize the LaserSensorsVeh object with all lidar sensors set to None."""
        self.LEFT: Optional[Lidar] = None
        self.TOP: Optional[Lidar] = None
        self.RIGHT: Optional[Lidar] = None
        self.REAR: Optional[Lidar] = None

    def __iter__(self):
        """Make the object iteratable over its LiDAR sensors."""
        for name in self._LIDAR_NAMES:
            lidar = getattr(self, name)
            if lidar is not None:
                yield name, lidar

    def __len__(self):
        """Return the number of non-None lidars."""
        return sum(1 for name in self._LIDAR_NAMES if getattr(self, name) is not None)

    def to_bytes(self) -> bytes:
        """Serialize all laser sensors to bytes.

        This method serializes all lidar sensor objects of the vehicle into bytes using
        the `serialize` function. Each lidar's serialized byte representation is preceded
        by a length prefix. If a sensor is `None`, a default placeholder is serialized.

        Returns:
            bytes: The serialized byte representation of all laser sensors, each prefixed by its length.
        """
        return b''.join(serialize(getattr(self, name)) for name in self._LIDAR_NAMES)

    @classmethod
    def from_bytes(cls, data) -> 'LaserSensorsVeh':
        """Deserialize bytes to create a LaserSensorsVeh object.

        This method deserializes a byte stream into a `LaserSensorsVeh` object,
        reconstructing each lidar sensor by calling `deserialize`. The byte stream
        contains each lidar's serialized data prefixed by its length.

        Args:
            data (bytes): The byte data to deserialize.

        Returns:
            LaserSensorsVeh: The deserialized LaserSensorsVeh object, with lidar sensors re-initialized.
        """
        instance = cls()
        for name in cls._LIDAR_NAMES:
            lidar, data = deserialize(data, Lidar)
            setattr(instance, name, lidar)
        return instance, data


class VisionSensorsTow:
    """Class representing a grouping of vision sensors for a tower.

    Attributes:
        VIEW_1 (Optional[Camera]): First camera view of the tower.
        VIEW_2 (Optional[Camera]): Second camera view of the tower.
    """

    _CAMERA_NAMES = ['VIEW_1', 'VIEW_2']

    def __init__(self):
        """Initialize the VisionSensorsTow object with all cameras set to None."""
        self.VIEW_1: Optional[Camera] = None
        self.VIEW_2: Optional[Camera] = None

    def __iter__(self):
        """Make the object iteratable over its cameras."""
        for name in self._CAMERA_NAMES:
            camera = getattr(self, name)
            if camera is not None:
                yield name, camera

    def __len__(self):
        """Return the number of non-None cameras."""
        return sum(1 for name in self._CAMERA_NAMES if getattr(self, name) is not None)

    def to_bytes(self) -> bytes:
        """Serialize all vision sensors to bytes.

        This method serializes all camera sensor objects of the tower into bytes using
        the `serialize` function. Each camera's serialized byte representation is preceded
        by a length prefix. If a sensor is `None`, a default placeholder is serialized.

        Returns:
            bytes: The serialized byte representation of all vision sensors, each prefixed by its length.
        """
        return b''.join(serialize(getattr(self, name)) for name in self._CAMERA_NAMES)

    @classmethod
    def from_bytes(cls, data) -> 'VisionSensorsTow':
        """Deserialize bytes to create a VisionSensorsTow object.

        This method deserializes a byte stream into a `VisionSensorsTow` object,
        reconstructing each camera sensor by calling `deserialize`. The byte stream
        contains each camera's serialized data prefixed by its length.

        Args:
            data (bytes): The byte data to deserialize.

        Returns:
            VisionSensorsTow: The deserialized VisionSensorsTow object, with camera sensors re-initialized.
        """
        instance = cls()
        for name in cls._CAMERA_NAMES:
            camera, data = deserialize(data, Camera)
            setattr(instance, name, camera)
        return instance, data


class LaserSensorsTow:
    """Class representing a grouping of laser sensors for a tower.

    Attributes:
        VIEW_1 (Optional[Lidar]): First lidar view of the tower.
        VIEW_2 (Optional[Lidar]): Second lidar view of the tower.
        UPPER_PLATFORM (Optional[Lidar]): Lidar sensor mounted on the upper platform of the tower.
    """

    _LIDAR_NAMES = ['VIEW_1', 'VIEW_2', 'UPPER_PLATFORM']

    def __init__(self):
        """Initialize the LaserSensorsTow object with all lidar sensors set to None."""
        self.VIEW_1: Optional[Lidar] = None
        self.VIEW_2: Optional[Lidar] = None
        self.UPPER_PLATFORM: Optional[Lidar] = None

    def __iter__(self):
        """Make the object iteratable over its LiDAR sensors."""
        for name in self._LIDAR_NAMES:
            lidar = getattr(self, name)
            if lidar is not None:
                yield name, lidar

    def __len__(self):
        """Return the number of non-None lidars."""
        return sum(1 for name in self._LIDAR_NAMES if getattr(self, name) is not None)

    def to_bytes(self) -> bytes:
        """Serialize all laser sensors to bytes.

        This method serializes all lidar sensor objects of the tower into bytes using
        the `serialize` function. Each lidar's serialized byte representation is preceded
        by a length prefix. If a sensor is `None`, a default placeholder is serialized.

        Returns:
            bytes: The serialized byte representation of all laser sensors, each prefixed by its length.
        """
        return b''.join(serialize(getattr(self, name)) for name in self._LIDAR_NAMES)

    @classmethod
    def from_bytes(cls, data) -> 'LaserSensorsTow':
        """Deserialize bytes to create a LaserSensorsTow object.

        This method deserializes a byte stream into a `LaserSensorsTow` object,
        reconstructing each lidar sensor by calling `deserialize`. The byte stream
        contains each lidar's serialized data prefixed by its length.

        Args:
            data (bytes): The byte data to deserialize.

        Returns:
            LaserSensorsTow: The deserialized LaserSensorsTow object, with lidar sensors re-initialized.
        """
        instance = cls()
        for name in cls._LIDAR_NAMES:
            lidar, data = deserialize(data, Lidar)
            setattr(instance, name, lidar)
        return instance, data


class Tower:
    """Class representing a tower with grouped sensors and relevant information.

    Attributes:
        info (Optional[TowerInformation]): Information about the tower.
        cameras (VisionSensorsTow): The grouped camera sensors for the tower.
        lidars (LaserSensorsTow): The grouped lidar sensors for the tower.
        GNSS (Optional[GNSS]): GNSS sensor data for the tower.
    """

    def __init__(self, info: Optional[TowerInformation] = None):
        """Initialize the Tower object.

        Args:
            info (Optional[TowerInformation]): Optional tower information.
        """
        self.info = info
        self.cameras: VisionSensorsTow = VisionSensorsTow()
        self.lidars: LaserSensorsTow = LaserSensorsTow()
        self.GNSS: Optional[GNSS] = GNSS()

    def to_bytes(self) -> bytes:
        """Serialize the tower object to bytes.

        This method serializes the tower's information, grouped camera and lidar sensors,
        and GNSS data into bytes. Each serialized data block is prefixed with its length
        using `obj_to_bytes`.

        Returns:
            bytes: The serialized byte representation of the tower, including sensor and GNSS data.
        """
        tower_bytes = obj_to_bytes(self.info) + self.cameras.to_bytes() + self.lidars.to_bytes() + serialize(self.GNSS)
        return len(tower_bytes).to_bytes(INT_LENGTH, 'big') + tower_bytes

    @classmethod
    def from_bytes(cls, data) -> 'Tower':
        """Deserialize bytes to create a Tower object.

        This method deserializes a byte stream into a `Tower` object, including its
        information, camera, lidar, and GNSS sensor data. Each data block is prefixed
        with its length and deserialized using the appropriate method (`obj_from_bytes`,
        `VisionSensorsTow.from_bytes`, `LaserSensorsTow.from_bytes`).

        Args:
            data (bytes): The byte data to deserialize.

        Returns:
            Tower: The deserialized Tower object, with all sensors and information re-initialized.
        """
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        instance.cameras, data = VisionSensorsTow.from_bytes(data)
        instance.lidars, data = LaserSensorsTow.from_bytes(data)
        instance.GNSS, _ = deserialize(data, GNSS)
        return instance


class Vehicle:
    """Class representing a vehicle with grouped sensors and relevant information.

    Attributes:
        info (Optional[VehicleInformation]): Information about the vehicle.
        cameras (VisionSensorsVeh): The grouped camera sensors for the vehicle.
        lidars (LaserSensorsVeh): The grouped lidar sensors for the vehicle.
        IMU (IMU): IMU sensor data for the vehicle.
        GNSS (GNSS): GNSS sensor data for the vehicle.
        DYNAMICS (Dynamics): Dynamic state data for the vehicle.
    """

    def __init__(self, info: Optional[VehicleInformation] = None):
        """Initialize the Vehicle object.

        Args:
            info (Optional[VehicleInformation]): Optional vehicle information.
        """
        self.info = info
        self.cameras: VisionSensorsVeh = VisionSensorsVeh()
        self.lidars: LaserSensorsVeh = LaserSensorsVeh()
        self.IMU: IMU = IMU()
        self.GNSS: GNSS = GNSS()
        self.DYNAMICS: Dynamics = Dynamics()

    def to_bytes(self) -> bytes:
        """Serialize the vehicle object to bytes.

        This method serializes the vehicle's information, grouped camera and lidar sensors,
        and sensor data (IMU, GNSS, Dynamics) into bytes. Each serialized data block is
        prefixed with its length using `obj_to_bytes`.

        Returns:
            bytes: The serialized byte representation of the vehicle, including sensor and dynamics data.
        """
        vehicle_bytes = (
                obj_to_bytes(self.info)
                + self.cameras.to_bytes()
                + self.lidars.to_bytes()
                + serialize(self.IMU)
                + serialize(self.GNSS)
                + serialize(self.DYNAMICS)
        )
        return len(vehicle_bytes).to_bytes(INT_LENGTH, 'big') + vehicle_bytes

    @classmethod
    def from_bytes(cls, data) -> 'Vehicle':
        """Deserialize bytes to create a Vehicle object.

        This method deserializes a byte stream into a `Vehicle` object, including its
        information, sensors (cameras, lidars, IMU, GNSS), and dynamics data. Each data
        block is prefixed with its length and deserialized using the appropriate method.

        Args:
            data (bytes): The byte data to deserialize.

        Returns:
            Vehicle: The deserialized Vehicle object, with all sensors and information re-initialized.
        """
        instance = cls()
        info_bytes, data = read_data_block(data)
        setattr(instance, 'info', obj_from_bytes(info_bytes))
        instance.cameras, data = VisionSensorsVeh.from_bytes(data)
        instance.lidars, data = LaserSensorsVeh.from_bytes(data)
        instance.IMU, data = deserialize(data, IMU)
        instance.GNSS, data = deserialize(data, GNSS)
        instance.DYNAMICS, _ = deserialize(data, Dynamics)
        return instance
