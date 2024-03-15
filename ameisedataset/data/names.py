class InfoBaseClass(type):
    """ Base metaclass to provide dictionary-like access to class attributes.
    Allows retrieval of class attributes using dictionary key access style.
    """
    def __getitem__(cls, key):
        """ Retrieve class attribute value using key access.
        Args:
            key (str): Attribute key.
        Returns:
            Any: Value of the specified class attribute.
        """
        return getattr(cls, key)

    def is_type_of(cls, value):
        """ Check if the provided value corresponds to a defined camera type.
        Args:
            value (Any): Value to be checked.
        Returns:
            bool: True if the value is a defined camera type, False otherwise.
        """
        return value in cls.__dict__

    def get_name_by_value(cls, value):
        """ Retrieve the name of the class constant based on its value. """
        constants = {k: v for k, v in vars(cls).items() if not callable(v) and not k.startswith("__")}
        for name, val in constants.items():
            if val == value:
                return name
        return None


class Camera(metaclass=InfoBaseClass):
    """ Defines constants representing different camera types.
    Attributes:
        MONO_LEFT (int): Represents a mono left camera.
        STEREO_LEFT (int): Represents a stereo left camera.
        STEREO_RIGHT (int): Represents a stereo right camera.
        MONO_RIGHT (int): Represents a mono right camera.
    """
    BACK_LEFT = 0
    FRONT_LEFT = 1
    STEREO_LEFT = 2
    STEREO_RIGHT = 3
    FRONT_RIGHT = 4
    BACK_RIGHT = 5
    TOWER_1 = 6     # TOWER_A
    TOWER_2 = 7     # TOWER_B


class Lidar(metaclass=InfoBaseClass):
    """ Defines constants representing different Lidar types.
    Attributes:
        OS0_LEFT (int): Represents a OS0 left Lidar.
        OS1_TOP (int): Represents a OS1 top Lidar.
        OS0_RIGHT (int): Represents a OS0 right Lidar.
    """
    OS0_LEFT = 0
    OS1_TOP = 1
    OS0_RIGHT = 2
    BLICKFELD_1 = 3     # QUBE_A
    BLICKFELD_2 = 4     # QUBE_B


class IMU(metaclass=InfoBaseClass):
    BUS = 0


class GNSS(metaclass=InfoBaseClass):
    BUS = 0
    TOWER = 1
