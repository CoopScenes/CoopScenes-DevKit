import dill
from decimal import Decimal
import numpy as np
from PIL import Image as PilImage
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta, timezone

from ameisedataset.miscellaneous import INT_LENGTH, NUM_CAMERAS, NUM_LIDAR, NUM_IMU, compute_checksum
from enum import Enum

from ameisedataset.data import Image, Points, Motion, Position


class Camera:
    def __init__(self):
        self.info = "hi"

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


class Lidar:
    def __init__(self):
        pass

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


class IMU:
    def __init__(self):
        self.raw: List[Motion] = []
        self.ekf: List[Motion] = []

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


class GNSS:
    def __init__(self):
        self.raw: List[Position] = []
        self.ekf: List[Position] = []

    def __getattr__(self, attr) -> np.array:
        if hasattr(self.ekf, attr):
            return getattr(self.ekf, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
