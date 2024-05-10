import dill

from decimal import Decimal
from ameisedataset.miscellaneous import INT_LENGTH, NUM_CAMERAS, NUM_LIDAR, NUM_IMU, compute_checksum
from ameisedataset.data import Camera, Lidar, IMU, Tower, Vehicle
from typing import List, Tuple, Optional, Dict
from data import Image, Points

def _read_data_block(data, offset):
    data_len = int.from_bytes(data[offset:offset + INT_LENGTH], 'big')
    offset += INT_LENGTH
    data_bytes = data[offset:offset + data_len]
    offset += data_len
    return data_bytes, offset


def _to_bytes(obj) -> bytes:
    obj_bytes = dill.dumps(obj)
    obj_bytes_len = len(obj_bytes).to_bytes(INT_LENGTH, 'big')
    return obj_bytes_len + obj_bytes


def _from_bytes(data: bytes):
    return dill.loads(data)


class Frame:
    """
    Represents a frame containing both images and points.
    Attributes:
        frame_id (int): Unique identifier for the frame.
        timestamp (str): Timestamp associated with the frame.
        cameras (List[Image]): List of images associated with the frame.
        lidar (List[Points]): List of point data associated with the frame.
    """

    def __init__(self, frame_id: int, timestamp: Decimal):
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


    @classmethod
    def from_bytes(cls, data, meta_info):
        """
        Creates a Frame instance from compressed byte data.
        Args:
            data (bytes): Compressed byte data representing the frame.
            meta_info (Infos): Metadata information about the frame's data types.
        Returns:
            Frame: An instance of the Frame class.
        """
        # Extract frame information length and data
        frame_info_len = int.from_bytes(data[:INT_LENGTH], 'big')
        frame_info_bytes = data[INT_LENGTH:INT_LENGTH + frame_info_len]
        frame_info = dill.loads(frame_info_bytes)
        # [self.frame_id, self.timestamp]
        frame_instance = cls(frame_info[0], frame_info[1])
        # Initialize offset for further data extraction
        offset = INT_LENGTH + frame_info_len
        for info_name in frame_info[2:]:
            # Check if the info name corresponds to a Camera type
            if Camera.is_type_of(info_name.upper()):
                # Extract image length and data
                camera_img_bytes, offset = _read_data_block(data, offset)
                # Extract timestamp
                ts_img_bytes, offset = _read_data_block(data, offset)
                # Create Image instance and store it in the frame instance
                frame_instance.cameras[Camera[info_name.upper()]] = Image.from_bytes(camera_img_bytes, ts_img_bytes,
                                                                                     meta_info.cameras[Camera[
                                                                                         info_name.upper()]].shape)
            # Check if the info name corresponds to a Lidar type
            elif Lidar.is_type_of(info_name.upper()):
                # Extract points length and data
                laser_pts_bytes, offset = _read_data_block(data, offset)
                # extract timestamp
                ts_laser_bytes, offset = _read_data_block(data, offset)
                # Create Points instance and store it in the frame instance
                # .lidar[Lidar.OS1_TOP].dtype
                frame_instance.lidar[Lidar[info_name.upper()]] = Points.from_bytes(laser_pts_bytes, ts_laser_bytes,
                                                                                   dtype=meta_info.lidar[
                                                                                       Lidar[info_name.upper()]].dtype)
            elif IMU.is_type_of(info_name.upper()):
                imu_bytes, offset = _read_data_block(data, offset)
                frame_instance.imu[IMU[info_name.upper()]] = _from_bytes(imu_bytes)
            elif info_name == 'GNSS':
                gnss_bytes, offset = _read_data_block(data, offset)
                frame_instance.gnss = _from_bytes(gnss_bytes)
        # Return the fully populated frame instance
        return frame_instance

    def to_bytes(self):
        """
        Converts the Frame instance to compressed byte data.
        Returns:
            bytes: Compressed byte representation of the Frame.
        """
        # convert data to bytes
        image_bytes = b""
        laser_bytes = b""
        imu_bytes = b""
        camera_indices, lidar_indices, imu_indices, gnss_available = self.get_data_lists()
        frame_info = [self.frame_id, self.timestamp]
        for data_index in camera_indices:
            frame_info.append(Camera.get_name_by_value(data_index))
        for data_index in lidar_indices:
            frame_info.append(Lidar.get_name_by_value(data_index))
        for data_index in imu_indices:
            frame_info.append(IMU.get_name_by_value(data_index))
        if gnss_available:
            frame_info.append("GNSS")
            gnss_bytes = _to_bytes(self.gnss)
        else:
            gnss_bytes = b""
        frame_info_bytes = dill.dumps(frame_info)
        frame_info_len = len(frame_info_bytes).to_bytes(4, 'big')
        # Encode images together with their time
        cam_msgs_to_write = [self.cameras[idx] for idx in camera_indices]
        for img_obj in cam_msgs_to_write:
            image_bytes += img_obj.to_bytes()
        # Encode laser points
        lidar_msgs_to_write = [self.lidar[idx] for idx in lidar_indices]
        for laser in lidar_msgs_to_write:
            laser_bytes += laser.to_bytes()
        imu_msgs_to_write = [self.imu[idx] for idx in imu_indices]
        for imu in imu_msgs_to_write:
            imu_bytes += _to_bytes(imu)

        # pack bytebuffer all together and compress them to one package
        combined_data = frame_info_len + frame_info_bytes + image_bytes + laser_bytes + imu_bytes + gnss_bytes
        # compressed_data = combined_data  #zlib.compress(combined_data)  # compress if something is compressable
        # calculate length and checksum
        combined_data_len = len(combined_data).to_bytes(4, 'big')
        combined_data_checksum = compute_checksum(combined_data)
        # return a header with the length and byteorder
        return combined_data_len + combined_data_checksum + combined_data

    def get_data_lists(self) -> Tuple[List[int], List[int], bool]:
        """
        Retrieves indices of cameras and lidars based on specific conditions.
        Returns:
            Tuple[List[int], List[int]]:
                - First list contains indices of cameras with non-null images.
                - Second list contains indices of lidar data with non-zero size.
        """
        camera_indices = [idx for idx, image in enumerate(self.cameras) if image.image is not None]
        lidar_indices = [idx for idx, points in enumerate(self.lidar) if points.points is not None]
        imu_indices = [idx for idx, imu in enumerate(self.imu) if imu[-1].timestamp is not None]
        gnss_available = False if self.gnss.status is None else True
        return camera_indices, lidar_indices, imu_indices, gnss_available