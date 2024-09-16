"""
This module provides classes for managing and accessing data records in the AMEISE-Record format (.4mse).
It includes functionality to load, serialize, and manipulate frames of data, as well as to scan a directory
for .4mse files.

Classes:
    DataRecord: Represents a data record in the AMEISE-Record format. Handles loading frames from a .4mse file,
                provides access to individual frames, and serializes frames into bytes.

    Dataloader: Manages the loading of AMEISE-Record files from a specified directory. Provides access to these
                records and allows for retrieval by index or filename.
"""
from typing import List, Optional, Iterator
import os
import glob
from aeifdataset.data import *
from aeifdataset.miscellaneous import InvalidFileTypeError, obj_to_bytes, obj_from_bytes, INT_LENGTH


class DataRecord:
    """Class representing a data record in the AMEISE-Record format (.4mse).

    This class is responsible for loading, accessing, and manipulating individual frames from
    an AMEISE-Record file. It stores the sequence of frame lengths and the raw frame data.

    Attributes:
        name (Optional[str]): The name of the record file.
        num_frames (int): The number of frames in the record.
        frame_lengths (List[int]): List of lengths for each frame in the record.
        frames_data (bytes): Raw bytes representing the frames in the record.
    """

    def __init__(self, record_file: Optional[str] = None):
        """Initialize a DataRecord object.

        Args:
            record_file (Optional[str]): Path to the AMEISE-Record file to load.
                                         If None, an empty record is created.

        Raises:
            InvalidFileTypeError: If the provided file is not in the .4mse format.
        """
        self.name: Optional[str] = record_file
        self.num_frames: int = 0
        self.frame_lengths: List[int] = []
        self.frames_data: bytes = b""
        if record_file is not None:
            if os.path.splitext(record_file)[1] != ".4mse":
                raise InvalidFileTypeError("This is not a valid AMEISE-Record file.")
            with open(record_file, 'rb') as file:
                # Read frame_lengths (array with num_frames entries)
                frame_lengths_len: int = int.from_bytes(file.read(INT_LENGTH), 'big')
                self.frame_lengths = obj_from_bytes(file.read(frame_lengths_len))
                # Read frames
                self.frames_data: bytes = file.read()
            self.num_frames: int = len(self.frame_lengths)
            self.name = os.path.splitext(os.path.basename(record_file))[0]

    def __len__(self):
        """Return the number of frames in the DataRecord."""
        return self.num_frames

    def __getitem__(self, frame_index) -> Frame:
        """Get a specific frame by its index.

        Args:
            frame_index (int): The index of the frame to retrieve.

        Returns:
            Frame: The frame at the specified index.

        Raises:
            ValueError: If the frame index is out of range.
        """
        if frame_index < 0 or frame_index >= len(self.frame_lengths):
            raise ValueError("Frame index out of range.")
        start_pos = sum(self.frame_lengths[:frame_index])
        end_pos = start_pos + self.frame_lengths[frame_index]
        return Frame.from_bytes(self.frames_data[start_pos:end_pos])

    def __iter__(self) -> Iterator['Frame']:
        """Return an iterator over frames in the DataRecord.

        Yields:
            Iterator[Frame]: An iterator that yields Frame objects.
        """
        start_pos = 0
        for length in self.frame_lengths:
            end_pos = start_pos + length
            yield Frame.from_bytes(self.frames_data[start_pos:end_pos])
            start_pos = end_pos

    @staticmethod
    def to_bytes(frames: List[Frame]) -> bytes:
        """Serialize a list of frames into bytes.

        Args:
            frames (List[Frame]): List of Frame objects to serialize.

        Returns:
            bytes: The serialized byte representation of the frames.
        """
        frame_lengths: List[int] = []
        frames_bytes = b""
        for _frame in frames:
            frame_bytes = _frame.to_bytes()
            frame_lengths.append(len(frame_bytes))
            frames_bytes += frame_bytes
        frame_lengths_bytes = obj_to_bytes(frame_lengths)
        return frame_lengths_bytes + frames_bytes


class Dataloader:
    """Class responsible for loading and managing AMEISE-Record files from a directory.

    This class scans a specified directory for all files with the .4mse extension
    and provides access to these records.

    Attributes:
        data_dir (str): The path to the directory containing .4mse files.
        record_map (List[str]): List of paths to .4mse files in the directory.
    """

    def __init__(self, data_dir: str):
        """Initialize a Dataloader object with the specified data directory.

        Args:
            data_dir (str): The directory containing .4mse record files.
        """
        self.data_dir: str = os.path.join(data_dir)
        self.record_map: List[str] = glob.glob(os.path.join(self.data_dir, '*.4mse'))

    def __len__(self):
        """Return the number of records found in the directory."""
        return len(self.record_map)

    def __getitem__(self, item) -> 'DataRecord':
        """Get a specific DataRecord by index.

        Args:
            item (int): The index of the record to retrieve.

        Returns:
            DataRecord: The DataRecord object at the specified index.
        """
        return DataRecord(record_file=self.record_map[item])

    def __iter__(self) -> Iterator['DataRecord']:
        """Return an iterator over DataRecord objects in the directory.

        Yields:
            Iterator[DataRecord]: An iterator that yields DataRecord objects.
        """
        for record_path in self.record_map:
            yield DataRecord(record_file=record_path)
