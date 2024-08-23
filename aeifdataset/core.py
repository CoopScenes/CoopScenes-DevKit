import os
import glob

from typing import List, Optional
from aeifdataset.data import *
from aeifdataset.miscellaneous import InvalidFileTypeError, obj_to_bytes, obj_from_bytes, INT_LENGTH


class DataRecord:
    def __init__(self, record_file: Optional[str] = None):
        # expect record_file to be an absolute readable path
        self.name: Optional[str] = record_file
        # TODO: store sequence name
        self.num_frames: int = 0
        self.frame_lengths: List[int] = []
        self.frames_data: bytes = b""
        if record_file is not None:
            if os.path.splitext(record_file)[1] != ".4mse":
                raise InvalidFileTypeError("This is not a valid AMEISE-Record file.")
            with open(record_file, 'rb') as file:
                """ TODO: implement checksum
                record_checksum = file.read(SHA256_CHECKSUM_LENGTH)
                if compute_checksum(combined_data) != record_checksum:
                    raise ChecksumError("Checksum mismatch. Data might be corrupted!")
                """
                # Read frame_lengths, array with num_frames entries (int)
                frame_lengths_len: int = int.from_bytes(file.read(INT_LENGTH), 'big')
                self.frame_lengths = obj_from_bytes(file.read(frame_lengths_len))
                # Read frames
                self.frames_data: bytes = file.read()
            self.num_frames: int = len(self.frame_lengths)
            self.name = os.path.splitext(os.path.basename(record_file))[0]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, frame_index) -> Frame:
        if frame_index < 0 or frame_index >= len(self.frame_lengths):
            raise ValueError("Frame-Index out of range.")
        start_pos = sum(self.frame_lengths[:frame_index])
        end_pos = start_pos + self.frame_lengths[frame_index]
        return Frame.from_bytes(self.frames_data[start_pos:end_pos])

    @staticmethod
    def to_bytes(frames: List[Frame]) -> bytes:
        # frame lengths
        frame_lengths: List[int] = []
        # frame bytes
        frames_bytes = b""
        for _frame in frames:
            frame_bytes = _frame.to_bytes()
            frame_lengths.append(len(frame_bytes))
            frames_bytes += frame_bytes
        frame_lengths_bytes = obj_to_bytes(frame_lengths)
        # pack data sequence
        record_bytes = frame_lengths_bytes + frames_bytes
        # record_bytes_checksum = compute_checksum(record_bytes)
        return record_bytes  # record_bytes_checksum +


class Dataloader:
    # TODO: implement __iter__()
    def __init__(self, data_dir: str):
        self.data_dir: os.path = os.path.join(data_dir)
        self.record_map: List[str] = glob.glob(os.path.join(self.data_dir, '*.4mse'))

    def __len__(self):
        return len(self.record_map)

    def __getitem__(self, item) -> DataRecord:
        return DataRecord(record_file=self.record_map[item])

    def get_record_by_name(self, filename: str) -> Optional[DataRecord]:
        for record_path in self.record_map:
            if filename in record_path:
                return DataRecord(record_file=record_path)
        print(f"No record with name {filename} found.")
        return None

# amse_dataloader = Dataloader("/records")
# myRecord: DataRecord = amse_dataloader.get_record_by_name("2024-03-07-17-42-28_3_tower.4mse")
# myFrame: Frame = myRecord[7]
# print(myFrame.vehicle.cameras.FRONT_LEFT)
