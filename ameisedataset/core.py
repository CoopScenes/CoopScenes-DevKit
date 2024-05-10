import os
import glob
from typing import List, Tuple

from ameisedataset.data import *
from ameisedataset.miscellaneous import compute_checksum, InvalidFileTypeError, ChecksumError, SHA256_CHECKSUM_LENGTH, INT_LENGTH, NUM_FRAMES_PER_RECORD



def _read_info_object(file):
    """Read and deserialize an info object from the file."""
    info_length = int.from_bytes(file.read(INT_LENGTH), 'big')
    info_checksum = file.read(SHA256_CHECKSUM_LENGTH)  # SHA-256 checksum length
    combined_info = file.read(info_length)

    # Verify checksum
    if compute_checksum(combined_info) != info_checksum:
        raise ChecksumError(f"Checksum of Info is not correct! Check file.")
    return Infos.from_bytes(combined_info)

"""
def _read_frame_object(file, meta_infos):
    combined_data_len = int.from_bytes(file.read(INT_LENGTH), 'big')
    combined_data_checksum = file.read(SHA256_CHECKSUM_LENGTH)  # SHA-256 checksum length
    combined_data = file.read(combined_data_len)
    # Verify checksum
    if compute_checksum(combined_data) != combined_data_checksum:
        raise ChecksumError("Checksum mismatch. Data might be corrupted!")
    return Frame.from_bytes(combined_data, meta_info=meta_infos)


def unpack_record(filename) -> Tuple[Infos, List[Frame]]:
    # Ensure the provided file has the correct extension
    if os.path.splitext(filename)[1] != ".4mse":
        raise InvalidFileTypeError("This is not a valid AMEISE-Record file.")
    frames: List[Frame] = []

    with open(filename, 'rb') as file:
        chunk_info, meta_info = _read_info_object(file)
        # Read num frames
        num_frames = int.from_bytes(file.read(INT_LENGTH), 'big')
        # Read frames
        for _ in range(num_frames):
            frames.append(_read_frame_object(file, meta_info))
    return meta_info, frames
"""


class DataRecord:
    def __init__(self, record_file: str):
        self.frame_lengths: List[int] = []
        with open(record_file, 'rb') as file:
            self.meta_information = _read_info_object(file)
            # Read num frames
            self.num_frames = int.from_bytes(file.read(INT_LENGTH), 'big')
            for _ in range(self.num_frames):
                self.frame_lengths.append(int.from_bytes(file.read(INT_LENGTH), 'big'))
            # Read frames
            self.frames_data = file.read()
        self.name = os.path.splitext(os.path.basename(record_file))[0]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, frame_index) -> Frame:
        if frame_index < 0 or frame_index >= len(self.frame_lengths):
            raise ValueError("Frame-Index out of range.")
        start_pos = sum(self.frame_lengths[:frame_index])
        end_pos = start_pos + self.frame_lengths[frame_index]
        return Frame.from_bytes(self.frames_data[start_pos:end_pos], self.meta_information)


class Dataloader:
    def __init__(self, data_dir: str):
        self.data_dir: os.path = os.path.join(data_dir)
        self.record_map: List[str] = glob.glob(os.path.join(self.data_dir, '*.4mse'))

    def __len__(self):
        return len(self.record_map)

    def __getitem__(self, item) -> DataRecord:
        return DataRecord(record_file=self.record_map[item])

    def get_record_by_name(self, filename: str) -> DataRecord:
        for record_path in self.record_map:
            if filename in record_path:
                return DataRecord(record_file=record_path)
        print(f"No record with name {filename} found.")
        return None


amse_dataloader = Dataloader("/records")
myRecord: DataRecord = amse_dataloader.get_record_by_name("2024-03-07-17-42-28_3_tower.4mse")
myFrame: Frame = myRecord[7]
print(myFrame.vehicle.cameras.FRONT_LEFT)
