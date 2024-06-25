import hashlib
import dill
from ameisedataset.miscellaneous import INT_LENGTH


def compute_checksum(data):
    # calculates the has value of a given bytestream - SHA256
    return hashlib.sha256(data).digest()


def read_data_block(data, dtype_length: int = INT_LENGTH):
    data_len = int.from_bytes(data[0:dtype_length], 'big')
    data_block_bytes = data[dtype_length:dtype_length + data_len]
    return data_block_bytes, data[dtype_length + data_len:]


def obj_to_bytes(obj) -> bytes:
    obj_bytes = dill.dumps(obj)
    obj_bytes_len = len(obj_bytes).to_bytes(INT_LENGTH, 'big')
    return obj_bytes_len + obj_bytes


def obj_from_bytes(data: bytes):
    return dill.loads(data)


def serialize(obj):
    if obj is None:
        return b'\x00\x00\x00\x00'
    obj_bytes = obj.to_bytes()
    obj_bytes_len = len(obj_bytes).to_bytes(INT_LENGTH, 'big')
    return obj_bytes_len + obj_bytes


def deserialize(data, cls, *args):
    obj_len = int.from_bytes(data[:INT_LENGTH], 'big')
    if obj_len == 0:
        return None, data[INT_LENGTH:]
    obj_data = data[INT_LENGTH:INT_LENGTH + obj_len]
    return cls.from_bytes(obj_data, *args), data[INT_LENGTH + obj_len:]
