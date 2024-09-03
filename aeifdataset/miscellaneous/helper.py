import hashlib
import dill
from aeifdataset.miscellaneous import INT_LENGTH
from decimal import Decimal
from datetime import datetime, timedelta, timezone


def unix_to_utc(unix_time: Decimal, precision='ns', timezone_offset_hours=2):
    # Convert the timestamp to nanosecond precision
    unix_time_str = str(unix_time).replace('.', '')
    unix_time_ns = Decimal(unix_time_str)

    seconds = int(unix_time_ns) // 1000000000
    nanoseconds = int(unix_time_ns) % 1000000000

    utc_time = datetime.utcfromtimestamp(seconds)
    utc_time += timedelta(seconds=nanoseconds / 1e9)
    # Compute the local time with the specified timezone offset
    local_time = utc_time + timedelta(hours=timezone_offset_hours)

    if precision == 'ns':
        local_time_str = local_time.strftime('%Y-%m-%d_%H:%M:%S') + f'.{nanoseconds:09d}'
    elif precision == 's':
        local_time_str = local_time.strftime('%Y-%m-%d_%H:%M:%S')
    else:
        raise ValueError("Precision must be 'ns' or 's'")

    return local_time_str


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
