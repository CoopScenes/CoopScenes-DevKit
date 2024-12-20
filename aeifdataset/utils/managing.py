"""
This module provides functions to group files into maneuvers based on timestamps.

The primary function `get_maneuver_split` reads files from a directory, sorts them by timestamp, and groups them into maneuvers separated by a time gap of more than 20 seconds. The maneuvers can either be returned as file paths or as Dataloader snippets.

Functions:
    get_maneuver_split(dataset_dir, return_paths=False):
        Groups files into maneuvers and returns them as file paths or Dataloader snippets.

    extract_timestamp(filename):
        Extracts the timestamp from a file name.
"""
from datetime import datetime
import os
from aeifdataset import Dataloader


def get_maneuver_split(dataset_dir, return_paths=False):
    """
    Groups files into maneuvers based on a time gap of 20 seconds.

    Args:
        dataset_dir (str): Path to the directory containing the files.
        return_paths (bool): If True, returns file paths. If False, returns Dataloader snippets.

    Returns:
        list: List of maneuvers, either as file paths or Dataloader snippets.
    """
    # Read files from the directory
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".4mse")]

    # Sort files by timestamp
    sorted_files = sorted(all_files, key=extract_timestamp)

    maneuvers = []
    current_start_id = None
    current_end_id = None
    last_timestamp = None

    # Initialize Dataloader
    dataset = Dataloader(dataset_dir)

    for idx, file in enumerate(sorted_files):
        current_timestamp = extract_timestamp(file)

        if last_timestamp is not None:
            # Check if the time gap is greater than 20 seconds
            time_diff = (current_timestamp - last_timestamp).total_seconds()
            if time_diff > 20:
                # Add section to maneuvers
                if current_start_id is not None and current_end_id is not None:
                    if return_paths:
                        maneuvers.append(sorted_files[current_start_id:current_end_id + 1])
                    else:
                        maneuvers.append(dataset[current_start_id:current_end_id + 1])
                # Start a new maneuver
                current_start_id = idx

        if current_start_id is None:
            current_start_id = idx
        current_end_id = idx

        last_timestamp = current_timestamp

    # Add the last maneuver
    if current_start_id is not None and current_end_id is not None:
        if return_paths:
            maneuvers.append(sorted_files[current_start_id:current_end_id + 1])
        else:
            maneuvers.append(dataset[current_start_id:current_end_id + 1])

    return maneuvers


def extract_timestamp(filename):
    """
    Extracts the timestamp from a file name.

    The file name is expected to follow the format:
    idXXXXX_YYYY-MM-DD_HH-MM-SS.4mse

    Args:
        filename (str): The file name to extract the timestamp from.

    Returns:
        datetime: The extracted timestamp as a datetime object.
    """
    base_name = os.path.splitext(filename)[0]  # Remove the file extension (.4mse)
    _, date_str, time_str = base_name.split('_')  # Split the file name
    return datetime.strptime(f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S")
