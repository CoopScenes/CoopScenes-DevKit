"""
This module provides functions to group files into maneuvers based on timestamps and to save images from frames or datasets.

The primary function `get_maneuver_split` reads files from a directory, sorts them by timestamp, and groups them into maneuvers separated by a time gap of more than 20 seconds. Additionally, functions are provided to save images efficiently, either from individual frames or entire datasets, with multithreading support.

Functions:
    get_maneuver_split(dataset_dir, return_paths):
        Groups files into maneuvers and returns them as file paths or Dataloader snippets.

    save_image(image, output_path, filename, dtype):
        Saves a single image to disk in the specified format ('JPEG' or 'PNG').

    save_all_images_in_frame(frame, output_path, create_subdir, use_raw, dtype):
        Saves all images from the cameras in a frame.

    save_dataset_images_multithreaded(dataset, save_dir, create_subdir, use_raw, dtype, num_cores):
        Saves images from a dataset using multithreading for faster processing.
"""
from datetime import datetime
import os
from coopscenes import Dataloader, Image
from typing import Union
import multiprocessing as mp
import sys
from PIL import Image as PilImage


def get_maneuver_split(dataset_dir, return_paths=False):
    """
    Groups files into maneuvers based on a time gap of 20 seconds.

    Args:
        dataset_dir (str): Path to the directory containing the files.
        return_paths (bool): If True, returns file paths. If False, returns Dataloader snippets.

    Returns:
        list: List of maneuvers, either as file paths or Dataloader snippets.
    """
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".4mse")]

    sorted_files = sorted(all_files, key=_extract_timestamp)

    maneuvers = []
    current_start_id = None
    current_end_id = None
    last_timestamp = None

    dataset = Dataloader(dataset_dir)

    for idx, file in enumerate(sorted_files):
        current_timestamp = _extract_timestamp(file)

        if last_timestamp is not None:
            time_diff = (current_timestamp - last_timestamp).total_seconds()
            if time_diff > 20:
                if current_start_id is not None and current_end_id is not None:
                    if return_paths:
                        maneuvers.append(sorted_files[current_start_id:current_end_id + 1])
                    else:
                        maneuvers.append(dataset[current_start_id:current_end_id + 1])
                current_start_id = idx

        if current_start_id is None:
            current_start_id = idx
        current_end_id = idx

        last_timestamp = current_timestamp

    if current_start_id is not None and current_end_id is not None:
        if return_paths:
            maneuvers.append(sorted_files[current_start_id:current_end_id + 1])
        else:
            maneuvers.append(dataset[current_start_id:current_end_id + 1])

    return maneuvers


def _extract_timestamp(filename):
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


def save_image(image: Union['Image', PilImage.Image], output_path: str, filename: str, dtype: str = 'PNG'):
    """Saves a single image to disk in the specified format.

    This function saves an image (raw or processed) to a specified directory with a given filename.
    The supported formats are 'JPEG' and 'PNG'.

    Args:
        image (Union[Image, PilImage.Image]): The image to be saved. If an `Image` object is provided,
            its internal `PilImage` representation is used.
        output_path (str): The directory where the image will be saved.
        filename (str): The name of the file (without extension).
        dtype (str, optional): The format in which to save the image ('JPEG' or 'PNG'). Defaults to 'PNG'.

    Raises:
        ValueError: If an unsupported format is specified.
    """
    if isinstance(image, Image):
        image = image.image

    dtype = dtype.upper()
    if dtype == "JPEG":
        ext = "jpg"
    elif dtype == "PNG":
        ext = "png"
    else:
        raise ValueError("Unsupported format. Use 'JPEG' or 'PNG'.")

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'{filename}.{ext}')
    image.save(output_file, format=dtype)


def save_all_images_in_frame(frame, output_path: str, create_subdir: bool = True, use_raw: bool = False,
                             dtype: str = 'PNG'):
    """Saves all images from the cameras in a frame.

    This function iterates through all cameras in the given frame and saves their images
    to the specified output directory. It optionally creates subdirectories for each camera
    and supports saving raw or processed images in the specified format.

    Args:
        frame: The frame object containing vehicle and tower cameras.
        output_path (str): The directory where images will be saved.
        create_subdir (bool, optional): Whether to create subdirectories for each camera. Defaults to True.
        use_raw (bool, optional): Whether to save raw images instead of processed images. Defaults to False.
        dtype (str, optional): The format in which to save the images ('JPEG' or 'PNG'). Defaults to 'PNG'.

    Raises:
        ValueError: If an unsupported format is specified.
    """
    os.makedirs(output_path, exist_ok=True)
    for agent in frame:
        for camera_name, camera in agent.cameras:
            image_to_save = camera._image_raw if use_raw else camera.image
            timestamp = image_to_save.get_timestamp()

            if create_subdir:
                camera_dir = os.path.join(output_path, camera_name.lower())
                os.makedirs(camera_dir, exist_ok=True)
                save_path = camera_dir
                save_image(image_to_save, output_path=save_path, filename=timestamp, dtype=dtype)
            else:
                save_path = output_path
                save_image(image_to_save, output_path=save_path, filename=f'{timestamp}_{camera_name.lower()}',
                           dtype=dtype)


def _save_datarecord_images(datarecord, save_dir, create_subdir, use_raw, dtype):
    """Saves all images from the frames in a datarecord.

    This function iterates through all frames in the given datarecord and saves the images
    from each frame's cameras. It supports saving raw or processed images in the specified format.

    Args:
        datarecord: The datarecord containing frames to process.
        save_dir (str): The directory where images will be saved.
        create_subdir (bool): Whether to create subdirectories for cameras.
        use_raw (bool): Whether to save raw images instead of processed images.
        dtype (str): The format in which to save the images ('JPEG' or 'PNG').
    """
    for frame in datarecord:
        save_all_images_in_frame(frame, save_dir, create_subdir, use_raw, dtype)


def save_dataset_images_multithreaded(dataset, save_dir: str, create_subdir: bool = True, use_raw: bool = False,
                                      dtype='PNG', num_cores: int = 2):
    """Saves images from a dataset using multithreading.

    Args:
        dataset: Iterable containing datarecords to process.
        save_dir (str): Directory where images will be saved.
        create_subdir (bool, optional): Whether to create subdirectories for cameras. Defaults to True.
        use_raw (bool, optional): Whether to use raw images instead of processed images. Defaults to False.
        dtype (str, optional): The data type in which to save the image ('PNG' or 'JPEG'). Defaults to 'PNG'.
        num_cores (int, optional): Number of cores to use for multithreading. Defaults to 2.
    """
    with mp.Pool(processes=num_cores) as pool:
        batch = []
        total_records = len(dataset)

        for i, datarecord in enumerate(dataset, start=1):
            batch.append(datarecord)

            if len(batch) == num_cores:
                results = [
                    pool.apply_async(_save_datarecord_images, args=(record, save_dir, create_subdir, use_raw, dtype))
                    for record in batch]

                for result in results:
                    try:
                        result.wait()
                    except Exception as e:
                        print(f"Error in worker process: {e}")

                batch.clear()

            percent_complete = (i / total_records) * 100
            sys.stdout.write(f"\rDataset Progress: {percent_complete:.2f}%")
            sys.stdout.flush()

        if batch:
            results = [pool.apply_async(_save_datarecord_images, args=(record, save_dir, create_subdir, use_raw, dtype))
                       for record in batch]
            for result in results:
                try:
                    result.wait()
                except Exception as e:
                    print(f"Error in worker process: {e}")

        sys.stdout.write("\rDataset Progress: 100.00%\n")
        sys.stdout.flush()
