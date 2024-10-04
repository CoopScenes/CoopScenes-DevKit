"""
This module provides functions for processing and handling images related to camera sensors.
It includes functionalities for image rectification, depth map computation, saving images with metadata,
loading images with embedded metadata, and saving all images from a frame.

Functions:
    get_rect_img(camera, performance_mode=False): Rectify the provided image using the camera's intrinsic and extrinsic parameters.
    get_depth_map(camera_left, camera_right): Compute a depth map from a pair of stereo images.
    save_image(image, output_path, suffix='', metadata=None): Save an image to disk with optional metadata.
    save_all_images_in_frame(frame, output_path, create_subdir=False): Save all images from a frame's vehicle and tower cameras.
    load_image_with_metadata(file_path): Load an image along with its embedded metadata.
"""
from typing import Optional, Tuple
import os
from PIL import Image as PilImage
from PIL.PngImagePlugin import PngInfo
from aeifdataset.data import CameraInformation, Camera, Image
import numpy as np
import cv2


def get_rect_img(camera: Camera, performance_mode: bool = False) -> Image:
    """Rectify the provided image using the camera's intrinsic and extrinsic parameters.

    Performs image rectification using the camera matrix, distortion coefficients, rectification matrix,
    and projection matrix. The rectified image is returned as an `Image` object.

    Args:
        camera (Camera): The camera object containing the image and calibration parameters.
        performance_mode (bool, optional): If True, use faster interpolation; otherwise, use higher quality. Defaults to False.

    Returns:
        Image: The rectified image wrapped in the `Image` class.
    """
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=camera.info.camera_mtx,
        distCoeffs=camera.info.distortion_mtx[:-1],
        R=camera.info.rectification_mtx,
        newCameraMatrix=camera.info.projection_mtx,
        size=camera.info.shape,
        m1type=cv2.CV_16SC2
    )

    interpolation_algorithm = cv2.INTER_LINEAR if performance_mode else cv2.INTER_LANCZOS4

    rectified_image = cv2.remap(np.array(camera._image_raw.image), mapx, mapy, interpolation=interpolation_algorithm)

    return Image(PilImage.fromarray(rectified_image), camera._image_raw.timestamp)


def get_depth_map(camera_left: Camera, camera_right: Camera) -> np.ndarray:
    """Compute a depth map from a pair of stereo images.

    Uses stereo block matching to compute a disparity map, then calculates the depth map using the disparity values,
    focal length, and baseline distance between the two cameras.

    Args:
        camera_left (Camera): The left camera of the stereo pair.
        camera_right (Camera): The right camera of the stereo pair.

    Returns:
        np.ndarray: The computed depth map.
    """
    rect_left = get_rect_img(camera_left)
    rect_right = get_rect_img(camera_right)

    img1 = np.array(rect_left.image.convert('L'))  # Convert to grayscale
    img2 = np.array(rect_right.image.convert('L'))  # Convert to grayscale

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(img1, img2)

    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    safe_disparity = np.where(disparity == 0, 0.000001, disparity)

    f = camera_right.info.focal_length
    b = abs(camera_right.info.stereo_transform.translation[0]) * 10 ** 3

    depth_map = f * b / safe_disparity

    return depth_map


def save_image(image: Image, output_path: str, suffix: str = '', metadata: Optional[CameraInformation] = None):
    """Save an image to disk with optional metadata.

    Saves an `Image` object to disk in PNG format. Optionally, metadata can be embedded into the image file.

    Args:
        image (Image): The image to be saved.
        output_path (str): The directory where the image will be saved.
        suffix (str, optional): Optional suffix to be added to the image filename. Defaults to ''.
        metadata (Optional[CameraInformation], optional): Metadata to embed in the image file. Defaults to None.
    """
    output_file = os.path.join(output_path, f'{image.get_timestamp()}{suffix}.png')

    info = PngInfo()
    if metadata:
        info_dict = metadata.to_dict()
        for key, value in info_dict.items():
            info.add_text(key, value)

    image.save(output_file, 'PNG', pnginfo=info, compress_level=0)


def save_all_images_in_frame(frame, output_path: str, create_subdir: bool = False):
    """Save all images from a frame's vehicle and tower cameras.

    Iterates through all cameras in the frame, saving each camera's image.
    If `create_subdir` is True, a subdirectory for each camera will be created.

    Args:
        frame: The frame object containing vehicle and tower cameras.
        output_path (str): The directory where images will be saved.
        create_subdir (bool, optional): If True, creates a subdirectory for each camera. Defaults to False.
    """
    for agent in frame:
        for camera_name, camera in agent.cameras:
            if create_subdir:
                camera_dir = os.path.join(output_path, camera_name.lower())
                os.makedirs(camera_dir, exist_ok=True)
                save_path = camera_dir
            else:
                save_path = output_path

            save_image(camera._image_raw, save_path, '', camera.info)


def load_image_with_metadata(file_path: str) -> Tuple[PilImage.Image, dict]:
    """Load an image along with its metadata.

    Loads an image file and extracts any embedded metadata.

    Args:
        file_path (str): The path to the image file.

    Returns:
        Tuple[PilImage.Image, dict]: The loaded image and a dictionary containing the metadata.
    """
    image = PilImage.open(file_path)

    metadata = image.info
    metadata_dict = {}
    for key, value in metadata.items():
        metadata_dict[key] = value.decode('utf-8') if isinstance(value, bytes) else value

    return image, metadata_dict
