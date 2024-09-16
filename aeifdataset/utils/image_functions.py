"""
This module provides functions for processing and handling images related to camera sensors.
It includes functionalities for image rectification, depth map computation, saving images
with metadata, and loading images with embedded metadata.

Functions:
    get_rect_img(camera): Rectify the provided image using the camera's intrinsic and extrinsic parameters.
    get_depth_map(camera_left, camera_right): Compute a depth map from a pair of stereo images.
    save_image(image, output_path, suffix, metadata): Save an image to disk with optional metadata.
    save_all_camera_images(frame, output_path): Save all images from a frame's vehicle and tower cameras.
    load_image_with_metadata(file_path): Load an image along with its embedded metadata.
"""
from typing import Optional, Tuple
import os
from PIL import Image as PilImage
from PIL.PngImagePlugin import PngInfo
from aeifdataset.data import CameraInformation, Camera, Image
import numpy as np
import cv2


def get_rect_img(camera: Camera) -> Image:
    """Rectify the provided image using the camera's intrinsic and extrinsic parameters.

    This function performs image rectification using the camera matrix, distortion coefficients,
    rectification matrix, and projection matrix. The rectified image is returned as an `Image` object.

    Args:
        camera (Camera): The camera object containing the image and calibration parameters.

    Returns:
        Image: The rectified image wrapped in the `Image` class.
    """
    # Init and calculate rectification matrix
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=camera.info.camera_mtx,
                                             distCoeffs=camera.info.distortion_mtx[:-1],
                                             R=camera.info.rectification_mtx,
                                             newCameraMatrix=camera.info.projection_mtx,
                                             size=camera.info.shape,
                                             m1type=cv2.CV_16SC2)
    # Apply matrix
    rectified_image = cv2.remap(np.array(camera._image_raw.image), mapx, mapy, interpolation=cv2.INTER_LANCZOS4)

    return Image(PilImage.fromarray(rectified_image), camera._image_raw.timestamp)


def get_depth_map(camera_left: Camera, camera_right: Camera) -> np.ndarray:
    """Compute a depth map from a pair of stereo images.

    This function uses stereo block matching to compute a disparity map and then
    calculates the depth map using the disparity values, the focal length, and the baseline
    distance between the two cameras.

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

    # Create the block matching algorithm with high-quality settings
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

    # Compute the disparity map
    disparity = stereo.compute(img1, img2)

    # Normalize for better visualization
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # To avoid division by zero, set disparity values of 0 to a small value
    safe_disparity = np.where(disparity == 0, 0.000001, disparity)

    f = camera_right.info.focal_length
    b = abs(camera_right.info.stereo_transform.translation[0]) * 10 ** 3

    depth_map = f * b / safe_disparity

    return depth_map


def save_image(image: Image, output_path: str, suffix: str = '', metadata: Optional[CameraInformation] = None):
    """Save an image to disk with optional metadata.

    This function saves an `Image` object to disk in PNG format. Metadata can be optionally
    embedded into the image file.

    Args:
        image (Image): The image to be saved.
        output_path (str): The directory where the image will be saved.
        suffix (str): Optional suffix to be added to the image filename.
        metadata (Optional[CameraInformation]): Optional metadata to embed in the image file.
    """
    output_file = os.path.join(output_path, f'{image.get_timestamp()}{suffix}.png')

    info = PngInfo()
    if metadata:
        info_dict = metadata.to_dict()
        for key, value in info_dict.items():
            info.add_text(key, value)

    image.save(output_file, 'PNG', pnginfo=info, compress_level=0)


def save_all_camera_images(frame, output_path: str):
    """Save all images from a frame's vehicle and tower cameras.

    This function iterates through all cameras in the vehicle and tower of the frame and saves
    each image to the specified output path.

    Args:
        frame: The frame object containing vehicle and tower cameras.
        output_path (str): The directory where images will be saved.
    """
    # Iterate through all attributes in the 'vehicle.cameras' and 'tower.cameras' objects
    for camera_attr in dir(frame.vehicle.cameras):
        camera = getattr(frame.vehicle.cameras, camera_attr, None)
        if camera and hasattr(camera, '_image_raw'):
            try:
                image = camera._image_raw
                metadata = camera.info
                suffix = f'_{camera_attr.lower()}'
                save_image(image, output_path, suffix, metadata)
            except AttributeError as e:
                print(f"Error processing {camera_attr}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {camera_attr}: {e}")

    # Do the same for 'tower.cameras' if necessary
    for camera_attr in dir(frame.tower.cameras):
        camera = getattr(frame.tower.cameras, camera_attr, None)
        if camera and hasattr(camera, '_image_raw'):
            try:
                image = camera._image_raw
                metadata = camera.info
                suffix = f'_{camera_attr.lower()}'
                save_image(image, output_path, suffix, metadata)
            except AttributeError as e:
                print(f"Error processing {camera_attr}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {camera_attr}: {e}")


def load_image_with_metadata(file_path: str) -> Tuple[PilImage.Image, dict]:
    """Load an image along with its metadata.

    This function loads an image file and extracts any embedded metadata.

    Args:
        file_path (str): The path to the image file.

    Returns:
        Tuple[PilImage, dict]: The loaded image and a dictionary containing the metadata.
    """
    image = PilImage.open(file_path)

    metadata = image.info
    metadata_dict = {}
    for key, value in metadata.items():
        metadata_dict[key] = value.decode('utf-8') if isinstance(value, bytes) else value

    return image, metadata_dict
