"""
This module provides functions for visualizing sensor data, including disparity maps from stereo camera images,
3D point clouds from Lidar sensors, and projections of Lidar points onto camera images. It also includes
utilities for displaying the effects of correcting the extrinsic parameters of the camera.

Functions:
    show_disparity_map(camera_left, camera_right, cmap_name, max_value):
        Display the disparity map between two stereo camera images.
    show_points(lidar):
        Display the point cloud from a Lidar sensor.
    show_projection(camera, lidar, lidar2, lidar3, static_color, static_color2, static_color3, max_range_factor, intensity):
        Project and visualize Lidar points onto a camera image.
    show_tf_correction(camera, lidar, roll_correction, pitch_correction, yaw_correction, intensity, static_color, max_range_factor):
        Display the effect of correcting the extrinsic parameters on Lidar projection.
"""
from typing import Optional
from PIL import Image
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from aeifdataset.data import Lidar, Camera
from aeifdataset.utils import get_projection_img, get_depth_map


def show_disparity_map(camera_left: Camera, camera_right: Camera, cmap_name: str = "viridis",
                       max_value: int = 40) -> None:
    """Display the disparity map between two stereo camera images.

    This function computes and visualizes the disparity map from a pair of rectified stereo images.
    The resulting disparity map is color-mapped and displayed using a specified colormap.

    Args:
        camera_left (Camera): The left camera of the stereo pair.
        camera_right (Camera): The right camera of the stereo pair.
        cmap_name (str): The name of the colormap to use for visualization. Defaults to "viridis".
        max_value (int): The maximum value for normalization. Defaults to 40.

    Returns:
        None
    """
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    disparity_map = get_depth_map(camera_left, camera_right)

    # Set the min and max values for normalization
    val_min = np.min(disparity_map)
    val_max = max_value

    # Create a mask for outliers (disparity values greater than 10 * val_max)
    mask = disparity_map > 10 * val_max

    # Normalize the disparity map
    norm_values = (disparity_map - val_min) / (val_max - val_min)
    norm_values = np.clip(norm_values, 0, 1)  # Ensure values are within [0, 1]

    # Apply the colormap
    colored_map = cmap(norm_values)

    # Set masked values to black
    colored_map[mask] = [0, 0, 0, 1]  # RGBA, with alpha=1

    # Convert to 8-bit per channel image
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)

    # Create and return the image
    img = Image.fromarray(colored_map)

    img.show()


def show_points(lidar: Lidar) -> None:
    """Display the point cloud from a Lidar sensor.

    This function visualizes the 3D point cloud data captured by the Lidar sensor using Open3D.

    Args:
        lidar (Lidar): The Lidar sensor containing the 3D point cloud data.

    Returns:
        None
    """
    if importlib.util.find_spec("open3d") is None:
        raise ImportError('Install open3d to use this function with: python -m pip install open3d')
    import open3d as o3d
    points = lidar.points.points

    # Convert structured NumPy array to a regular 3D NumPy array with contiguous memory.
    xyz_points = np.stack((points['x'], points['y'], points['z']), axis=-1)

    # Ensure the data type is float64, which is expected by Open3D.
    xyz_points = xyz_points.astype(np.float64)

    # Create Open3D point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)

    # Estimate normals.
    pcd.estimate_normals()

    # Visualize the point cloud.
    o3d.visualization.draw_geometries([pcd])


def show_projection(camera: Camera, lidar: Lidar, lidar2: Optional[Lidar] = None, lidar3: Optional[Lidar] = None,
                    static_color=None, static_color2=None, static_color3=None, max_range_factor: float = 0.5,
                    intensity: bool = False) -> None:
    """Project and visualize Lidar points onto a camera image.

    This function projects points from up to three Lidar sensors onto a camera image,
    optionally highlighting points by intensity or using static colors.

    Args:
        camera (Camera): The camera onto which the Lidar points are projected.
        lidar (Lidar): The primary Lidar sensor.
        lidar2 (Optional[Lidar]): An optional second Lidar sensor. Defaults to None.
        lidar3 (Optional[Lidar]): An optional third Lidar sensor. Defaults to None.
        static_color: Static color for the points from the primary Lidar. Defaults to None.
        static_color2: Static color for the points from the second Lidar. Defaults to None.
        static_color3: Static color for the points from the third Lidar. Defaults to None.
        max_range_factor (float): Factor to scale the max range for normalization. Defaults to 0.5.
        intensity (bool): If True, use intensity values for coloring. Defaults to False.

    Returns:
        None
    """
    camera.image.image = get_projection_img(camera, lidar, intensity, static_color, max_range_factor)
    if lidar2 is not None:
        camera.image.image = get_projection_img(camera, lidar2, intensity, static_color2, max_range_factor, False)
    if lidar3 is not None:
        camera.image.image = get_projection_img(camera, lidar3, intensity, static_color3, max_range_factor, False)
    camera.image.image.show()


def show_tf_correction(camera: Camera, lidar: Lidar, roll_correction: float, pitch_correction: float,
                       yaw_correction: float,
                       intensity: bool = False, static_color=None, max_range_factor: float = 0.5) -> None:
    """Display the effect of correcting the extrinsic parameters on Lidar projection.

    This function visualizes the projection of Lidar points onto a camera image before and after
    applying a correction to the extrinsic parameters of the camera.

    Args:
        camera (Camera): The camera with extrinsic parameters to correct.
        lidar (Lidar): The Lidar sensor providing the points to project.
        roll_correction (float): Correction to apply to the roll angle (in radians).
        pitch_correction (float): Correction to apply to the pitch angle (in radians).
        yaw_correction (float): Correction to apply to the yaw angle (in radians).
        intensity (bool): If True, use intensity values for coloring. Defaults to False.
        static_color: Static color for the points. Defaults to None.
        max_range_factor (float): Factor to scale the max range for normalization. Defaults to 0.5.

    Returns:
        None
    """
    proj_img = get_projection_img(camera, lidar, intensity, static_color, max_range_factor)

    # Adjust extrinsic parameters
    x, y, z = camera.info.extrinsic.xyz
    roll, pitch, yaw = camera.info.extrinsic.rpy
    camera.info.extrinsic.xyz = np.array([x, y, z])
    camera.info.extrinsic.rpy = np.array([roll + roll_correction, pitch + pitch_correction, yaw + yaw_correction])

    # Display corrected projection
    proj_img_corrected = get_projection_img(camera, lidar, intensity, static_color, max_range_factor)

    fig, axes = plt.subplots(1, 2, figsize=(40, 26))

    # Display the first image
    axes[0].imshow(proj_img)
    axes[0].set_title('Raw')
    axes[0].axis('off')  # Hide axes

    # Display the second image
    axes[1].imshow(proj_img_corrected)
    axes[1].set_title(f'Corrected [Roll:{roll_correction}, Pitch:{pitch_correction}, Yaw:{yaw_correction}]')
    axes[1].axis('off')  # Hide axes

    print(camera.info.extrinsic)

    # Display the images
    plt.show()
