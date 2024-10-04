"""
This module provides functions for visualizing sensor data from autonomous vehicles, including disparity maps from stereo camera images,
3D point clouds from LiDAR sensors, and projections of LiDAR points onto camera images. It also includes utilities for displaying
the effects of correcting the extrinsic parameters of the camera, allowing users to visualize before-and-after corrections for roll, pitch, and yaw.

Functions:
    get_colored_stereo_image(camera_left, camera_right, cmap_name, min_value, max_value):
        Compute and return the depth map between two stereo camera images as a color-mapped image.
    plot_points_on_image(image, points, points_3d, cmap_name, radius, static_color, max_range_factor):
        Plot 2D points on a camera image with optional color mapping based on range values.
    get_projection_img(camera, lidars, max_range_factor):
        Generate an image with LiDAR points projected onto the camera image.
    show_points(lidar):
        Display the 3D point cloud from a LiDAR sensor using Open3D.
    show_tf_correction(camera, lidar_with_color, roll_correction, pitch_correction, yaw_correction, max_range_factor):
        Display the effect of correcting the extrinsic parameters on the projection of LiDAR points onto a camera image, showing both raw and corrected projections side-by-side.
"""
from typing import Optional, Union, Tuple, List
from PIL import Image as PilImage, ImageDraw, ImageColor
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from aeifdataset.data import Lidar, Camera
from aeifdataset.utils import get_depth_map, get_projection


def get_colored_stereo_image(camera_left: Camera, camera_right: Camera, cmap_name: str = "viridis",
                             min_value: int = 0, max_value: int = 40) -> PilImage:
    """Compute and return the depth map between two stereo camera images as a color-mapped image.

    This function computes the depth map from a pair of rectified stereo images.
    The resulting depth map is color-mapped and returned as a PIL image.

    Args:
        camera_left (Camera): The left camera of the stereo pair.
        camera_right (Camera): The right camera of the stereo pair.
        cmap_name (str): The name of the colormap to use for visualization. Defaults to "viridis".
        min_value (int): The minimum depth value to be considered in the map. Values below this threshold will be masked (set to black). Defaults to 0.
        max_value (int): The maximum value for normalization. Depth values will be normalized between this and the `min_value`. Defaults to 40.

    Returns:
        PilImage: The generated depth map with the specified colormap applied.
    """
    cmap = plt.get_cmap(cmap_name)
    depth_map = get_depth_map(camera_left, camera_right)

    mask = depth_map > min_value
    norm_values = (depth_map - min_value) / (max_value - min_value)
    norm_values = np.clip(norm_values, 0, 1)

    colored_map = cmap(norm_values)
    colored_map[mask] = [0, 0, 0, 1]  # Set masked values to black
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)

    img = PilImage.fromarray(colored_map).convert('RGB')
    return img


def plot_points_on_image(image: PilImage, points: List[Tuple[float, float]], points_3d: np.array,
                         cmap_name: str = "inferno", radius: int = 2,
                         static_color: Optional[Union[str, Tuple[int, int, int]]] = None,
                         max_range_factor: float = 0.5) -> PilImage:
    """Plot 2D points on a camera image with optional color mapping.

    This function plots a list of 2D points onto a camera image. If a static color is provided,
    all points will be plotted in that color. Otherwise, the points will be dynamically colored
    based on their range values using the specified colormap.

    Args:
        image (PilImage): The camera image onto which the points will be plotted.
        points (List[Tuple[float, float]]): The 2D coordinates of the points to plot.
        points_3d (np.array): The corresponding 3D points used to calculate the range.
        cmap_name (str): The name of the matplotlib colormap to use for color mapping. Defaults to "inferno".
        radius (int): The radius of the points to plot. Defaults to 2.
        static_color (Optional[Union[str, Tuple[int, int, int]]]): A string representing a color name (e.g., "red")
            or an RGB tuple. If provided, this color is used for all points. Defaults to None.
        max_range_factor (float): A factor used to scale the maximum range for normalization. Defaults to 0.5.

    Returns:
        PilImage: The image with the points plotted on it.
    """
    draw = ImageDraw.Draw(image)

    if static_color is not None:
        if isinstance(static_color, str):
            static_color = ImageColor.getrgb(static_color)
        for x, y in points:
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=static_color)
    else:
        cmap = plt.get_cmap(cmap_name)
        ranges = np.linalg.norm(points_3d, axis=1)
        val_min = np.min(ranges)
        val_max = np.max(ranges) * max_range_factor
        norm_values = (ranges - val_min) / (val_max - val_min)

        for (x, y), value in zip(points, norm_values):
            rgba = cmap(value)
            color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)

    return image


def get_projection_img(camera: Camera,
                       lidars: Union[Lidar, Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]],
                       List[Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]]]],
                       max_range_factor: float = 0.5) -> PilImage:
    """Generate an image with LiDAR points projected onto it.

    Args:
        camera (Camera): The camera onto which the LiDAR points are projected.
        lidars (Union[Lidar, Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]],
                 List[Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]]]]): A single LiDAR sensor or
                 a tuple containing a LiDAR and an optional static color, or a list of such tuples.
        max_range_factor (float): A factor used to scale the maximum range for normalization. Defaults to 0.5.

    Returns:
        PilImage: The image with the LiDAR points projected onto it.
    """
    proj_img = camera.image.image.copy()

    if isinstance(lidars, Lidar):
        lidars = [(lidars, None)]
    elif isinstance(lidars, tuple):
        lidars = [lidars]

    for lidar, static_color in lidars:
        pts, proj = get_projection(lidar, camera)
        proj_img = plot_points_on_image(proj_img, proj, pts, static_color=static_color,
                                        max_range_factor=max_range_factor)

    return proj_img


def show_points(lidar: Lidar) -> None:
    """Display the point cloud from a LiDAR sensor.

    Visualize the 3D point cloud data captured by the LiDAR sensor using Open3D.

    Args:
        lidar (Lidar): The LiDAR sensor containing the 3D point cloud data.

    Returns:
        None
    """
    if importlib.util.find_spec("open3d") is None:
        raise ImportError('Install open3d to use this function with: python -m pip install open3d')
    import open3d as o3d
    points = lidar

    xyz_points = np.stack((points['x'], points['y'], points['z']), axis=-1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)

    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd])


def show_tf_correction(camera: Camera,
                       lidar_with_color: Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]],
                       roll_correction: float, pitch_correction: float, yaw_correction: float,
                       max_range_factor: float = 0.5) -> None:
    """Display the effect of correcting the extrinsic parameters on LiDAR projection.

    This function visualizes the projection of LiDAR points onto a camera image before and after
    applying a correction to the camera's extrinsic parameters. The user can specify corrections
    to the roll, pitch, and yaw angles. The raw and corrected projections are displayed side-by-side.

    Args:
        camera (Camera): The camera whose extrinsic parameters will be adjusted.
        lidar_with_color (Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]]): A tuple containing the LiDAR sensor
            and an optional static color for the points.
        roll_correction (float): Correction to apply to the roll angle (in radians).
        pitch_correction (float): Correction to apply to the pitch angle (in radians).
        yaw_correction (float): Correction to apply to the yaw angle (in radians).
        max_range_factor (float): A factor used to scale the maximum range of values for normalization. Defaults to 0.5.

    Returns:
        None
    """
    lidar, static_color = lidar_with_color

    proj_img = get_projection_img(camera, [(lidar, static_color)], max_range_factor=max_range_factor)

    original_rpy = np.array(camera.info.extrinsic.rpy)
    camera.info.extrinsic.rpy = original_rpy + np.array([roll_correction, pitch_correction, yaw_correction])

    proj_img_corrected = get_projection_img(camera, [(lidar, static_color)], max_range_factor=max_range_factor)
    camera.info.extrinsic.rpy = original_rpy  # Restore original parameters

    fig, axes = plt.subplots(1, 2, figsize=(40, 26))

    axes[0].imshow(proj_img)
    axes[0].set_title('Raw')
    axes[0].axis('off')

    axes[1].imshow(proj_img_corrected)
    axes[1].set_title(f'Corrected [Roll: {roll_correction}, Pitch: {pitch_correction}, Yaw: {yaw_correction}]')
    axes[1].axis('off')

    print("Camera extrinsic parameters after correction:", camera.info.extrinsic)

    plt.show()
