"""
This module provides functions for visualizing sensor data from autonomous vehicles, including disparity maps from stereo camera images,
3D point clouds from LiDAR sensors, and projections of LiDAR points onto camera images. It also includes utilities for displaying
the effects of correcting the extrinsic parameters of the camera, allowing users to visualize before-and-after corrections for roll, pitch, and yaw.

Functions:
    get_colored_stereo_image(camera_left, camera_right, cmap_name, min_value, max_value):
        Computes and returns the depth map between two stereo camera images as a color-mapped image.
    plot_points_on_image(image, points, points_3d, cmap_name, radius, static_color, max_range):
        Plots 2D points on a camera image with optional color mapping based on range values or static color.
    get_projection_img(camera, lidars, cmap_name, radius, max_range):
        Generates an image with LiDAR points projected onto the camera image with optional colormap and radius settings.
    show_points(points, colors, point_size):
        Displays the 3D point cloud from a LiDAR sensor or NumPy arrays of points using Open3D.
    show_tf_correction(camera, lidar_with_color, roll_correction, pitch_correction, yaw_correction, max_range_factor):
        Displays the effect of correcting the extrinsic parameters on the projection of LiDAR points onto a camera image, showing both raw and corrected projections side-by-side.
"""
from typing import Optional, Union, Tuple, List
from PIL import Image as PilImage, ImageDraw, ImageColor
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from aeifdataset.data import Lidar, Camera
from aeifdataset.utils import get_projection
from aeifdataset.utils.image_functions import get_disparity_map


def get_colored_stereo_image(camera_left: Camera, camera_right: Camera, cmap_name: str = "viridis",
                             min_value: int = 0, max_value: int = 1000) -> PilImage:
    """Compute and return the disparity map between two stereo camera images as a color-mapped image.

       This function computes the disparity map from a pair of rectified stereo images using disparity calculation.
       The resulting disparity map is normalized between the specified `min_value` and `max_value`, color-mapped
       using the specified colormap, and returned as a PIL image.

       Disparity values below `min_value` are set to `min_value` for normalization, and values above `max_value`
       can optionally be masked and set to black in the final image.

       Args:
           camera_left (Camera): The left camera of the stereo pair.
           camera_right (Camera): The right camera of the stereo pair.
           cmap_name (str): The name of the colormap to use for visualization. Defaults to "viridis".
           min_value (int): The minimum disparity value to be considered for normalization. Disparity values below this
                            threshold are clamped to this value. Defaults to 0.
           max_value (int): The maximum disparity value for normalization. Disparity values will be normalized between
                            this and `min_value`. Values above this threshold are set to black. Defaults to 1000.

       Returns:
           PilImage: The generated disparity map with the specified colormap applied, returned as an RGB PIL image.
       """
    cmap = plt.get_cmap(cmap_name)
    disparity_map = get_disparity_map(camera_left, camera_right)[:, 128:]

    norm_values = (disparity_map - min_value) / (max_value - min_value)

    colored_map = cmap(norm_values)

    mask = disparity_map > max_value
    colored_map[mask] = [0, 0, 0, 1]  # Set masked values to black

    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)

    img = PilImage.fromarray(colored_map).convert('RGB')

    return img


def plot_points_on_image(image: PilImage, points: np.ndarray, points_3d: np.ndarray,
                         cmap_name: str = "Spectral", radius: float = 2.5,
                         static_color: Optional[Union[str, Tuple[int, int, int]]] = None,
                         min_range: Optional[float] = 3,
                         max_range: Optional[float] = 50, opacity: float = 0.5) -> PilImage:
    """Plot 2D points on a camera image with optional color mapping, range limits, and opacity.

    This function plots a list of 2D points onto a camera image. If a static color is provided,
    all points will be plotted in that color with the specified opacity. Otherwise, the points will be
    dynamically colored based on their range values using the specified colormap. You can also set
    minimum and maximum range values for color normalization. The opacity value controls the transparency of the points.

    Args:
        image (PilImage): The camera image onto which the points will be plotted.
        points (np.ndarray): The 2D coordinates of the points to plot.
        points_3d (np.ndarray): The corresponding 3D points used to calculate the range.
        cmap_name (str): The name of the matplotlib colormap to use for dynamic color mapping. Defaults to "Spectral".
        radius (float): The radius of the points to plot. Defaults to 2.5.
        static_color (Optional[Union[str, Tuple[int, int, int]]]): A string representing a color name (e.g., "red")
            or an RGB tuple. If provided, this color is used for all points. Defaults to None.
        min_range (Optional[float]): The minimum range value for normalization. Defaults to 3.
        max_range (Optional[float]): The maximum range value for normalization. Defaults to 50.
        opacity (float): A value between 0 (completely transparent) and 1 (fully opaque) to control the transparency of the points. Defaults to 0.5.

    Returns:
        PilImage: The image with the points plotted on it.
    """
    if points.size == 0:
        return image

    draw = ImageDraw.Draw(image, "RGBA")

    opacity = int(np.clip(opacity * 255, 0, 255))

    if static_color is not None:
        if isinstance(static_color, str):
            static_color = ImageColor.getrgb(static_color)
        for x, y in points:
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=(*static_color, opacity))
    else:
        cmap = plt.get_cmap(cmap_name)
        ranges = np.linalg.norm(points_3d, axis=1)
        val_min = min_range
        val_max = max_range
        norm_values = (ranges - val_min) / (val_max - val_min)

        for (x, y), value in zip(points, norm_values):
            rgba = cmap(value)
            color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), opacity)
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)

    return image


def get_projection_img(camera: Camera,
                       *lidars: Union[Lidar, Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]]],
                       cmap_name: str = "Spectral", radius: float = 2.5,
                       min_range: Optional[float] = 5, max_range: Optional[float] = 50,
                       opacity: float = 0.5) -> PilImage:
    """Generate an image with LiDAR points projected onto it.

    This function projects LiDAR points onto a camera image and allows for optional
    coloring of the points. It supports a colormap for dynamic coloring based on
    range or other factors, as well as a static color option for individual LiDAR sensors.

    Args:
        camera (Camera): The camera onto which the LiDAR points are projected.
        *lidars (Union[Lidar, Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]]]): One or more LiDAR sensors,
                 or tuples containing a LiDAR and an optional static color.
        cmap_name (str): The name of the colormap used for dynamic point coloring. Defaults to 'inferno_r'.
        radius (float): The radius for plotting the LiDAR points on the image. Defaults to 2.5.
        min_range (Optional[float]): The minimum range value for normalization. Defaults to 3.
        max_range (Optional[float]): The maximum range value for normalization. Defaults to 50.
        opacity (float): The opacity value for the plotted points, ranging from 0 (transparent) to 1 (opaque).
                         Defaults to 0.5.

    Returns:
        PilImage: The image with the LiDAR points projected onto it.
    """
    proj_img = camera.image.image.copy()

    lidar_list = []
    for lidar in lidars:
        if isinstance(lidar, Lidar):
            lidar_list.append((lidar, None))
        elif isinstance(lidar, tuple) and isinstance(lidar[0], Lidar):
            lidar_list.append(lidar)
        else:
            raise ValueError("Each argument must be either a Lidar object or a tuple of (Lidar, optional color).")

    for lidar, static_color in lidar_list:
        pts, proj = get_projection(lidar, camera)
        proj_img = plot_points_on_image(proj_img, proj, pts, static_color=static_color, cmap_name=cmap_name,
                                        radius=radius, min_range=min_range, max_range=max_range, opacity=opacity)

    return proj_img


def show_points(points: Union[Lidar, np.ndarray],
                colors: Optional[np.ndarray] = None,
                point_size: Optional[float] = None) -> None:
    """Display the 3D point cloud from a LiDAR sensor or NumPy arrays of points and colors.

    This function visualizes the 3D point cloud data from a LiDAR sensor or NumPy arrays
    using Open3D for 3D visualization. If colors are provided, they will be applied to the
    points in the visualization. The point size can also be adjusted.

    Args:
        points (Union[Lidar, np.ndarray]): The LiDAR sensor or a NumPy array containing the 3D point cloud data.
        colors (Optional[np.ndarray]): An optional NumPy array containing RGB colors for each point.
        point_size (float): The size of the points in the visualization. Defaults to 8.0 if colors are provided,
                            otherwise 1.0.

    Raises:
        ImportError: If Open3D is not installed.

    Returns:
        None
    """
    if importlib.util.find_spec("open3d") is None:
        raise ImportError('Install open3d to use this function with: python -m pip install open3d')

    import open3d as o3d
    if isinstance(points, Lidar):
        points = np.stack((points.points['x'], points.points['y'], points.points['z']), axis=-1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if point_size is None:
        point_size = 8.0 if colors is not None else 1.0

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Adjust the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()


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
