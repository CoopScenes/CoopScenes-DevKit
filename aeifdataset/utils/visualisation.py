"""
This module provides functions for visualizing sensor data from autonomous vehicles, including depth maps from stereo camera images,
3D point clouds from LiDAR sensors, and projections of LiDAR points onto camera images.

Functions:
    get_colored_stereo_image(camera_left, camera_right, cmap_name, min_value, max_value):
        Computes and returns the depth map between two stereo camera images as a color-mapped image.
    plot_points_on_image(image, points, points_3d, cmap_name, radius, static_color, min_range, max_range, opacity):
        Plots 2D points on a camera image with optional color mapping based on range values or static color.
    get_projection_img(camera, *lidars, cmap_name, radius, min_range, max_range, opacity):
        Generates an image with LiDAR points projected onto the camera image, with optional colormap, radius, and opacity settings.
    show_points(*lidars, point_size, cmap_name, min_range, max_range):
        Displays 3D point clouds from multiple LiDAR sensors or NumPy arrays with customizable colors and point size.
"""
from typing import Optional, Union, Tuple
from PIL import Image as PilImage, ImageDraw, ImageColor
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from aeifdataset.data import Lidar, Camera, LidarInformation, Pose
from aeifdataset.utils import get_projection, get_disparity_map, transform_points_to_origin


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
                         min_range: Optional[float] = 4,
                         max_range: Optional[float] = 50, opacity: float = 0.6) -> PilImage:
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
                       min_range: Optional[float] = 4, max_range: Optional[float] = 50,
                       opacity: float = 0.6) -> PilImage:
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


def show_points(*lidars: Union[
    np.ndarray, Lidar, Tuple[Union[np.ndarray, Lidar], Optional[Union[Tuple[int, int, int], np.ndarray]]]],
                point_size: Optional[float] = 3, cmap_name: str = "Spectral", min_range: float = 4,
                max_range: float = 50) -> None:
    """Display the 3D point cloud from multiple LiDAR sensors or point arrays with optional colors.

    Args:
        *lidars (Union[np.ndarray, Lidar, Tuple[Union[np.ndarray, Lidar], Optional[Union[Tuple[int, int, int], np.ndarray]]]]):
            One or more LiDAR sensors or point arrays, optionally paired with an RGB color tuple or an array of colors.
        point_size (float, optional): The size of the points in the visualization.
        cmap_name (str, optional): The name of the colormap to use for range-based coloring.
        min_range (float, optional): The minimum range value for color normalization. Defaults to 4.
        max_range (float, optional): The maximum range value for color normalization. Defaults to 50.

    Raises:
        ImportError: If Open3D is not installed.
        ValueError: If the provided color array does not match the number of points.

    Returns:
        None
    """
    if importlib.util.find_spec("open3d") is None:
        raise ImportError('Install open3d to use this function with: python -m pip install open3d')

    import open3d as o3d

    all_points = []
    all_colors = []

    cmap = plt.get_cmap(cmap_name)  # Load colormap for range-based coloring

    for lidar_data in lidars:
        # Check if lidar_data is a tuple with an optional color
        if isinstance(lidar_data, tuple):
            lidar_or_points, color = lidar_data
        else:
            lidar_or_points = lidar_data
            color = None

        # Extract points and transform them
        if isinstance(lidar_or_points, Lidar):
            points = transform_points_to_origin(lidar_or_points)  # Transform points for Lidar object
        elif isinstance(lidar_or_points, np.ndarray) and lidar_or_points.shape[1] == 3:  # Assuming it's a (n, 3) array
            points = transform_points_to_origin(
                (lidar_or_points, LidarInformation(name='',
                                                   extrinsic=Pose(np.array([0, 0, 0]), np.array(
                                                       [0, 0, 0])))))  # Transform raw np.ndarray points
        else:
            raise ValueError(
                "Each entry must be a Lidar object, a (n, 3) ndarray, or a tuple of either with an optional color.")

        all_points.append(points)

        # Color based on range with colormap if no static color is provided
        if color is None:
            ranges = np.linalg.norm(points, axis=1)
            norm_values = (ranges - min_range) / (max_range - min_range)
            norm_values = np.clip(norm_values, 0, 1)  # Ensure values are between 0 and 1
            colors = cmap(norm_values)[:, :3]  # Apply colormap and ignore alpha channel
            all_colors.append(colors)
        elif isinstance(color, tuple) and len(color) == 3:
            # Normalize the RGB color to [0, 1] range and apply to all points
            static_color = np.tile(np.array(color) / 255.0, (points.shape[0], 1))
            all_colors.append(static_color)
        elif isinstance(color, np.ndarray) and color.shape == points.shape:
            all_colors.append(color / 255.0 if color.max() > 1 else color)
        else:
            raise ValueError(
                "Color must be an RGB tuple, an (n, 3) color array matching the number of points, or None.")

    # Combine all points and colors
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()
