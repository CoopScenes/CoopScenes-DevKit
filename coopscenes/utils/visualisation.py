"""
This module provides functions for visualizing sensor data from autonomous vehicles, including:
- Depth maps generated from stereo camera images.
- 3D point clouds visualized from LiDAR sensors.
- Projections of LiDAR points onto camera images.

Functions:
    get_colored_stereo_image(camera_left, camera_right, cmap_name, min_value, max_value):
        Computes and returns a depth map between two stereo camera images as a color-mapped image.
    plot_points_on_image(image, points, points_3d, cmap_name, radius, static_color, min_range, max_range, opacity):
        Overlays 2D points on a camera image with optional color mapping or static coloring.
    get_projection_img(camera, *lidars, cmap_name, radius, min_range, max_range, opacity, vehicle_info):
        Generates an image with LiDAR points projected onto the camera image with optional colormap and opacity.
    show_points(*args, point_size, cmap_name, min_range, max_range, vehicle_info):
        Displays 3D point clouds from multiple LiDAR sensors, frames, or agents with customizable visualization settings.
"""
from typing import Optional, Union, Tuple
from PIL import Image as PilImage, ImageDraw, ImageColor
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from coopscenes.data import Lidar, Camera, LidarInformation, VehicleInformation, Frame, Vehicle, Tower
from coopscenes.utils import get_projection, get_disparity_map, transform_points_to_origin


def get_colored_stereo_image(camera_left: Camera, camera_right: Camera, cmap_name: str = "viridis",
                             min_value: int = 0, max_value: int = 1000) -> PilImage:
    """Compute and return the disparity map between two stereo camera images as a color-mapped image.

       This function computes the disparity map from a pair of rectified stereo images using disparity calculation.
       The resulting disparity map is normalized between the specified `min_value` and `max_value`, color-mapped
       using the specified colormap, and returned as a PIL image.

       Args:
           camera_left (Camera): The left camera of the stereo pair.
           camera_right (Camera): The right camera of the stereo pair.
           cmap_name (str): The name of the colormap to use for visualization. Defaults to "viridis".
           min_value (int): The minimum disparity value to normalize. Values below this are clamped. Defaults to 0.
           max_value (int): The maximum disparity value for normalization. Values above this are masked. Defaults to 1000.

       Returns:
           PilImage: A color-mapped disparity map as a PIL image.
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
    """Overlay 2D points on a camera image with optional color mapping, range limits, and opacity.

    This function plots 2D points onto a camera image. Points can be dynamically colored based on their range
    values using a colormap or a static color. Supports customizing point radius and transparency.

    Args:
        image (PilImage): The camera image onto which the points will be plotted.
        points (np.ndarray): The 2D coordinates of the points to plot.
        points_3d (np.ndarray): Corresponding 3D points used to calculate the range.
        cmap_name (str): The name of the matplotlib colormap to use for dynamic color mapping. Defaults to "Spectral".
        radius (float): The radius of the points to plot. Defaults to 2.5.
        static_color (Optional[Union[str, Tuple[int, int, int]]]): A color for all points. Defaults to None.
        min_range (Optional[float]): Minimum range value for normalization. Defaults to 4.
        max_range (Optional[float]): Maximum range value for normalization. Defaults to 50.
        opacity (float): Transparency of the points (0 for transparent, 1 for opaque). Defaults to 0.6.

    Returns:
        PilImage: The image with points plotted on it.
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
                       *args: Union[
                           Lidar,
                           Frame,
                           Vehicle,
                           Tower,
                           Tuple[Union[Lidar, Frame, Vehicle, Tower], Optional[Union[str, Tuple[int, int, int]]]]],
                       cmap_name: str = "Spectral",
                       radius: float = 2.5,
                       min_range: Optional[float] = 4,
                       max_range: Optional[float] = 50,
                       opacity: float = 0.6,
                       vehicle_info: Optional[VehicleInformation] = None) -> PilImage:
    """Generate an image with LiDAR points projected onto a camera image.

    Projects LiDAR points from various sources (LiDARs, Frame, Vehicle, Tower) onto a camera image
    with optional colormap-based or static coloring.

    Args:
        camera (Camera): The camera onto which the LiDAR points are projected.
        *args: One or more sources of LiDAR points, such as Frame, Vehicle, Tower, Lidar, or
            tuples of (source, optional color).
        cmap_name (str): The name of the colormap for dynamic coloring. Defaults to "Spectral".
        radius (float): The radius of the plotted points. Defaults to 2.5.
        min_range (Optional[float]): Minimum range value for normalization. Defaults to 4.
        max_range (Optional[float]): Maximum range value for normalization. Defaults to 50.
        opacity (float): Transparency of the plotted points. Defaults to 0.6.
        vehicle_info (Optional[VehicleInformation]): Vehicle information for global transformations.

    Returns:
        PilImage: The image with LiDAR points projected onto it.
    """
    proj_img = camera.image.image.copy()

    lidar_list = []

    for source in args:
        if isinstance(source, Lidar):
            lidar_list.append((source, None))
        elif isinstance(source, tuple) and isinstance(source[0], (Lidar, Frame, Vehicle, Tower)):
            obj, static_color = source
            if isinstance(obj, Frame):
                vehicle_info = vehicle_info or obj.vehicle.info
                for _, lidar in (*obj.vehicle.lidars, *obj.tower.lidars):
                    lidar_list.append((lidar, static_color))
            elif isinstance(obj, Vehicle):
                vehicle_info = vehicle_info or obj.info
                for _, lidar in obj.lidars:
                    lidar_list.append((lidar, static_color))
            elif isinstance(obj, Tower):
                for _, lidar in obj.lidars:
                    lidar_list.append((lidar, static_color))
            elif isinstance(obj, Lidar):
                lidar_list.append((obj, static_color))
        elif isinstance(source, Frame):
            vehicle_info = vehicle_info or source.vehicle.info
            for _, lidar in (*source.vehicle.lidars, *source.tower.lidars):
                lidar_list.append((lidar, None))
        elif isinstance(source, Vehicle):
            vehicle_info = vehicle_info or source.info
            for _, lidar in source.lidars:
                lidar_list.append((lidar, None))
        elif isinstance(source, Tower):
            for _, lidar in source.lidars:
                lidar_list.append((lidar, None))
        else:
            raise ValueError(
                "Each source must be a Lidar, Frame, Vehicle, Tower, or a tuple of (source, optional color)."
            )

    for lidar, static_color in lidar_list:
        pts, proj = get_projection(lidar, camera, vehicle_info)
        pts = transform_points_to_origin((pts, lidar.info), vehicle_info=vehicle_info)
        proj_img = plot_points_on_image(proj_img, proj, pts, static_color=static_color, cmap_name=cmap_name,
                                        radius=radius, min_range=min_range, max_range=max_range, opacity=opacity)

    return proj_img


def show_points(*args: Union[
    Frame, Vehicle, Tower, np.ndarray, Lidar, Tuple[
        Union[np.ndarray, Lidar, Vehicle, Tower, Frame], Optional[Union[Tuple[int, int, int], np.ndarray]]]],
                point_size: Optional[float] = 3, cmap_name: str = "Spectral", min_range: float = 4,
                max_range: float = 50, vehicle_info: Optional[VehicleInformation] = None) -> None:
    """Visualize 3D point clouds from multiple LiDARs, frames, or agents with Open3D.

    Combines points from multiple sources and displays them in a 3D visualizer with customizable settings.

    Args:
        *args (Union[Frame, Vehicle, Tower, np.ndarray, Lidar, Tuple[Union[np.ndarray, Lidar, Vehicle, Tower, Frame], Optional[Union[Tuple[int, int, int], np.ndarray]]]]):
            Sources of 3D points, including frames, vehicles, towers, or individual LiDAR sensors.
        point_size (float, optional): Size of the points in the visualization. Defaults to 3.
        cmap_name (str, optional): Colormap name for range-based coloring. Defaults to "Spectral".
        min_range (float, optional): Minimum range value for normalization. Defaults to 4.
        max_range (float, optional): Maximum range value for normalization. Defaults to 50.
        vehicle_info (Optional[VehicleInformation]): Vehicle information for point transformations.

    Raises:
        ImportError: If Open3D is not installed.
        ValueError: If invalid data or colors are provided.

    Returns:
        None
    """
    if importlib.util.find_spec("open3d") is None:
        raise ImportError('Install open3d to use this function with: python -m pip install open3d')

    import open3d as o3d

    all_points = []
    all_colors = []

    cmap = plt.get_cmap(cmap_name)

    lidar_list = []
    for lidar_data in args:
        if isinstance(lidar_data, tuple):
            obj, color = lidar_data
            if isinstance(obj, Frame):
                vehicle_info = vehicle_info or obj.vehicle.info
                for _, lidar_obj in (*obj.vehicle.lidars, *obj.tower.lidars):
                    lidar_list.append((lidar_obj, color))
            elif isinstance(obj, Vehicle):
                vehicle_info = vehicle_info or obj.info
                for _, lidar_obj in obj.lidars:
                    lidar_list.append((lidar_obj, color))
            elif isinstance(obj, Tower):
                for _, lidar_obj in obj.lidars:
                    lidar_list.append((lidar_obj, color))
            elif isinstance(obj, (Lidar, np.ndarray)):
                lidar_list.append((obj, color))  # Direkt zur Liste hinzufügen
            else:
                raise ValueError("Unsupported object type in tuple: {}".format(type(obj)))
        elif isinstance(lidar_data, Frame):
            vehicle_info = vehicle_info or lidar_data.vehicle.info
            for _, lidar_obj in (*lidar_data.vehicle.lidars, *lidar_data.tower.lidars):
                lidar_list.append(lidar_obj)
        elif isinstance(lidar_data, Vehicle):
            vehicle_info = vehicle_info or lidar_data.info
            for _, lidar_obj in lidar_data.lidars:
                lidar_list.append(lidar_obj)
        elif isinstance(lidar_data, Tower):
            for _, lidar_obj in lidar_data.lidars:
                lidar_list.append(lidar_obj)
        elif isinstance(lidar_data, (Lidar, np.ndarray)):
            lidar_list.append(lidar_data)
        else:
            raise ValueError(
                "Each entry must be a Frame, Vehicle, Tower, Lidar object, a (n, 3) ndarray, or a tuple of either with an optional color.")

    for lidar_data in lidar_list:
        if isinstance(lidar_data, tuple):
            lidar_or_points, color = lidar_data
        else:
            lidar_or_points = lidar_data
            color = None

        if isinstance(lidar_or_points, Lidar):
            points = transform_points_to_origin(lidar_or_points, vehicle_info)  # Transform points for Lidar object
        elif isinstance(lidar_or_points, np.ndarray) and lidar_or_points.shape[1] == 3:  # Assuming it's a (n, 3) array
            points = lidar_or_points
        else:
            raise ValueError(
                "Each entry must be a Lidar object, a (n, 3) ndarray, or a tuple of either with an optional color.")

        all_points.append(points)

        if color is None:
            ranges = np.linalg.norm(points, axis=1)
            norm_values = (ranges - min_range) / (max_range - min_range)
            norm_values = np.clip(norm_values, 0, 1)
            colors = cmap(norm_values)[:, :3]
            all_colors.append(colors)
        elif isinstance(color, tuple) and len(color) == 3:
            static_color = np.tile(np.array(color) / 255.0, (points.shape[0], 1))
            all_colors.append(static_color)
        elif isinstance(color, np.ndarray) and color.shape == points.shape:
            all_colors.append(color / 255.0 if color.max() > 1 else color)
        else:
            raise ValueError(
                "Color must be an RGB tuple, an (n, 3) color array matching the number of points, or None.")

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()
