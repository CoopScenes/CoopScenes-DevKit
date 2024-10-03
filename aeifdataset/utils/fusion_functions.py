"""
This module provides functions for fusing data from Lidar and camera sensors. It includes functionalities
to project 3D Lidar points onto a 2D camera image plane, plot these points on images, and generate
images with Lidar points overlaid.

Functions:
    get_projection(lidar, camera): Project Lidar points onto a camera image plane.
    plot_points_on_image(camera, points, values, cmap_name, radius, static_color, max_range_factor, raw_image):
        Plot 2D points on a camera image with color mapping.
    get_projection_img(camera, lidar, intensity, static_color, max_range_factor, raw_image):
        Generate an image with Lidar points projected onto it.
"""
from typing import Tuple, List, Optional, Union
from PIL import Image as PilImage, ImageDraw, ImageColor
import numpy as np
import matplotlib.pyplot as plt
from aeifdataset.data import Lidar, Camera, Image
from aeifdataset.utils import get_transformation


def get_projection(lidar: Lidar, camera: Camera) -> Tuple[np.array, np.array]:
    """Project Lidar points onto a camera image plane with improved performance.

    Args:
        lidar (Lidar): The Lidar sensor containing 3D points to project.
        camera (Camera): The camera onto which the Lidar points are projected.

    Returns:
        Tuple[np.array, np.array]:
            - A NumPy array containing the 3D points within the camera's field of view.
            - A NumPy array of shape (N, 2) representing the 2D image coordinates of the projected points.
    """
    lidar_tf = get_transformation(lidar)
    camera_tf = get_transformation(camera)

    camera_inverse_tf = camera_tf.invert_transformation()
    lidar_to_cam_tf = lidar_tf.combine_transformation(camera_inverse_tf)

    # Apply rectification and projection matrices
    rect_mtx = np.eye(4)
    rect_mtx[:3, :3] = camera.info.rectification_mtx
    proj_mtx = camera.info.projection_mtx

    # Prepare points
    points_3d = np.array([point.tolist()[:3] for point in lidar.points.points])
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Transform points to camera coordinates
    points_in_camera = lidar_to_cam_tf.transformation_mtx.dot(points_3d_homogeneous.T).T

    # Apply rectification and projection
    points_in_camera = rect_mtx.dot(points_in_camera.T).T
    points_2d_homogeneous = proj_mtx.dot(points_in_camera.T).T

    # Normalize by the third (z) component to get image coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]

    # Filter points that are behind the camera
    valid_indices = points_2d_homogeneous[:, 2] > 0

    # Filter points within the image bounds
    u = points_2d[valid_indices, 0]
    v = points_2d[valid_indices, 1]
    within_bounds = (u >= 0) & (u < camera.info.shape[0]) & (v >= 0) & (v < camera.info.shape[1])

    final_points_3d = points_3d[valid_indices][within_bounds]
    final_projections = points_2d[valid_indices][within_bounds]

    return final_points_3d, final_projections


def plot_points_on_image(image: PilImage, points: List[Tuple[float, float]], points_3d: np.array,
                         cmap_name: str = "inferno", radius: int = 2,
                         static_color: Optional[Union[str, Tuple[int, int, int]]] = None,
                         max_range_factor: float = 0.5) -> PilImage:
    """Plot 2D points on a camera image with color mapping.

    This function plots a list of 2D points onto a camera image. If a static color is provided,
    all points will be plotted in that color. Otherwise, the points will be colored dynamically
    based on their range values using a specified colormap.

    Args:
        image (PilImage): The camera image to plot the points onto.
        points (List[Tuple[float, float]]): The 2D coordinates of the points to plot.
        points_3d (np.array): The corresponding 3D points for calculating the range.
        cmap_name (str): The name of the matplotlib colormap to use. Defaults to "inferno".
        radius (int): The radius of the points to plot. Defaults to 2.
        static_color (Optional[Union[str, Tuple[int, int, int]]]): A string representing a color name (e.g., "red")
            or an RGB tuple. If provided, this color is used for all points.
        max_range_factor (float): Factor to scale the max range of values for normalization. Defaults to 0.5.

    Returns:
        PilImage: The image with the points plotted on it.
    """
    draw = ImageDraw.Draw(image)

    if static_color is not None:
        # Convert string color to RGB if needed
        if isinstance(static_color, str):
            static_color = ImageColor.getrgb(static_color)

        # Plot all points with the static color
        for x, y in points:
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=static_color)
    else:
        # Calculate the range dynamically from 3D points
        cmap = plt.get_cmap(cmap_name)
        ranges = np.linalg.norm(points_3d, axis=1)
        val_min = np.min(ranges)
        val_max = np.max(ranges) * max_range_factor

        # Normalize the range values
        norm_values = (ranges - val_min) / (val_max - val_min)

        # Plot points with colormap based on range values
        for (x, y), value in zip(points, norm_values):
            rgba = cmap(value)
            color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)

    return image


def get_projection_img(camera: Camera, lidar: Lidar, lidar2: Optional[Lidar] = None, lidar3: Optional[Lidar] = None,
                       static_color: Optional[Union[str, Tuple[int, int, int]]] = None,
                       static_color2: Optional[Union[str, Tuple[int, int, int]]] = None,
                       static_color3: Optional[Union[str, Tuple[int, int, int]]] = None,
                       max_range_factor: float = 0.5) -> PilImage:
    """Generate an image with Lidar points projected onto it.

    Args:
        camera (Camera): The camera onto which the Lidar points are projected.
        lidar (Lidar): The first Lidar sensor containing points to project.
        lidar2 (Optional[Lidar]): An optional second Lidar sensor. Defaults to None.
        lidar3 (Optional[Lidar]): An optional third Lidar sensor. Defaults to None.
        static_color (Optional[Union[str, Tuple[int, int, int]]]): A static color for the first Lidar points.
        static_color2 (Optional[Union[str, Tuple[int, int, int]]]): A static color for the second Lidar points.
        static_color3 (Optional[Union[str, Tuple[int, int, int]]]): A static color for the third Lidar points.
        max_range_factor (float): Factor to scale the max range for normalization. Defaults to 0.5.

    Returns:
        PilImage: The image with the Lidar points projected onto it.
    """
    # Project points from the first Lidar and apply dynamic/static coloring
    pts, proj = get_projection(lidar, camera)
    range_vals = np.linalg.norm(pts[:, :3], axis=1)
    proj_img = plot_points_on_image(camera.image.image, proj, pts, static_color=static_color,
                                    max_range_factor=max_range_factor)

    # Repeat for lidar2 if provided
    if lidar2 is not None:
        pts2, proj2 = get_projection(lidar2, camera)
        range_vals2 = np.linalg.norm(pts2[:, :3], axis=1)
        proj_img = plot_points_on_image(proj_img, proj2, pts2, static_color=static_color2,
                                        max_range_factor=max_range_factor)

    # Repeat for lidar3 if provided
    if lidar3 is not None:
        pts3, proj3 = get_projection(lidar3, camera)
        range_vals3 = np.linalg.norm(pts3[:, :3], axis=1)
        proj_img = plot_points_on_image(proj_img, proj3, pts3, static_color=static_color3,
                                        max_range_factor=max_range_factor)

    return proj_img
