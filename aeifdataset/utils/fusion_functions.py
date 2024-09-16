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
from typing import Tuple, List, Optional
from PIL import ImageDraw
from PIL import Image as PilImage
import numpy as np
import matplotlib
from aeifdataset.data import Lidar, Camera
from aeifdataset.utils import get_transformation, get_rect_img


def get_projection(lidar: Lidar, camera: Camera) -> Tuple[np.array, List[Tuple]]:
    """Project Lidar points onto a camera image plane.

    This function projects the 3D Lidar points onto the 2D image plane of a given camera
    using their respective transformations. Only points within the camera's field of view
    are returned along with their image coordinates.

    Args:
        lidar (Lidar): The Lidar sensor containing 3D points to project.
        camera (Camera): The camera onto which the Lidar points are projected.

    Returns:
        Tuple[np.array, List[Tuple]]:
            - A NumPy array containing the 3D points within the camera's field of view.
            - A list of tuples representing the 2D image coordinates of the projected points.
    """
    lidar_tf = get_transformation(lidar)
    camera_tf = get_transformation(camera)

    camera_inverse_tf = camera_tf.invert_transformation()
    lidar_to_cam_tf = lidar_tf.combine_transformation(camera_inverse_tf)
    rect_mtx = np.eye(4)
    rect_mtx[:3, :3] = camera.info.rectification_mtx
    proj_mtx = camera.info.projection_mtx

    projection = []
    points = []
    # TODO: change to matrix operation
    for point in lidar.points.points:
        point_vals = np.array(point.tolist()[:3])
        # Transform points to new coordinate system
        point_in_camera = proj_mtx.dot(
            rect_mtx.dot(lidar_to_cam_tf.transformation_mtx.dot(np.append(point_vals[:3], 1))))
        # Check if points are behind the camera
        u = point_in_camera[0] / point_in_camera[2]
        v = point_in_camera[1] / point_in_camera[2]
        if point_in_camera[2] <= 0:
            continue
        elif 0 <= u < camera.info.shape[0] and 0 <= v < camera.info.shape[1]:
            projection.append((u, v))
            points.append(point)
        else:
            continue
    return np.array(points, dtype=points[0].dtype), projection


def plot_points_on_image(camera: Camera, points: List[Tuple[float, float]], values: np.array,
                         cmap_name: str = "inferno", radius: int = 2,
                         static_color: Optional[Tuple[int, int, int]] = None,
                         max_range_factor: float = 0.5, raw_image: bool = True) -> PilImage:
    """Plot 2D points on a camera image with color mapping.

    This function plots a list of 2D points onto a rectified or raw camera image,
    using a specified colormap to represent the values associated with each point.
    The points can be plotted with varying colors or a static color.

    Args:
        camera (Camera): The camera object containing the image.
        points (List[Tuple[float, float]]): The 2D coordinates of the points to plot.
        values (np.array): The values to map to colors, used for color normalization.
        cmap_name (str): The name of the matplotlib colormap to use. Defaults to "inferno".
        radius (int): The radius of the points to plot. Defaults to 2.
        static_color (Optional[Tuple[int, int, int]]): RGB tuple for a static color. If provided,
            this color is used instead of the colormap.
        max_range_factor (float): Factor to scale the max range of values for normalization. Defaults to 0.5.
        raw_image (bool): If True, use the raw image for plotting. If False, use the rectified image. Defaults to True.

    Returns:
        PilImage: The image with the points plotted on it.
    """
    if raw_image:
        rect_img = get_rect_img(camera)
    else:
        rect_img = camera.image.image

    draw = ImageDraw.Draw(rect_img)
    cmap = matplotlib.colormaps[cmap_name + "_r"]
    val_min = np.min(values)
    val_max = np.max(values) * max_range_factor

    norm_values = (values - val_min) / (val_max - val_min)

    for punkt, value in zip(points, norm_values):
        x, y = punkt
        if static_color is None:
            rgba = cmap(value)
            color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        else:
            color = static_color
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
    return rect_img


def get_projection_img(camera: Camera, lidar: Lidar, intensity: bool = False,
                       static_color: Optional[Tuple[int, int, int]] = None, max_range_factor: float = 0.5,
                       raw_image: bool = True) -> PilImage:
    """Generate an image with Lidar points projected onto it.

    This function projects Lidar points onto a camera image, highlighting either intensity
    or range values. The function returns an image with the projected points overlaid.

    Args:
        camera (Camera): The camera onto which the Lidar points are projected.
        lidar (Lidar): The Lidar sensor containing points to project.
        intensity (bool): If True, use intensity values for coloring. If False, use range values. Defaults to False.
        static_color (Optional[Tuple[int, int, int]]): If specified, use a static color for all points.
        max_range_factor (float): Factor to scale the max range for normalization. Defaults to 0.5.
        raw_image (bool): If True, use the raw image for projection. If False, use the rectified image. Defaults to True.

    Returns:
        PilImage: The image with the Lidar points projected onto it.
    """
    highlight = 'intensity' if intensity else 'range'

    pts, proj = get_projection(lidar, camera)

    # Check if the desired highlight exists; needed only for old data
    if highlight not in pts:
        pts[highlight] = np.sqrt(pts['x'] ** 2 + pts['y'] ** 2 + pts['z'] ** 2)

    proj_img = plot_points_on_image(camera, proj, pts[highlight], static_color=static_color,
                                    max_range_factor=max_range_factor, raw_image=raw_image)
    return proj_img
