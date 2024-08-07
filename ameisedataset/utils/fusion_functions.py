import ameisedataset.utils.transformation as tf
import ameisedataset.utils.image_functions as img_fkt
from ameisedataset.data import Lidar, Camera
from PIL import ImageDraw
import numpy as np
import matplotlib
from typing import Tuple, List


def get_projection(lidar: Lidar, camera: Camera) -> Tuple[np.array, List[Tuple]]:
    lidar_tf = tf.get_transformation(lidar)
    camera_tf = tf.get_transformation(camera)

    camera_inverse_tf = camera_tf.invert_transformation()
    lidar_to_cam_tf = lidar_tf.combine_transformation(camera_inverse_tf)
    rect_mtx = np.eye(4)
    rect_mtx[:3, :3] = camera.info.rectification_mtx
    proj_mtx = camera.info.projection_mtx

    projection = []
    points = []
    for point in lidar.points.points:
        point_vals = np.array(point.tolist()[:3])
        # Transform points to new coordinate system
        point_in_camera = proj_mtx.dot(
            rect_mtx.dot(lidar_to_cam_tf.transformation_mtx.dot(np.append(point_vals[:3], 1))))
        # check if pts are behind the camera
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


def get_projection_img(camera: Camera, lidar: Lidar, intensity=False, static_color=None, max_range_factor=0.5):
    if intensity:
        highlight = 'intensity'
    elif 'view' in lidar.info.name:
        highlight = 'y'
    else:
        highlight = 'range'

    # Original projection
    pts, proj = get_projection(lidar, camera)

    # Display original projection
    proj_img = plot_points_on_image(camera, proj, pts[highlight], static_color=static_color,
                                    max_range_factor=max_range_factor)
    return proj_img


def plot_points_on_image(camera, points, values, cmap_name="inferno", radius=2, static_color=None,
                         max_range_factor=0.5):
    rect_img = img_fkt.get_rect_img(camera)

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
