"""
This module provides functions for fusing data from LiDAR and camera sensors. It includes functionalities
to project 3D LiDAR points onto a 2D camera image plane, plot these points on images, and generate
images with LiDAR points overlaid.

Functions:
    get_projection(lidar, camera): Projects LiDAR points onto a camera image plane.
    plot_points_on_image(camera, points, values, cmap_name, radius, static_color, max_range_factor, raw_image):
        Plots 2D points on a camera image with color mapping.
    get_projection_img(camera, lidar, intensity, static_color, max_range_factor, raw_image):
        Generates an image with LiDAR points projected onto it.
"""
from typing import Tuple
import numpy as np
from aeifdataset.data import Lidar, Camera
from aeifdataset.utils import get_transformation


def get_projection(lidar: Lidar, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Projects LiDAR points onto a camera image plane with improved performance.

    This function transforms the 3D points from a LiDAR sensor into the camera's coordinate frame
    and projects them onto the 2D image plane of the camera using the camera's intrinsic and extrinsic parameters.

    Args:
        lidar (Lidar): The LiDAR sensor containing 3D points to project.
        camera (Camera): The camera onto which the LiDAR points will be projected.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A NumPy array of shape (N, 3) containing the 3D points that are within the camera's field of view.
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

    # Prepare points in homogeneous coordinates
    points_3d = np.array([point.tolist()[:3] for point in lidar.points.points])
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Transform points to camera coordinates
    points_in_camera = lidar_to_cam_tf.transformation_mtx.dot(points_3d_homogeneous.T).T

    # Apply rectification and projection to points
    points_in_camera = rect_mtx.dot(points_in_camera.T).T
    points_2d_homogeneous = proj_mtx.dot(points_in_camera.T).T

    # Normalize by the third (z) component to get 2D image coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]

    # Filter points that are behind the camera
    valid_indices = points_2d_homogeneous[:, 2] > 0

    # Filter points that are within the image bounds
    u = points_2d[valid_indices, 0]
    v = points_2d[valid_indices, 1]
    within_bounds = (u >= 0) & (u < camera.info.shape[0]) & (v >= 0) & (v < camera.info.shape[1])

    # Select the final 3D points and their 2D projections
    final_points_3d = points_3d[valid_indices][within_bounds]
    final_projections = points_2d[valid_indices][within_bounds]

    return final_points_3d, final_projections
