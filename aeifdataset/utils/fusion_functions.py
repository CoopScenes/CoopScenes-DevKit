"""
This module provides functions for fusing data from LiDAR and camera sensors, including the projection of 3D
LiDAR points onto 2D camera image planes and combining 3D points from multiple LiDAR sensors.

Functions:
    get_projection(lidar, camera):
        Projects 3D LiDAR points onto a camera image plane using the camera's intrinsic, extrinsic, and rectification matrices.

    get_rgb_projection(lidar, camera):
        Projects 3D LiDAR points onto a camera image plane and retrieves the corresponding RGB values for each projected point.

    combine_lidar_points(agent=None, *lidars):
        Combines 3D points from multiple LiDAR sensors (either from an agent object such as a Tower or Vehicle, or individual LiDAR sensors)
        and returns them as a single NumPy array.
"""
from typing import Tuple, Union, Optional
import numpy as np
from aeifdataset.data import Lidar, Camera, Tower, Vehicle
from aeifdataset.utils import get_transformation, transform_points_to_origin


def get_projection(lidar: Lidar, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Projects LiDAR points onto a camera image plane.

    This function transforms the 3D points from a LiDAR sensor into the camera's coordinate frame
    and projects them onto the 2D image plane of the camera using the camera's intrinsic, extrinsic,
    and rectification matrices. The function filters points that are behind the camera or outside
    the image bounds.

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


def get_rgb_projection(lidar: Lidar, camera: Camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects LiDAR points onto a camera image plane and retrieves their corresponding RGB values.

    This function first projects the LiDAR points onto the camera's 2D image plane. Then, for each
    projected 2D point, it retrieves the corresponding RGB color from the camera's image.

    Args:
        lidar (Lidar): The LiDAR sensor containing 3D points to project.
        camera (Camera): The camera onto which the LiDAR points will be projected.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - A NumPy array of shape (N, 3) containing the 3D points that are within the camera's field of view.
            - A NumPy array of shape (N, 2) representing the 2D image coordinates of the projected points.
            - A NumPy array of shape (N, 3) representing the RGB color for each 3D point.
    """
    points_color = []
    rgb_image = np.array(camera)

    pts_3d, proj_2d = get_projection(lidar, camera)

    for proj_pt in proj_2d:
        u, v = int(proj_pt[0]), int(proj_pt[1])
        # Hole den RGB-Wert aus dem Bildarray
        r, g, b = rgb_image[v, u, :]
        points_color.append([r / 255.0, g / 255.0, b / 255.0])

    points_color = np.array(points_color)

    return pts_3d, proj_2d, points_color


def combine_lidar_points(agent: Union[Tower, Vehicle] = None,
                         *lidars: Optional[Lidar]) -> np.ndarray:
    """Combines 3D points from one or multiple LiDAR sensors into a single array.

    This function can take either an agent (such as a Tower or Vehicle) containing multiple LiDAR sensors,
    or individual LiDAR sensor objects. The 3D points from all the provided LiDAR sensors are transformed
    into the agent's coordinate frame and combined into a single NumPy array.

    Args:
        agent (Union[Tower, Vehicle], optional): An agent object containing LiDAR sensors. If provided, all LiDARs
                                                from the agent will be combined. Defaults to None.
        *lidars (Optional[Lidar]): One or more individual LiDAR objects to combine points from, if no agent is provided.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3) containing the combined 3D points from all the LiDAR sensors.
    """
    all_points = []
    if isinstance(agent, (Tower, Vehicle)):
        lidars = tuple(lidar for _, lidar in agent.lidars)

    for lidar_obj in lidars:
        if hasattr(lidar_obj, 'lidars'):  # Agent object case
            for _, lidar_sensor in lidar_obj.lidars:
                points = transform_points_to_origin(lidar_sensor)
                all_points.append(points)
        else:  # LiDAR object case
            points = transform_points_to_origin(lidar_obj)
            all_points.append(points)

    all_points = np.vstack(all_points)
    all_points = all_points[:, :3]
    return all_points
