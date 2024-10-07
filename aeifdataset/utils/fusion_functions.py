"""
This module provides functions for fusing data from LiDAR and camera sensors. It includes functionality
to project 3D LiDAR points onto a 2D camera image plane, plot these points on images, and generate
images with LiDAR points overlaid.

Functions:
    get_projection(lidar, camera): Projects LiDAR points onto a camera image plane.
    combine_lidar_points(agent=None, *lidars): Combines points from multiple LiDARs and returns them as a NumPy array.
    _transform_lidar_to_origin(lidar_sensor): Transforms LiDAR points to a common coordinate origin.
"""

from typing import Tuple, Union, Optional
import numpy as np
from aeifdataset.data import Lidar, Camera, Tower, Vehicle
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


"""
def projection_rgb_function:
  points = []
points_color = []

# Schleife durch Kameras und LiDARs
for _, camera in frame.vehicle.cameras:
    if _ == "STEREO_RIGHT":
        continue
    rgb_image = np.array(camera)
    for _, lidar in frame.vehicle.lidars:

        pts_3d, proj_2d = ad.get_projection(lidar, camera)

        # Füge 3D-Punkte zur Liste hinzu
        # todo: get_transformation
        pts_new = np.stack((pts_3d[:,0],
                           pts_3d[:,1],
                           pts_3d[:,2],
                           np.ones((pts_3d.shape[0]))))

        # Get the transformation matrix and apply it
        trans = ad.get_transformation(lidar)
        transformed_points = (trans.transformation_mtx @ pts_new).T
        # new_pts = _transform_lidar_to_origin(lidar)
        points.append(transformed_points[:,:3])

        # Extrahiere die Farbwerte für die projizierten 2D-Punkte
        for proj_pt in proj_2d:
            u, v = int(proj_pt[0]), int(proj_pt[1])
            # Hole den RGB-Wert aus dem Bildarray
            r, g, b = rgb_image[v, u, :]
            points_color.append([r / 255.0, g / 255.0, b / 255.0])

# Konvertiere die Listen in NumPy-Arrays
points = np.vstack(points)  # Stapelt alle 3D-Punkte vertikal zu einem großen Array
points_color = np.array(points_color)



point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(points_color)

# Erstelle ein Visualizer-Objekt
vis = o3d.visualization.Visualizer()
vis.create_window()

# Füge die Punktwolke hinzu
vis.add_geometry(point_cloud)

# Setze die Punktgröße
opt = vis.get_render_option()
opt.point_size = 8.0  # Setzt die Punktgröße (Standard ist 1.0)
view_control = vis.get_view_control()
view_control.set_lookat([0, 0, 0])
# Starte die Visualisierung
vis.run()  
"""


def combine_lidar_points(agent: Union[Tower, Vehicle] = None,
                         *lidars: Optional[Lidar]) -> np.ndarray:
    """Combines points from one or multiple LiDAR sensors and returns them as a NumPy array.

    This function can either take an agent (like a Tower or Vehicle) containing LiDAR sensors,
    or individual LiDAR objects. The points from all the provided sensors are transformed
    and combined into a single NumPy array.

    Args:
        agent (Union[Tower, Vehicle], optional): An agent object containing LiDAR sensors (default is None).
        *lidars (Optional[Lidar]): One or more LiDAR objects to combine points from.

    Returns:
        np.ndarray: A NumPy array containing the 3D points from all the LiDAR sensors.
    """
    all_points = []
    if isinstance(agent, (Tower, Vehicle)):
        lidars = tuple(lidar for _, lidar in agent.lidars)

    for lidar_obj in lidars:
        if hasattr(lidar_obj, 'lidars'):  # Agent object case
            for _, lidar_sensor in lidar_obj.lidars:
                points = _transform_lidar_to_origin(lidar_sensor)
                all_points.append(points)
        else:  # LiDAR object case
            points = _transform_lidar_to_origin(lidar_obj)
            all_points.append(points)

    all_points = np.vstack(all_points)
    all_points = all_points[:, :3]
    return all_points


def _transform_lidar_to_origin(lidar_sensor: Lidar) -> np.ndarray:
    """Transforms LiDAR points to a common coordinate origin.

    This function takes a LiDAR sensor's points and applies its transformation matrix to
    convert the points into a common coordinate frame.

    Args:
        lidar_sensor (Lidar): The LiDAR sensor object containing point data.

    Returns:
        np.ndarray: A NumPy array containing the transformed 3D points.
    """
    points = np.stack((lidar_sensor.points.points['x'],
                       lidar_sensor.points.points['y'],
                       lidar_sensor.points.points['z'],
                       np.ones((lidar_sensor.points.points.shape[0]))))

    # Get the transformation matrix and apply it
    trans = get_transformation(lidar_sensor)
    transformed_points = trans.transformation_mtx @ points

    return transformed_points.T