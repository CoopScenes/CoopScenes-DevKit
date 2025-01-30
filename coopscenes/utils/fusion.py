"""
This module provides functions for fusing data from LiDAR and camera sensors. It includes functionality for projecting
3D LiDAR points onto 2D camera image planes, retrieving corresponding RGB values, and combining 3D points from
multiple LiDAR sensors.

Functions:
    get_projection(lidar, camera, vehicle_info):
        Projects 3D LiDAR points onto a camera image plane using the camera's intrinsic, extrinsic, and rectification matrices.

    get_rgb_projection(lidar, camera, vehicle_info):
        Projects 3D LiDAR points onto a camera image plane and retrieves the corresponding RGB values for each projected point.

    combine_lidar_points(*args, vehicle_info):
        Combines 3D points from multiple LiDAR sensors (from a Frame, Tower, Vehicle, or individual sensors) and returns them as a single NumPy array.
"""
from typing import Tuple, Union, Optional
import numpy as np
from coopscenes.data import Lidar, Camera, Tower, Vehicle, Frame, VehicleInformation, LidarInformation
from coopscenes.utils import get_transformation, transform_points_to_origin
import importlib.util


def get_projection(lidar: Lidar, camera: Camera, vehicle_info: Optional[VehicleInformation] = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """Projects LiDAR points onto a camera image plane.

    Transforms the 3D points from a LiDAR sensor into the camera's coordinate frame and projects them onto the 2D
    image plane of the camera using the camera's intrinsic, extrinsic, and rectification matrices. Filters points
    that are behind the camera or outside the image bounds.

    Args:
        lidar (Lidar): The LiDAR sensor containing 3D points to project.
        camera (Camera): The camera onto which the LiDAR points will be projected.
        vehicle_info (Optional[VehicleInformation]): Optional VehicleInformation for global transformation.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A NumPy array of shape (N, 3) containing the 3D points that are within the camera's field of view.
            - A NumPy array of shape (N, 2) representing the 2D image coordinates of the projected points.
    """
    lidar_tf = get_transformation(lidar)
    camera_tf = get_transformation(camera)

    if lidar_tf.to != camera_tf.to:
        if lidar_tf.to == "lidar_top":
            if vehicle_info is None:
                raise ValueError("vehicle_info must be provided when transforming between agents.")
            vehicle_tf = get_transformation(vehicle_info)
            lidar_tf = lidar_tf.combine_transformation(vehicle_tf)
        elif camera_tf.to == "lidar_top":
            if vehicle_info is None:
                raise ValueError("vehicle_info must be provided when transforming between agents.")
            vehicle_tf = get_transformation(vehicle_info)
            camera_tf = camera_tf.combine_transformation(vehicle_tf)

    camera_inverse_tf = camera_tf.invert_transformation()
    lidar_to_cam_tf = lidar_tf.combine_transformation(camera_inverse_tf)

    # Apply rectification and projection matrices
    rect_mtx = np.eye(4)
    rect_mtx[:3, :3] = camera.info.rectification_mtx
    proj_mtx = camera.info.projection_mtx

    # Prepare points in homogeneous coordinates
    points_3d = np.array([point.tolist()[:3] for point in lidar.points])
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Transform points to camera coordinates
    points_in_camera = lidar_to_cam_tf.mtx.dot(points_3d_homogeneous.T).T

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


def get_rgb_projection(lidar: Lidar, camera: Camera, vehicle_info: Optional[VehicleInformation] = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Projects LiDAR points onto a camera image plane and retrieves their corresponding RGB values.

    First projects the LiDAR points onto the camera's 2D image plane. Then, for each projected 2D point, it
    retrieves the corresponding RGB color from the camera's image.

    Args:
        lidar (Lidar): The LiDAR sensor containing 3D points to project.
        camera (Camera): The camera onto which the LiDAR points will be projected.
        vehicle_info (Optional[VehicleInformation]): Optional VehicleInformation for global transformation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - A NumPy array of shape (N, 3) containing the 3D points that are within the camera's field of view.
            - A NumPy array of shape (N, 2) representing the 2D image coordinates of the projected points.
            - A NumPy array of shape (N, 3) representing the RGB color for each 3D point.
    """
    points_color = []
    rgb_image = np.array(camera)

    pts_3d, proj_2d = get_projection(lidar, camera, vehicle_info)

    for proj_pt in proj_2d:
        u, v = int(proj_pt[0]), int(proj_pt[1])
        r, g, b = rgb_image[v, u, :]
        points_color.append([r / 255.0, g / 255.0, b / 255.0])

    points_color = np.array(points_color)

    return pts_3d, proj_2d, points_color


def combine_lidar_points(*args: Union[Frame, Tower, Vehicle, Lidar, Tuple[np.ndarray, LidarInformation]],
                         vehicle_info: Optional[VehicleInformation] = None) -> np.ndarray:
    """Combines 3D points from one or multiple LiDAR sensors into a single array.

    Args:
        *args: Either a Frame, Vehicle, and/or Tower object containing LiDAR sensors,
               one or more individual Lidar objects, or tuples of (np.ndarray, LidarInformation).
        vehicle_info (Optional[VehicleInformation]): Optional VehicleInformation for global transformation.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3) containing the combined 3D points from all the LiDAR sensors.

    Raises:
        ValueError: If multiple Frame objects are provided.
    """
    all_points = []
    frame = None
    vehicle = None
    tower = None
    lidars = []
    tuples = []

    for arg in args:
        if isinstance(arg, Frame):
            if frame:
                raise ValueError("Only one Frame object can be provided.")
            frame = arg
        elif isinstance(arg, Vehicle):
            if vehicle:
                raise ValueError("Only one Vehicle object can be provided outside of a Frame.")
            vehicle = arg
        elif isinstance(arg, Tower):
            if tower:
                raise ValueError("Only one Tower object can be provided outside of a Frame.")
            tower = arg
        elif isinstance(arg, Lidar):
            lidars.append(arg)
        elif isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], np.ndarray) and isinstance(arg[1],
                                                                                                        LidarInformation):
            tuples.append(arg)
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")

    if frame:
        for _, lidar_obj in (*frame.vehicle.lidars, *frame.tower.lidars):
            points = transform_points_to_origin(lidar_obj, vehicle_info or frame.vehicle.info)
            all_points.append(points)

    if vehicle:
        for _, lidar_obj in vehicle.lidars:
            points = transform_points_to_origin(lidar_obj, vehicle_info or vehicle.info)
            all_points.append(points)

    if tower:
        for _, lidar_obj in tower.lidars:
            points = transform_points_to_origin(lidar_obj, vehicle_info)
            all_points.append(points)

    for lidar_obj in lidars:
        points = transform_points_to_origin(lidar_obj, vehicle_info)
        all_points.append(points)

    for points_array, lidar_info in tuples:
        points = transform_points_to_origin((points_array, lidar_info), vehicle_info)
        all_points.append(points)

    if not all_points:
        raise ValueError("No LiDAR points provided or found in the specified input.")

    all_points = np.vstack(all_points)
    return all_points[:, :3]


def remove_hidden_points(lidar: Lidar,
                         camera: Camera,
                         vehicle_info: Optional[VehicleInformation] = None,
                         radius: int = 300000,
                         return_mask: bool = False
                         ) -> np.array:
    """
    Removes points that are in line of sight of a camera.
    Args:
        param: ['rm_los'] with field: radius
        points: A numpy array containing structured data with fields 'x', 'y', 'z'.
        camera_pose: Camera translation as a list of [x, y, z] to the LiDAR.
    Returns:
        filtered_points: A numpy array containing structured data with the same fields as 'points',
        with the points that are in line of sight removed.
    """
    if importlib.util.find_spec("open3d") is None:
        raise ImportError('Install open3d to use this function with: python -m pip install open3d')
    import open3d as o3d

    lidar_tf = get_transformation(lidar)
    camera_tf = get_transformation(camera)

    if lidar_tf.to != camera_tf.to:
        if lidar_tf.to == "lidar_top":
            if vehicle_info is None:
                raise ValueError("vehicle_info must be provided when transforming between agents.")
            vehicle_tf = get_transformation(vehicle_info)
            lidar_tf = lidar_tf.combine_transformation(vehicle_tf)
        elif camera_tf.to == "lidar_top":
            if vehicle_info is None:
                raise ValueError("vehicle_info must be provided when transforming between agents.")
            vehicle_tf = get_transformation(vehicle_info)
            camera_tf = camera_tf.combine_transformation(vehicle_tf)

    lidar_to_cam_tf = lidar_tf.combine_transformation(camera_tf).mtx
    camera_transition = lidar_to_cam_tf[:, -1]
    camera_position = camera_transition[:3].reshape(3, 1)

    # Manually extract x, y, z raw from the structured array
    pcd = o3d.geometry.PointCloud()
    xyz = np.vstack((lidar.points['x'], lidar.points['y'], lidar.points['z'])).T
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, pt_map = pcd.hidden_point_removal(camera_position, radius)
    mask = np.zeros(len(np.asarray(xyz)), dtype=bool)
    mask[pt_map] = True
    filtered_points = xyz[mask]
    if return_mask:
        return filtered_points, mask
    return filtered_points
