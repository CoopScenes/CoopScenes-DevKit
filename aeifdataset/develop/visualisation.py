from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import pypatchworkpp
import numpy as np
import open3d as o3d

from aeifdataset.data import Lidar, Camera
from aeifdataset.utils import get_projection_img, get_transformation, Transformation


def show_tf_correction(camera: Camera,
                       lidar_with_color: Union[Lidar, Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]]],
                       roll_correction: float = 0, pitch_correction: float = 0, yaw_correction: float = 0,
                       min_range: float = 4, max_range: float = 50, opacity: float = 0.6) -> None:
    """Display the effect of correcting the extrinsic parameters on LiDAR projection.

    This function visualizes the projection of LiDAR points onto a camera image before and after
    applying a correction to the camera's extrinsic parameters. The user can specify corrections
    to the roll, pitch, and yaw angles. The raw and corrected projections are displayed side-by-side.

    Args:
        camera (Camera): The camera whose extrinsic parameters will be adjusted.
        lidar_with_color (Union[Lidar, Tuple[Lidar, Optional[Union[str, Tuple[int, int, int]]]]]): Either a Lidar sensor or a tuple
            containing the LiDAR sensor and an optional static color for the points.
        roll_correction (float): Correction to apply to the roll angle (in radians).
        pitch_correction (float): Correction to apply to the pitch angle (in radians).
        yaw_correction (float): Correction to apply to the yaw angle (in radians).
        min_range (float): The minimum range value for normalization. Defaults to 4.
        max_range (float): The maximum range value for normalization. Defaults to 50.
        opacity (float): The opacity value for the plotted points, ranging from 0 (transparent) to 1 (opaque). Defaults to 0.6.

    Returns:
        None
    """
    # Handle the case where only a Lidar object is provided
    if isinstance(lidar_with_color, Lidar):
        lidar = lidar_with_color
        static_color = None
    elif isinstance(lidar_with_color, tuple) and isinstance(lidar_with_color[0], Lidar):
        lidar, static_color = lidar_with_color
    else:
        raise ValueError("Argument must be either a Lidar object or a tuple (Lidar, optional color).")

    # Project LiDAR points onto the camera image before applying corrections
    proj_img_raw = get_projection_img(camera, (lidar, static_color),
                                      min_range=min_range, max_range=max_range, opacity=opacity)

    # Apply the RPY corrections to the camera's extrinsic parameters
    camera.info.extrinsic.rpy += np.array([roll_correction, pitch_correction, yaw_correction])

    # Project LiDAR points onto the camera image after applying corrections
    proj_img_corrected = get_projection_img(camera, (lidar, static_color),
                                            min_range=min_range, max_range=max_range, opacity=opacity)

    # Plot the raw and corrected projections side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(40, 26))

    axes[0].imshow(proj_img_raw)
    axes[0].set_title('Raw Projection')
    axes[0].axis('off')

    axes[1].imshow(proj_img_corrected)
    axes[1].set_title(
        f'Corrected Projection [Roll: {roll_correction}, Pitch: {pitch_correction}, Yaw: {yaw_correction}]')
    axes[1].axis('off')

    new_transform = get_transformation(camera)
    lidar_transform = get_transformation(lidar)
    inverse_camera_transform = new_transform.invert_transformation()
    lidar_to_cam = lidar_transform.combine_transformation(inverse_camera_transform)
    cam_to_rotated = Transformation('cam_view_1', 'rotated_cam_view_1', 0, 0, 0, -1.57079632679, 0, -1.57079632679, )

    print(f'New Origin Transformation:\n{new_transform}')
    print(
        f'New lidar to rotated cam Transformation:\n{lidar_to_cam.combine_transformation(cam_to_rotated)}')

    plt.show()


def show_ground_segmentation(points):
    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = True
    params.sensor_height = 3.45
    params.min_range = 15
    params.max_range = 20

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    pointcloud = np.stack((points['x'], points['y'], points['z'], points['intensity']), axis=-1).astype(np.float64)

    # Estimate Ground
    PatchworkPLUSPLUS.estimateGround(pointcloud)

    # Get Ground and Nonground
    ground = PatchworkPLUSPLUS.getGround()
    nonground = PatchworkPLUSPLUS.getNonground()
    time_taken = PatchworkPLUSPLUS.getTimeTaken()

    ground_idx = PatchworkPLUSPLUS.getGroundIndices()
    nonground_idx = PatchworkPLUSPLUS.getNongroundIndices()

    # Get centers and normals for patches
    centers = PatchworkPLUSPLUS.getCenters()
    normals = PatchworkPLUSPLUS.getNormals()

    print("Origianl Points  #: ", pointcloud.shape[0])
    print("Ground Points    #: ", ground.shape[0])
    print("Nonground Points #: ", nonground.shape[0])
    print("Time Taken : ", time_taken / 1000000, "(sec)")
    print("Press ... \n")
    print("\t H  : help")
    print("\t N  : visualize the surface normals")
    print("\tESC : close the Open3D window")

    # Visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=600, height=400)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    ground_o3d = o3d.geometry.PointCloud()
    ground_o3d.points = o3d.utility.Vector3dVector(ground)
    ground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float)  # RGB
    )

    nonground_o3d = o3d.geometry.PointCloud()
    nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
    nonground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float)  # RGB
    )

    centers_o3d = o3d.geometry.PointCloud()
    centers_o3d.points = o3d.utility.Vector3dVector(centers)
    centers_o3d.normals = o3d.utility.Vector3dVector(normals)
    centers_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float)  # RGB
    )

    vis.add_geometry(mesh)
    vis.add_geometry(ground_o3d)
    vis.add_geometry(nonground_o3d)
    vis.add_geometry(centers_o3d)

    vis.run()
    vis.destroy_window()
