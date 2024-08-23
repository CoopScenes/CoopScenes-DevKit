from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Optional

from aeifdataset.data import Lidar, Camera
from aeifdataset.utils import get_projection_img
from aeifdataset.utils import get_depth_map


def show_disparity_map(camera_left, camera_right, cmap_name="viridis", max_value=40):
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    disparity_map = get_depth_map(camera_left, camera_right)

    # Set the min and max values for normalization
    val_min = np.min(disparity_map)
    val_max = max_value

    # Create a mask for outliers (disparity values greater than 10 * val_max)
    mask = disparity_map > 10 * val_max

    # Normalize the disparity map
    norm_values = (disparity_map - val_min) / (val_max - val_min)
    norm_values = np.clip(norm_values, 0, 1)  # Ensure values are within [0, 1]

    # Apply the colormap
    colored_map = cmap(norm_values)

    # Set masked values to black
    colored_map[mask] = [0, 0, 0, 1]  # RGBA, with alpha=1

    # Convert to 8-bit per channel image
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)

    # Create and return the image
    img = Image.fromarray(colored_map)

    img.show()


def show_points(lidar: Lidar):
    points = lidar.points.points

    # Convert structured NumPy array to a regular 3D NumPy array with contiguous memory.
    xyz_points = np.stack((points['x'], points['y'], points['z']), axis=-1)

    # Ensure the data type is float64, which is expected by Open3D.
    xyz_points = xyz_points.astype(np.float64)

    # Create Open3D point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)

    # Estimate normals.
    pcd.estimate_normals()

    # Visualize the point cloud.
    o3d.visualization.draw_geometries([pcd])


def show_projection(camera: Camera, lidar: Lidar, lidar2: Optional[Lidar] = None, lidar3: Optional[Lidar] = None,
                    static_color=None, static_color2=None,
                    static_color3=None, max_range_factor=0.5, intensity=False):
    camera.image.image = get_projection_img(camera, lidar, intensity, static_color, max_range_factor)
    if lidar2 is not None:
        camera.image.image = get_projection_img(camera, lidar2, intensity, static_color2, max_range_factor, False)
    if lidar3 is not None:
        camera.image.image = get_projection_img(camera, lidar3, intensity, static_color3, max_range_factor, False)
    camera.image.image.show()


def show_tf_correction(camera: Camera, lidar: Lidar, roll_correction, pitch_correction, yaw_correction, intensity=False,
                       static_color=None, max_range_factor=0.5):
    proj_img = get_projection_img(camera, lidar, intensity, static_color, max_range_factor)

    # Adjust extrinsic parameters
    x, y, z = camera.info.extrinsic.xyz
    roll, pitch, yaw = camera.info.extrinsic.rpy
    camera.info.extrinsic.xyz = np.array([x, y, z])
    camera.info.extrinsic.rpy = np.array([roll + roll_correction, pitch + pitch_correction, yaw + yaw_correction])

    # Display corrected projection
    proj_img_corrected = get_projection_img(camera, lidar, intensity, static_color, max_range_factor)

    fig, axes = plt.subplots(1, 2, figsize=(40, 26))

    # Display the first image
    axes[0].imshow(proj_img)
    axes[0].set_title('Raw')
    axes[0].axis('off')  # Hide axes

    # Display the second image
    axes[1].imshow(proj_img_corrected)
    axes[1].set_title(f'Corrected [Roll:{roll_correction}, Pitch:{pitch_correction}, Yaw:{yaw_correction}]')
    axes[1].axis('off')  # Hide axes

    print(camera.info.extrinsic)

    # Display the images
    plt.show()
