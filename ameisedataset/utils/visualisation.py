from PIL import ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
import ameisedataset.utils.transformation as tf
import ameisedataset.utils.image_functions as img_fkt


def show_disparity_map(disparity_map, cmap_name="viridis", val_min=None, val_max=None):
    cmap = matplotlib.colormaps[cmap_name]
    val_min = val_min if val_min is not None else np.min(disparity_map)
    val_max = val_max if val_max is not None else np.max(disparity_map)
    mask = disparity_map > 10 * val_max
    norm_values = np.where(mask, 0, (disparity_map - val_min) / (val_max - val_min))
    colored_map = cmap(norm_values)
    colored_map[mask] = [0, 0, 0, 1]  # Set masked values to black
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored_map)
    return img


def plot_points_on_image(img, points, values, cmap_name="inferno", radius=2):
    draw = ImageDraw.Draw(img)
    cmap = matplotlib.colormaps[cmap_name + "_r"]
    val_min = np.min(values)
    val_max = np.max(values) * 0.5

    norm_values = (values - val_min) / (val_max - val_min)

    for punkt, value in zip(points, norm_values):
        x, y = punkt
        rgba = cmap(value)
        farbe = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=farbe)
    return img


def visualize_points(points):
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


def vis_lidar_temporal(points):
    max_timestamp = np.max(points['t'])
    min_timestamp = np.min(points['t'])

    threshold_ts = min_timestamp + 0.05 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.1 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.2 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.3 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.4 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)


def check_tf_correction(camera: Camera, lidar: Lidar, roll_correction, pitch_correction, yaw_correction):
    if 'view' in lidar.info.name:
        range_variable = 'y'
    else:
        range_variable = 'range'

    # Original projection
    pts, proj = tf.get_projection(lidar, camera)

    # Display original projection
    stereo_left_rect_1 = img_fkt.rectify_image(camera)
    proj_img = plot_points_on_image(stereo_left_rect_1, proj, pts[range_variable])

    # Adjust extrinsic parameters
    x, y, z = camera.info.extrinsic.xyz
    roll, pitch, yaw = camera.info.extrinsic.rpy
    camera.info.extrinsic.xyz = np.array([x, y, z])
    camera.info.extrinsic.rpy = np.array([roll + roll_correction, pitch + pitch_correction, yaw + yaw_correction])

    # New projection with corrected extrinsic parameters
    pts2, proj2 = tf.get_projection(lidar, camera)

    # Display corrected projection
    stereo_left_rect_2 = img_fkt.rectify_image(camera)
    proj_img_corrected = plot_points_on_image(stereo_left_rect_2, proj2, pts2[range_variable])

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
