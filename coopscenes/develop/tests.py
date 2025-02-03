import coopscenes as ad
# from coopscenes.utils.transformation import apply_transformation_to_points

import numpy as np
import open3d as o3d
from math import radians, cos, sqrt
from decimal import Decimal
from kiss_icp.preprocess import get_preprocessor
from kiss_icp.config import KISSConfig
import h5py


def filter_points(points, x_range, y_range, z_range):
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    mask = (points['x'] < x_min) | (points['x'] > x_max) | \
           (points['y'] < y_min) | (points['y'] > y_max) | \
           (points['z'] < z_min) | (points['z'] > z_max)
    return points[mask]


def numpy_to_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Nur die x, y, z-Koordinaten
    return pcd


def structured_to_xyz(points):
    xyz = np.vstack((points['x'], points['y'], points['z'])).T  # Shape: (n, 3)
    return xyz


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


def flat_distance(lat1, lon1, lat2, lon2):
    # Umrechnung von Grad zu Metern
    lat_dist = (lat2 - lat1) * 111320  # Breitengradabstand in Metern
    lon_dist = (lon2 - lon1) * 111320 * cos(radians(lat1))  # Längengradabstand in Metern, angepasst an die Breite

    # Euklidischer Abstand
    distance = sqrt(lat_dist ** 2 + lon_dist ** 2)
    return distance


def find_matching_row(file_path: str, target_timestamp: Decimal) -> int:
    """
    Find the line number in the file where the timestamp matches the target timestamp with millisecond precision.

    Args:
        file_path (str): Path to the file containing timestamped data.
        target_timestamp (Decimal): Target timestamp to match.

    Returns:
        int: The line number that matches the timestamp, or -1 if no match is found.
    """
    # Convert the target timestamp to milliseconds
    target_timestamp_ms = target_timestamp // Decimal(1_000_000)

    # Open and read the file
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):  # Start line_number at 1
            # Split each line and parse the timestamp
            parts = line.strip().split()
            file_timestamp_ms = Decimal(parts[0]) // Decimal(1_000_000)

            # Check for match with millisecond precision
            if file_timestamp_ms == target_timestamp_ms:
                return line_number  # Return the line number

    return -1  # Return -1 if no match is found


def read_line_by_number(file_path: str, line_number: int) -> str:
    """
    Read a specific line from a file by its line number.

    Args:
        file_path (str): Path to the file.
        line_number (int): The line number to read.

    Returns:
        str: The content of the line.
    """
    with open(file_path, 'r') as file:
        for current_line_number, line in enumerate(file, start=1):
            if current_line_number == line_number:
                return line.strip()

    raise ValueError(f"Line number {line_number} not found in file.")


def format_to_4x4_matrix(line_content: str) -> np.ndarray:
    """
    Convert a line of 12 pose values into a 4x4 transformation matrix.

    Args:
        line_content (str): Line containing 12 values as a string.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # Extract the 12 values from the line
    values = [float(x) for x in line_content.split()]

    if len(values) != 12:
        raise ValueError("Line content does not contain 12 values.")

    # Convert to a 4x4 matrix
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[:3, :4] = np.array(values).reshape(3, 4)
    matrix[3, :] = [0, 0, 0, 1]  # Set the last row to [0, 0, 0, 1]
    return matrix

    # id01814_2024-10-12_11-44-04.4mse
    # id00865_2024-10-12_11-36-35.4mse


if __name__ == '__main__':
    """
    # save_dir = '/mnt/dataset/anonymisation/validation/27_09_seq_1/png'
    # dataset = ad.Dataloader("/mnt/hot_data/dataset/seq_1")
    example_record_1 = ad.DataRecord("example_record_1.4mse")
    frame = example_record_1[0]
    lidar = frame.tower.lidars.UPPER_PLATFORM
    camera = frame.vehicle.cameras.STEREO_LEFT
    xyz_without_hidden, mask = ad.remove_hidden_points(lidar=lidar, camera=camera, vehicle_info=frame.vehicle.info, return_mask=True)

    points = lidar._points_raw
    fields = ['intensity', 't', 'reflectivity', 'ring', 'ambient']
    points_additional = np.stack([lidar[field] for field in fields], axis=-1)
    filtered_add_points = points_additional[mask]
    combined_data = np.hstack((xyz_without_hidden, filtered_add_points))
    points_structured_without_hidden = np.array([tuple(row) for row in combined_data],
                                          dtype=np.dtype(LidarInformation._os_dtype_structure()))

    #lidar._points_deskewd = points_structured_without_hidden
    _, proj = ad.get_projection(lidar, camera, frame.vehicle.info)
    proj_img = ad.plot_points_on_image(camera.image.image.copy(), proj, _)
    proj_img.show()
    """
    """
    def _get_timestamps(points):
        points_ts = points['t']
        normalized_points_ts = (points_ts - points_ts.min()) / (points_ts.max() - points_ts.min())
        return normalized_points_ts


    lidar_points = frame.vehicle.lidars.TOP
    lidar_points2 = frame2.vehicle.lidars.TOP
    k_config = KISSConfig()
    k_config.data.max_range = 200
    k_config.data.min_range = 0
    k_config.data.deskew = True
    k_config.registration.max_num_threads = 0

    pts = structured_to_xyz(lidar_points)
    pts2 = structured_to_xyz(lidar_points2)

    preprocessor = get_preprocessor(k_config)
    # points_scam = ad.transform_points_to_origin(lidar_points)
    ts = _get_timestamps(lidar_points)
    flipped_arr = 1 - ts
    # transfom to top
    lidar_points_new = preprocessor.preprocess(pts2, flipped_arr, tf3.mtx)
    lidar_points_new_new = preprocessor.preprocess(pts, ts, tf4.mtx)

    # ad.show_points((lidar_points, (255, 0, 0)), (lidar_points_new, (0, 255, 0)))
    # back transform
    image = ad.get_projection_img(frame2.vehicle.cameras.STEREO_LEFT,
                                  ad.Lidar(lidar_points.info, ad.Points(lidar_points_new)))
    #image2 = ad.get_projection_img(frame2.vehicle.cameras.STEREO_LEFT,
    #                               lidar_points)
    image2 = ad.get_projection_img(frame2.vehicle.cameras.STEREO_LEFT,
                                  ad.Lidar(lidar_points.info, ad.Points(lidar_points_new_new)))
    image.show()
    image2.show()

    # frame.tower.cameras.VIEW_2.show()
    # frame.tower.cameras.VIEW_1.show()
    # frame.vehicle.cameras.STEREO_LEFT.show()

    # ad.show_points(frame)

    # ad.show_points((frame.vehicle, (64, 200, 200)), (frame.tower, (219, 48, 130)))

    points = []
    points_color = []

    for agent in frame:
        for _, camera in agent.cameras:
            for _, lidar in agent.lidars:
                pts_3d, proj_2d, color = ad.get_rgb_projection(lidar, camera, vehicle_info=frame.vehicle.info)
                if pts_3d.size > 0:
                    points.append(ad.transform_points_to_origin((pts_3d, lidar.info), vehicle_info=frame.vehicle.info))
                    points_color.append(color)

    points = np.vstack(points)
    points_color = np.vstack(points_color)

    # Visualize the fused point cloud with attached RGB values
    ad.show_points((points, points_color), point_size=5, vehicle_info=frame.vehicle.info)


    maneuvers = ad.get_maneuver_split('/mnt/hot_data/dataset/seq_1')

    for maneuver in maneuvers:
        import simplekml
        from datetime import datetime

        # KML-Datei erstellen
        kml = simplekml.Kml()

        # Positionen aus dem Manöver hinzufügen
        for record in maneuver:
            for frame in record:
                for pos in [frame.vehicle.GNSS.position[0]]:
                    # Sekunden und Nanosekunden extrahieren
                    raw_timestamp = pos.timestamp
                    seconds = raw_timestamp // 1_000_000_000  # Ganzzahlige Sekunden
                    nanoseconds = raw_timestamp % 1_000_000_000  # Restliche Nanosekunden

                    # Zeitstempel formatieren
                    timestamp = datetime.utcfromtimestamp(int(seconds))
                    formatted_timestamp = f"{timestamp.isoformat()}.{int(nanoseconds):09d}Z"

                    # Punkt hinzufügen
                    point = kml.newpoint(name="Position", coords=[(pos.longitude, pos.latitude)])
                    point.timestamp.when = formatted_timestamp  # Zeitstempel mit Nanosekunden

        # Fahrverlauf als Linie mit Zeitstempeln
        line = kml.newlinestring(name="Fahrverlauf")
        line.coords = [
            (frame.vehicle.GNSS.position[0].longitude, frame.vehicle.GNSS.position[0].latitude)
            for record in maneuver
            for frame in record
        ]
        line.style.linestyle.color = simplekml.Color.blue
        line.style.linestyle.width = 3

        # KML speichern
        kml.save("fahrverlauf_with_nanoseconds.kml")
        print("Fahrverlauf mit Nanosekundenpräzision in fahrverlauf_with_nanoseconds.kml gespeichert.")

    # Save one image as png or jpeg. Optional suffix can be applied.
    # ad.save_image(camera.image.image, output_path, f'{camera.image.get_timestamp()}_{camera_name}', dtype='jpeg')
    # ad.save_all_images_in_frame(frame, output_path, create_subdir=True)
    # ad.save_dataset_images_multithreaded(dataset, output_path, use_raw=True, num_cores=8)
    # Aktuelle Übereinstimmungen: 1831


    # Transformiere die Fahrzeug-Punktwolke mit der resultierenden Transformationsmatrix
    pcd_vehicle_transformed = pcd_vehicle.transform(transformation_matrix)

    # Visualisierung vorbereiten
    pcd_tower.paint_uniform_color([1, 0, 0])  # Turm-Punktwolke rot färben
    pcd_vehicle_transformed.paint_uniform_color([0, 1, 0])  # Fahrzeug-Punktwolke grün färben

    # Open3D-Visualizer starten
    o3d.visualization.draw_geometries(
        [pcd_tower, pcd_vehicle_transformed],
        zoom=0.7,
        front=[0.0, 0.0, -1.0],  # Ansicht nach vorne
        lookat=[0.0, 0.0, 0.0],  # Mittelpunkt der Ansicht
        up=[0.0, -1.0, 0.0]  # Oben-Achse nach oben ausgerichtet
    )

    frame = dataset[0][0]
    for datarecord in dataset:
        for frame in datarecord:
            speed = np.linalg.norm(frame.vehicle.DYNAMICS.velocity[0].linear_velocity) * 3.6
            if speed < 1:
                print(f'Datarecord: {datarecord.name}, Frame: {frame.frame_id}')

    image = frame.vehicle.cameras.STEREO_LEFT


    points_left = frame.vehicle.lidars.LEFT
    points_top = frame.vehicle.lidars.TOP
    points_right = frame.vehicle.lidars.RIGHT

        ad.show_points(
        (points_left, (255, 0, 0)),
        (points_top, (0, 255, 0)),
        (points_right, (0, 0, 255))
    )

    ad.show_points(
        (points_left, (255, 0, 0)),
        (points_top, (0, 255, 0))
    )

    xyz_points = np.stack(
        (points_left['x'], points_left['y'], points_left['z']), axis=-1)
    visualize_lidar_points(xyz_points, title='Upper Platform LiDAR Point Cloud')

    LEFT
    x_range = (-2.9, 1.8)
    y_range = (-1.7, 1.6)
    z_range = (-2.8, 0.2)

    RIGHT
    x_range = (-1.2, 1.5)
    y_range = (-0.6, 1.7)
    z_range = (-1.1, 0)

    new_pts = filter_points(points_right, x_range, y_range, z_range)
    coordinates = np.vstack((new_pts['x'], new_pts['y'], new_pts['z'])).T
    ad.show_points(points_right)

    ad.save_image(image, '/mnt/hot_data/samples')
    ad.show_points(points)

    ad.show_tf_correction(image, points, -0.003, -0.01, -0.004)
    ad.get_projection_img(image, points).show()
    ad.get_projection_img(image2, points).show()
    """
