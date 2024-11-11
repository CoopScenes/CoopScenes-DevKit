import multiprocessing as mp
import aeifdataset as ad
from aeifdataset import DataRecord
from aeifdataset.develop import show_tf_correction
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import numpy as np


# Funktion, die in jedem Prozess ausgeführt wird
def save_datarecord_images(datarecord, save_dir):
    for frame in datarecord:
        ad.save_all_images_in_frame(frame, save_dir, create_subdir=True)


def save_dataset_images_multithreaded(dataset, save_dir):
    # Anzahl der Prozessoren festlegen
    num_workers = 6

    # Pool erstellen
    with mp.Pool(processes=num_workers) as pool:
        # Erstellen der Aufgaben für jeden datarecord
        for datarecord in tqdm(dataset, desc="Submitting tasks for datarecords"):
            pool.apply_async(save_datarecord_images, args=(datarecord, save_dir))

        # Warten, bis alle Prozesse abgeschlossen sind
        pool.close()
        pool.join()


def filter_points(points, x_range, y_range, z_range):
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    mask = (points['x'] < x_min) | (points['x'] > x_max) | \
           (points['y'] < y_min) | (points['y'] > y_max) | \
           (points['z'] < z_min) | (points['z'] > z_max)
    return points[mask]


def filter_points(points, x_range, y_range, z_range):
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    mask = (points['x'] < x_min) | (points['x'] > x_max) | \
           (points['y'] < y_min) | (points['y'] > y_max) | \
           (points['z'] < z_min) | (points['z'] > z_max)
    return points[mask]


if __name__ == '__main__':
    save_dir = '/mnt/dataset/anonymisation/validation/27_09_seq_1/png'
    dataset = ad.Dataloader("/mnt/hot_data/dataset/seq_1_maille")
    # frame = DataRecord('/mnt/hot_data/dataset/seq_1_maille/id02315_2024-09-27_10-44-30.4mse')[90]
    # frame = DataRecord('/mnt/hot_data/dataset/seq_1_maille/id03878_2024-09-27_11-03-43.4mse')[0]
    frame = DataRecord('/mnt/hot_data/dataset/seq_1_maille/id04645_2024-09-27_11-22-33.4mse')[0]
    print(np.linalg.norm(frame.vehicle.DYNAMICS.velocity[0].linear_velocity) * 3.6)

    points_left = frame.vehicle.lidars.LEFT
    points_top = frame.vehicle.lidars.TOP
    points_right = frame.vehicle.lidars.RIGHT

    ad.show_points(
        (points_left, (255, 0, 0)),
        (points_top, (0, 255, 0)),
        (points_right, (0, 0, 255))
    )

    '''
    frame = dataset[0][0]
    for datarecord in dataset:
        for frame in datarecord:
            speed = np.linalg.norm(frame.vehicle.DYNAMICS.velocity[0].linear_velocity) * 3.6
            if speed < 1:
                print(f'Datarecord: {datarecord.name}, Frame: {frame.frame_id}')
    
    # image = frame.vehicle.cameras.STEREO_LEFT
    
    
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
    '''
    # ad.save_image(image, '/mnt/hot_data/samples')
    # ad.show_points(points)

    # ad.show_tf_correction(image, points, -0.003, -0.01, -0.004)
    # ad.get_projection_img(image, points).show()
    # ad.get_projection_img(image2, points).show()
