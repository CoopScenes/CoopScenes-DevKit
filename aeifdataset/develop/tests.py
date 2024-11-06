import multiprocessing as mp
import aeifdataset as ad
from aeifdataset import DataRecord
from aeifdataset.develop import show_tf_correction
from tqdm import tqdm
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


if __name__ == '__main__':
    save_dir = '/mnt/dataset/anonymisation/validation/27_09_seq_1/png'
    dataset = ad.Dataloader("/mnt/hot_data/dataset/seq_3_bushaltestelle")

    frame = dataset[3][0]

    image = frame.tower.cameras.VIEW_1
    points = frame.tower.lidars.UPPER_PLATFORM

    image2 = frame.tower.cameras.VIEW_2

    # ad.save_image(image, '/mnt/hot_data/samples', '_png')
    # ad.show_points(points)

    # ad.show_tf_correction(image, points, -0.003, -0.01, -0.004)
    # ad.get_projection_img(image, points).show()
    ad.get_projection_img(image2, points).show()
