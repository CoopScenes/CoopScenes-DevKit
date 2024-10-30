import multiprocessing as mp
import aeifdataset as ad
from aeifdataset import DataRecord
from aeifdataset.develop import show_tf_correction
from tqdm import tqdm


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


if __name__ == '__main__':
    save_dir = '/mnt/dataset/anonymisation/validation/27_09_seq_1/png'
    dataset = ad.Dataloader("/mnt/hot_data/dataset/seq_3_bushaltestelle")

    # frame = dataset[22][0]
    frames = DataRecord('/mnt/hot_data/dataset/seq_3_bushaltestelle/incomplete_frames.4mse')
    frame = frames[400:430]

    image = frame.tower.cameras.VIEW_1
    points = frame.tower.lidars.VIEW_1

    # ad.show_tf_correction(image, points, -0.003, -0.01, -0.004)
    ad.get_projection_img(image, points).show()
