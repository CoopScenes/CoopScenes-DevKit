from ameisedataset import DataRecord
import ameisedataset.utils.visualisation as viz
import ameisedataset.utils.image_functions as imf

# id04390_2024-07-18_18-11-45.4mse
# id03960_2024-07-18_18-11-02.4mse
# id04770_2024-07-18_18-12-23.4mse

# id09940_2024-07-18_18-21-00.4mse

# id09700_2024-07-18_18-20-36.4mse

dataloader = DataRecord("/media/ameise/Extreme SSD/datasets/record_1/packed/id04390_2024-07-18_18-11-45.4mse")
frame = dataloader[9]
camera = frame.vehicle.cameras.BACK_RIGHT
lidar = frame.vehicle.lidars.TOP
stereo_left = frame.vehicle.cameras.STEREO_LEFT
stereo_right = frame.vehicle.cameras.STEREO_RIGHT
imf.get_rect_img(stereo_left).show()
# viz.show_projection(camera, lidar)
# viz.show_tf_correction(camera, lidar, roll_correction=0, pitch_correction=0, yaw_correction=0)
viz.show_disparity_map(stereo_left, stereo_right)
