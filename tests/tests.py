from ameisedataset import DataRecord
import ameisedataset.utils.visualisation as viz

# id04390_2024-07-18_18-11-45.4mse
# id03960_2024-07-18_18-11-02.4mse
# id04770_2024-07-18_18-12-23.4mse

# id09940_2024-07-18_18-21-00.4mse

# id09700_2024-07-18_18-20-36.4mse

dataloader = DataRecord("/media/slam/Extreme SSD/datasets/record_1/packed/id03060_2024-07-18_18-09-32.4mse")
frame = dataloader[9]
camera = frame.tower.cameras.VIEW_2
lidar = frame.tower.lidars.VIEW_2

viz.check_tf_correction(camera, lidar, roll_correction=0, pitch_correction=0, yaw_correction=0)
