from decimal import Decimal
from typing import Optional
import numpy as np
from aeifdataset import Dataloader, DataRecord
import aeifdataset as ad
import os

# id04390_2024-07-18_18-11-45.4mse
# id03960_2024-07-18_18-11-02.4mse
# id04770_2024-07-18_18-12-23.4mse

# id09940_2024-07-18_18-21-00.4mse

# id09700_2024-07-18_18-20-36.4mse

example_record_1 = DataRecord("/mnt/dataset/dataset/seq_1_maille/packed/id00501_2024-09-27_10-32-20.4mse")

frame = example_record_1[0]

proj_img = ad.get_projection_img(frame.vehicle.cameras.STEREO_LEFT,
                                 frame.vehicle.lidars.TOP, frame.vehicle.lidars.LEFT, frame.vehicle.lidars.RIGHT)

proj_img.show()
# proj_img2 = ad.get_projection_img(frame.tower.cameras.VIEW_1,
#                                  frame.tower.lidars.UPPER_PLATFORM, frame.tower.lidars.VIEW_1,
#                                  frame.tower.lidars.VIEW_2)
# proj_img2.show()

'''
points = []
points_color = []

for _, camera in frame.vehicle.cameras:
    if _ == "STEREO_RIGHT":
        continue
    for _, lidar in frame.vehicle.lidars:
        pts_3d, proj_2d, color = ad.get_rgb_projection(lidar, camera)
        points.append(ad.transform_points_to_origin((pts_3d, lidar.info)))
        points_color.append(color)

points = np.vstack(points)
points_color = np.vstack(points_color)
ad.show_points(points, points_color)




stereo_img = ad.get_depth_map(frame.vehicle.cameras.STEREO_LEFT, frame.vehicle.cameras.STEREO_RIGHT)

stereo_img.show()
'''
