from decimal import Decimal
from typing import Optional
import numpy as np
from aeifdataset import Dataloader, DataRecord
from aeifdataset.utils import visualisation as vis
from aeifdataset.utils import image_functions as imf
import os

# id04390_2024-07-18_18-11-45.4mse
# id03960_2024-07-18_18-11-02.4mse
# id04770_2024-07-18_18-12-23.4mse

# id09940_2024-07-18_18-21-00.4mse

# id09700_2024-07-18_18-20-36.4mse


example_record_1 = DataRecord("/mnt/dataset/record_1/packed/id00020_2024-07-18_18-04-28.4mse")
frame = example_record_1[1]
test = frame.vehicle.lidars.TOP.info
print(test)

print('stop')
pass
'''
back_left = frame.vehicle.cameras.BACK_LEFT
front_left = frame.vehicle.cameras.FRONT_LEFT
stereo_left = frame.vehicle.cameras.STEREO_LEFT
stereo_right = frame.vehicle.cameras.STEREO_RIGHT
front_right = frame.vehicle.cameras.FRONT_RIGHT
back_right = frame.vehicle.cameras.BACK_RIGHT

c_view_1 = frame.tower.cameras.VIEW_1
c_view_2 = frame.tower.cameras.VIEW_2

top = frame.vehicle.lidars.TOP
left = frame.vehicle.lidars.LEFT
right = frame.vehicle.lidars.RIGHT

l_view_1 = frame.tower.lidars.VIEW_1
l_view_2 = frame.tower.lidars.VIEW_2
upper_platform = frame.tower.lidars.UPPER_PLATFORM

imf.save_image(back_left._image_raw, '/mnt/dataset/record_1/png', '_back_left',back_left.info)

pass
# vis.show_projection(c_view_2, upper_platform, l_view_2, static_color='blue', static_color2='red')
# viz.show_projection(back_left, left)
# viz.show_projection(front_left, left)
# viz.show_projection(stereo_left, top)
# viz.show_projection(front_right, right)
# viz.show_projection(back_right, right)

# viz.show_disparity_map(stereo_left, stereo_right)
'''
