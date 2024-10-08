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

ad.get_projection_img(frame.tower.cameras.VIEW_2, frame.tower.lidars.VIEW_2).show()
