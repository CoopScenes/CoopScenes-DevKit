from aeifdataset import DataRecord

dataloader = DataRecord("/media/slam/Extreme SSD/datasets/record_1/packed/id04770_2024-07-18_18-12-23.4mse")
myFrame = dataloader[0]
myFrame.vehicle.cameras.STEREO_LEFT.image.show()
pass
