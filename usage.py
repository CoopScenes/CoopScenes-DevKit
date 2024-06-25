from ameisedataset import DataRecord

dataloader = DataRecord("/home/slam/datasets/packed/id00000_2024-05-27_16-45-04.4mse")
myFrame = dataloader[0]
myFrame.vehicle.cameras.FRONT_LEFT.image.show()
pass
