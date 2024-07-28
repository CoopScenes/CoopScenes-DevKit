from ameisedataset import DataRecord

dataloader = DataRecord("/media/slam/data/bus_packed/id00140_2024-07-18_13-54-47.4mse")
myFrame = dataloader[18]
myFrame.vehicle.cameras.FRONT_LEFT.image.show()
myFrame.vehicle.cameras.BACK_RIGHT.image.show()
pass
