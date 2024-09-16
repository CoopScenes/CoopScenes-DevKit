# Frame

## Frame

| Attribute   | Data Type | Description                          |
|-------------|-----------|--------------------------------------|
| `frame_id`  | `int`     | Unique identifier for the frame.     |
| `timestamp` | `Decimal` | Timestamp associated with the frame. |
| `version`   | `float`   | Version of the frame format.         |
| `vehicle`   | `Vehicle` | Vehicle sensor data.                 |
| `tower`     | `Tower`   | Tower sensor data.                   |

-------------------------------------------------------------------

# Agents

### Vehicle

| Attribute  | Data Type            | Description                         |
|------------|----------------------|-------------------------------------|
| `info`     | `VehicleInformation` | Metadata about the vehicle.         |
| `cameras`  | `VisionSensorsVeh`   | Grouping of vehicle cameras.        |
| `lidars`   | `LaserSensorsVeh`    | Grouping of vehicle lidars.         |
| `IMU`      | `IMU`                | IMU sensor data for the vehicle.    |
| `GNSS`     | `GNSS`               | GNSS sensor data for the vehicle.   |
| `DYNAMICS` | `Dynamics`           | Dynamic state data for the vehicle. |

### VisionSensorsVeh

| Camera Position | Data Type | Description                         |
|-----------------|-----------|-------------------------------------|
| `BACK_LEFT`     | `Camera`  | Back-left camera of the vehicle.    |
| `FRONT_LEFT`    | `Camera`  | Front-left camera of the vehicle.   |
| `STEREO_LEFT`   | `Camera`  | Left stereo camera of the vehicle.  |
| `STEREO_RIGHT`  | `Camera`  | Right stereo camera of the vehicle. |
| `FRONT_RIGHT`   | `Camera`  | Front-right camera of the vehicle.  |
| `BACK_RIGHT`    | `Camera`  | Back-right camera of the vehicle.   |
| `REAR`          | `Camera`  | Rear camera of the vehicle.         |

### LaserSensorsVeh

| Lidar Position | Data Type | Description                                    |
|----------------|-----------|------------------------------------------------|
| `LEFT`         | `Lidar`   | Lidar sensor on the left side of the vehicle.  |
| `TOP`          | `Lidar`   | Lidar sensor on the top of the vehicle.        |
| `RIGHT`        | `Lidar`   | Lidar sensor on the right side of the vehicle. |
| `REAR`         | `Lidar`   | Lidar sensor at the rear of the vehicle.       |

### Tower

| Attribute | Data Type          | Description                     |
|-----------|--------------------|---------------------------------|
| `info`    | `TowerInformation` | Metadata about the tower.       |
| `cameras` | `VisionSensorsTow` | Grouping of tower cameras.      |
| `lidars`  | `LaserSensorsTow`  | Grouping of tower lidars.       |
| `GNSS`    | `GNSS`             | GNSS sensor data for the tower. |

### VisionSensorsTow

| Camera View | Data Type | Description                      |
|-------------|-----------|----------------------------------|
| `VIEW_1`    | `Camera`  | First camera view of the tower.  |
| `VIEW_2`    | `Camera`  | Second camera view of the tower. |

### LaserSensorsTow

| Lidar View       | Data Type | Description                                              |
|------------------|-----------|----------------------------------------------------------|
| `VIEW_1`         | `Lidar`   | First lidar view of the tower.                           |
| `VIEW_2`         | `Lidar`   | Second lidar view of the tower.                          |
| `UPPER_PLATFORM` | `Lidar`   | Lidar sensor mounted on the upper platform of the tower. |

### Vehicle Information

| Attribute    | Data Type | Description                      |
|--------------|-----------|----------------------------------|
| `model_name` | `str`     | Model name of the vehicle.       |
| `extrinsic`  | `Pose`    | Pose information of the vehicle. |

----------------------------------------------------------------

# Sensor

### Camera

| Attribute    | Data Type           | Description                                  |
|--------------|---------------------|----------------------------------------------|
| `info`       | `CameraInformation` | Metadata about the camera.                   |
| `_image_raw` | `Image`             | Raw image data captured by the camera.       |
| `image`      | `Image`             | Rectified image data captured by the camera. |

### Lidar

| Attribute | Data Type          | Description                             |
|-----------|--------------------|-----------------------------------------|
| `info`    | `LidarInformation` | Metadata about the Lidar sensor.        |
| `points`  | `Points`           | Point cloud data captured by the Lidar. |

### IMU

| Attribute | Data Type        | Description                    |
|-----------|------------------|--------------------------------|
| `info`    | `IMUInformation` | Metadata about the IMU sensor. |
| `motion`  | `List[Motion]`   | Motion data from the IMU.      |

### GNSS

| Attribute  | Data Type         | Description                         |
|------------|-------------------|-------------------------------------|
| `info`     | `GNSSInformation` | Metadata about the GNSS sensor.     |
| `position` | `List[Position]`  | Position data from the GNSS sensor. |

### Dynamics

| Attribute  | Data Type             | Description                         |
|------------|-----------------------|-------------------------------------|
| `info`     | `DynamicsInformation` | Metadata about vehicle dynamics.    |
| `velocity` | `List[Velocity]`      | Velocity data of the vehicle.       |
| `heading`  | `List[Heading]`       | Heading information of the vehicle. |

-----------------------------------------------------------------------------

# Data

### Image

| Attribute   | Data Type   | Description                            |
|-------------|-------------|----------------------------------------|
| `timestamp` | `Decimal`   | Timestamp when the image was captured. |
| `image`     | `PIL.Image` | The image data.                        |

### Points

| Attribute   | Data Type  | Description                                |
|-------------|------------|--------------------------------------------|
| `points`    | `np.array` | Array of 3D points (x, y, z).              |
| `timestamp` | `Decimal`  | Timestamp associated with the point cloud. |

### Pose

| Attribute | Data Type  | Description                             |
|-----------|------------|-----------------------------------------|
| `xyz`     | `np.array` | Position in 3D space (x, y, z).         |
| `rpy`     | `np.array` | Rotation in roll, pitch, yaw (r, p, y). |

### Camera Information

| Attribute            | Data Type         | Description                          |
|----------------------|-------------------|--------------------------------------|
| `name`               | `str`             | Name of the camera.                  |
| `model_name`         | `str`             | Model name of the camera.            |
| `shape`              | `Tuple[int, int]` | Image resolution (width, height).    |
| `distortion_type`    | `str`             | Type of lens distortion.             |
| `camera_mtx`         | `np.array`        | Intrinsic camera matrix.             |
| `distortion_mtx`     | `np.array`        | Distortion coefficients.             |
| `rectification_mtx`  | `np.array`        | Rectification matrix.                |
| `projection_mtx`     | `np.array`        | Projection matrix.                   |
| `region_of_interest` | `ROI`             | Region of interest within the image. |
| `focal_length`       | `int`             | Focal length of the camera in mm.    |
| `extrinsic`          | `Pose`            | Pose of the camera.                  |

### Lidar Information

| Attribute              | Data Type  | Description                     |
|------------------------|------------|---------------------------------|
| `name`                 | `str`      | Name of the Lidar sensor.       |
| `model_name`           | `str`      | Model name of the Lidar sensor. |
| `beam_altitude_angles` | `np.array` | Beam altitude angles.           |
| `beam_azimuth_angles`  | `np.array` | Beam azimuth angles.            |
| `extrinsic`            | `Pose`     | Pose of the Lidar sensor.       |





