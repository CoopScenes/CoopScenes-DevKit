# Frame

### Frame

| Attribute   | Data Type | Description                                            |
|-------------|-----------|--------------------------------------------------------|
| `frame_id`  | `int`     | Unique identifier for the frame.                       |
| `timestamp` | `Decimal` | Timestamp associated with the frame.                   |
| `version`   | `str`     | Version of aeif-dataset package used for packing       |
| `vehicle`   | `Vehicle` | Vehicle sensor data.                                   |
| `tower`     | `Tower`   | Tower sensor data.                                     |

-------------------------------------------------------------------

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

-------------------------------------------------------------------

-------------------------------------------------------------------

# Sensors

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

-------------------------------------------------------------------

-------------------------------------------------------------------

# Data

## Metadata

### Vehicle Information

| Attribute    | Data Type        | Description                                                               |
|--------------|------------------|---------------------------------------------------------------------------|
| `model_name` | `str`            | Model name of the vehicle.                                                |
| `extrinsic`  | `Optional[Pose]` | The extrinsic pose of the TOP Lidar relative to the UPPER_PLATFORM Lidar. |

### TowerInformation

| Attribute    | Data Type        | Description                                                               |
|--------------|------------------|---------------------------------------------------------------------------|
| `model_name` | `str`            | Model name of the tower.                                                  |
| `extrinsic`  | `Optional[Pose]` | The extrinsic pose of the UPPER_PLATFORM Lidar relative to the TOP Lidar. |

### Camera Information

| Attribute            | Data Type         | Description                                                                                                                  |
|----------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------|
| `name`               | `str`             | Name of the camera.                                                                                                          |
| `model_name`         | `str`             | Model name of the camera.                                                                                                    |
| `shape`              | `Tuple[int, int]` | Image resolution (width, height).                                                                                            |
| `distortion_type`    | `str`             | Type of lens distortion.                                                                                                     |
| `camera_mtx`         | `np.array`        | Intrinsic camera matrix.                                                                                                     |
| `distortion_mtx`     | `np.array`        | Distortion coefficients.                                                                                                     |
| `rectification_mtx`  | `np.array`        | Rectification matrix.                                                                                                        |
| `projection_mtx`     | `np.array`        | Projection matrix.                                                                                                           |
| `region_of_interest` | `ROI`             | Region of interest within the image.                                                                                         |
| `focal_length`       | `int`             | Focal length of the camera in mm.                                                                                            |
| `extrinsic`          | `Optional[Pose]`  | The extrinsic pose of the Camera sensor relative to the TOP Lidar for the vehicle or the UPPER_PLATFORM Lidar for the tower. |

### LidarInformation

| Attribute                        | Data Type            | Description                                                                                                                 |
|----------------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `name`                           | `str`                | The name of the Lidar sensor.                                                                                               |
| `model_name`                     | `Optional[str]`      | The model name of the Lidar sensor.                                                                                         |
| `extrinsic`                      | `Optional[Pose]`     | The extrinsic pose of the Lidar sensor relative to the TOP Lidar for the vehicle or the UPPER_PLATFORM Lidar for the tower. |
| `vertical_fov`                   | `Optional[float]`    | The vertical field of view of the Lidar (for Blickfeld sensors).                                                            |
| `horizontal_fov`                 | `Optional[float]`    | The horizontal field of view of the Lidar (for Blickfeld sensors).                                                          |
| `beam_altitude_angles`           | `Optional[np.array]` | Beam altitude angles (for Ouster sensors).                                                                                  |
| `beam_azimuth_angles`            | `Optional[np.array]` | Beam azimuth angles (for Ouster sensors).                                                                                   |
| `lidar_origin_to_beam_origin_mm` | `Optional[np.array]` | Distance from the Lidar origin to the beam origin in mm (for Ouster sensors).                                               |
| `horizontal_scanlines`           | `Optional[int]`      | The number of horizontal scanlines (for Ouster sensors).                                                                    |
| `vertical_scanlines`             | `Optional[int]`      | The number of vertical scanlines (for Ouster sensors).                                                                      |
| `phase_lock_offset`              | `Optional[int]`      | The phase lock offset (for Ouster sensors).                                                                                 |
| `lidar_to_sensor_transform`      | `Optional[np.array]` | Transformation matrix from the Lidar frame to the sensor frame (for Ouster sensors).                                        |
| `horizontal_angle_spacing`       | `Optional[float]`    | The horizontal angle spacing of the Lidar (for Blickfeld sensors).                                                          |
| `frame_mode`                     | `Optional[str]`      | The frame mode of the Lidar (for Blickfeld sensors).                                                                        |
| `scan_pattern`                   | `Optional[str]`      | The scan pattern of the Lidar (for Blickfeld sensors).                                                                      |
| `dtype`                          | `np.dtype`           | Data type structure of the Lidar point cloud data.                                                                          |

### IMUInformation

| Attribute    | Data Type        | Description                                                                     |
|--------------|------------------|---------------------------------------------------------------------------------|
| `model_name` | `str`            | Model name of the IMU sensor.                                                   |
| `extrinsic`  | `Optional[Pose]` | The extrinsic pose of the IMU sensor relative to the TOP Lidar for the vehicle. |

### GNSSInformation

| Attribute    | Data Type        | Description                                                                      |
|--------------|------------------|----------------------------------------------------------------------------------|
| `model_name` | `str`            | Model name of the GNSS sensor.                                                   |
| `extrinsic`  | `Optional[Pose]` | The extrinsic pose of the GNSS sensor relative to the TOP Lidar for the vehicle. |

### DynamicsInformation

| Attribute         | Data Type | Description              |
|-------------------|-----------|--------------------------|
| `velocity_source` | `str`     | Source of velocity data. |
| `heading_source`  | `str`     | Source of heading data.  |

-------------------------------------------------------------------

## Sensor Data

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

### Motion

| Attribute                        | Data Type  | Description                            |
|----------------------------------|------------|----------------------------------------|
| `timestamp`                      | `Decimal`  | Timestamp of the motion data.          |
| `orientation`                    | `np.array` | Orientation vector.                    |
| `orientation_covariance`         | `np.array` | Covariance of the orientation.         |
| `angular_velocity`               | `np.array` | Angular velocity vector.               |
| `angular_velocity_covariance`    | `np.array` | Covariance of the angular velocity.    |
| `linear_acceleration`            | `np.array` | Linear acceleration vector.            |
| `linear_acceleration_covariance` | `np.array` | Covariance of the linear acceleration. |

### Position

| Attribute         | Data Type         | Description                        |
|-------------------|-------------------|------------------------------------|
| `timestamp`       | `Decimal`         | Timestamp of the position data.    |
| `status`          | `str`             | Status of the GNSS signal.         |
| `services`        | `Dict[str, bool]` | Status of satellite services.      |
| `latitude`        | `Decimal`         | Latitude in decimal degrees.       |
| `longitude`       | `Decimal`         | Longitude in decimal degrees.      |
| `altitude`        | `Decimal`         | Altitude in meters.                |
| `covariance`      | `np.array`        | Covariance matrix of the position. |
| `covariance_type` | `str`             | Type of covariance.                |

### Velocity

| Attribute          | Data Type  | Description                        |
|--------------------|------------|------------------------------------|
| `timestamp`        | `Decimal`  | Timestamp of the velocity data.    |
| `linear_velocity`  | `np.array` | Linear velocity vector.            |
| `angular_velocity` | `np.array` | Angular velocity vector.           |
| `covariance`       | `np.array` | Covariance matrix of the velocity. |

### Heading

| Attribute     | Data Type  | Description                       |
|---------------|------------|-----------------------------------|
| `timestamp`   | `Decimal`  | Timestamp of the heading data.    |
| `orientation` | `np.array` | Orientation vector.               |
| `covariance`  | `np.array` | Covariance matrix of the heading. |

-------------------------------------------------------------------

## Misc

### Pose

| Attribute | Data Type  | Description                                            |
|-----------|------------|--------------------------------------------------------|
| `xyz`     | `np.array` | Position in the reference coordinate system (x, y, z). |
| `rpy`     | `np.array` | Rotation in roll, pitch, and yaw (r, p, y).            |

### ROI (Region of Interest)

| Attribute  | Data Type | Description                          |
|------------|-----------|--------------------------------------|
| `x_offset` | `int`     | X-coordinate of the top-left corner. |
| `y_offset` | `int`     | Y-coordinate of the top-left corner. |
| `width`    | `int`     | Width of the ROI.                    |
| `height`   | `int`     | Height of the ROI.                   |

### TransformationMtx

| Attribute     | Data Type  | Description                   |
|---------------|------------|-------------------------------|
| `rotation`    | `np.array` | Rotation matrix (3x3).        |
| `translation` | `np.array` | Translation vector (x, y, z). |
