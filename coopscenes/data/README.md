# Frame

### Frame

| Attribute   | Data Type | Description                                            |
|-------------|-----------|--------------------------------------------------------|
| `frame_id`  | `int`     | Unique identifier for the frame.                       |
| `timestamp` | `Decimal` | Timestamp associated with the frame.                   |
| `version`   | `str`     | Version of aeif-dataset package used for packing       |
| `vehicle`   | `Vehicle` | Vehicle sensor data.                                   |
| `tower`     | `Tower`   | Tower sensor data.                                     |

---

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

---

# Sensors

### Camera

| Attribute         | Data Type           | Description                                                                 |
|-------------------|---------------------|-----------------------------------------------------------------------------|
| `info`            | `CameraInformation` | Metadata about the camera.                                                  |
| `_image_raw`      | `Image`             | Raw image data captured by the camera.                                      |
| `image`           | `Image`             | Rectified image data captured by the camera.                                |

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

---

# Data

## Metadata

### VehicleInformation

| Attribute    | Data Type        | Description                                      |
|--------------|------------------|--------------------------------------------------|
| `model_name` | `str`            | Model name of the vehicle.                       |
| `extrinsic`  | `Optional[np.ndarray]` | Extrinsic pose relative to TOP Lidar.    |
| `height`     | `Optional[np.ndarray]` | Height of TOP Lidar above ground.        |

### TowerInformation

| Attribute    | Data Type        | Description                                             |
|--------------|------------------|---------------------------------------------------------|
| `model_name` | `str`            | Model name of the tower.                                |
| `height`     | `Optional[np.ndarray]` | Height of UPPER_PLATFORM Lidar above ground.     |

### CameraInformation

| Attribute         | Data Type         | Description                                             |
|-------------------|-------------------|---------------------------------------------------------|
| `name`            | `str`             | Name of the camera.                                     |
| `model_name`      | `str`             | Model name of the camera.                               |
| `shape`           | `Tuple[int, int]` | Image resolution (width, height).                       |
| `distortion_type` | `str`             | Type of lens distortion.                                |
| `camera_mtx`      | `np.ndarray`      | Intrinsic camera matrix.                                |
| `distortion_mtx`  | `np.ndarray`      | Distortion coefficients.                                |
| `rectification_mtx`| `np.ndarray`     | Rectification matrix.                                   |
| `projection_mtx`  | `np.ndarray`      | Projection matrix.                                      |
| `region_of_interest` | `ROI`          | Region of interest within the image.                    |
| `camera_type`     | `str`             | Camera type (e.g. monocular, stereo).                   |
| `focal_length`    | `int`             | Focal length in mm.                                     |
| `aperture`        | `int`             | Aperture size in mm.                                    |
| `exposure_time`   | `int`             | Exposure time in microseconds.                          |
| `extrinsic`       | `np.ndarray`      | Extrinsic pose relative to TOP/UPPER_PLATFORM Lidar.    |
| `stereo_transform`| `np.ndarray`      | Extrinsic pose of STEREO_LEFT relative to STEREO_RIGHT. |

### LidarInformation

| Attribute                        | Data Type            | Description                                                    |
|----------------------------------|----------------------|----------------------------------------------------------------|
| `name`                           | `str`                | Name of the Lidar sensor.                                      |
| `model_name`                     | `str`                | Model name of the Lidar sensor.                                |
| `extrinsic`                      | `np.ndarray`         | Extrinsic pose relative to TOP/UPPER_PLATFORM Lidar.           |
| `vertical_fov`                   | `float`              | Vertical FoV (Blickfeld).                                      |
| `horizontal_fov`                 | `float`              | Horizontal FoV (Blickfeld).                                    |
| `horizontal_angle_spacing`       | `float`              | Horizontal angle spacing (Blickfeld).                          |
| `frame_mode`                     | `str`                | Frame mode (Blickfeld).                                        |
| `scan_pattern`                   | `str`                | Scan pattern (Blickfeld).                                      |
| `beam_altitude_angles`           | `np.ndarray`         | Altitude angles (Ouster).                                      |
| `beam_azimuth_angles`            | `np.ndarray`         | Azimuth angles (Ouster).                                       |
| `lidar_origin_to_beam_origin_mm` | `np.ndarray`         | Distance from Lidar to beam origin (Ouster).                   |
| `horizontal_scanlines`           | `int`                | Horizontal scanlines (Ouster).                                 |
| `vertical_scanlines`             | `int`                | Vertical scanlines (Ouster).                                   |
| `phase_lock_offset`              | `int`                | Phase lock offset (Ouster).                                    |
| `lidar_to_sensor_transform`      | `np.ndarray`         | Transform from Lidar to Sensor (Ouster).                       |
| `motion_transform`               | `np.ndarray`         | Motion compensation transform (Ouster).                        |
| `dtype`                          | `np.dtype`           | Data type structure of the point cloud.                        |

### DynamicsInformation

| Attribute         | Data Type | Description              |
|-------------------|-----------|--------------------------|
| `velocity_source` | `str`     | Source of velocity data. |
| `heading_source`  | `str`     | Source of heading data.  |

---

## Sensor Data

### Image

| Attribute   | Data Type     | Description                                  |
|-------------|---------------|----------------------------------------------|
| `timestamp` | `Decimal`     | Timestamp when the image was captured.       |
| `image`     | `PIL.Image`   | The image data.                              |
| `labels`    | `ImageLabels` | Optional labels, e.g., bounding boxes.       |

### Points

| Attribute   | Data Type  | Description                                |
|-------------|------------|--------------------------------------------|
| `points`    | `np.array` | Array of 3D points or structured array.    |
| `timestamp` | `Decimal`  | Timestamp associated with the point cloud. |

### ImageLabels

| Attribute | Data Type   | Description                                 |
|-----------|-------------|---------------------------------------------|
| `bbox_2d` | `np.array`  | Structured array of 2D bounding boxes.      |

### Motion / Velocity / Heading / Position

(Datenstruktur wie bisher – keine Änderungen notwendig)

---

## Misc

### ROI (Region of Interest)

| Attribute  | Data Type | Description                          |
|------------|-----------|--------------------------------------|
| `x_offset` | `int`     | X-coordinate of the top-left corner. |
| `y_offset` | `int`     | Y-coordinate of the top-left corner. |
| `width`    | `int`     | Width of the ROI.                    |
| `height`   | `int`     | Height of the ROI.                   |
