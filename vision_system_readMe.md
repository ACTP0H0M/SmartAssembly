# Vision System Documentation

The vision system comprises two main features:
- Visual referencing
- Metallic Pixel Detection (MPD)

The ROS package that contains the vision system components is located in folder ``src/vision``.

The working principle behind the developed vision system and the experimental setup used in this project is explained by Polikarpov et. al. [[1]].

The `vision `package is divided into three main components:
 - Vision Sensor: Cameras data acquisition and ROS communication.
 - Visual Servo: Algorithm for localizing the THD w.r.t. robot end-effector (EEF).
 - Metallic Pixel Detection: Algorithm for classification of THD pin position based on the metallic pixel count.
 - Clicking Goal: A convenient teach-in method for getting the goal position by clicking the goal in the camera images.

## 1. Configuration File
The config file `config_hexaglide.yaml` is located in folder ``src/vision/config`` and describes the following parameters of vision system:

- ``ros_param``: Describes the ROS parameters for the cameras
- ``extrinsic_params``: Contains the parameters that describes the experimental setup and cameras extrinsic properties.
- ``image_roi``: Region Of Interest (ROI) in pixel values for camera1 and camera2.

## 2. Cameras Communication
The class `VisionSensor` in the script `src/vision/src/vision/vision_sensor.py` establishes the communication with 
the Intel RealSense cameras ROS topics for sensor data acquisition and processing.

For synchronising the data streams from the two cameras, [ApproximateTime Policy](http://wiki.ros.org/message_filters#Policy-Based_Synchronizer_.5BROS_1.1.2B-.5D) is utilised.

## 3. Visual Servo
The class `VisualServo` in the script `src/vision/src/vision/visual_servo.py` contains the algorithm for visual referencing. There are two main components of the class:

### 1. Pin tip detection:
The function `pin_tips` for detecting the pin tips of THD is implemented in the file `src/vision/src/vision/pin_detector.py`.
This function is used by the class `VisualServo` for getting all the pin tips within the defined ROI.

### 2. Transformation matrix calculation:
For selecting the correct pin tip and determining the transformation matrix between the selected pin tip in cameras pixel coordinate system 
and the robot EEF cartesian coordinate system, following class methods are used:
- `autoSelectRightPinTip`: First the sorting algorithm selects the pin tip to be controlled. Once the pixel values of the selected pin tip is determined, 
the function `get_pin_tip_location` (defined in the script `src/vision/src/vision/pin_detector.py`) is used to calculate the transformation from cameras pixel
coordinates system to a virtual coordinates system (tilda coordinates). Tilda coordinate origin is located at the intersection of the line of sight of the two cameras.
This method works for Single In-line Package (SIP) components.
- `autoSelectDIPPinTip`: This method is used for calculating the pin tip coordinates of Dual In-line Package (DIP) components. 
The sorting algorithm used for selecting the pin tip for DIP components is described in this method. For calculating the Tilda coordinates from cameras pixel coordinates, the same function, `get_pin_tip_location` is utilised.

## 4. Metallic Pixel Detection
The class `MetallicPixelDetection` in the script `src/vision/src/vision/metallic_pixel_detection.py` contains the algorithm for determining the fine adjustment needed to center
the controlled pin of THD on top of the goal position.

The class method `mpd` determines the location of the controlled pin, based on the zoomed in image near the goal position. Assumption of the algorithm is that the goal position is the
center pixel of the image. The deviation of the pin from the goal position is determined by this method.

## 5. Click Goal Image
The class `GoalPixelSetter` in the script `src/vision/src/vision/click_goal.py` enables the user to click the goal point in the images from the two cameras. With this method, the
precise location of the goal position is not required.

### Fine-Tuning Goal Position
To fine-tune the clicked goal position, before starting the program, set ``wait_for_input`` in the class ``Hexaglide`` 
(file folder: `src/hexaglide_tht_control/src/hexaglide_tht_control/robot.py`) method ``setEEFPose()`` to True. Then after the approach correction, 
carefully check if there is an adjustment needed in the goal position. 

Rule of thumb, if the pin is around 0.5mm on left of the hole from camera1 perspective, then add 2 or 3 pixels, and if 
it is on right of the hole, then subtract 2 or 3 pixels. (Same step for the second camera).

Save this goal pixel values for future insertion in the same goal position.

## References

1. M. Polikarpov, Y. Mehmood, D. Boiar, L. Schulte and J. Deuse, "A Cost-Efficient Robotic Through-Hole Assembly System for Dual-Pin Component Insertion," 2023 11th International Conference on Control, Mechatronics and Automation (ICCMA), Grimstad, Norway, 2023, pp. 271-277, doi: 10.1109/ICCMA59762.2023.10375039.


[1]: (https://ieeexplore.ieee.org/document/10375039)