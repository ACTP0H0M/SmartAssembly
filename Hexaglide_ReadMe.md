# Hexaglide Documentation for SmartAssembly

Hexaglide is a closed-kinematic robot with 6 Degrees of Freedom made by [SFA GmbH & Co KG](https://sfa-aichstetten.de/en/). 
The Hexaglide manuplator is a modified adaptation of the robot by Honegger et. al. [[1]]. 

The ROS package ``src/hexaglide_tht_control`` 

- The inverse kinematics (IK) solver for the Hexaglide is an adaptation of inverse kinematic analysis of Hexaglide by Palpacelli et. al. [[2]].

- The forward kinematics (FK) solver relies on the motor encoder readings and is based on simple open-loop control.  

## User Guide
In this section, the components of the ``hexaglide_tht_control`` package are explained.

### 1. Hexaglide Components

- Intel RealSense D405 cameras ([Product docs](https://www.intelrealsense.com/depth-camera-d405/))
- Phidget stepper motors ([Product docs](https://www.phidgets.com/?prodid=351))
- Phidget stepper motor controller ([Product docs](https://www.phidgets.com/?prodid=1121))
- Phidget stepper motor encoder interface ([Product docs](https://www.phidgets.com/?prodid=1332))
- Limit switches ([Product docs](https://www.phidgets.com/?prodid=641))
- Load cells ([Product docs](https://www.phidgets.com/?prodid=1192))
- Phidget load cells bridge ([Product docs](https://www.phidgets.com/?prodid=1270))

### 2. Inverse Kinematics Solver

The implementation of IK Solver for the Hexaglide is in the file: ``src/hexaglide_tht_control/src/hexaglide_tht_control/kinematic_solver_v2.py``

The parameters used in the calculation of Inverse Kinematics according to [[2]] are measured from the CAD model of the Hexaglide (located in the folder
``hexaglide_model/``).

### 3. Configuration File

The config file `config_hexaglide.yaml` is located in folder ``src/hexaglide_tht_control/config/`` and describes the following parameters for initializing the Hexaglide:

- Phidget parameters (`phidget_param`): 
  - Describes the serial number and the port ID for the Phidget devices.
  - Rescale factor converts the default unit of movement (1/16 step) of motor controller to desired unit. ([More info](https://www.phidgets.com/?prodid=1029#Tab_User_Guide))
- Hexaglide parameters (`hexaglide_param`): 
  - Movement parameters/limits
  - Rotation between Hexaglide end-effector (EEF) and base coordinate frame (``init_rotation_eef_base``)
- Referencing parameters (`referencing_param`): Parameters for referencing the Hexaglide above the lower limit switches
- Workspace parameters (``workspace``): Describes the workspace limits of the Hexaglide

### 4. Initialize Hexaglide

The class ``Hexaglide`` in the file `src/hexaglide_tht_control/src/hexaglide_tht_control/robot.py` initializes the kinematic solver and 
Phidget devices using the config file. 

### 5. Referencing Hexaglide

The current version of Hexaglide does not have an on-board memory and the joints position are lost upon restart. Thus, the 
joints needs to be referenced everytime upon start-up. 

The class ``Hexaglide`` method `doReference()` is used for the referencing.

General procedure for referencing follows:

- Move all the joints DOWN until the lower limit switch is ON.
- Move joints UP Until the lower limit switch is OFF.
- Move each joint sequentially down at lower speed to ensure not more than one limit switch is ON, simultaneously.
- Since the lower limit switches have mounting error (means not all switches are at the same level), add the ``referencing_param/joints_zero_error`` from the config
file to ensure all the joints are at the same level.

### 6. Hexaglide Movement

- The movement commands for the Hexaglide are interpreted as relative commands in the EEF frame. The class ``Hexaglide`` method `setEEFPose()` is used sending relative movement command 
  to the position controllers. The units of the relative movement commands for position is meters and for orientation is degrees 
  and the format of the command is a list: ``[delta_X, delta_Y, delta_Z, rot_X, rot_Y, rot_Z]``.

  Where ``delta_X``, ``delta_Y`` and ``delta_Z`` are the displacement and ``rot_X``, ``rot_Y`` and ``rot_Z`` are the relative rotation in X, Y and Z axis of the EEF frame, respectively.

- The class ``Hexaglide`` method `getEEFState()` returns the current EEF position and orientation in meters and quaternions, respectively.
- The joints velocity are scaled according to the displacement of the joint variable. The class ``Hexaglide`` method `getVariableJointVelocity()` is used for setting the variable joint velocity.
- To ensure smooth movement, Python [Multithreading](https://docs.python.org/3/library/threading.html) is used for simultaneously commanding all the individual motor position controllers.
- To run the program in automatic mode (without user input), set ``wait_for_input`` in the class ``Hexaglide`` method ``setEEFPose()`` to False.

### 7. ROS Communication

The Python script ``src/tht_control/scripts/hexaglide_sub_pub`` establishes the communication between ROS and the Hexaglide. 

The ROS topics used:
- ``eef_state_pub``: The Hexaglide EEF state is published by the script `hexaglide_sub_pub` on this topic. The message format is an array with first 3 entries
  as the EEF position (in meters) and 4 entries of EEF orientation (in quaternions).
- ``visual_servo_cmd``: The direct movement commands in EEF coordinates are published on this topic. The script `hexaglide_sub_pub` interprets the commands and send it to the position controller.
- ``action``: The discrete movements commands in EEF coordinates are published on this topic. The script `hexaglide_sub_pub` interprets the message according
  to the parameter ``env_param/action_dict`` described in the config file `src/tht_control/config/config_hexaglide.yaml`. The stepsize for the 
  movement is also described in the same config file by the parameters ``env_param/translation_step_mm`` and ``env_param/rotation_step_deg`` for translation and rotation, respectively.


## References
1. M. Honegger, A. Codourey and E. Burdet, "Adaptive control of the Hexaglide, a 6 dof parallel manipulator," Proceedings of International Conference on Robotics and Automation, Albuquerque, NM, USA, 1997, pp. 543-548 vol.1, doi: https://doi.org/10.1109/ROBOT.1997.620093.
2. Palpacelli, M.-C.; Carbonari, L.; Palmieri, G.; D’Anca, F.; Landini, E.; Giorgi, G. Functional Design of a 6-DOF Platform for Micro-Positioning. Robotics 2020, 9, 99. https://doi.org/10.3390/robotics9040099

[1]: https://ieeexplore.ieee.org/abstract/document/620093
[2]: https://www.mdpi.com/2218-6581/9/4/99