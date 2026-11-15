# YuMi Documentation for SmartAssembly
YuMi is dual-arm robot with 7 Degrees of Freedom robot made by ABB.

## Initialize YuMi connection with ROS
-    Turn on the robot and  wait for it to boot up.
-    In Teach Pendant, select bottom-right menu (Gears logo), and select the first option (with robot logo).
-    On the new window, select Operator mode to be ‘Manual’ and turn the motors on.
-    Select the top-left drop down menu and select ‘Program Editor’
-    In the Program Editor window, select the first program (T_ROB_L), click on Debug and select ‘PP to main’. 
-    Select ‘Tasks and Programs’ drop down. Now, select the second program (T_ROB_R) and click ‘Open’. 
     From ‘Debug’ menu, select ‘PP to Main’.
-    Repeat step 6 for the remaining programs (T_GRIP_Motion, T_GRIP_Motion_L, T_GRIP_Motion_R, T_GRIP_State).
-    From the bottom-right menu, again select the first option and now change the Operator mode to ‘Auto’.
-    Press the Play ► physical button on the Teach Pendant.
-    The robot grippers should first completely open, then close and then back to home position.
-    Once the gripper routine is done, robot is ready for connecting with PC.
-    Open the terminal and type following commands in order 
        ```
        cd ~/smartassembly_ws
        source devel/setup.bash
        roslaunch yumi_support robot_interface.launch
        ```
-    Once the connection is successfully initialized, invoke Moveit! Controllers in a new terminal:
        ``` 
        cd ~/smartassembly_ws
        source devel/setup.bash
        roslaunch yumi_moveit_config demo_online.launch
        ```
-   Make sure that in smartassembly_ws/src/yumi/yumi_moveit_config/config/controllers.yaml file, last two controller 
uncommented and the rest commented.
