#!/usr/bin/env python3

# Python Modules
import argparse
import operator
import numpy as np
import math as m
import time
import cv2
from matplotlib import pyplot as plt
# ROS Modules
import rospy
from std_msgs.msg import Int64, Float64MultiArray, Bool
from sensor_msgs.msg import Image
# Local Modules
from tht_control.observer import Observer
from tht_control.utils import *
from tht_control.common import *
from vision.metallic_pixel_detection import MetallicPixelDetection
from vision.visual_servo import VisualServo
from vision.click_goal import GoalPixelSetter
from vision.vision_sensor import convertNumpyImgToGrayscale, calculateGrayscaleImageDiff, convert16SC1Torgb8


class THTVisualServo():
    def __init__(self, robot_type=None):
        print("Started initializing THTVisualServo...")
        config_file_name, self.robot_type = load_config_file(robot_type)
        ros_param_path = load_ros_params(rospackage_name='tht_control',
                                         rel_path_from_package_to_file='config', yaml_file_name=config_file_name)
        self.number_actions = rospy.get_param('/env_param/n_actions')
        self.action_pub = rospy.Publisher(rospy.get_param('/ros_param/action_pub'), Int64, queue_size=10)
        self.eef_state_pub = rospy.get_param('/ros_param/eef_state_pub')
        self.cropped_img1_pub = rospy.Publisher(rospy.get_param('/ros_param/cropped_img1_pub'), Image, queue_size=10)
        self.cropped_img2_pub = rospy.Publisher(rospy.get_param('/ros_param/cropped_img2_pub'), Image, queue_size=10)
        self.visual_servo_cmd_pub = rospy.Publisher('visual_servo_cmd', Float64MultiArray, queue_size=10)
        self.img1_pub = rospy.get_param('/ros_param/img1_pub')
        self.img2_pub = rospy.get_param('/ros_param/img2_pub')
        self.translation_stepsize = rospy.get_param('/env_param/translation_step_mm')
        self.rotation_stepsize = rospy.get_param('/env_param/rotation_step_deg')
        self.output_dir = rospy.get_param('/log_param/output_dir')
        self.workspace_x = rospy.get_param('/robot_workspace/x_limit')
        self.workspace_y = rospy.get_param('/robot_workspace/y_limit')
        self.workspace_z = rospy.get_param('/robot_workspace/z_limit')
        self.workspace_thetax = rospy.get_param('/robot_workspace/thetax_limit')
        self.workspace_thetay = rospy.get_param('/robot_workspace/thetay_limit')
        self.workspace_thetaz = rospy.get_param('/robot_workspace/thetaz_limit')
        self.tildaorigin_to_pcb_z = rospy.get_param('/robot_workspace/tildaorigin_to_pcb_z')
        self.H1origin_to_tildaorigin = rospy.get_param('/robot_workspace/H1origin_to_tildaorigin')
        self.robot_init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.rotation_stepsize = (rospy.get_param('/env_param/rotation_step_deg')) * m.pi / 180
        self.orientation_as_euler = rospy.get_param('/env_param/orientation_as_euler')
        self.image_as_input_without_cv = rospy.get_param('/env_param/image_as_input_without_cv')
        self.diff_image_as_input = rospy.get_param('/env_param/diff_image_as_input')
        self.pressure_hist_as_obs = rospy.get_param('/env_param/pressure_hist_as_obs')
        self.max_step_count = rospy.get_param('/env_param/max_step_count')
        self.max_pressure = rospy.get_param('/env_param/max_pressure')
        self.max_depth = rospy.get_param('/env_param/max_depth')
        self.obs_logfile = 'testlogs'  # TODO:
        self.obs_per_step = []
        self.action_dict = rospy.get_param('/env_param/action_dict')
        h = rospy.get_param('/cropped_image_pixels/h')
        w = rospy.get_param('/cropped_image_pixels/w')
        self.obs = Observer(rospy.get_param('/ros_param/eef_state_pub'), h,
                            w)  # TODO not used: x1, y1, x2, y2. Remove these
        self.visual_servo = VisualServo(self.robot_type)
        self.init_step = True

        # Capture complete images from both cameras for goal hole clicking
        _current_image1, _current_image2 = self.obs.current_visualdata()
        _current_cropped_image1 = None
        _current_cropped_image2 = None

        # Setting goal hole by clicking it in both camera images or use default values
        self.goal_pose_pixel_image1 = []
        self.goal_pose_pixel_image2 = []
        self.click_goal_image = rospy.get_param('/env_param/click_goal_image')
        if self.click_goal_image:
            print("THTVisualServo: Please set the goal hole by clicking on it in both images!")
            self.goal_pixel_setter = GoalPixelSetter()
            self.goal_pixel_setter.click_set_goal_in_pixels(_current_image1)
            if self.goal_pixel_setter.x is None or self.goal_pixel_setter.y is None:
                raise RuntimeError("Goal pixel in image1 not clicked")
            self.goal_pose_pixel_image1.append((int(self.goal_pixel_setter.x), int(self.goal_pixel_setter.y)))
            print("Clicked goal pixel in image 1: ", self.goal_pose_pixel_image1)
            self.goal_pixel_setter = GoalPixelSetter()
            self.goal_pixel_setter.click_set_goal_in_pixels(_current_image2)
            if self.goal_pixel_setter.x is None or self.goal_pixel_setter.y is None:
                raise RuntimeError("Goal pixel in image2 not clicked")
            self.goal_pose_pixel_image2.append((int(self.goal_pixel_setter.x), int(self.goal_pixel_setter.y)))
            print("Clicked goal pixel in image 2: ", self.goal_pose_pixel_image2)
        else:
            self.goal_pose_pixel_image1.append(tuple(rospy.get_param('/env_param/goal_pixel_img1')))
            self.goal_pose_pixel_image2.append(tuple(rospy.get_param('/env_param/goal_pixel_img2')))
        print('Goal image1 (pixels): ', self.goal_pose_pixel_image1)
        print('Goal image2 (pixels): ', self.goal_pose_pixel_image2)
        self.goal_pixel_correction_img1 = rospy.get_param('/env_param/goal_pixel_correction_img1')
        self.goal_pixel_correction_img2 = rospy.get_param('/env_param/goal_pixel_correction_img2')
        self.corrected_goal_pose_pixel_image1 = [
            (self.goal_pose_pixel_image1[0][0] + self.goal_pixel_correction_img1[0],
             self.goal_pose_pixel_image1[0][1] + self.goal_pixel_correction_img1[1])]
        self.corrected_goal_pose_pixel_image2 = [
            (self.goal_pose_pixel_image2[0][0] + self.goal_pixel_correction_img2[0],
             self.goal_pose_pixel_image2[0][1] + self.goal_pixel_correction_img2[1])]
        print('Corrected goal image1 (pixels): ', self.corrected_goal_pose_pixel_image1)
        print('Corrected goal image2 (pixels): ', self.corrected_goal_pose_pixel_image2)
        self.corrected_goal_pose_tilda = None

        # Get initial cropped images (without THD or gripper) to work with image differences
        self.init_cropped_image1, self.init_cropped_image2 = self.obs.current_cropped_visualdata(
            self.goal_pose_pixel_image1,
            self.goal_pose_pixel_image2)
        self.img_center_x = self.init_cropped_image1.shape[1] / 2
        self.img_center_y = self.init_cropped_image1.shape[0] / 2
        self.center_tolerance_px = 1
        self.mpd = MetallicPixelDetection(self.init_cropped_image1, self.init_cropped_image2, self.center_tolerance_px,
                                          self.img_center_x)
        if self.diff_image_as_input:
            # Convert both images to grayscale
            self.init_cropped_image1 = convertNumpyImgToGrayscale(self.init_cropped_image1)
            self.init_cropped_image2 = convertNumpyImgToGrayscale(self.init_cropped_image2)
        # Calculate tilda (w.r.t. cameras' optical axis intersection) coordinates of the goal hole
        if self.robot_type == 'yumi':
            self.goal_pose_tilda = self.visual_servo.calc_1_tip_coords(self.goal_pose_pixel_image1,
                                                                       self.goal_pose_pixel_image2,
                                                                       self.visual_servo.x_tilda_c1,
                                                                       self.visual_servo.y_tilda_c2)
            print('Goal pose tilda: ', self.goal_pose_tilda)

        elif self.robot_type == 'hexaglide':

            z_tilda_goal = self.tildaorigin_to_pcb_z
            ######## CALCULATED TILDA GOAL #########
            self.goal_pose_tilda = self.visual_servo.calc_1_tip_coords(self.goal_pose_pixel_image1,
                                                                       self.goal_pose_pixel_image2,
                                                                       self.visual_servo.x_tilda_c1,
                                                                       self.visual_servo.y_tilda_c2)
            self.goal_pose_tilda = list(self.goal_pose_tilda)
            self.goal_pose_tilda[2] = z_tilda_goal * 1000
            self.goal_pose_tilda = tuple(self.goal_pose_tilda)
            print('calculated goal_pose_tilda: ', self.goal_pose_tilda)

            ###### ESTIMATED TILDA GOAL WITH FITTING FUNCTION ######
            # x_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image2[0][0],
            #                                       self.corrected_goal_pose_pixel_image2[0][1],
            #                                       self.visual_servo.x_tilda_params[0],
            #                                       self.visual_servo.x_tilda_params[1],
            #                                       self.visual_servo.x_tilda_params[2],
            #                                       self.visual_servo.x_tilda_params[3],
            #                                       self.visual_servo.x_tilda_params[4],
            #                                       self.visual_servo.x_tilda_params[5])
            #
            # y_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image1[0][0],
            #                                       self.corrected_goal_pose_pixel_image1[0][1],
            #                                       self.visual_servo.y_tilda_params[0],
            #                                       self.visual_servo.y_tilda_params[1],
            #                                       self.visual_servo.y_tilda_params[2],
            #                                       self.visual_servo.y_tilda_params[3],
            #                                       self.visual_servo.y_tilda_params[4],
            #                                       self.visual_servo.y_tilda_params[5])
            # self.goal_pose_tilda = (x_tilda_goal, y_tilda_goal, z_tilda_goal*1000)
            # print('Goal pose tilda by Fitting function: ', self.goal_pose_tilda)

        self.hole_position_H1 = self.H1origin_to_tildaorigin + np.dot(self.visual_servo.rot_tilda_to_base,
                                                                      (np.array(self.goal_pose_tilda)) / 1000)
        self.pixel_goal_error_hist_1 = []
        self.pixel_goal_error_hist_2 = []
        self.tilda_goal_error_hist = []
        self.old_z_relative_coord = [0.0]
        self.old_robot_pose_roll = []
        self.old_robot_pose_pitch = []
        self.old_robot_pose_yaw = []
        self.goal_error_coords = rospy.get_param('/env_param/goal_error_coords')
        # self._set_init_pose(self.robot_init_pose)  # TODO: Add init pose in config file
        ros_param_path = load_ros_params(rospackage_name="tht_control",
                                         rel_path_from_package_to_file='config',
                                         yaml_file_name='tht_insertion_plan.yaml')
        self.tht_component_dict = rospy.get_param('/tht_components/component_dict')
        if rospy.get_param('/insertion_plan/insert_single_part'):
            self.part_ID = rospy.get_param('/insertion_plan/part_ID')
            self.component_package = self.tht_component_dict[self.part_ID][0]
            self.n_pins = self.tht_component_dict[self.part_ID][1]
            self.pin_separation = self.tht_component_dict[self.part_ID][2]
        else:
            self.THT_insertion_sequence = rospy.get_param('/insertion_plan/THT_insertion_sequence')
            self.component_package_list = [self.tht_component_dict[part_id][0] for part_id in
                                           self.THT_insertion_sequence]
            self.n_pins_list = [self.tht_component_dict[part_id][1] for part_id in
                                self.THT_insertion_sequence]
            self.pin_separation_list = [self.tht_component_dict[part_id][2] for part_id in
                                        self.THT_insertion_sequence]
        # ToDo: Actually each component in an insertion plan can have its own reorientation angle
        self.part_reorientation_angle = rospy.get_param('/insertion_plan/part_reorientation_angle')
        self.part_reorientation_angle = self.part_reorientation_angle * m.pi / 180  # radians

        self.cam1_tip_window_px = 10
        self.cam1_tip_window_px = 10
        self._set_init_pose()
        self.step_count = 0
        self.success = False

        # Load THT Insertion Plan Config

        print("Finished initializing THTVisualServo")

    def _set_init_pose(self):
        """
        Asks the user to prepare the robot for a RL training episode.
        This includes bringing the robot in its init pose (in front of the cameras to do visual calibration)
        and placing the THT element in the robot gripper.
        """
        self.step_count = 0
        print("Started THTVisualServo._set_init_pose()...")
        print("Please prepare the robot for a new RL training episode!")
        print("1. Bring the robot gripper in front of the cameras by executing:")
        print("   rosrun tht_control yumi_move_test.py --direction cameras")
        print("2. Adjust or place the THT element in robot gripper if required:")
        print("   rosrun tht_control grip_part.py --gripper open")
        print("   rosrun tht_control grip_part.py --gripper close")
        print("   Hold the THT element still by hand between the closing grippers.")
        gripper_in_front_of_cameras = input(
            "   When the robot holds the THT element in front of the cameras and dark background is present, press y")
        print("3. Executing visual calibration and approach phase:")

        start_set_init_pose = time.time()

        pin_side_cam2 = None
        wait_for_input = True

        selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.moveToTildaOrigin(pin_side_cam2,
                                                                                           wait_for_input)

        # TODO: Get pin separation from planner config
        if self.component_package == 'SIP':
            self.graspingAngleErrorCompensation(pin_side_cam2, self.pin_separation, wait_for_input)
        # pin_separation = self.tht_component_dict['T'][2]
        #

        # Then execute final translation
        pin_side_cam2 = 'lowest'
        selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.moveToTildaOrigin(pin_side_cam2,
                                                                                           wait_for_input)

        # ToDo remove - just for visualization to check how accurate we are with the pin tip in origin
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_eef_pos = _current_eef_state_msg.data[:3]
        _current_eef_orientation = _current_eef_state_msg.data[3:]
        if self.component_package == 'SIP':
            eef_to_tilda, selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.visual_servo.autoSelectRightPinTip(
                _current_eef_pos, pin_side_cam2)
        elif self.component_package == 'DIP':
            eef_to_tilda, selected_pin_tip_tilda = self.visual_servo.autoSelectDIPPinTip(_current_eef_pos, self.n_pins,
                                                                                         self.cam1_tip_window_px)

        # Approach Goal Position
        self.approachGoal(selected_pin_tip_tilda, wait_for_input)

        # Correct Approach Pose
        self.approachCorrection(wait_for_input)

        if self.robot_type == 'hexaglide':
            _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
            self.robot_init_pose = _current_eef_state_msg.data

    def _set_visual_servo_cmd(self, cmd):
        self.visual_servo_cmd_pub.publish(cmd)  # Float64MultiArray

    def _set_action(self, action, wait_for_input=False):
        """
        Move the robot based on the action variable given
        """
        self.step_count = self.step_count + 1
        if self.init_step:  # TODO done to display images without lag
            self.init_step = False

        if self.robot_type == 'hexaglide':
            cmd_vector = Float64MultiArray()
            cmd_vector.data = [0, 0, 0]
            H1origin_to_tildaorigin_updated = []  # Coordinates of Tilda origin w.r.t H1 frame after rotation of theta about H1 z-axis
            if action >= 6:     # For rotation actions, rotate about the goal hole
                if action == 6:  # rotation command about z_H1 axis
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[0] * m.cos(self.rotation_stepsize) - self.hole_position_H1[1] * m.sin(self.rotation_stepsize))
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[1] * m.cos(self.rotation_stepsize) + self.hole_position_H1[0] * m.sin(self.rotation_stepsize))
                    cmd_vector.data[0] = cmd_vector.data[0] + (H1origin_to_tildaorigin_updated[0] - self.hole_position_H1[0])
                    cmd_vector.data[1] = cmd_vector.data[1] + (H1origin_to_tildaorigin_updated[1] - self.hole_position_H1[1])
                    eef_orientation_cmd = [0, 0, self.rotation_stepsize]
                    eef_orientation_cmd = quaternion_from_euler_theta(eef_orientation_cmd)
                    cmd_vector.data = np.concatenate((cmd_vector.data, eef_orientation_cmd, np.array([int(wait_for_input)])), axis=0)
                elif action == 7:
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[0] * m.cos(-self.rotation_stepsize) - self.hole_position_H1[1] * m.sin(-self.rotation_stepsize))
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[1] * m.cos(-self.rotation_stepsize) + self.hole_position_H1[0] * m.sin(-self.rotation_stepsize))
                    cmd_vector.data[0] = cmd_vector.data[0] + (H1origin_to_tildaorigin_updated[0] - self.hole_position_H1[0])
                    cmd_vector.data[1] = cmd_vector.data[1] + (H1origin_to_tildaorigin_updated[1] - self.hole_position_H1[1])
                    eef_orientation_cmd = [0, 0, -self.rotation_stepsize]
                    eef_orientation_cmd = quaternion_from_euler_theta(eef_orientation_cmd)
                    cmd_vector.data = np.concatenate((cmd_vector.data, eef_orientation_cmd, np.array([int(wait_for_input)])), axis=0)

                self._set_visual_servo_cmd(cmd_vector)
                appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
                if appr_success_msg.data == True:
                    appr_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
                    self.robot_init_pose = appr_eef_state_msg.data
                    print("Finished Action "+str(action)+" with success")
                else:
                    print("Action "+str(action)+" execution failed!")
                    exit(0)
            else:   # for translation actions: Move hexaglide UP, DOWN, LEFT or RIGHT
                self.action_pub.publish(action)
                appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)  # timeout = 5
                if appr_success_msg.data == True:
                    print("THTVisualServo: Action completed ")
                else:
                    print("THTVisualServo: Action failed")
                    exit(0)

        elif self.robot_type == 'yumi':
            self.action_pub.publish(action)
            if self.init_step:  # TODO done to display images without lag
                self.init_step = False
            appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)  # timeout = 5
            if appr_success_msg.data == True:
                print("THTVisualServo: Action completed ")
            else:
                print("THTVisualServo: Action failed")
                exit(0)



    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have access to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """

        print("THTVisualServo: Getting observations in _get_obs()...")
        start = time.time()
        _current_pressure = self.obs.current_pressure()
        print("time pressure: ", time.time() - start)
        _current_pressure_hist = self.obs.pressure_history(_current_pressure)

        # _current_image1, _current_image2 = self.obs.current_visualdata()
        start = time.time()
        _current_cropped_image1, _current_cropped_image2 = self.obs.current_cropped_visualdata(
            self.goal_pose_pixel_image1,
            self.goal_pose_pixel_image2)  # IS THIS NEEDED? New images in get_pin_tip fn.

        print("time images: ", time.time() - start)
        # Publish cropped images for visualization
        _cropped_img1 = None
        _cropped_img2 = None
        if self.diff_image_as_input:
            print("THTVisualServo: Working with grayscale diff images")
            _cropped_img1 = calculateGrayscaleImageDiff(self._init_cropped_image1,
                                                        convertNumpyImgToGrayscale(_current_cropped_image1))
            _cropped_img2 = calculateGrayscaleImageDiff(self._init_cropped_image2,
                                                        convertNumpyImgToGrayscale(_current_cropped_image2))
            _cropped_img1 = convert16SC1Torgb8(_cropped_img1)
            _cropped_img2 = convert16SC1Torgb8(_cropped_img2)
            # TODO Set _current_cropped_image1/2 and adjust the model to work with grayscale differences
            _cropped_img1 = self.obs.numpy_to_image(_cropped_img1, 'mono8')  # was '16SC1'
            _cropped_img2 = self.obs.numpy_to_image(_cropped_img2, 'mono8')
            print("THTVisualServo: Type of cropped_img1.data is ", type(_cropped_img1.data))
        else:
            print("THTVisualServo: Working with raw images")
            _cropped_img1 = self.obs.numpy_to_image(_current_cropped_image1, 'bgr8')
            _cropped_img2 = self.obs.numpy_to_image(_current_cropped_image2, 'bgr8')
            print("THTVisualServo: Type of cropped_img1.data is ", type(_cropped_img1.data))

        _robot_pose = self.obs.robot_state.eef_state
        _current_robot_position = _robot_pose[:3]
        _current_robot_orientation = _robot_pose[3:]

        if self.orientation_as_euler:  # TODO: changes in the observation space
            # CHECK FORMAT FOR RECEIVING ROBOT ORIENTATION (x,y,z,w) or (w,x,y,z)
            _current_robot_orientation = euler_from_quaternion(_current_robot_orientation[0],
                                                               _current_robot_orientation[1],
                                                               _current_robot_orientation[2],
                                                               _current_robot_orientation[3])

        # Use relative robot pose as observation (current pose - init pose after approach phase)
        _relative_robot_position = [_robot_pose[i] - self.robot_init_pose[i] for i in range(3)]
        _relative_robot_orientation = relativeQuaternionRotation(q0=_robot_pose[3:], q1=self.robot_init_pose[3:])
        # Debugging
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_robot_position = _current_eef_state_msg.data[:3]
        _current_robot_orientation = _current_eef_state_msg.data[3:]
        _current_robot_pose = _current_robot_position + _current_robot_orientation
        print("_robot_init_pose: ", self.robot_init_pose)
        print("_current_robot_pose: ", _current_robot_pose)
        _current_robot_pose = np.array(_current_robot_pose)
        _relative_robot_pose = _relative_robot_position + _relative_robot_orientation
        print("_relative_robot_pose: ", _relative_robot_pose)
        _relative_robot_pose = np.array(_relative_robot_pose)
        _current_pressure = np.array([_current_pressure])
        print("_current_pressure", _current_pressure)

        self.old_z_relative_coord.append(_relative_robot_position[2])

        # TODO: Improvement in observation selection (Pixel error or pixel pose?)
        observations = {"image1": _current_cropped_image1, "image2": _current_cropped_image2,
                        "pressure": _current_pressure,
                        "pose": _relative_robot_pose}
        print("THTVisualServo: Finished getting observations in _get_obs()")
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        pressure_exceeded = observations["pressure"][0].item() > self.max_pressure
        if self.robot_type == 'yumi':
            depth_exceeded = observations["pose"][2].item() < self.max_depth / 1000
        elif self.robot_type == 'hexaglide':
            print('observations["pose"][2].item(): ', observations["pose"][2].item())
            depth_exceeded = observations["pose"][2].item() >= self.max_depth / 1000
        time_up = self.step_count >= self.max_step_count  # TODO incorporate steps
        component_lost_from_gripper = False  # TODO from opencv: vision task
        # Only ask whether goal is reached when target depth is exceeded
        if depth_exceeded:
            success = input("Goal reached? Press y for success, any other key otherwise!")
            if success == "y":
                self.success = True
            elif success == "w" or success == "a" or success == "s" or success == "d" or success == "e":
                print("Repeat w/a/s/d/e input, because depth is reached")
                depth_exceeded = False
                self.success = False
            else:
                self.success = False

        # Alternative to input:
        # goal_reached = True if distance_norm < 0.001 & orientation_difference < 0.001 & _current_robot_pose[2] < zlimit else False
        # eef_not_in_working_space = not self.check_eef_in_working_space(observations.get_eef)
        # component_not_in_working_space = not self.check_eef_in_working_space(
        #    observations.get_component_eef)  # maybe remove this, we dont have 3d coord, or pin detection is not working

        done = pressure_exceeded or depth_exceeded or time_up or component_lost_from_gripper or self.success  # TODO add component_in_image
        if done:
            final_str = "pressure_exceeded: ", pressure_exceeded, "depth_exceeded: ", depth_exceeded, "time up: ", time_up, \
                        ", component_lost_from_gripper: ", component_lost_from_gripper, "success: ", self.success
            print(final_str)
        #       eef_not_in_working_space  # | component_not_in_working_space
        print("THTVisualServo _is_done(): ", done)
        return done

    def moveToTildaOrigin(self, pin_side_cam2=None, wait_for_input=True):
        '''
        Finds primary pin tip and moves it (only translation) to the intersection of cameras' main optical axes
        (so called 'tilda origin', since tilda coordinate system is based on this optical intersection)
        '''
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        eef_position = _current_eef_state_msg.data[:3]
        eef_orientation = _current_eef_state_msg.data[3:]
        # Case for SIP and DIP:
        if self.component_package == 'DIP':
            free_pin_tip_tilda = None
            eef_to_tilda, selected_pin_tip_tilda = self.visual_servo.autoSelectDIPPinTip(eef_position, self.n_pins,
                                                                                         self.cam1_tip_window_px)
        elif self.component_package == 'SIP':
            eef_to_tilda, selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.visual_servo.autoSelectRightPinTip(
                eef_position, pin_side_cam2)
        eef_cmd = Float64MultiArray()
        '''
        Warning:
        YuMi controller interprets translation commands as relative but rotations as absolute!
        Hexaglide controller interprets all commands (both translation and rotation) as relative!
        Since moveToTildaOrigin only involves translations, quaternion of the command is set to no relative rotation
        in case of Hexaglide.
        '''
        if self.robot_type == 'yumi':
            eef_cmd.data = np.concatenate(
                (eef_to_tilda - eef_position, eef_orientation, np.array([int(wait_for_input)])), axis=0)
        elif self.robot_type == 'hexaglide':
            eef_cmd.data = np.concatenate(
                (eef_to_tilda - eef_position, [0, 0, 0, 1], np.array([int(wait_for_input)])), axis=0)
        self._set_visual_servo_cmd(eef_cmd)
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        if appr_success_msg.data == True:
            print("THTVisualServo: Successfully finished visual calibration iteration 1")
        else:
            print("THTVisualServo: Failed at visual calibration iteration 1")
            exit(0)
        return selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2

    def graspingAngleErrorCompensation(self, pin_side_cam2, pin_separation, wait_for_input=True):
        # Compensate the THD rotation around vertical axis
        # For 2 short pins p1, p2 that means orienting them so that:
        # the line through p1 and p2 goes directly through the line of sight of camera 2
        # print("THTVisualServo: Getting current complete images from cameras...")
        _current_image1, _current_image2 = self.obs.current_visualdata()
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_eef_pos = _current_eef_state_msg.data[:3]
        _current_eef_orientation = _current_eef_state_msg.data[3:]
        eef_to_tilda, selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.visual_servo.autoSelectRightPinTip(
            _current_eef_pos, pin_side_cam2)
        p1 = selected_pin_tip_tilda
        p2 = free_pin_tip_tilda
        print('p1: ', p1)
        print('p2: ', p2)
        pin_separation_measured = m.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
        p2_d = np.add(p1, np.array([0, pin_separation_measured, p2[2]]))  # d means 'desired'
        diff_p1_p2 = p2 - p1
        diff_p1_p2_d = p2_d - p1
        print('Measured pin separation: ', pin_separation_measured)
        print('Nominal pin separation: ', pin_separation)

        if abs(pin_separation - pin_separation_measured) > 0.001:
            print('Pin separation is too different from the nominal! Exiting.')
            exit(0)

        # Theta = Rotation of THD around vertical axis (z-axis Yumi)
        theta = m.acos(np.dot(diff_p1_p2_d, diff_p1_p2) / (np.linalg.norm(diff_p1_p2_d) * np.linalg.norm(diff_p1_p2)))
        theta_degrees = theta / m.pi * 180.0
        eef_to_tilda_right_desired = eef_to_tilda
        eef_to_tilda_right_desired[0] = eef_to_tilda[0] + pin_separation

        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_eef_pos = _current_eef_state_msg.data[:3]
        _current_eef_orientation = _current_eef_state_msg.data[3:]
        sign = -1 if pin_side_cam2 == 'right' else 1
        theta = sign * theta
        print('Theta (degrees): ', sign * theta_degrees)

        eef_cmd = Float64MultiArray()
        _updated_eef_pos = [0, 0, 0]

        if self.robot_type == 'yumi':
            roll, pitch, yaw = euler_from_quaternion_theta(_current_eef_orientation)
            yaw = yaw + theta
            _updated_eef_orientation_euler = [roll, pitch, yaw]
            _updated_eef_orientation = quaternion_from_euler_theta(_updated_eef_orientation_euler)
            eef_cmd.data = np.concatenate((_updated_eef_pos, _updated_eef_orientation, np.array([int(wait_for_input)])),
                                          axis=0)

        elif self.robot_type == 'hexaglide':
            roll = 0.0
            pitch = 0.0
            yaw = -theta
            _updated_eef_orientation_euler = [roll, pitch, yaw]

            H1origin_to_tildaorigin_updated = []  # Coordinates of Tilda origin w.r.t H1 frame after rotation of theta about H1 z-axis
            H1origin_to_tildaorigin_updated.append(
                self.H1origin_to_tildaorigin[0] * m.cos(yaw) - self.H1origin_to_tildaorigin[1] * m.sin(yaw))
            H1origin_to_tildaorigin_updated.append(
                self.H1origin_to_tildaorigin[1] * m.cos(yaw) + self.H1origin_to_tildaorigin[0] * m.sin(yaw))

            _updated_eef_pos[0] = H1origin_to_tildaorigin_updated[0] - self.H1origin_to_tildaorigin[0]
            _updated_eef_pos[1] = H1origin_to_tildaorigin_updated[1] - self.H1origin_to_tildaorigin[1]

            _updated_eef_orientation = quaternion_from_euler_theta(_updated_eef_orientation_euler)
            # print('H1origin_to_tildaorigin_updated: ', H1origin_to_tildaorigin_updated)
            print('updated_eef_pos: ', _updated_eef_pos)

            eef_cmd.data = np.concatenate((_updated_eef_pos, _updated_eef_orientation, np.array([int(wait_for_input)])),
                                          axis=0)

        self._set_visual_servo_cmd(eef_cmd)
        start = time.time()
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        print('APPROACH MESSAGE 2 IN: ', time.time() - start)
        if appr_success_msg.data == True:
            print("Part reorientation SUCCESSFUL")
        else:
            print("Part reorientation UNSUCCESSFUL")
            exit(0)

    def approachGoal(self, selected_pin_tip_tilda, wait_for_input):
        cmd_vector = Float64MultiArray()
        cmd_vector.data = np.dot(self.visual_servo.rot_tilda_to_base,
                                 (np.array(self.goal_pose_tilda)) / 1000 - selected_pin_tip_tilda)
        cmd_vector.data[2] = self.tildaorigin_to_pcb_z if cmd_vector.data[2] < self.tildaorigin_to_pcb_z else \
            cmd_vector.data[2]
        if self.robot_type == 'yumi':
            _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
            _current_eef_pos = _current_eef_state_msg.data[:3]
            _current_eef_orientation = _current_eef_state_msg.data[3:]
            cmd_vector.data = np.concatenate((cmd_vector.data, _current_eef_orientation,
                                              np.array([int(wait_for_input)])), axis=0)
        elif self.robot_type == 'hexaglide':
            if self.part_reorientation_angle != 0:
                roll = 0.0
                pitch = 0.0
                yaw = self.part_reorientation_angle
                _updated_eef_orientation_euler = [roll, pitch, yaw]

                H1origin_to_tildaorigin_updated = []  # Coordinates of Tilda origin w.r.t H1 frame after rotation of theta about H1 z-axis
                H1origin_to_tildaorigin_updated.append(
                    self.hole_position_H1[0] * m.cos(yaw) - self.hole_position_H1[1] * m.sin(yaw))
                H1origin_to_tildaorigin_updated.append(
                    self.hole_position_H1[1] * m.cos(yaw) + self.hole_position_H1[0] * m.sin(yaw))

                cmd_vector.data[0] = cmd_vector.data[0] + (H1origin_to_tildaorigin_updated[0] - self.hole_position_H1[0])
                cmd_vector.data[1] = cmd_vector.data[1] + (H1origin_to_tildaorigin_updated[1] - self.hole_position_H1[1])

                _updated_eef_orientation = quaternion_from_euler_theta(_updated_eef_orientation_euler)
                cmd_vector.data = np.concatenate(
                    (cmd_vector.data, _updated_eef_orientation, np.array([int(wait_for_input)])),
                    axis=0)
            else:
                cmd_vector.data = np.concatenate((cmd_vector.data, [0, 0, 0, 1], np.array([int(wait_for_input)])),
                                                 axis=0)
        # Send command to YuMi
        print('updated_eef_pose: ', cmd_vector.data)
        self._set_visual_servo_cmd(cmd_vector)
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        if appr_success_msg.data == True:
            appr_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
            self.robot_init_pose = appr_eef_state_msg.data
            print("Finished Approach Phase with success")
        else:
            print("Finished THTVisualServo._set_init_pose() without success, exiting script")
            exit(0)

    def approachCorrection(self, wait_for_input):
        # Approach correction phase
        _current_cropped_image1, _current_cropped_image2 = self.obs.current_cropped_visualdata(
            self.goal_pose_pixel_image1,
            self.goal_pose_pixel_image2)

        _cropped_img1 = self.obs.numpy_to_image(_current_cropped_image1, 'bgr8')
        _cropped_img2 = self.obs.numpy_to_image(_current_cropped_image2, 'bgr8')
        self.cropped_img1_pub.publish(_cropped_img1)
        self.cropped_img2_pub.publish(_cropped_img2)

        image1_pin_location, image2_pin_location, pin_dev_cam1, pin_dev_cam2 = self.mpd.mpd(_current_cropped_image1,
                                                                                            _current_cropped_image2)
        if abs(pin_dev_cam1) > self.center_tolerance_px or abs(pin_dev_cam2) > self.center_tolerance_px:
            orientation_cmd = None
            if self.robot_type == 'yumi':
                self.corrected_goal_pose_pixel_image1 = self.corrected_goal_pose_pixel_image1 + pin_dev_cam1
                self.corrected_goal_pose_pixel_image2 = self.corrected_goal_pose_pixel_image2 + pin_dev_cam2
                self.corrected_goal_pose_tilda = self.visual_servo.calc_1_tip_coords(
                    self.corrected_goal_pose_pixel_image1,
                    self.corrected_goal_pose_pixel_image2,
                    self.visual_servo.x_tilda_c1,
                    self.visual_servo.y_tilda_c2)
                orientation_cmd = self.robot_init_pose[3:]
            elif self.robot_type == 'hexaglide':
                # CALCULATE DIFF GOAL POSE WITH GEOMETRY:
                self.corrected_goal_pose_pixel_image1 = self.corrected_goal_pose_pixel_image1 + pin_dev_cam1
                self.corrected_goal_pose_pixel_image2 = self.corrected_goal_pose_pixel_image2 + pin_dev_cam2
                self.corrected_goal_pose_tilda = self.visual_servo.calc_1_tip_coords(
                    self.corrected_goal_pose_pixel_image1,
                    self.corrected_goal_pose_pixel_image2,
                    self.visual_servo.x_tilda_c1,
                    self.visual_servo.y_tilda_c2)

                # ESTIMATE DIFF GOAL POSE WITH FITTING FUNCTION
                # self.corrected_goal_pose_pixel_image1 = [(self.corrected_goal_pose_pixel_image1[0][0] + pin_dev_cam1,
                #                                           self.corrected_goal_pose_pixel_image1[0][1])]
                # self.corrected_goal_pose_pixel_image2 = [(self.corrected_goal_pose_pixel_image2[0][0] + pin_dev_cam2,
                #                                           self.corrected_goal_pose_pixel_image2[0][1])]
                # corrected_x_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image2[0][0],
                #                                                        self.corrected_goal_pose_pixel_image2[0][1],
                #                                                        self.visual_servo.x_tilda_params[0],
                #                                                        self.visual_servo.x_tilda_params[1],
                #                                                        self.visual_servo.x_tilda_params[2],
                #                                                        self.visual_servo.x_tilda_params[3],
                #                                                        self.visual_servo.x_tilda_params[4],
                #                                                        self.visual_servo.x_tilda_params[5])
                #
                # corrected_y_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image1[0][0],
                #                                                        self.corrected_goal_pose_pixel_image1[0][1],
                #                                                        self.visual_servo.y_tilda_params[0],
                #                                                        self.visual_servo.y_tilda_params[1],
                #                                                        self.visual_servo.y_tilda_params[2],
                #                                                        self.visual_servo.y_tilda_params[3],
                #                                                        self.visual_servo.y_tilda_params[4],
                #                                                        self.visual_servo.y_tilda_params[5])
                # self.corrected_goal_pose_tilda = (
                # corrected_x_tilda_goal, corrected_y_tilda_goal, self.goal_pose_tilda[2])
                orientation_cmd = [0, 0, 0, 1]

            diff_goal_pose_tilda = list(map(operator.sub, self.goal_pose_tilda, self.corrected_goal_pose_tilda))
            print('image1_pin_location, image2_pin_location', image1_pin_location, image2_pin_location)
            print('pin_dev_cam1, pin_dev_cam2: ', pin_dev_cam1, pin_dev_cam2)
            diff_goal_pose_tilda[2] = 0
            print('diff_goal_pose_tilda: ', diff_goal_pose_tilda)
            eef_cmd = np.dot(self.visual_servo.rot_tilda_to_base, np.array(diff_goal_pose_tilda) / 1000)
            # eef_cmd = [diff_goal_pose_tilda[1] / 1000, -diff_goal_pose_tilda[0] / 1000, 0]

            cmd_vector = Float64MultiArray()
            cmd_vector.data = np.concatenate((eef_cmd, orientation_cmd,
                                              np.array([int(wait_for_input)])), axis=0)
            self._set_visual_servo_cmd(cmd_vector)
            appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)

            if appr_success_msg.data == True:
                appr_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
                self.robot_init_pose = appr_eef_state_msg.data
                print("Finished THTRLEnv Approach Correction Phase successful")
            else:
                print("Finished THTRLEnv Approach Correction Phase Unsuccessful, exiting script")
                exit(0)
        else:
            self.corrected_goal_pose_tilda = self.visual_servo.calc_1_tip_coords(self.corrected_goal_pose_pixel_image1,
                                                                                 self.corrected_goal_pose_pixel_image2,
                                                                                 self.visual_servo.x_tilda_c1,
                                                                                 self.visual_servo.y_tilda_c2)
            print("Skipping Approach Correction Phase")


'''
This script uses Metallic Pixel Detection (MPD) to directly derive actions based on images.
Pressure and pose data are used to detect important states.
'''
if __name__ == '__main__':
    rospy.init_node('tht_visualservo', anonymous=True)
    # Initialization of the controller
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_type', type=str, help='yumi/hexaglide', default=None, required=True)
    args = parser.parse_args()
    robot_type = args.robot_type
    tht_controller = THTVisualServo(robot_type)
    # Approach the target
    # tht_controller._set_init_pose()
    # - Retrieve observations
    observations = tht_controller._get_obs()
    image_without_pin_cam1 = tht_controller.init_cropped_image1
    image_without_pin_cam2 = tht_controller.init_cropped_image2
    down_count = 0
    # Execute movements
    while True:
        print('--------------------------')
        print('Down count: ', down_count)
        # - Retrieve observations
        observations = tht_controller._get_obs()
        # - Check whether done or not
        if tht_controller._is_done(observations):
            down_count = 0
            user_input = input('Episode completed. Reset init pose? y/n ')
            if user_input == 'y':
                tht_controller.step_count = 0
                tht_controller.old_z_relative_coord = [0.0]
                tht_controller.init_step = True
                tht_controller.success = False
                tht_controller._set_init_pose()
                gray_image_without_pin_cam1 = cv2.cvtColor(tht_controller.init_cropped_image1, cv2.COLOR_BGR2GRAY)
                gray_image_without_pin_cam2 = cv2.cvtColor(tht_controller.init_cropped_image2, cv2.COLOR_BGR2GRAY)
                observations = tht_controller._get_obs()
            else:
                break

        print('THTVisualServo: Step ', tht_controller.step_count)

        # - Search strategy if significant pressure detected
        if observations["pressure"][0].item() > 250.0:
            # movement_cycle = [2, 3, 3, 2] # right left left right
            movement_cycle = [7, 6, 6, 7, 2, 3, 3, 2]  # yaw_p yaw_n yaw_n yaw_p right left left right
            unsuccessful_cycle = True
            for movement in movement_cycle:
                # tht_controller._set_action(5)  # only useful for parts with very bendable pins and small pin separation!
                tht_controller._set_action(movement)
                # tht_controller._set_action(4)  # only useful for part with very bendable pins and small pin separation!
                observations = tht_controller._get_obs()
                cropped_img1 = tht_controller.obs.numpy_to_image(observations['image1'], 'bgr8')
                cropped_img2 = tht_controller.obs.numpy_to_image(observations['image2'], 'bgr8')
                tht_controller.cropped_img1_pub.publish(cropped_img1)
                tht_controller.cropped_img2_pub.publish(cropped_img2)
                if observations["pressure"][0].item() < 75.0:
                    tht_controller._set_action(4)  # down
                    unsuccessful_cycle = False
                    observations = tht_controller._get_obs()
                    cropped_img1 = tht_controller.obs.numpy_to_image(observations['image1'], 'bgr8')
                    cropped_img2 = tht_controller.obs.numpy_to_image(observations['image2'], 'bgr8')
                    tht_controller.cropped_img1_pub.publish(cropped_img1)
                    tht_controller.cropped_img2_pub.publish(cropped_img2)
                    break
            if unsuccessful_cycle:
                tht_controller._set_action(5)  # up
                down_count = down_count - 1
                observations = tht_controller._get_obs()
                cropped_img1 = tht_controller.obs.numpy_to_image(observations['image1'], 'bgr8')
                cropped_img2 = tht_controller.obs.numpy_to_image(observations['image2'], 'bgr8')
                tht_controller.cropped_img1_pub.publish(cropped_img1)
                tht_controller.cropped_img2_pub.publish(cropped_img2)
            continue

        # - Get classification left/center/right for both images
        image_with_pin_cam1 = observations['image1']
        image_with_pin_cam2 = observations['image2']
        mpd = MetallicPixelDetection(image_without_pin_cam1, image_without_pin_cam2, tht_controller.center_tolerance_px,
                                     tht_controller.img_center_x)
        pin_location_cam1, pin_location_cam2, pin_dev_cam1, pin_dev_cam2 = mpd.mpd(image_with_pin_cam1,
                                                                                   image_with_pin_cam2)

        print('Pin deviation from center in cam1 (pixels): ', pin_dev_cam1)
        print('Pin deviation from center in cam2 (pixels): ', pin_dev_cam2)

        pin_location_cam1 = 'Left' if pin_dev_cam1 < -tht_controller.center_tolerance_px else 'Right' \
            if pin_dev_cam1 > tht_controller.center_tolerance_px else 'Center'
        pin_location_cam2 = 'Left' if pin_dev_cam2 < -tht_controller.center_tolerance_px else 'Right' \
            if pin_dev_cam2 > tht_controller.center_tolerance_px else 'Center'

        print('Pin location cam1: ', pin_location_cam1)
        print('Pin location cam2: ', pin_location_cam2)

        print('Reducing positioning error in ', 'cam1' if abs(pin_dev_cam1) > abs(pin_dev_cam2) else 'cam2')

        cropped_img1 = tht_controller.obs.numpy_to_image(observations['image1'], 'bgr8')
        cropped_img2 = tht_controller.obs.numpy_to_image(observations['image2'], 'bgr8')

        tht_controller.cropped_img1_pub.publish(cropped_img1)
        tht_controller.cropped_img2_pub.publish(cropped_img2)

        # STEPWISE MOVEMENTS:
        if down_count >= 2:
            recommended_action = 4  # Keyboard:E Yumi perspective: Down
            tht_controller._set_action(recommended_action)
            down_count = down_count + 1
            continue

        # DIRECT MOVEMENT:
        if pin_location_cam1 != 'Center' or pin_location_cam2 != 'Center':
            if robot_type == 'yumi':
                tht_controller.corrected_goal_pose_pixel_image1 = tht_controller.corrected_goal_pose_pixel_image1 + pin_dev_cam1
                tht_controller.corrected_goal_pose_pixel_image2 = tht_controller.corrected_goal_pose_pixel_image2 + pin_dev_cam2
                _corrected_goal_pose_tilda = tht_controller.visual_servo.calc_1_tip_coords(
                    tht_controller.corrected_goal_pose_pixel_image1,
                    tht_controller.corrected_goal_pose_pixel_image2,
                    tht_controller.visual_servo.x_tilda_c1,
                    tht_controller.visual_servo.y_tilda_c2)
                orientation_cmd = tht_controller.robot_init_pose[3:]
            elif robot_type == 'hexaglide':
                # CALCULATE DIFF GOAL POSE WITH GEOMETRY:
                tht_controller.corrected_goal_pose_pixel_image1 = tht_controller.corrected_goal_pose_pixel_image1 + pin_dev_cam1
                tht_controller.corrected_goal_pose_pixel_image2 = tht_controller.corrected_goal_pose_pixel_image2 + pin_dev_cam2
                _corrected_goal_pose_tilda = tht_controller.visual_servo.calc_1_tip_coords(
                    tht_controller.corrected_goal_pose_pixel_image1,
                    tht_controller.corrected_goal_pose_pixel_image2,
                    tht_controller.visual_servo.x_tilda_c1,
                    tht_controller.visual_servo.y_tilda_c2)

                # ESTIMATE DIFF GOAL POSE WITH FITTING FUNCTION
                # tht_controller.corrected_goal_pose_pixel_image1 = [
                #     (tht_controller.corrected_goal_pose_pixel_image1[0][0] + pin_dev_cam1,
                #      tht_controller.corrected_goal_pose_pixel_image1[0][1])]
                # tht_controller.corrected_goal_pose_pixel_image2 = [
                #     (tht_controller.corrected_goal_pose_pixel_image2[0][0] + pin_dev_cam2,
                #      tht_controller.corrected_goal_pose_pixel_image2[0][1])]
                # corrected_x_tilda_goal = tht_controller.visual_servo.square_func(
                #     tht_controller.corrected_goal_pose_pixel_image2[0][0],
                #     tht_controller.corrected_goal_pose_pixel_image2[0][1],
                #     tht_controller.visual_servo.x_tilda_params[0],
                #     tht_controller.visual_servo.x_tilda_params[1],
                #     tht_controller.visual_servo.x_tilda_params[2],
                #     tht_controller.visual_servo.x_tilda_params[3],
                #     tht_controller.visual_servo.x_tilda_params[4],
                #     tht_controller.visual_servo.x_tilda_params[5])
                #
                # corrected_y_tilda_goal = tht_controller.visual_servo.square_func(
                #     tht_controller.corrected_goal_pose_pixel_image1[0][0],
                #     tht_controller.corrected_goal_pose_pixel_image1[0][1],
                #     tht_controller.visual_servo.y_tilda_params[0],
                #     tht_controller.visual_servo.y_tilda_params[1],
                #     tht_controller.visual_servo.y_tilda_params[2],
                #     tht_controller.visual_servo.y_tilda_params[3],
                #     tht_controller.visual_servo.y_tilda_params[4],
                #     tht_controller.visual_servo.y_tilda_params[5])
                # _corrected_goal_pose_tilda = (
                # corrected_x_tilda_goal, corrected_y_tilda_goal, tht_controller.corrected_goal_pose_tilda[2])
                orientation_cmd = [0, 0, 0, 1]

            print(tht_controller.corrected_goal_pose_tilda, _corrected_goal_pose_tilda)
            diff_goal_pose_tilda = list(
                map(operator.sub, tht_controller.corrected_goal_pose_tilda, _corrected_goal_pose_tilda))
            print('diff_goal_pose_tilda: ', diff_goal_pose_tilda[0] / 1000, diff_goal_pose_tilda[1] / 1000)
            eef_cmd = np.dot(tht_controller.visual_servo.rot_tilda_to_base, np.array(diff_goal_pose_tilda) / 1000)
            cmd_vector = Float64MultiArray()
            cmd_vector.data = np.concatenate((eef_cmd, orientation_cmd, np.array([int(False)])), axis=0)
            tht_controller._set_visual_servo_cmd(cmd_vector)
            appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
            # tht_controller.corrected_goal_pose_tilda = _corrected_goal_pose_tilda
        else:
            recommended_action = 4  # Keyboard:E Yumi perspective: Down
            tht_controller._set_action(recommended_action)
            down_count = down_count + 1
