#!/usr/bin/env python

import os
from yumi_tht_control.arm import Arm
from vision.grasp_circle_detection import get_object_position, get_object_position_by_polygons
from tht_control.common import *
from vision.click_goal import GoalPixelSetter
from vision.visual_servo import VisualServo
from vision.vision_sensor import convertNumpyImgToGrayscale, calculateGrayscaleImageDiff, convert16SC1Torgb8
from std_msgs.msg import Int64, Float64MultiArray, Bool
from sensor_msgs.msg import Image
import numpy as np
import pickle
import rospy
import argparse
import cv2
import rospkg
import matplotlib.pyplot as plt
from std_msgs.msg import String, Bool
import time



'''
This script is used to control the left arm of YuMi
'''
if __name__ == '__main__':
    # Initialize the left YuMi arm and empty lists of waypoints
    rospy.init_node('move_group_python_interface', anonymous=True)
    yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    # visual_servo = VisualServo()
    cropped_img1_pub = rospy.Publisher('cropped_img1', Image, queue_size=10)
    cropped_img2_pub = rospy.Publisher('cropped_img2', Image, queue_size=10)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('yumi_tht_control')
    path = os.path.abspath(os.curdir)
    # Initialize the interface between RL controller and robot controller
    rospy.set_param('translation_step_mm', 0.5)
    rospy.set_param('rotation_step_deg', 2)
    reached_pub = rospy.Publisher("reached_position", Bool, queue_size=10)
    # Navigate to the file with serialized (saved/recorded) waypoints and read them
    ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                     yaml_file_name='config_yumi.yaml')
    grasp_waypoint_list_filename = rospy.get_param('ros_param/grasp_waypoint_list_filename')
    grasp_waypoints = get_waypoints(grasp_waypoint_list_filename)
    insertion_waypoint_list_filename = rospy.get_param('ros_param/insertion_waypoint_list_filename')
    insertion_waypoints = get_waypoints(insertion_waypoint_list_filename)
    home_waypoints = []
    translation_step_size = 0.3  # mm

    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', type=str, help='forwards/backwards/cameras/insertion', default=None,
                        required=False)
    args = parser.parse_args()

    object_center_x, object_center_y, img = get_object_position()
    object_center_x_lying, object_center_y_lying, img_lying, angles_lying = get_object_position_by_polygons()

    if args.direction == 'forwards':
        home_waypoints = list(grasp_waypoints)
        yumi.followWaypoints(home_waypoints)

    elif args.direction == 'backwards':
        home_waypoints = list(reversed(grasp_waypoints))
        yumi.followWaypoints(home_waypoints)
        exit(0)

    # cv2.imshow('Image', img)
    # cv2.waitKey(0)

    user_input = raw_input('Continue Movement? y/n')
    if not (user_input == 'y'):
        exit(0)

    eef_position = [None, None, None]
    eef_orientation = [None, None, None, None]

    # Camera lens center in EEF frame approximately
    camera_center_position = [0.17934, 0.4134, 0.4627]
    eef_position[0] = camera_center_position[0]
    eef_position[1] = camera_center_position[1]
    eef_position[2] = camera_center_position[2]
    eef_orientation[0] = -0.9991170568679713
    eef_orientation[1] = -0.03441299405798008
    eef_orientation[2] = -0.014942181567382563
    eef_orientation[3] = 0.018910074545026218
    eef_cmd = np.concatenate((eef_position, eef_orientation), axis=0)
    print('Current EEF Pose: ', yumi.getEEFState())
    print('Commanded EEF Pose: ', eef_cmd)

    user_input = raw_input('Continue Movement? y/n')
    if not (user_input == 'y'):
        exit(0)
    else:
        yumi.setEEFPose(eef_cmd)

    current_eef_position, current_eef_orientation = yumi.getEEFState()

    print('Moving to object: ', object_center_x[0], object_center_y[0])

    # Move EEF to 1st object center (x, y)
    eef_position[0] = eef_position[0] - (object_center_x[0] / 1000)
    eef_position[1] = eef_position[1] + (object_center_y[0] / 1000)
    eef_position[2] = current_eef_position[2]
    eef_orientation[0] = current_eef_orientation[0]
    eef_orientation[1] = current_eef_orientation[1]
    eef_orientation[2] = current_eef_orientation[2]
    eef_orientation[3] = current_eef_orientation[3]
    eef_cmd = np.concatenate((eef_position, eef_orientation), axis=0)
    print('Current EEF Pose: ', yumi.getEEFState())
    print('Commanded EEF Pose: ', eef_cmd)
    user_input = raw_input('Continue Movement? y/n')
    if not (user_input == 'y'):
        exit(0)
    else:
        yumi.setEEFPose(eef_cmd)

    # Open the gripper
    yumi.openGripper()

    # Move EEF down to Grab the object
    yumi.getEEFState()
    current_eef_position, current_eef_orientation = yumi.getEEFState()

    eef_position[0] = current_eef_position[0]
    eef_position[1] = current_eef_position[1]
    eef_position[2] = current_eef_position[2] - (12.0 / 100.0)  # Move down -z by 10 cm
    eef_orientation[0] = current_eef_orientation[0]
    eef_orientation[1] = current_eef_orientation[1]
    eef_orientation[2] = current_eef_orientation[2]
    eef_orientation[3] = current_eef_orientation[3]
    eef_cmd = np.concatenate((eef_position, eef_orientation), axis=0)
    print('Current EEF Pose: ', yumi.getEEFState())
    print('Commanded EEF Pose: ', eef_cmd)
    user_input = raw_input('Continue Movement? y/n')
    if not (user_input == 'y'):
        exit(0)
    else:
        yumi.setEEFPose(eef_cmd)

    # Close gripper and move UP
    yumi.closeGripper()

    current_eef_position, current_eef_orientation = yumi.getEEFState()
    eef_position[0] = current_eef_position[0]
    eef_position[1] = current_eef_position[1]
    eef_position[2] = current_eef_position[2] + (12.0 / 100.0)  # Move up +z by 10 cm
    eef_orientation[0] = current_eef_orientation[0]
    eef_orientation[1] = current_eef_orientation[1]
    eef_orientation[2] = current_eef_orientation[2]
    eef_orientation[3] = current_eef_orientation[3]
    eef_cmd = np.concatenate((eef_position, eef_orientation), axis=0)
    print('Current EEF Pose: ', yumi.getEEFState())
    print('Commanded EEF Pose: ', eef_cmd)
    user_input = raw_input('Continue Movement? y/n')
    if not (user_input == 'y'):
        exit(0)
    else:
        yumi.setEEFPose(eef_cmd)

    # Move to home position
    home_waypoints = list(reversed(grasp_waypoints))
    yumi.followWaypoints(home_waypoints)

    home_waypoints = list(insertion_waypoints)
    yumi.followWaypoints(home_waypoints)
