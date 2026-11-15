#!/usr/bin/env python

import rospy
import rospkg
import pickle
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision.vision_sensor import VisionSensor
from vision.pin_detector import pin_tips, get_pin_tip_location
from tht_control.common import *
from yumi_tht_control.arm import Arm
import std_msgs.msg

'''
This script determines the transformation between camera coordinate system and robot coordinate system
by moving a short steel wire along 3 robot axes, detecting the tip point and calculating best fit orthogonal system.
'''

global B, H, L
global x_hat_c1, x_hat_c2, y_hat_c1, y_hat_c2, z_hat

def main():
    global B, H, L
    global x_hat_c1, x_hat_c2, y_hat_c1, y_hat_c2, z_hat

    ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                     yaml_file_name='config_yumi.yaml')

    rospy.init_node('visual_coord_sys_referencing', anonymous=False)

    # Left arm follows the waypoints
    # We assume the steel wire is already gripped
    yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    waypoint_list_filename = rospy.get_param('ros_param/waypoint_list_filename')
    waypoints = get_waypoints(waypoint_list_filename)
    # yumi.followWaypoints(waypoints)

    camera_id = rospy.get_param('ros_param/camera_id')
    # rospy.init_node('pin_localization', anonymous=True)
    dataserver = VisionSensor(cameras=camera_id)

    # Input distances measured to the adjustable bolts inside the profile in mm
    d1 = rospy.get_param('extrinsic_params/d1')
    d2 = rospy.get_param('extrinsic_params/d2')

    # Calculate or input camera positions in corner coordinates (hat)
    x_hat_c1 = rospy.get_param('extrinsic_params/x_hat_c1')
    y_hat_c1 = d1 + 30.0
    x_hat_c2 = d2 + 12.0
    y_hat_c2 = rospy.get_param('extrinsic_params/y_hat_c2')
    z_hat = rospy.get_param('extrinsic_params/z_hat')

    # Calculate camera positions in image centre coordinates (tilda)
    x_tilda_c1 = -x_hat_c2 + x_hat_c1
    y_tilda_c1 = 0  # per definition of image centre coordinate system
    x_tilda_c2 = 0  # per definition of image centre coordinate system
    y_tilda_c2 = -y_hat_c1 + y_hat_c2

    # (Ideal) rotation between YuMi and camera coord.sys.
    rot_tilda_to_base = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    # Intrinsic parameters of the cameras
    cam_info = dataserver.getCameraInfo()
    print len(cam_info)
    B = cam_info[0]
    H = cam_info[1]
    L = cam_info[2]  # Fx~Fy
    print 'Camera info: ', cam_info

    ### VISUAL REFERENCING ###

    # Repeat visual servoing twice to get to tilda origin

    # Get images synchronously
    for i in range(2):
        img = dataserver.readCamera()
        rospy.sleep(1)
        img = img[:len(camera_id)]
        move_to_tilda_origin(x_tilda_c1, y_tilda_c2, img, yumi)
        rospy.sleep(3)

    # Record eef position
    yumi.getEEFState()
    tilda_origin_eef_pos, tilda_origin_eef_orientation = yumi.getEEFState()
    print 'EEF position for tilda origin [m]: ', tilda_origin_eef_pos

    # Move along x_Y, y_Y, z_Y by 20 mm, get pin tip tilda, move back to saved origin
    vector_length = 0.02
    axes_tips = []
    for i in range(3):
        # eef_pos = tilda_origin_eef_pos + np.dot(rot_tilda_to_base, rot_tilda_to_base[i]*vector_length)
        eef_pos, eef_orientation = yumi.getEEFState()
        eef_pos[i] = eef_pos[i] + vector_length
        # more readable version!
        yumi.getEEFState()
        print 'updated eef_pos [m]: ', eef_pos
        print 'change in eef_pos (m): ', np.array([eef_pos]) - np.array([yumi.getEEFState()[0]])
        eef_cmd = np.concatenate((eef_pos, eef_orientation), axis=0)
        usr_input = raw_input('Continue robot motion? y/n')
        check = False
        if usr_input == 'y':
            while check == False:
                check = yumi.setEEFPose(eef_cmd)

        # check command execution instead of sleep!
        img = dataserver.readCamera()
        rospy.sleep(1)
        img = img[:len(camera_id)]

        # Get list of tuples with pin tip coordinates in the image
        tips_1 = pin_tips(img[0], limits_x=[300, 600], limits_y=[150, 300], cam=1)
        tips_2 = pin_tips(img[1], limits_x=[300, 620], limits_y=[150, 300], cam=2)
        pin_tip_tilda = calc_1_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2)
        print 'Pin tip tilda (Calibration): ', pin_tip_tilda
        axes_tips.append(pin_tip_tilda)

        # Move back to saved origin
        eef_pos = tilda_origin_eef_pos
        print 'updated eef_pos [m]: ', eef_pos
        print 'change in eef_pos (m): ', np.array([eef_pos]) - np.array([yumi.getEEFState()[0]])
        eef_cmd = np.concatenate((eef_pos, eef_orientation), axis=0)
        check = False
        usr_input = raw_input('Continue robot motion? y/n')
        if usr_input == 'y':
            while check == False:
                check = yumi.setEEFPose(eef_cmd)

    # Divide the entries by vector length im metres to get a calibration matrix
    axes_tips = np.array(axes_tips)
    axes_tips = axes_tips / vector_length / 1000
    axes_tips = axes_tips.T
    print axes_tips

    img = dataserver.readCamera()
    rospy.sleep(1)
    # img = img[:len(camera_id)]
    img = img[-2:]

    # Get list of tuples with pin tip coordinates in the image
    tips_1 = pin_tips(img[0], limits_x=[300, 600], limits_y=[150, 300], cam=1)
    tips_2 = pin_tips(img[1], limits_x=[300, 620], limits_y=[150, 300], cam=2)
    # tips_2 = pin_tips(img[1], limits_x=[300, 500], limits_y=[150, 300])  # Not working

    pin_tip_tilda = calc_1_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2)
    pin_tip_hat = [pin_tip_tilda[0] + x_hat_c2,
                   pin_tip_tilda[1] + y_hat_c1,
                   pin_tip_tilda[2] + z_hat]
    print 'pin_tip_hat [mm]: ', pin_tip_hat
    pin_tip_hat = np.array(pin_tip_hat)
    print 'tip tilda [mm]: ', pin_tip_tilda
    pin_tip_tilda = (np.array(pin_tip_tilda)) / 1000
    # eef_pose = eefpos_to_geo_msg(position)

    pin_tip_hat_goal = np.array([174.5, 139.5, -3])

    diff_pin_tip_hat = pin_tip_hat_goal - pin_tip_hat
    diff_pin_tip_hat = (np.array(diff_pin_tip_hat)) / 1000

    yumi.getEEFState()
    eef_pos, eef_orientation = yumi.getEEFState()
    print 'initial eef_pos [m]: ', eef_pos

    # eef_pos = eef_pos - (np.dot(transform_base_eef, (np.concatenate(((pin_tip_tilda)/1000, [1]), axis=0))))[:-1]

    #eef_pos = eef_pos - np.dot(rot_tilda_to_base, pin_tip_tilda)  # Move to tilda origin
    #eef_pos = eef_pos + np.dot(rot_tilda_to_base, diff_pin_tip_hat)

    # Use calibration matrix to determine the eef command
    command = np.linalg.solve(axes_tips, diff_pin_tip_hat)
    eef_pos = eef_pos + command

    # transform_tilda_to_yumi(eef_pos, pin_tip_tilda)
    print 'updated eef_pos [m]: ', eef_pos
    print 'change in eef_pos (m): ', eef_pos - yumi.getEEFState()[0]

    eef_cmd = np.concatenate((eef_pos, eef_orientation), axis=0)
    usr_input = raw_input('Continue robot motion? y/n')
    if usr_input == 'y':
        yumi.setEEFPose(eef_cmd)

def move_to_tilda_origin(x_tilda_c1, y_tilda_c2, img, yumi):
    # Get list of tuples with pin tip coordinates in the image
    tips_1 = pin_tips(img[0], limits_x=[300, 600], limits_y=[150, 300], cam=1)
    tips_2 = pin_tips(img[1], limits_x=[300, 620], limits_y=[150, 300], cam=2)

    pin_tip_tilda = calc_1_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2)
    pin_tip_hat = [pin_tip_tilda[0] + x_hat_c2,
                   pin_tip_tilda[1] + y_hat_c1,
                   pin_tip_tilda[2] + z_hat]
    print 'pin_tip_hat [mm]: ', pin_tip_hat
    print 'tip tilda [mm]: ', pin_tip_tilda
    pin_tip_tilda = (np.array(pin_tip_tilda)) / 1000

    yumi.getEEFState()
    eef_pos, eef_orientation = yumi.getEEFState()
    print 'initial eef_pos [m]: ', eef_pos

    rot_tilda_to_base = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    eef_pos = eef_pos - np.dot(rot_tilda_to_base, pin_tip_tilda)  # Move to tilda origin

    # transform_tilda_to_yumi(eef_pos, pin_tip_tilda)
    print 'updated eef_pos [m]: ', eef_pos
    print 'change in eef_pos (m): ', eef_pos - yumi.getEEFState()[0]

    eef_cmd = np.concatenate((eef_pos, eef_orientation), axis=0)
    usr_input = raw_input('Continue robot motion? y/n')
    if usr_input == 'y':
        yumi.setEEFPose(eef_cmd)


def transform_tilda_to_yumi(x_t_tilda, x_eef):
    x_t_tilda = np.array(x_t_tilda)  # Pin tip in camera coordinates
    x_eef = np.array(x_eef)  # EEF pose at pin tip calculation
    x_eef = x_eef * 1000
    rot_tilda_to_base = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    temp = np.dot(rot_tilda_to_base, x_t_tilda)
    temp = temp - x_eef
    temp = np.array([temp])
    transform_tilda_to_yumi = np.concatenate((rot_tilda_to_base, temp.T), axis=1)
    transform_tilda_to_yumi = np.array([transform_tilda_to_yumi])
    transform_tilda_to_yumi = np.concatenate((transform_tilda_to_yumi, np.array([0, 0, 0, 1])), axis=0)
    print transform_tilda_to_yumi


def calc_1_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2):
    # Take the lowest detected tip in the image
    # Sort tip coordinates by increasing cy value
    sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
    sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
    pin_tip_tilda = get_pin_tip_location(sorted_tips_1[-1], sorted_tips_2[-1], x_tilda_c1, y_tilda_c2, B, H, L)
    return pin_tip_tilda


'''
For 2 straight pins of different length
'''


def calc_2_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2):
    if len(tips_1) == 0:
        rospy.logerr('No tips were detected by Camera 1')
    if len(tips_2) == 0:
        rospy.logerr('No tips were detected by Camera 2')

    if len(tips_1) == 2 and len(tips_2) == 1:
        rospy.logwarn('Pins obscure each other from the POV of Camera 2, THT element needs to be rotated')
    if len(tips_1) == 1 and len(tips_2) == 2:
        rospy.logwarn('Pins obscure each other from the POV of Camera 1, THT element needs to be rotated')

    if len(tips_1) > 2 or len(tips_2) > 2:
        rospy.logwarn('Too many pin tips detected')

    if len(tips_1) == 2 and len(tips_2) == 2:
        print('Both pin tips are seen by both cameras')
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        for i in range(2):
            print(get_pin_tip_location(sorted_tips_1[i], sorted_tips_2[i], x_tilda_c1, y_tilda_c2, B, H, L))


if __name__ == '__main__':
    main()
