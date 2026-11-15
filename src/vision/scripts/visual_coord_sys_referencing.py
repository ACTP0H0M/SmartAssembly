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


def main():
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
    y_hat_c1 = 248 - d1 - 24.5 - 12.0
    x_hat_c2 = 358.5 - d2 - 24.5 - 30.0
    y_hat_c2 = rospy.get_param('extrinsic_params/y_hat_c2')
    z_hat = rospy.get_param('extrinsic_params/z_hat')

    # Calculate camera positions in image centre coordinates (tilda)
    x_tilda_c1 = -x_hat_c2 + x_hat_c1
    y_tilda_c1 = 0  # per definition of image centre coordinate system
    x_tilda_c2 = 0  # per definition of image centre coordinate system
    y_tilda_c2 = -y_hat_c1 + y_hat_c2

    # Intrinsic parameters of the cameras
    cam_info = dataserver.getCameraInfo()
    B = cam_info[0]
    H = cam_info[1]
    L = cam_info[2]  # Fx~Fy

    # Get images synchronously
    img = dataserver.readCamera
    rospy.sleep(1)
    img = img[:len(camera_id)]

    # scale_x = rospy.get_param('image_roi/scale_x')
    # scale_y = rospy.get_param('image_roi/scale_y')
    # tips_1 = pin_tips(img[0], limits_x=[B/2-B/2*scale_x, B/2+B/2*scale_x], limits_y=[H/2-H/2*scale_y, H/2+H/2*scale_y])
    # tips_2 = pin_tips(img[1], limits_x=[B/2-B/2*scale_x, B/2+B/2*scale_x], limits_y=[H/2-H/2*scale_y, H/2+H/2*scale_y])

    # Get list of tuples with pin tip coordinates in the image
    tips_1 = pin_tips(img[0], limits_x=[400, 600], limits_y=[150, 300])
    tips_2 = pin_tips(img[1], limits_x=[400, 600], limits_y=[150, 300])
    # tips_2 = pin_tips(img[1], limits_x=[300, 500], limits_y=[150, 300])  # Not working

    pin_tip_tilda = calc_1_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2, B, H, L)
    pin_tip_hat = (pin_tip_tilda[0] + x_hat_c2,
                   pin_tip_tilda[1] + y_hat_c1,
                   pin_tip_tilda[2] + z_hat)
    print(pin_tip_hat)
    eef_to_camera_transform(pin_tip_tilda, yumi.getEEFState()[0])
    move_to_tilda_origin()


def eef_to_camera_transform(x_t_tilda, x_eef):
    x_t_tilda = np.array(x_t_tilda)     #Pin tip in camera coordinates
    x_eef = np.array(x_eef)     #EEF pose at pin tip calculation
    x_eef = x_eef*1000
    rot_tilda_to_base = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # t_base_to_eef = get_transformation_matrix(target_frame='yumi_link_6_l', source_frame='yumi_body')
    # temp = np.delete(t_base_to_eef, 3, 1)
    # # temp = np.delete(temp, 3, 0)
    # temp = np.dot(temp, rot_cam_to_base)
    print x_eef
    temp = np.dot(rot_tilda_to_base, x_t_tilda)
    tran_tilda_to_base = temp - x_eef
    transform_cam_to_base = np.concatenate((rot_tilda_to_base, tran_tilda_to_base), axis=1)
    transform_cam_to_base = np.concatenate((rot_tilda_to_base, tran_tilda_to_base), axis=1)
    print transform_cam_to_base

def move_to_tilda_origin:






def calc_1_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2, B, H, L):
    # Take the lowest detected tip in the image
    # Sort tip coordinates by increasing cy value
    sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
    sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
    pin_tip_tilda = get_pin_tip_location(sorted_tips_1[-1], sorted_tips_2[-1], x_tilda_c1, y_tilda_c2, B, H, L)
    return pin_tip_tilda


'''
For 2 straight pins of different length
'''


def calc_2_tip_coords(tips_1, tips_2, x_tilda_c1, y_tilda_c2, B, H, L):
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
