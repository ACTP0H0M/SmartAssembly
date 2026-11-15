#!/usr/bin/env python

import glob
import os
from yumi_tht_control.arm import Arm
from tht_control.observer import Observer
from tht_control.common import *
from vision.click_goal import GoalPixelSetter
from vision.visual_servo import VisualServo
from sensor_msgs.msg import Image
import math as m
import cv2
import rospkg
import numpy as np
import rospy


def record_cropped_image(set_number=None):
    rospy.init_node('move_group_python_interface', anonymous=True)
    yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    visual_servo = VisualServo()
    cropped_img1_pub = rospy.Publisher('cropped_img1', Image, queue_size=10)
    cropped_img2_pub = rospy.Publisher('cropped_img2', Image, queue_size=10)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('tht_control')
    path = os.path.abspath(os.curdir)
    # Navigate to the file with serialized (saved/recorded) waypoints and read them
    ros_param_path = load_ros_params(rospackage_name="tht_control", rel_path_from_package_to_file='config',
                                     yaml_file_name='config_yumi.yaml')
    x1 = rospy.get_param('/image_window/x1')
    y1 = rospy.get_param('/image_window/y1')
    x2 = rospy.get_param('/image_window/x2')
    y2 = rospy.get_param('/image_window/y2')
    h = rospy.get_param('/image_window/h')
    ph = rospy.get_param('/image_window/ph')
    tildaorigin_to_pcb_z = rospy.get_param('/robot_workspace/tildaorigin_to_pcb_z')
    obs = Observer(x1, y1, x2, y2, h, ph)
    current_image1, current_image2 = obs.current_visualdata()
    goal_pixel_setter = GoalPixelSetter()
    goal_pixel_setter.click_set_goal_in_pixels(current_image1)
    goal_pose_pixel_image1 = [(int(goal_pixel_setter.x), int(goal_pixel_setter.y))]
    goal_pixel_setter.click_set_goal_in_pixels(current_image2)
    goal_pose_pixel_image2 = [(int(goal_pixel_setter.x), int(goal_pixel_setter.y))]
    init_cropped_image1, init_cropped_image2 = obs.current_cropped_visualdata(
        goal_pose_pixel_image1,
        goal_pose_pixel_image2)
    goal_pose_tilda = visual_servo.calc_1_tip_coords(goal_pose_pixel_image1,
                                                     goal_pose_pixel_image2,
                                                     visual_servo.x_tilda_c1,
                                                     visual_servo.y_tilda_c2)

    def record_data(filename=None, set_number=None):
        cropped_image1, cropped_image2 = obs.current_cropped_visualdata(goal_pose_pixel_image1,
                                                                        goal_pose_pixel_image2)
        cv2.imwrite(path + '/dataset_pin_detection/' + set_number + filename + '_cam1' + '.png',
                    cropped_image1)
        cv2.imwrite(path + '/dataset_pin_detection/' + set_number + filename + '_cam2' + '.png',
                    cropped_image2)

    def visualCalib():
        print('VISUAL CALIBRATION INITIALIZED')
        for i in range(2):
            yumi.getEEFState()
            eef_position, eef_orientation = yumi.getEEFState()
            eef_to_tilda = visual_servo.EEFtoTildaOrigin(eef_position)
            eef_cmd = np.concatenate((eef_to_tilda, eef_orientation), axis=0)
            print(eef_cmd)
            usr_input = raw_input('Continue robot motion? y/n')
            if usr_input == 'y':
                yumi.setEEFPose(eef_cmd)
            else:
                print('EXITING')
                exit(0)

        current_image1, current_image2 = obs.current_visualdata()
        current_tilda_pin_tip, _ = visual_servo.getPinTipFromImgs(current_image1, current_image2, click_pin=True)
        eef_cmd = np.dot(visual_servo.rot_tilda_to_base, (np.array(goal_pose_tilda)) / 1000 - current_tilda_pin_tip)
        eef_cmd[2] = tildaorigin_to_pcb_z if eef_cmd[2] < tildaorigin_to_pcb_z else eef_cmd[2]
        print('Commanded EEF Pose (Hole location): ', eef_cmd)
        usr_input = raw_input('Continue robot motion? y/n')
        if usr_input == 'y':
            curr_eef_pos, curr_eef_orientation = yumi.getEEFState()
            goal = curr_eef_pos + curr_eef_orientation
            goal[0] = goal[0] + eef_cmd[0]
            goal[1] = goal[1] + eef_cmd[1]
            goal[2] = goal[2] + eef_cmd[2]
            print('Goal EEF Pose', goal)
            yumi.setEEFPose(goal)
        else:
            print('EXITING')
            exit(0)

    record_data('without_pin', set_number)
    visualCalib()
    # yumi.moveAlongTCPAxis(1.0/1000, axis='z')
    # record_data('without_pin', set_number)
    # yumi.moveAlongTCPAxis(-1.0/1000, axis='z')
    record_data('with_pin', set_number)
    # yumi.moveAlongTCPAxis(1.0/1000, axis='z')
    # for i in range(10, 90, 10):
    #     yumi.rotateAlongTCPAxis(i * m.pi / 180, axis='z')
    #     record_data('without_pin_' + str(i) + '_degrees_z', set_number)
    # yumi.rotateAlongTCPAxis(-90 * m.pi / 180, axis='z')
    # yumi.moveAlongTCPAxis(-1.0/1000, axis='z')
    # for i in range(10, 90, 10):
    #     yumi.rotateAlongTCPAxis(i * m.pi / 180, axis='z')
    #     record_data('without_pin_' + str(i) + '_degrees_z', set_number)
    # yumi.rotateAlongTCPAxis(-90 * m.pi / 180, axis='z')

if __name__ == "__main__":
    set_name = 'set16/'
    data_path = './dataset_pin_detection/' + set_name
    if not os.path.isdir(data_path):
        usr_input = raw_input("folder " + data_path + " not exists! Create new set folder?")
        if usr_input == 'y':
            os.mkdir(data_path)
            print('Set dirctory created')
        else:
            print('EXITING')
            exit(0)
    record_cropped_image(set_number=set_name)
