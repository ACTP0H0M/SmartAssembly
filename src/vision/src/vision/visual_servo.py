#!/usr/bin/env python

import rospy
import rospkg
import pickle
import cv2
import numpy as np
from vision.click_goal import GoalPixelSetter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision.vision_sensor import VisionSensor
from vision.pin_detector import pin_tips, get_pin_tip_location
from tht_control.common import *
from tht_control.utils import manhattan_distance, weighted_manhattan_distance
import std_msgs.msg

'''
This script determines the transformation between camera coordinate system and robot coordinate system
by moving a short steel wire along 3 robot axes, detecting the tip point and calculating best fit orthogonal system.
'''


class VisualServo(VisionSensor):

    def __init__(self, robot_type=None):
        config_file_name, self.robot_type = load_config_file(robot_type)
        self.ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                              yaml_file_name=config_file_name)
        self.camera_id = rospy.get_param('ros_param/camera_id')
        VisionSensor.__init__(self, cameras=self.camera_id)
        self.d1 = rospy.get_param('extrinsic_params/d1')
        self.d2 = rospy.get_param('extrinsic_params/d2')
        self.x_tilda_params = rospy.get_param('extrinsic_params/x_tilda_params')
        self.y_tilda_params = rospy.get_param('extrinsic_params/y_tilda_params')

        # Calculate or input camera positions in corner coordinates (hat)
        self.x_hat_c1 = rospy.get_param('extrinsic_params/x_hat_c1')
        self.y_hat_c1 = self.d1 + 30.0
        self.x_hat_c2 = self.d2 + 12.0
        self.y_hat_c2 = rospy.get_param('extrinsic_params/y_hat_c2')
        self.z_hat = rospy.get_param('extrinsic_params/z_hat')
        self.vertical_shift_cam1 = rospy.get_param('extrinsic_params/vertical_shift_cam1')
        self.vertical_shift_cam2 = rospy.get_param('extrinsic_params/vertical_shift_cam2')
        self.horizontal_shift_cam1 = rospy.get_param('extrinsic_params/horizontal_shift_cam1')
        self.horizontal_shift_cam2 = rospy.get_param('extrinsic_params/horizontal_shift_cam2')

        # Calculate camera positions in image centre coordinates (tilda)
        self.x_tilda_c1 = -self.x_hat_c2 + self.x_hat_c1
        self.y_tilda_c1 = 0  # per definition of image centre coordinate system
        self.x_tilda_c2 = 0  # per definition of image centre coordinate system
        self.y_tilda_c2 = -self.y_hat_c1 + self.y_hat_c2
        # x_roi[0], y_roi[0] for cam=1. For cam 2: x_roi[1], y_roi[1]

        # Set ROI
        self.x_roi = rospy.get_param('image_roi/x_roi')
        self.y_roi = rospy.get_param('image_roi/y_roi')
        self.cam_info = self.getCameraInfo()
        # Rotation matrix from Tilda frame to YuMi base frame
        self.rot_tilda_to_base = np.array(rospy.get_param('extrinsic_params/rot_tilda_to_base'))
        self.mask_1 = None
        self.mask_2 = None

    def getPinTipPixels(self, img_1, img_2):
        self.cam_img = [img_1, img_2]
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)
        return tips_1, tips_2

    def getPinTipFromImgs(self, img_1, img_2, click_pin=False, dual_pin=False):
        self.cam_img = [img_1, img_2]
        print('VisualServo: Number of images received by getPinTipFromImgs() = ', len(self.cam_img))
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)
        # print("tips1,2:", tips_1, tips_2)
        if not click_pin:
            pin_tip_tilda = self.calc_1_tip_coords(tips_1, tips_2, self.x_tilda_c1, self.y_tilda_c2)
        else:
            print("nearby pin")
            # click component pin before approach. This defines the pin that should be inserted first
            # set pin_pose_pixel_image1 to None if not needed

            img_1_with_circles = img_1.copy()
            img_2_with_circles = img_2.copy()
            for cx, cy in tips_1:
                img_1_with_circles = cv2.circle(img_1_with_circles, (cx, cy), 2, (0, 0, 255), -1)
            for cx, cy in tips_2:
                img_2_with_circles = cv2.circle(img_2_with_circles, (cx, cy), 2, (0, 0, 255), -1)
            goal_pixel_setter = GoalPixelSetter()
            while goal_pixel_setter.x is None or goal_pixel_setter.y is None:
                goal_pixel_setter.click_set_goal_in_pixels(img_1_with_circles)
            pin_pose_pixel_image1 = [(int(goal_pixel_setter.x), int(goal_pixel_setter.y))]
            mask_1 = img_1[pin_pose_pixel_image1[0][1] - 20: pin_pose_pixel_image1[0][1] + 5,
                     pin_pose_pixel_image1[0][0] - 7: pin_pose_pixel_image1[0][0] + 7]
            mask_2 = img_2[pin_pose_pixel_image1[0][1] - 20: pin_pose_pixel_image1[0][1] + 5,
                     pin_pose_pixel_image1[0][0] - 7: pin_pose_pixel_image1[0][0] + 7]
            self.mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
            self.mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2GRAY)
            goal_pixel_setter.x = None
            goal_pixel_setter.y = None
            while goal_pixel_setter.x is None or goal_pixel_setter.y is None:
                goal_pixel_setter.click_set_goal_in_pixels(img_2_with_circles)
            pin_pose_pixel_image2 = [(int(goal_pixel_setter.x), int(goal_pixel_setter.y))]
            pin_tip_tilda = self.calc_1_tip_coords_nearby_clicked_pin(tips_1, tips_2,
                                                                      pin_pose_pixel_image1,
                                                                      pin_pose_pixel_image2)
        pin_tip_hat = [pin_tip_tilda[0] + self.x_hat_c2,
                       pin_tip_tilda[1] + self.y_hat_c1,
                       pin_tip_tilda[2] + self.z_hat]
        print('pin_tip_hat [mm]: ', pin_tip_hat)
        print('pin_tip_tilda [mm]: ', pin_tip_tilda)
        pin_tip_tilda = (np.array(pin_tip_tilda)) / 1000
        pin_tip_hat = (np.array(pin_tip_tilda)) / 1000
        if dual_pin:
            pin_tip_tilda1, pin_tip_tilda2 = self.calc_2_tip_coords(tips_1, tips_2, self.x_tilda_c1, self.y_tilda_c2)
            return pin_tip_tilda1, pin_tip_tilda2
        else:
            return pin_tip_tilda, pin_tip_hat

    def selectPinTip(self, eef_pose):
        self.cam_img = self.readCamera()
        rospy.sleep(0.2)  # TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        # print('number of images', len(self.cam_img))
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)

        img_1_with_circles = self.cam_img[0].copy()
        img_2_with_circles = self.cam_img[1].copy()

        for cx, cy in tips_1:
            img_1_with_circles = cv2.circle(img_1_with_circles, (cx, cy), 2, (0, 0, 255), -1)
        for cx, cy in tips_2:
            img_2_with_circles = cv2.circle(img_2_with_circles, (cx, cy), 2, (0, 0, 255), -1)
        goal_pixel_setter = GoalPixelSetter()
        goal_pixel_setter.x = None
        goal_pixel_setter.y = None
        while goal_pixel_setter.x is None or goal_pixel_setter.y is None:
            goal_pixel_setter.click_set_goal_in_pixels(img_1_with_circles)
        pin_pose_pixel_image1 = [(int(goal_pixel_setter.x), int(goal_pixel_setter.y))]
        goal_pixel_setter.x = None
        goal_pixel_setter.y = None
        while goal_pixel_setter.x is None or goal_pixel_setter.y is None:
            goal_pixel_setter.click_set_goal_in_pixels(img_2_with_circles)
        pin_pose_pixel_image2 = [(int(goal_pixel_setter.x), int(goal_pixel_setter.y))]
        self.pin_tip_tilda = self.calc_1_tip_coords(pin_pose_pixel_image1, pin_pose_pixel_image2, self.x_tilda_c1,
                                                    self.y_tilda_c2)
        pin_tip_hat = [self.pin_tip_tilda[0] + self.x_hat_c2,
                       self.pin_tip_tilda[1] + self.y_hat_c1,
                       self.pin_tip_tilda[2] + self.z_hat]
        print('pin_tip_hat [mm]: ', pin_tip_hat)
        print('pin_tip_tilda [mm]: ', self.pin_tip_tilda)
        self.pin_tip_tilda = (np.array(self.pin_tip_tilda)) / 1000
        self.pin_tip_hat = (np.array(self.pin_tip_tilda)) / 1000
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3, ))
        print('Current EEF Pose: ', eef_pose)
        print('Updated EEF Pose: ', self.goal_eef_pose)
        print('Change in EEF Pose: ', self.goal_eef_pose - eef_pose)
        return self.goal_eef_pose, self.pin_tip_tilda

    def autoSelectPinTip(self, eef_pose, pin_side_cam2=None):
        self.cam_img = self.readCamera()
        rospy.sleep(0.2)  # TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        # print('number of images', len(self.cam_img))
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)
        # Find two lowest pin tips in both cameras
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        if len(sorted_tips_1) <= 0 or len(sorted_tips_2) <= 0:
            raise RuntimeError(
                "Error Visual Servo: No Tip found. Please check if part is in the gripper and if good lighting conditions are given.")
        two_lowest_pin_tips_1 = [sorted_tips_1[-2], sorted_tips_1[-1]]
        two_lowest_pin_tips_2 = [sorted_tips_2[-2], sorted_tips_2[-1]]
        # In cam 1 take the left pin tip
        left_pin_tip_1 = sorted(two_lowest_pin_tips_1, key=lambda tup: tup[0])[0]
        pin_pose_pixel_image1 = left_pin_tip_1
        print('VisualServo: pin_pose_pixel_image1: ', pin_pose_pixel_image1)
        # In cam 2 take the lowest pin tip or the tip on the saved side from the first auto selection
        pin_pose_pixel_image2 = None
        free_pin_pixel_image2 = None
        if pin_side_cam2 is None:
            # Save the side on which we see the lowest pin tip in cam 2 image
            higher_pin_2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[1])[0]
            lower_pin_2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[1])[1]
            if higher_pin_2[0] < lower_pin_2[0]:
                print('VisualServo: lower of the 2 pins in cam 2 is on the right')
                pin_side_cam2 = 'right'
            else:
                print('VisualServo: lower of the 2 pins in cam 2 is on the left')
                pin_side_cam2 = 'left'
            pin_pose_pixel_image2 = lower_pin_2
            free_pin_pixel_image2 = higher_pin_2
            print('VisualServo: pin_pose_pixel_image2: ', pin_pose_pixel_image2)
        else:
            if pin_side_cam2 == 'right':
                print('VisualServo: using pin tip on the right in cam 2 as selected')
                pin_pose_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[1]
                free_pin_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[0]
            elif pin_side_cam2 == 'left':
                print('VisualServo: using pin tip on the left in cam 2 as selected')
                pin_pose_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[0]
                free_pin_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[1]
            elif pin_side_cam2 == 'lowest':
                print('VisualServo: using pin tip lowest in cam 2 as selected')
                pin_pose_pixel_image2 = two_lowest_pin_tips_2[1]
                free_pin_pixel_image2 = two_lowest_pin_tips_2[1]

        # Return gripper movement command and selected pin tip tilda coordinates
        self.pin_tip_tilda = get_pin_tip_location(pin_pose_pixel_image1, pin_pose_pixel_image2, self.x_tilda_c1,
                                                  self.y_tilda_c2,
                                                  self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        # Return free pin tip tilda coordinates
        # Free means the pin that is not being brought to tilda origin
        right_pin_tip_1 = sorted(two_lowest_pin_tips_1, key=lambda tup: tup[0])[1]
        free_pin_tip_tilda = get_pin_tip_location(right_pin_tip_1, free_pin_pixel_image2, self.x_tilda_c1,
                                                  self.y_tilda_c2,
                                                  self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        print('pin_tip_tilda [mm]: ', self.pin_tip_tilda)
        self.pin_tip_tilda = (np.array(self.pin_tip_tilda)) / 1000
        free_pin_tip_tilda = (np.array(free_pin_tip_tilda)) / 1000
        self.pin_tip_hat = (np.array(self.pin_tip_tilda)) / 1000
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3, ))
        print('Current EEF Pose: ', eef_pose)
        print('Updated EEF Pose: ', self.goal_eef_pose)
        print('Change in EEF Pose: ', self.goal_eef_pose - eef_pose)
        return self.goal_eef_pose, self.pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2

    def autoSelectRightPinTip(self, eef_pose, pin_side_cam2=None, horizontal_shift_cam1=0, vertical_shift_cam1=0,
                              horizontal_shift_cam2=0, vertical_shift_cam2=0):
        self.cam_img = self.readCamera()
        rospy.sleep(0.1)  # TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        # print('number of images', len(self.cam_img))
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1, plot=True,
                          vertical_shift_cam=self.vertical_shift_cam1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2, plot=True,
                          vertical_shift_cam=self.vertical_shift_cam2)
        # Find two lowest pin tips in both cameras
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        if len(sorted_tips_1) <= 0 or len(sorted_tips_2) <= 0:
            raise RuntimeError(
                "Error Visual Servo: No Tip found. Please check if part is in the gripper and if good lighting conditions are given.")
        two_lowest_pin_tips_1 = [sorted_tips_1[-2], sorted_tips_1[-1]]
        two_lowest_pin_tips_2 = [sorted_tips_2[-2], sorted_tips_2[-1]]
        # In cam 1 take the right pin tip
        right_pin_tip_1 = sorted(two_lowest_pin_tips_1, key=lambda tup: tup[0])[1]
        pin_pose_pixel_image1 = right_pin_tip_1
        # print('VisualServo: pin_pose_pixel_image1: ', pin_pose_pixel_image1)
        # In cam 2 take the lowest pin tip or the tip on the saved side from the first auto selection
        pin_pose_pixel_image2 = None
        free_pin_pixel_image2 = None
        if pin_side_cam2 is None:
            # Save the side on which we see the lowest pin tip in cam 2 image
            higher_pin_2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[1])[0]
            lower_pin_2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[1])[1]
            if higher_pin_2[0] < lower_pin_2[0]:
                # print('VisualServo: higher of the 2 pins in cam 2 is on the left')
                pin_side_cam2 = 'left'
            else:
                # print('VisualServo: higher of the 2 pins in cam 2 is on the right')
                pin_side_cam2 = 'right'
            pin_pose_pixel_image2 = higher_pin_2
            free_pin_pixel_image2 = lower_pin_2
            # print('VisualServo: pin_pose_pixel_image2: ', pin_pose_pixel_image2)
        else:
            if pin_side_cam2 == 'right':
                # print('VisualServo: using pin tip on the right in cam 2 as selected')
                pin_pose_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[1]
                free_pin_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[0]
            elif pin_side_cam2 == 'left':
                # print('VisualServo: using pin tip on the left in cam 2 as selected')
                pin_pose_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[0]
                free_pin_pixel_image2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])[1]
            elif pin_side_cam2 == 'lowest':
                # print('VisualServo: using pin tip lowest in cam 2 as selected')
                pin_pose_pixel_image2 = two_lowest_pin_tips_2[1]
                free_pin_pixel_image2 = two_lowest_pin_tips_2[1]

        # Return gripper movement command and selected pin tip tilda coordinates
        print('Pin tip pixel coordinates image1: ', pin_pose_pixel_image1)
        print('Pin tip pixel coordinates image2: ', pin_pose_pixel_image2)

        ### TEST ###
        # pin_pose_pixel_image1 = (pin_pose_pixel_image1[0], pin_pose_pixel_image1[1] + self.vertical_shift_cam1)
        # pin_pose_pixel_image2 = (pin_pose_pixel_image2[0], pin_pose_pixel_image2[1] + self.vertical_shift_cam2)
        pin_pose_pixel_image1 = (pin_pose_pixel_image1[0] + horizontal_shift_cam1, pin_pose_pixel_image1[1] + vertical_shift_cam1)
        pin_pose_pixel_image2 = (pin_pose_pixel_image2[0] + horizontal_shift_cam2, pin_pose_pixel_image2[1] + vertical_shift_cam2)
        print('Corrected pin tip pixel coordinates image1: ', pin_pose_pixel_image1)
        print('Corrected pin tip pixel coordinates image2: ', pin_pose_pixel_image2)
        ### END TEST ###

        self.pin_tip_tilda = get_pin_tip_location(pin_pose_pixel_image1, pin_pose_pixel_image2, self.x_tilda_c1,
                                                  self.y_tilda_c2,
                                                  self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        # Return free pin tip tilda coordinates
        # Free means the pin that is not being brought to tilda origin
        left_pin_tip_1 = sorted(two_lowest_pin_tips_1, key=lambda tup: tup[0])[0]
        left_pin_tip_1 = (left_pin_tip_1[0], left_pin_tip_1[1] + self.vertical_shift_cam1)
        free_pin_pixel_image2 = (free_pin_pixel_image2[0], free_pin_pixel_image2[1] + self.vertical_shift_cam2)
        print('left_pin_tip_1: ', left_pin_tip_1)
        print('free_pin_pixel_image2', free_pin_pixel_image2)
        free_pin_tip_tilda = get_pin_tip_location(left_pin_tip_1, free_pin_pixel_image2, self.x_tilda_c1,
                                                  self.y_tilda_c2,
                                                  self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        # pin_tip_hat = [self.pin_tip_tilda[0] + self.x_hat_c2,
        #                self.pin_tip_tilda[1] + self.y_hat_c1,
        #                self.pin_tip_tilda[2] + self.z_hat]
        # print('pin_tip_hat [mm]: ', pin_tip_hat)
        print('pin_tip_tilda [mm]: ', self.pin_tip_tilda)
        print('free_pin_tip_tilda [mm]: ', free_pin_tip_tilda)
        self.pin_tip_tilda = (np.array(self.pin_tip_tilda)) / 1000
        free_pin_tip_tilda = (np.array(free_pin_tip_tilda)) / 1000
        self.pin_tip_hat = (np.array(self.pin_tip_tilda)) / 1000
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3, ))
        # print('Current EEF Pose: ', eef_pose)
        # print('Updated EEF Pose: ', self.goal_eef_pose)
        # print('Change in EEF Pose: ', self.goal_eef_pose - eef_pose)
        return self.goal_eef_pose, self.pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2

    def autoSelectDIPPinTip(self, eef_pose, n_pins_total, cam1_tip_window_px, horizontal_shift_cam1=0, vertical_shift_cam1=0,
                            horizontal_shift_cam2=0, vertical_shift_cam2=0):
        self.cam_img = self.readCamera()
        img_bottom_left_corner = (0, self.cam_info[0][1])  # cam_info[0][1]: Image pixels height
        img_bottom_right_corner = (self.cam_info[0][0], self.cam_info[0][1])    # cam_info[0][1]: Image pixels width
        print('img_bottom_right_corner', img_bottom_right_corner)
        manhattan_distance_tips_1 = []
        manhattan_distance_tips_2 = []
        rospy.sleep(0.2)  # TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1, plot=True)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2, plot=True)
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        print('sorted_tips_1: ', sorted_tips_1)
        print('sorted_tips_2: ', sorted_tips_2)
        # # Find n/2 lowest pin tips in cam1
        # if len(sorted_tips_1) > n_pins_total / 2:
        #     sorted_tips_1 = sorted_tips_1[-int(n_pins_total / 2):]
        # Find 2 lowest pin tips in cam2
        two_lowest_pin_tips_2 = [sorted_tips_2[-2], sorted_tips_2[-1]]
        # # Cam1:
        # # Calculate average height of selected pin tips in the image
        # cam1_avg_img_cy = sum(cy for _, cy in sorted_tips_1) / len(sorted_tips_1)
        # # Remove 'outliers' too far from the avg. pin tip height (e.g. reflections on the THD body)
        # filtered_tips_1 = []
        # for tip in sorted_tips_1:
        #     if abs(tip[1] - cam1_avg_img_cy) < cam1_tip_window_px:
        #         filtered_tips_1.append(tip)
        # # Pick the outermost right pin tip
        # sorted_x_tips_1 = sorted(filtered_tips_1, key=lambda tup: tup[0])
        # right_pin_tip_1 = sorted_x_tips_1[-1]
        # Pick the pin tip with the lowest Manhattan distance with image lower right corner:
        for i in range(len(sorted_tips_1)):
            manhattan_distance_tips_1.append(weighted_manhattan_distance(img_bottom_right_corner, sorted_tips_1[i]))
        right_pin_tip_1_idx = sorted(range(len(manhattan_distance_tips_1)), key=lambda k: manhattan_distance_tips_1[k])[0]
        right_pin_tip_1 = sorted_tips_1[right_pin_tip_1_idx]
        # Cam2:
        # Pick the left of the 2 lowest pin tips
        # sorted_x_tips_2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])
        # left_pin_tip_2 = sorted_x_tips_2[0]
        # Pick the pin tip with the lowest Manhattan distance with image lower left corner:
        sorted_x_tips_2 = sorted(two_lowest_pin_tips_2, key=lambda tup: tup[0])
        if weighted_manhattan_distance(img_bottom_left_corner, sorted_x_tips_2[0]) < \
                weighted_manhattan_distance(img_bottom_left_corner, sorted_x_tips_2[1]):
            left_pin_tip_2 = sorted_x_tips_2[0]
        else:
            left_pin_tip_2 = sorted_x_tips_2[1]

        # Calculate primary pin tip tilda
        print('right_pin_tip_1 before adding horizontal and vertical shifts: ', right_pin_tip_1)
        print('left_pin_tip_2 before adding horizontal and vertical shifts: ', left_pin_tip_2)
        right_pin_tip_1 = (right_pin_tip_1[0] + horizontal_shift_cam1, right_pin_tip_1[1] + vertical_shift_cam1)
        left_pin_tip_2 = (left_pin_tip_2[0] + horizontal_shift_cam2, left_pin_tip_2[1] + vertical_shift_cam2)
        print('right_pin_tip_1 after adding horizontal and vertical shifts: ', right_pin_tip_1)
        print('left_pin_tip_2 after adding horizontal and vertical shifts: ', left_pin_tip_2)
        self.pin_tip_tilda = get_pin_tip_location(right_pin_tip_1, left_pin_tip_2, self.x_tilda_c1, self.y_tilda_c2,
                                                  self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        print('pin_tip_tilda [mm]: ', self.pin_tip_tilda)
        self.pin_tip_tilda = (np.array(self.pin_tip_tilda)) / 1000
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3, ))
        return self.goal_eef_pose, self.pin_tip_tilda

    def getPinTip(self):
        self.cam_img = self.readCamera()
        rospy.sleep(0.2)  # TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        # print('number of images', len(self.cam_img))
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)
        pin_tip_tilda = self.calc_1_tip_coords(tips_1, tips_2, self.x_tilda_c1, self.y_tilda_c2)
        pin_tip_hat = [pin_tip_tilda[0] + self.x_hat_c2,
                       pin_tip_tilda[1] + self.y_hat_c1,
                       pin_tip_tilda[2] + self.z_hat]
        # print('pin_tip_hat [mm]: ', pin_tip_hat)
        # print('pin_tip_tilda [mm]: ', pin_tip_tilda)
        pin_tip_tilda = (np.array(pin_tip_tilda)) / 1000
        pin_tip_hat = (np.array(pin_tip_tilda)) / 1000
        return pin_tip_tilda, pin_tip_hat

    def calc_1_tip_coords(self, tips_1, tips_2, x_tilda_c1, y_tilda_c2):
        # Take the lowest detected tip in the image
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        # print("VisualServo cam_info: ", self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        if len(sorted_tips_1) <= 0 or len(sorted_tips_2) <= 0:
            raise RuntimeError(
                "Error Visual Servo: No Tip found. Please check if part is in the gripper and if good lighting conditions are given.")
        pin_tip_tilda = get_pin_tip_location(sorted_tips_1[-1], sorted_tips_2[-1], self.x_tilda_c1, self.y_tilda_c2,
                                             self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        print('Pixel coordinates image1: ', sorted_tips_1[-1])
        print('Pixel coordinates image2: ', sorted_tips_2[-1])
        return pin_tip_tilda

    def calc_2_tip_coords(self, tips_1, tips_2, x_tilda_c1, y_tilda_c2):
        # Take the lowest detected tip in the image
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        print("VisualServo cam_info: ", self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        if len(sorted_tips_1) <= 0 or len(sorted_tips_2) <= 0:
            raise RuntimeError(
                "Error Visual Servo: No Tip found. Please check if part is in the gripper and if good lighting conditions are given.")
        pin_tip_tilda1 = get_pin_tip_location(sorted_tips_1[-1], sorted_tips_2[-1], self.x_tilda_c1, self.y_tilda_c2,
                                              self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        pin_tip_tilda2 = get_pin_tip_location(sorted_tips_1[-2], sorted_tips_2[-2], self.x_tilda_c1, self.y_tilda_c2,
                                              self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        return pin_tip_tilda1, pin_tip_tilda2

    # Linear function for getting tilda coordinates using optimal parameters (a, b, c)
    # returns x_tilda (from image 2 w, h) and y_tilda (from image 1 w, h)
    def linear_func(self, w, h, a, b, c):
        return a * w + b * h + c

    # Square function for getting tilda coordinates using optimal parameters (a, b, c, d, e, f)
    def square_func(self, w, h, a, b, c, d, e, f):
        return a*w**2 + b*w + c*h**2 + d*h + e*w*h + f  # square function

    def calc_1_tip_coords_nearby_clicked_pin(self, tips_1, tips_2, clicked_pin_1, clicked_pin_2):
        if len(tips_1) <= 0 or len(tips_2) <= 0:
            raise RuntimeError(
                "Error Visual Servo: No Tip found. Please check if part is in the gripper and if good lighting conditions are given.")
        # Take the nearest detected tip to the clicked pin
        # todo pairwise distance: a[:, None, :] - b[None, :, :] # https://sparrow.dev/pairwise-distance-in-numpy/
        if len(clicked_pin_1) > 1:
            raise RuntimeError("distance is not correct: use pairwise distance")
        manhatten_dist1 = np.linalg.norm(np.array(tips_1) - np.array(clicked_pin_1), axis=1, ord=1)
        manhatten_dist2 = np.linalg.norm(np.array(tips_2) - np.array(clicked_pin_2), axis=1, ord=1)
        print(tips_1, "dist:", manhatten_dist1, "argmin", np.argmin(manhatten_dist1), "clicked pin:", clicked_pin_1)
        pin_tip_tilda = get_pin_tip_location(tips_1[np.argmin(manhatten_dist1)], tips_2[np.argmin(manhatten_dist2)],
                                             self.x_tilda_c1, self.y_tilda_c2,
                                             self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        return pin_tip_tilda

    def EEFtoTildaOrigin(self, eef_pose, offset=[0.0, 0.0, 0.0]):
        self.pin_tip_tilda = None
        self.pin_tip_hat = None
        self.goal_eef_pose = None
        offset = np.array([offset])
        self.cam_img = self.readCamera()
        rospy.sleep(0.2)  # TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        self.pin_tip_tilda, self.pin_tip_hat = self.getPinTipFromImgs(img_1=self.cam_img[0], img_2=self.cam_img[1])
        self.pin_tip_tilda = np.subtract(self.pin_tip_tilda, offset)
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3, ))
        print('Current EEF Pose: ', eef_pose)
        print('Updated EEF Pose: ', self.goal_eef_pose)
        print('Change in EEF Pose: ', self.goal_eef_pose - eef_pose)
        return self.goal_eef_pose

    def LeftPintoTildaOrigin(self, eef_pose, offset=[0.0, 0.0, 0.0]):
        self.pin_tip_tilda = None
        self.pin_tip_hat = None
        self.goal_eef_pose = None
        offset = np.array([offset])
        self.pin_tip_tilda, self.pin_tip_hat = self.getLeftPinTip()
        self.pin_tip_tilda = np.subtract(self.pin_tip_tilda, offset)
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3, ))
        print('Current EEF Pose: ', eef_pose)
        print('Updated EEF Pose: ', self.goal_eef_pose)
        print('Change in EEF Pose: ', self.goal_eef_pose - eef_pose)
        return self.goal_eef_pose

    def EEFtoTildaGoal(self, goal_pose_tilda, eef_pose):
        # eef_pose: Robot position only (x,y,z)
        goal_pose_eef = eef_pose - np.dot(self.rot_tilda_to_base, goal_pose_tilda.reshape(3, ))
        return goal_pose_eef

    def PinTipClosestToOrigin(self, eef_pose):
        self.cam_img = self.readCamera()
        rospy.sleep(0.2)  # TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        # print('number of images', len(self.cam_img))
        tips_1 = np.array(pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1))
        tips_2 = np.array(pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2))
        img_center = [self.cam_info[0][0] / 2, self.cam_info[0][1] / 2]
        manhatten_dist1 = np.linalg.norm(np.array(tips_1) - np.array(img_center), axis=1, ord=1)
        manhatten_dist2 = np.linalg.norm(np.array(tips_2) - np.array(img_center), axis=1, ord=1)
        dist_indices1 = np.argsort(manhatten_dist1)
        dist_indices2 = np.argsort(manhatten_dist2)
        tips_sorted1 = tips_1[dist_indices1]
        tips_sorted2 = tips_2[dist_indices2]
        tips_sorted1 = tips_sorted1[np.where(manhatten_dist1[dist_indices1] > 1.0 / 1000)]
        tips_sorted2 = tips_sorted2[np.where(manhatten_dist2[dist_indices2] > 1.0 / 1000)]
        closest_pin_to_origin1 = tips_sorted1[0]
        closest_pin_to_origin2 = tips_sorted2[0]
        self.pin_tip_tilda = get_pin_tip_location(closest_pin_to_origin1, closest_pin_to_origin2,
                                                  self.x_tilda_c1, self.y_tilda_c2,
                                                  self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])

        self.pin_tip_tilda = (np.array(self.pin_tip_tilda)) / 1000
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3, ))
        return self.goal_eef_pose, self.pin_tip_tilda
