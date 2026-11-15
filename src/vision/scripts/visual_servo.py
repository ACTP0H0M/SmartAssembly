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
# from tht_control.arm import Arm
import std_msgs.msg

'''
This script determines the transformation between camera coordinate system and robot coordinate system
by moving a short steel wire along 3 robot axes, detecting the tip point and calculating best fit orthogonal system.
'''


class VisualServo(VisionSensor):

    def __init__(self):
        self.ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                              yaml_file_name='config_yumi.yaml')
        self.camera_id = rospy.get_param('ros_param/camera_id')
        VisionSensor.__init__(self, cameras=self.camera_id)
        self.d1 = rospy.get_param('extrinsic_params/d1')
        self.d2 = rospy.get_param('extrinsic_params/d2')

        # Calculate or input camera positions in corner coordinates (hat)
        self.x_hat_c1 = rospy.get_param('extrinsic_params/x_hat_c1')
        self.y_hat_c1 = self.d1 + 30.0
        self.x_hat_c2 = self.d2 + 12.0
        self.y_hat_c2 = rospy.get_param('extrinsic_params/y_hat_c2')
        self.z_hat = rospy.get_param('extrinsic_params/z_hat')

        # Calculate camera positions in image centre coordinates (tilda)
        self.x_tilda_c1 = -self.x_hat_c2 + self.x_hat_c1
        self.y_tilda_c1 = 0  # per definition of image centre coordinate system
        self.x_tilda_c2 = 0  # per definition of image centre coordinate system
        self.y_tilda_c2 = -self.y_hat_c1 + self.y_hat_c2
        self.x_roi = [[440, 740], [485, 840]]  # ROI along x-axis for image 1 and 2
        self.y_roi = [[150, 460], [130, 460]]  # ROI along y-axis for image 1 and 2
        self.cam_info = self.getCameraInfo()
        # print 'B: ', self.cam_info[0], ' H: ', self.cam_info[1], ' L: ', self.cam_info[2]
        # print len(self.cam_info)
        # Rotation matrix from Tilda frame to YuMi base frame
        self.rot_tilda_to_base = np.array(rospy.get_param('extrinsic_params/rot_tilda_to_base'))

    def getPinTipPixels(self, img_1, img_2):
        self.cam_img = [img_1, img_2]
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)
        return tips_1, tips_2

    def getPinTipPixels2(self):
        print("test meth")

    def getPinTipFromImgs(self, img_1, img_2):
        self.cam_img = [img_1, img_2]
        print('number of images', len(self.cam_img))
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)
        pin_tip_tilda = self.calc_1_tip_coords(tips_1, tips_2, self.x_tilda_c1, self.y_tilda_c2)
        pin_tip_hat = [pin_tip_tilda[0] + self.x_hat_c2,
                       pin_tip_tilda[1] + self.y_hat_c1,
                       pin_tip_tilda[2] + self.z_hat]
        print('pin_tip_hat [mm]: ', pin_tip_hat)
        print('tip tilda [mm]: ', pin_tip_tilda)
        pin_tip_tilda = (np.array(pin_tip_tilda)) / 1000
        pin_tip_hat = (np.array(pin_tip_tilda)) / 1000
        return pin_tip_tilda, pin_tip_hat

    def getPinTip(self):
        self.cam_img = self.readCamera()
        #rospy.sleep(1) #TODO: Set the least amount of time for sleep
        self.cam_img = self.cam_img[-2:]
        print 'number of images', len(self.cam_img)
        tips_1 = pin_tips(self.cam_img[0], limits_x=self.x_roi[0], limits_y=self.y_roi[0], cam=1)
        tips_2 = pin_tips(self.cam_img[1], limits_x=self.x_roi[1], limits_y=self.y_roi[1], cam=2)
        pin_tip_tilda = self.calc_1_tip_coords(tips_1, tips_2, self.x_tilda_c1, self.y_tilda_c2)
        pin_tip_hat = [pin_tip_tilda[0] + self.x_hat_c2,
                       pin_tip_tilda[1] + self.y_hat_c1,
                       pin_tip_tilda[2] + self.z_hat]
        print 'pin_tip_hat [mm]: ', pin_tip_hat
        print 'tip tilda [mm]: ', pin_tip_tilda
        pin_tip_tilda = (np.array(pin_tip_tilda)) / 1000
        pin_tip_hat = (np.array(pin_tip_tilda)) / 1000
        return pin_tip_tilda, pin_tip_hat

    def calc_1_tip_coords(self, tips_1, tips_2, x_tilda_c1, y_tilda_c2):
        # Take the lowest detected tip in the image
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        pin_tip_tilda = get_pin_tip_location(sorted_tips_1[-1], sorted_tips_2[-1], self.x_tilda_c1, self.y_tilda_c2,
                                             self.cam_info[0][0], self.cam_info[0][1], self.cam_info[0][2])
        return pin_tip_tilda

    def EEFtoTildaOrigin(self, eef_pose, offset=[0.0, 0.0, 0.0]):
        self.pin_tip_tilda = None
        self.pin_tip_hat = None
        self.goal_eef_pose = None
        offset = np.array([offset])
        self.pin_tip_tilda, self.pin_tip_hat = self.getPinTip()
        self.pin_tip_tilda = np.subtract(self.pin_tip_tilda, offset)
        self.goal_eef_pose = eef_pose - np.dot(self.rot_tilda_to_base, self.pin_tip_tilda.reshape(3,))
        print 'Current EEF Pose: ', eef_pose
        print 'Updated EEF Pose: ', self.goal_eef_pose
        print 'Change in EEF Pose: ', self.goal_eef_pose - eef_pose
        return self.goal_eef_pose

    def EEFtoTildaGoal(self, goal_pose_tilda, eef_pose):
        # eef_pose: Robot position only (x,y,z)
        goal_pose_eef = eef_pose - np.dot(self.rot_tilda_to_base, goal_pose_tilda.reshape(3,))
        return goal_pose_eef
