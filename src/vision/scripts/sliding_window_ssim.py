#!/usr/bin/env python

import glob
import os
from yumi_tht_control.arm import Arm
from tht_control.observer import Observer
from tht_control.common import *
from vision.click_goal import GoalPixelSetter
from vision.visual_servo import VisualServo
from sensor_msgs.msg import Image
import cv2
import rospkg
import numpy as np
import rospy
import imutils
from skimage.measure import compare_ssim as compare_ssim # python2 version: method name is changing in python3
import math as m

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

    visualCalib()
    yumi.moveAlongTCPAxis(2.0/1000, axis='z')
    record_data('without_pin', set_number)
    yumi.moveAlongTCPAxis(-2.0/1000, axis='z')
    record_data('with_pin', set_number)

    return visual_servo.mask_1, visual_servo.mask_2

def save_fittest_rectangle_with_max_area(data_dir, img, img_without_pin, cam_id):
    gray_image_with_pin = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image_without_pin = cv2.cvtColor(img_without_pin, cv2.COLOR_RGB2GRAY)
    (score_cam1, diff_cam1) = compare_ssim(gray_image_without_pin, gray_image_with_pin,
                                           gaussian_weights=True, full=True)
    diff_cam1 = (diff_cam1 * 255).astype("uint8")
    thresh_cam1 = cv2.threshold(diff_cam1, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts_1 = cv2.findContours(thresh_cam1.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    cnts_1 = imutils.grab_contours(cnts_1)

    max_contour1 = cnts_1[0]
    for idx, c in enumerate(cnts_1):
        if idx > 0 and np.shape(cnts_1[idx])[0] > np.shape(cnts_1[idx - 1])[0]:
            max_contour1 = c

    # (x1, y1, w1, h1) = cv2.boundingRect(max_contour1)
    rect1 = cv2.minAreaRect(max_contour1)
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)
    cv2.drawContours(img, [box1], 0, (0, 0, 255), 2)

    print(box1)
    pt2_cam1 = ((rect1[0][0] + m.cos(rect1[2] * m.pi / 180) * (rect1[1][0] / 2)),
                (rect1[0][1] + m.sin(rect1[2] * m.pi / 180) * (rect1[1][0] / 2)))


    cv2.imwrite(data_dir + 'threshold_img_cam'+str(cam_id)+'.png', thresh_cam1)
    cv2.imwrite(data_dir + 'diff_img_cam'+str(cam_id)+'.png', diff_cam1)
    cv2.imwrite(data_dir + 'pin_img_cam'+str(cam_id)+'.png', img)
def save_grapcut(data_dir, img, cam_id):
    height, width, _ = img.shape
    left_margin_proportion = 0.05
    right_margin_proportion = 0.05
    up_margin_proportion = 0.05
    down_margin_proportion = 0.05

    boundary_rectangle = (
        int(width * left_margin_proportion),
        int(height * up_margin_proportion),
        int(width * (1 - right_margin_proportion)),
        int(height * (1 - down_margin_proportion)),
    )
    # Binarize input image
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(data_dir + 'gray_cam' + str(cam_id) + '.png', gray_image)
    #gray_image = (gray_image * 255).astype("uint8") # missing?
    binarized_image = cv2.threshold(gray_image, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cv2.imwrite(data_dir + 'binarized_cam' + str(cam_id) + '.png', binarized_image)
    # Initialize the mask with known information
    mask = np.zeros((height, width), np.uint8)
    #mask[:] = cv2.GC_PR_BGD # cv2.GC_INIT_WITH_MASK,
    #mask[binarized_image == 0] = cv2.GC_FGD

    # Arrays used by the algorithm internally
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Set the seed for reproducibility purposes
    cv2.setRNGSeed(0)
    number_of_iterations = 5

    cv2.grabCut(
        img,
        mask,
        boundary_rectangle,
        background_model,
        foreground_model,
        number_of_iterations,
        cv2.GC_INIT_WITH_RECT,#cv2.GC_INIT_WITH_MASK,
    )

    grabcut_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(
        "uint8"
    )
    cv2.imwrite(data_dir + 'grabcut_mask_cam' + str(cam_id) + '.png', grabcut_mask)
if __name__ == "__main__":
    set_name = 'set22/'
    data_path = './dataset_pin_detection/' + set_name
    if not os.path.isdir(data_path):
        usr_input = raw_input("folder " + data_path + " not exists! Create new set folder?")
        if usr_input == 'y':
            os.mkdir(data_path)
            print('Set dirctory created')
        else:
            print('EXITING')
            exit(0)
    image_mask_pin_cam1, image_mask_pin_cam2 = record_cropped_image(set_number=set_name)
    print('Mask1: ', image_mask_pin_cam1)
    print('Mask2: ', image_mask_pin_cam2)
    cv2.imwrite(data_path + 'Mask1.png', image_mask_pin_cam1)
    cv2.imwrite(data_path + 'Mask2.png', image_mask_pin_cam2)
    image_with_pin_cam1 = cv2.imread(data_path + 'with_pin_cam1.png')
    image_with_pin_cam2 = cv2.imread(data_path + 'with_pin_cam2.png')
    image_without_pin_cam1 = cv2.imread(data_path + 'without_pin_cam1.png')
    image_without_pin_cam2 = cv2.imread(data_path + 'without_pin_cam2.png')

    gray_image_with_pin_cam1 = cv2.cvtColor(image_with_pin_cam1, cv2.COLOR_BGR2GRAY)
    gray_image_mask_pin_cam1 = image_mask_pin_cam1
    gray_image_with_pin_cam2 = cv2.cvtColor(image_with_pin_cam2, cv2.COLOR_BGR2GRAY)
    gray_image_mask_pin_cam2 = image_mask_pin_cam2

    w, h = gray_image_mask_pin_cam1.shape[::-1]
    #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] # not used ssim
    methods = ['cv2.TM_CCORR_NORMED']
    for meth in methods:
        method = eval(meth)
        # Apply template Matching
        res_cam1 = cv2.matchTemplate(gray_image_with_pin_cam1, gray_image_mask_pin_cam1, method)
        min_val_cam1, max_val_cam1, min_loc_cam1, max_loc_cam1 = cv2.minMaxLoc(res_cam1)
        res_cam2 = cv2.matchTemplate(gray_image_with_pin_cam1, gray_image_mask_pin_cam1, method)
        min_val_cam2, max_val_cam2, min_loc_cam2, max_loc_cam2 = cv2.minMaxLoc(res_cam2)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left_1 = min_loc_cam1
            top_left_2 = min_loc_cam2
        else:
            top_left_1 = max_loc_cam1
            top_left_2 = max_loc_cam2

        bottom_right_1 = (top_left_1[0] + w, top_left_1[1] + h)
        bottom_right_2 = (top_left_2[0] + w, top_left_2[1] + h)
        image_with_pin_cam1_rect = cv2.rectangle(image_with_pin_cam1.copy(), top_left_1, bottom_right_1, 255, 1)
        image_with_pin_cam2_rect = cv2.rectangle(image_with_pin_cam2.copy(), top_left_2, bottom_right_2, 255, 1)
        cv2.imwrite(data_path + 'matching_results' + meth + '_cam1.png', image_with_pin_cam1_rect)
        cv2.imwrite(data_path + 'matching_results' + meth + '_cam2.png', image_with_pin_cam2_rect)

        #save_grapcut(data_path, image_with_pin_cam1.copy()[top_left_1[1]:bottom_right_1[1],
        #                        top_left_1[0]:bottom_right_1[0]], 1)
        #save_grapcut(data_path, image_with_pin_cam2.copy()[top_left_2[1]:bottom_right_2[1],
        #                        top_left_2[0]:bottom_right_2[0]], 2)

        save_fittest_rectangle_with_max_area(data_path, image_with_pin_cam1.copy()[top_left_1[1]:bottom_right_1[1],
                                                                                   top_left_1[0]:bottom_right_1[0]],
                                                     image_without_pin_cam1.copy()[top_left_1[1]:bottom_right_1[1],
                                                                                   top_left_1[0]:bottom_right_1[0]],
                                                     cam_id=1)
        save_fittest_rectangle_with_max_area(data_path, image_with_pin_cam2.copy()[top_left_2[1]:bottom_right_2[1],
                                                        top_left_2[0]:bottom_right_2[0]],
                                             image_without_pin_cam2.copy()[top_left_2[1]:bottom_right_2[1],
                                             top_left_2[0]:bottom_right_2[0]],
                                             cam_id=2)

