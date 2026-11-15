#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from realsense2_camera.vision_sensor import VisionSensor


# class DataServer:
#     def __init__(self):
#         self.cam_1_img_raw = Image()
#         self.cam_2_img_raw = Image()
#         self.sub_cam_1 = None
#         self.sub_cam_2 = None
#
#     def subCallback_cam1(self, msg):
#         self.cam_1_img_raw = msg
#         self.sub_cam_1.unregister()
#         print("Camera 1 image saved")
#
#     def subCallback_cam2(self, msg):
#         self.cam_2_img_raw = msg
#         self.sub_cam_2.unregister()
#         print("Camera 2 image saved")


def pin_tips(image, limits_x, limits_y):
    # Convert image to correct color space
    image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    print("image_cvt.shape: " + str(image_cvt.shape))

    # Convert image to grayscale
    gray = cv2.cvtColor(image_cvt, cv2.COLOR_RGBA2GRAY)

    # Apply a brightness threshold to create a binary image
    thresh = 200
    binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    print("binary.shape: " + str(binary.shape))

    # Apply pin tip detection kernel to the binary image and erode
    ddepth = -1
    kernel = np.array([[0, 0, 1, 2, 1, 0, 0],
                       [0, 1, 2, 3, 2, 1, 0],
                       [1, 2, 3, 4, 3, 2, 1],
                       [0, 0, 0, 0, 0, 0, 0],
                       [-19, -19, -19, -19, -19, -19, -19],
                       [-19, -19, -19, -19, -19, -19, -19],
                       [-19, -19, -19, -19, -19, -19, -19]])
    result = cv2.filter2D(binary, ddepth, kernel)
    kernel2 = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel2)
    print("result.shape: " + str(result.shape))

    # Find small contours and their centres

    keypoints = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    max_area = 5
    white_dots = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < max_area:
            # M = cv2.moments(c)
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            cx = 0
            cy = 0
            for p in c:
                cx += p[0][0]
                cy += p[0][1]
            cx = int(cx / len(c))
            cy = int(cy / len(c))
            if limits_x[0] < cx < limits_x[1] and limits_y[0] < cy < limits_y[1]:
                cv2.drawContours(keypoints, [c], -1, (36, 255, 12), 1)
                cv2.circle(keypoints, (cx, cy), 2, (0, 0, 255), -1)
                white_dots.append((cx, cy))

    print(len(white_dots))
    cv2.imshow('Contours', keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # The result is overlayed with the analyzed image for validation

    a_channel = np.ones(image_cvt.shape, dtype=float) / 2.0
    overlay = image_cvt * a_channel
    alpha = cv2.split(overlay)[3]
    mask = (np.multiply(alpha, 1.0 / 255))[:, :, np.newaxis]

    output = cv2.convertScaleAbs(image_cvt * mask + cv2.cvtColor(keypoints, cv2.COLOR_BGR2BGRA))  # * (1 - mask)
    cv2.imshow('With overlay', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return white_dots

    # For visualization purposes: overlay detected 'pin tips' on the source image
    # a_channel = np.ones(image_cvt.shape, dtype=float) / 2.0
    # overlay = image_cvt * a_channel
    # alpha = cv2.split(overlay)[3]
    # mask = (np.multiply(alpha, 1.0 / 255))[:, :, np.newaxis]
    # output = cv2.convertScaleAbs(image_cvt * mask + cv2.cvtColor(result, cv2.COLOR_GRAY2RGBA) * (1 - mask))
    # output = cv2.convertScaleAbs(image_cvt * mask + result * (1 - mask))
    # return output


'''
Given a pair of pixel coordinates from two cameras and the camera parameters, returns point coordinates in space
'''


def get_pin_tip_location(tip_coords_1, tip_coords_2, x_tilda_c1, y_tilda_c2, B, H, L):
    x1 = tip_coords_1[0]
    y1 = tip_coords_1[1]
    x2 = tip_coords_2[0]
    y2 = tip_coords_2[1]
    a = 0.0
    b = 0.0
    if x1 == B / 2:
        if x2 == B / 2:
            # point is on the z-tilda axis
            if y1 == H / 2:
                if y2 == H / 2:
                    # point is in the origin of tilda coordinate system
                    return (0.0, 0.0, 0.0)
                else:
                    print('Warning! Optical axis of cameras do not cross!')
            else:
                if y2 == H / 2:
                    print('Warning! Optical axis of cameras do not cross!')
                else:
                    # point is on the z-tilda axis but not in the origin
                    a = -x_tilda_c1 / L
                    b = -y_tilda_c2 / L
                    return (0.0, 0.0, a * (H / 2 - y1))
        else:
            # point is on the xz-tilda plane
            if y1 == H / 2:
                if y2 == H / 2:
                    # point is on the x-tilda axis
                    b = -y_tilda_c2 / L
                    return (b * (x2 - B / 2), 0.0, 0.0)
                else:
                    print('Warning! Optical axis of cameras do not cross!')
            else:
                if y2 == H / 2:
                    print('Warning! Optical axis of cameras do not cross!')
                else:
                    # point is on the xz-tilda plane but not on the x-tilda axis
                    b = -y_tilda_c2 / L
                    return (b * (x2 - B / 2), 0.0, b * (y2 - H / 2))
    else:
        if x2 == B / 2:
            # point is on the yz-tilda plane
            if y1 == H / 2:
                if y2 == H / 2:
                    # point is on the y-tilda axis
                    a = -x_tilda_c1 / L
                    return (0.0, a * (B / 2 - x1), 0.0)
                else:
                    print('Warning! Optical axis of cameras do not cross!')
            else:
                if y2 == H / 2:
                    print('Warning! Optical axis of cameras do not cross!')
                else:
                    # point is on the yz-tilda plane but not on the y-tilda axis
                    a = -x_tilda_c1 / L
                    return (0.0, a * (B / 2 - x1), a * (H / 2 - y1))
        else:
            # point is not on the xz-tilda and not on the yz-tilda planes
            if y1 == H / 2:
                if y2 == H / 2:
                    # point is on the xy-tilda plane but not on the x-tilda or y-tilda axes
                    b = (x_tilda_c1 * (B / 2 - x1) + y_tilda_c2 * L) / ((x2 - B / 2) * (B / 2 - x1) - L * L)
                    a = (y_tilda_c2 + b * L) / (B / 2 - x1)
                    return (b * (x2 - B / 2), a * (B / 2 - x1), 0.0)
                else:
                    print('Warning! Optical axis of cameras do not cross!')
            else:
                # point is also not on xy-tilda plane
                if y2 == H / 2:
                    print('Warning! Optical axis of cameras do not cross!')
                else:
                    # general case (point is not on the planes or the axes)
                    b = (x_tilda_c1 * (B / 2 - x1) + y_tilda_c2 * L) / ((x2 - B / 2) * (B / 2 - x1) - L * L)
                    a = (y_tilda_c2 + b * L) / (B / 2 - x1)
                    return (b * (x2 - B / 2), a * (B / 2 - x1), a * (H / 2 - y1))


def main():
    bridge = CvBridge()
    camera_id = ['cam_1', 'cam_2']
    rospy.init_node('pin_localization', anonymous=True)
    dataserver = DataServer(camera=camera_id)
    # dataserver.sub_cam_1 = rospy.Subscriber("/cam_1/color/image_raw", Image, dataserver.subCallback_cam1)
    # dataserver.sub_cam_2 = rospy.Subscriber("/cam_2/color/image_raw", Image, dataserver.subCallback_cam2)
    rospy.sleep(1)

    # Input distances measured to the adjustable bolts inside the profile in mm

    d1 = 0  # input
    d2 = 0  # input

    # Calculate or input camera positions in corner coordinates (hat)

    x_hat_c1 = 0  # input
    y_hat_c1 = 248 - d1 - 24.5

    x_hat_c2 = 358.5 - d2 - 24.5
    y_hat_c2 = 0  # input

    # Calculate camera positions in image centre coordinates (tilda)

    x_tilda_c1 = -x_hat_c2 + x_hat_c1
    y_tilda_c1 = 0  # per definition of image centre coordinate system

    x_tilda_c2 = 0  # per definition of image centre coordinate system
    y_tilda_c2 = -y_hat_c1 + y_hat_c2

    # Intrinsic parameters of the cameras
    B = 1280
    H = 720
    L = 646  # Fx~Fy

    img1 = bridge.imgmsg_to_cv2(dataserver.cam_1_img_raw, desired_encoding="rgb8")
    img2 = bridge.imgmsg_to_cv2(dataserver.cam_2_img_raw, desired_encoding="rgb8")

    # TODO get images from the cameras
    tips_1 = pin_tips(img1, limits_x=[300, 500], limits_y=[140, 340])
    tips_2 = pin_tips(img2, limits_x=[300, 500], limits_y=[140, 340])

    '''
    Draft for 2 straight pins
    '''

    if len(tips_1) == 0:
        print('No tips were detected by Camera 1')
    if len(tips_2) == 0:
        print('No tips were detected by Camera 2')

    if len(tips_1) == 2 and len(tips_2) == 1:
        print('Pins obscure each other from the POV of Camera 2, THT element needs to be rotated')
    if len(tips_1) == 1 and len(tips_2) == 2:
        print('Pins obscure each other from the POV of Camera 1, THT element needs to be rotated')

    if len(tips_1) > 2 or len(tips_2) > 2:
        print('Too many pin tips detected')

    if len(tips_1) == 2 and len(tips_2) == 2:
        print('Both pin tips are seen by both cameras')
        # Sort tip coordinates by increasing cy value
        sorted_tips_1 = sorted(tips_1, key=lambda tup: tup[1])
        sorted_tips_2 = sorted(tips_2, key=lambda tup: tup[1])
        for i in range(2):
            get_pin_tip_location(sorted_tips_1[i], sorted_tips_2[i], x_tilda_c1, y_tilda_c2, B, H, L)

    # cv2.imshow('Cam image 1', processImage(img1))
    # cv2.imshow('Cam image 2', processImage(img2))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
