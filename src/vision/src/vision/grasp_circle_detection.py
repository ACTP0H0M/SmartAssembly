#!/usr/bin/env python3

from tht_control.common import *
#from vision.record_hole_location import record_cropped_image
import cv2
import glob
import numpy as np
import imutils
import math as m
from matplotlib import pyplot as plt
#from scipy.linalg import eigh
from numpy import linalg as LA

def count_bright_pixels(img):
    blue_channel = img[:, :, 0]
    blue_channel = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    green_channel = img[:, :, 1]
    green_channel = cv2.threshold(green_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    red_channel = img[:, :, 2]
    red_channel = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blue_counts = np.sum(blue_channel > 127, axis=0)
    green_counts = np.sum(green_channel > 127, axis=0)
    red_counts = np.sum(red_channel>127, axis=0)
    return blue_counts, green_counts, red_counts, blue_channel, green_channel, red_channel

def display_hist(b_counts, g_counts, r_counts, title):
    plt.figure()
    X_axis = np.arange(len(b_counts))
    plt.bar(X_axis - 0.3, b_counts, 0.2, color='b', label='B')
    plt.bar(X_axis, g_counts, 0.2, color='g', label='G')
    plt.bar(X_axis + 0.3, r_counts, 0.2, color='r', label='R')
    plt.xlabel('Image columns')
    plt.ylabel('Number of bright pixels')
    plt.legend()
    plt.title(title)

def display_one_hist(counts, title):
    plt.figure()
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Columns')
    plt.ylabel(title)
    plt.title(title)

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def find_pin_img_x(diff_b, diff_g, diff_r):
    diff_weighted = 0.5 * diff_b + 0.5 * diff_r
    display_one_hist(diff_weighted, title='weighted diff')
    # Weight diff histogram with Gauss around the middle of the image
    l = len(diff_weighted)
    mid = l / 2
    sigma = 7.5
    for i in range(l):
        diff_weighted[i] = gaussian(x=i - mid, mu=0, sig=sigma) * diff_weighted[i]
    print('Detected pin x coordinate in the image: ', np.argmax(diff_weighted))
    pin_location_x = np.argmax(diff_weighted)
    return diff_weighted, pin_location_x
def get_image():
    for i in ["/dev/video12","/dev/video13"]: # linux command: v4l2-ctl --list-devices
    #for i in range(14):  # Try to open up to 10 camera devices
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        print("Camera %s: %d FPS, %dx%d" % (i, cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    cam = cv2.VideoCapture("/dev/video12")
    img = None
    img_counter = 0

    while True:
        img_counter = img_counter + 1
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        if img_counter > 30:
            img = frame
            break

    cam.release()
    return img
def get_intrinsics():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for i in range(16):
        img = get_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("ret", ret,"mtx", mtx,"dist", dist,"rvecs", rvecs,"tvecs", tvecs)


def get_object_position():
    #get_intrinsics() # fx=627.36686331, fy=625.65208741, cx=323.67031767, cy=242.16938535
    #exit(0)
    #record_cropped_image(set_number='set5/')
    #exit(0)
    #data_dir = '/home/ips/smartassembly_ws/dataset_pin_detection/set17/'
    # Camera Intrinsic Parameter
    fx = 627.36686331   # focal length
    fy = 625.65208741
    cx = 323.67031767   # principal focus
    cy = 242.16938535

    # Camera Extrinsic Parameter
    dist_to_object = 550    # Distance from camera to object in Z (mm)

    img = get_image()
    img_copy = img.copy()
    ROI_x = [327, 408]
    ROI_y = [256, 390]
    img = img[ROI_y[0]:ROI_y[1], ROI_x[0]:ROI_x[1]] # ROI
    # cv2.imshow("cropped img", img)
    # cv2.waitKey(0)

    img_center_x = img.shape[0] / 2
    img_center_y = img.shape[1] / 2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_cam = cv2.threshold(gray, 12, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.imshow("threshold img", thresh_cam)
    # cv2.waitKey(0)

    minDist = 20
    param1 = 50
    param2 = 5     # Smaller values give more circles
    minRadius = 5
    maxRadius = 10
    circles = cv2.HoughCircles(thresh_cam, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    circles_center = []
    u_norm = []
    v_norm = []
    x_c = []
    y_c = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        temp = 0
        for x, y, r in circles[0, :]:
            cv2.circle(img_copy, (x + ROI_x[0], y + ROI_y[0]), r, (0, 255 - temp, 0), 2)
            circles_center.append((x, y))
            u_norm.append((x + ROI_x[0] - cx) / fx)
            v_norm.append((y + ROI_y[0] - cy) / fy)
            x_c.append(dist_to_object * u_norm[-1])
            y_c.append(dist_to_object * v_norm[-1])
            temp = temp + 40 # for 6 objects approx 255/6 = 40 for a better color visualisation. Alternative use numbers

    print('Distance to capacitor in camera frame (X): ', x_c)
    print('Distance to capacitor in camera frame (Y): ', y_c)
    cv2.circle(img_copy, (int(cx), int(cy)), 2, (255, 0, 0), 2)
    cv2.imshow('img_copy', img_copy)
    cv2.waitKey(500)
    return x_c, y_c, img_copy

def get_object_orientation(src, contours):
    def angle_between(p1, p2): # not normalized
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    def get_orientation(pts, img):
        print(img)
        image= img.copy()
        sz = len(pts)
        print("szlength",sz)
        print("ptsShape", pts.shape)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]
        m = np.mean(data_pts, axis=0)
        cov_matrix = np.cov(data_pts, rowvar=False)
        eigenvalues, eigenvectors = LA.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # Select the eigenvector(s) you want to visualize
        eigenvector_to_visualize = eigenvectors[:, 0]
        # Normalize and scale the eigenvector for visualization
        scale_factor = 2  # Adjust the scale factor as needed
        eigenvector_to_visualize = eigenvector_to_visualize / np.max(eigenvector_to_visualize) * scale_factor

        # Plot the image
        plt.imshow(image, cmap='gray')

        # Overlay the scaled eigenvector
        start_point = m  # Choose a starting point for the eigenvector
        end_point = start_point + eigenvector_to_visualize
        plt.arrow(start_point[0], start_point[1], eigenvector_to_visualize[0], eigenvector_to_visualize[1], color='red', head_width=1)
        plt.show()
        return angle_between((1.0, 0.0), (eigenvector_to_visualize[0],eigenvector_to_visualize[1])) # not normalized! please check


    for i,c in enumerate(contours):

      # draw each contour only for visualization
      cv2.drawContours(src, [c], i, (0, 0, 255), 2)
      # find orientation of each shape
      print("countour",c)
      get_orientation(c,src)
def get_object_position_by_polygons():
    #get_intrinsics() # fx=627.36686331, fy=625.65208741, cx=323.67031767, cy=242.16938535
    #exit(0)
    #record_cropped_image(set_number='set5/')
    #exit(0)
    #data_dir = '/home/ips/smartassembly_ws/dataset_pin_detection/set17/'
    # Camera Intrinsic Parameter
    fx = 627.36686331   # focal length
    fy = 625.65208741
    cx = 323.67031767   # principal focus
    cy = 242.16938535

    # Camera Extrinsic Parameter
    dist_to_object = 560    # Distance from camera to object in Z (mm)

    img = get_image()
    img_copy = img.copy()
    #plt.imshow(img)
    #plt.show()
    ROI_x = [202, 298]
    ROI_y = [289, 400]
    img = img[ROI_y[0]:ROI_y[1], ROI_x[0]:ROI_x[1]]  # ROI
    # cv2.imshow("cropped img", img)
    # cv2.waitKey(0)

    img_center_x = img.shape[0] / 2
    img_center_y = img.shape[1] / 2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_cam = cv2.threshold(gray, 12, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cv2.imshow("threshold img", thresh_cam)
    cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(thresh_cam, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#CHAIN_APPROX_SIMPLE
    print("Number of objects detected:", len(contours))

    def aspect_ratio(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w) / h
        return ratio
    circles_center = []
    u_norm = []
    v_norm = []
    x_c = []
    y_c = []
    angles = []
    temp = 0
    if contours is not None:
        # loop over all the contours
        for i, cnt in enumerate(contours):
            print(cnt)
            ar = aspect_ratio(cnt)
            ar = round(ar, 3)
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 20 or w < 5: # remove big bounding boxes
                continue

            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "%.3f" % ar, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print ('Aspect Ratio of object {i + 1} =', ar)
            cv2.rectangle(img_copy, (x + ROI_x[0], y + ROI_y[0]), (x + ROI_x[0] + w, y + ROI_y[0] + h), (0, 255 - temp, 0), 2)
            hw=min(h,w)
            print("thresh_cam",thresh_cam)
            cv2.imshow("img_copy", img_copy)
            cv2.waitKey(0)
            thresh_cam_copy = cv2.cvtColor(thresh_cam.copy(),cv2.COLOR_GRAY2RGB)
            print("hw", hw)
            print("cnt",cnt)
            for ar in cnt:
                x_cnt = ar[0][0]
                y_cnt = ar[0][1]

                cv2.drawMarker(thresh_cam_copy, (x_cnt, y_cnt), (255, 0, 0), 5, 2, 1)
            cv2.imshow("thresh_cam_copy", thresh_cam_copy)
            cv2.waitKey(0)
            #get_object_orientation(img_copy[y + ROI_y[0]: y + ROI_y[0] + hw, x + ROI_x[0]:x + ROI_x[0] + hw],
            #                       [cnt[y + ROI_y[0]: y + ROI_y[0] + hw, x + ROI_x[0]:x + ROI_x[0] + hw]])
            cnt_in_boundingbox_coord = cnt.copy()
            for i in range(cnt_in_boundingbox_coord.shape[0]):
                cnt_in_boundingbox_coord[i, 0, 0] = cnt_in_boundingbox_coord[i, 0, 0] - x
                cnt_in_boundingbox_coord[i, 0, 1] = cnt_in_boundingbox_coord[i, 0, 1] - y
            print("cnt_in_boundingbox_coord", cnt_in_boundingbox_coord)
            angle = get_object_orientation(thresh_cam[y : y + hw, x :x + hw],
                                           [cnt_in_boundingbox_coord])
            circles_center.append((x, y))
            u_norm.append((x + ROI_x[0] + (w/2) - cx) / fx)
            v_norm.append((y + ROI_y[0] + (h/2) - cy) / fy)
            x_c.append(dist_to_object * u_norm[-1])
            y_c.append(dist_to_object * v_norm[-1])
            angles.append(angle)
            temp = temp + 40

        cv2.imshow("Aspect Ratio", img)
        cv2.waitKey(0)

    print('Distance to capacitor in camera frame (X): ', x_c)
    print('Distance to capacitor in camera frame (Y): ', y_c)
    cv2.circle(img_copy, (int(cx), int(cy)), 2, (255, 0, 0), 2)
    cv2.imshow('img_copy', img_copy)
    cv2.waitKey(0)
    return x_c, y_c, img_copy, angles


if __name__ == "__main__":
    x_c, y_c, img_copy, angles = get_object_position_by_polygons()



