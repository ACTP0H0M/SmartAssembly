#!/usr/bin/env python3

#from vision.record_hole_location import record_cropped_image
import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim
import math as m
from matplotlib import pyplot as plt


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

if __name__ == "__main__":

    #record_cropped_image(set_number='set5/')
    #exit(0)
    data_dir = '/home/ips/smartassembly_ws/dataset_pin_detection/set17/'
    filename_without_pin_cam1 = 'without_pin_cam1.png'
    filename_with_pin_cam1 = 'with_pin_cam1.png'
    filename_without_pin_cam2 = 'without_pin_cam2.png'
    filename_with_pin_cam2 = 'with_pin_cam2.png'

    image_without_pin_cam1 = cv2.imread(data_dir + filename_without_pin_cam1)
    image_with_pin_cam1 = cv2.imread(data_dir + filename_with_pin_cam1)
    image_without_pin_cam2 = cv2.imread(data_dir + filename_without_pin_cam2)
    image_with_pin_cam2 = cv2.imread(data_dir + filename_with_pin_cam2)

    img_center_x = image_with_pin_cam1.shape[0] / 2
    img_center_y = image_with_pin_cam1.shape[1] / 2

    b_with_pin_cam1, g_with_pin_cam1, r_with_pin_cam1 = cv2.split(image_with_pin_cam1)
    b_without_pin_cam1, g_without_pin_cam1, r_without_pin_cam1 = cv2.split(image_without_pin_cam1)
    b_with_pin_cam2, g_with_pin_cam2, r_with_pin_cam2 = cv2.split(image_with_pin_cam2)
    b_without_pin_cam2, g_without_pin_cam2, r_without_pin_cam2 = cv2.split(image_without_pin_cam2)

    # image_rgb_with_pin_cam1 = cv2.threshold(r_with_pin_cam1, 0, 255,
    #                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # image_rgb_without_pin_cam1 = cv2.threshold(r_without_pin_cam1, 0, 255,
    #                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # image_rgb_with_pin_cam2 = cv2.threshold(r_with_pin_cam2, 0, 255,
    #                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # image_rgb_without_pin_cam2 = cv2.threshold(r_without_pin_cam2, 0, 255,
    #                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #
    # image_rgb_with_pin_cam1 = cv2.cvtColor(image_with_pin_cam1, cv2.COLOR_BGR2RGB)
    # image_rgb_without_pin_cam1 = cv2.cvtColor(image_without_pin_cam1, cv2.COLOR_BGR2RGB)
    # image_rgb_with_pin_cam2 = cv2.cvtColor(image_with_pin_cam2, cv2.COLOR_BGR2RGB)
    # image_rgb_without_pin_cam2 = cv2.cvtColor(image_without_pin_cam2, cv2.COLOR_BGR2RGB)

    # Set camera to analyze (1 or 2)
    cam_idx = 1

    # Count bright pixels column-wise for each color channel
    bright_counts_without_pin = []
    bright_counts_with_pin = []

    if cam_idx == 1:
        bright_counts_without_pin = count_bright_pixels(image_without_pin_cam1)
        bright_counts_with_pin = count_bright_pixels(image_with_pin_cam1)
    elif cam_idx == 2:
        bright_counts_without_pin = count_bright_pixels(image_without_pin_cam2)
        bright_counts_with_pin = count_bright_pixels(image_with_pin_cam2)

    # Recalculate diff histogram from R, G, B to a single value (red and blue with higher importance)
    diff_b = bright_counts_with_pin[0] - bright_counts_without_pin[0]
    diff_g = bright_counts_with_pin[1] - bright_counts_without_pin[1]
    diff_r = bright_counts_with_pin[2] - bright_counts_without_pin[2]
    diff_weighted = find_pin_img_x(diff_b, diff_g, diff_r)


    #max_count_column = np.argmax(bright_counts_diff_cam1)
    #print(max_count_column)

    display_hist(bright_counts_without_pin[0],
                 bright_counts_without_pin[1],
                 bright_counts_without_pin[2],
                 'bright_counts_without_pin_cam'+str(cam_idx))
    display_hist(bright_counts_with_pin[0],
                 bright_counts_with_pin[1],
                 bright_counts_with_pin[2],
                 'bright_counts_with_pin_cam1'+str(cam_idx))
    display_hist(diff_b,
                 diff_g,
                 diff_r,
                 'bright_counts_diff_cam'+str(cam_idx))
    display_one_hist(diff_weighted, 'weighted diff with Gauss cam '+str(cam_idx))
    #display_hist(bright_counts_diff_cam1, 'bright_counts_diff')
    plt.show()

    # Save thresholded
    # cv2.imwrite(data_dir + 'blue_thresh_with_pin_cam1.png', bright_counts_with_pin_cam1[3])
    # cv2.imwrite(data_dir + 'blue_thresh_without_pin_cam1.png', bright_counts_without_pin_cam1[3])
    # cv2.imwrite(data_dir + 'green_thresh_with_pin_cam1.png', bright_counts_with_pin_cam1[4])
    # cv2.imwrite(data_dir + 'green_thresh_without_pin_cam1.png', bright_counts_without_pin_cam1[4])
    # cv2.imwrite(data_dir + 'red_thresh_with_pin_cam1.png', bright_counts_with_pin_cam1[5])
    # cv2.imwrite(data_dir + 'red_thresh_without_pin_cam1.png', bright_counts_without_pin_cam1[5])
    exit(0)

    # image_without_pin_cam1 = cv2.cvtColor(image_without_pin_cam1, cv2.COLOR_BGR2HSV)
    # image_with_pin_cam1 = cv2.cvtColor(image_with_pin_cam1, cv2.COLOR_BGR2HSV)
    # image_without_pin_cam2 = cv2.cvtColor(image_without_pin_cam2, cv2.COLOR_BGR2HSV)
    # image_with_pin_cam2 = cv2.cvtColor(image_with_pin_cam2, cv2.COLOR_BGR2HSV)

    gray_image_with_pin_cam1 = cv2.cvtColor(image_with_pin_cam1, cv2.COLOR_BGR2GRAY)
    histr = cv2.calcHist([image_with_pin_cam1], [0], None, [256], [0, 256])
    color = ('b', 'g', 'r')
    plt.plot(histr)
    plt.show()
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([image_with_pin_cam1], [i], None, [256], [0, 256])
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 256])
    # plt.show()
    # (score_cam1, diff_g_cam1) = compare_ssim(g_without_pin_cam1, g_with_pin_cam1, gaussian_weights=False, full=True)
    # (score_cam2, diff_g_cam2) = compare_ssim(g_without_pin_cam2, g_with_pin_cam2, gaussian_weights=False, full=True)
    #
    # (score_cam1, diff_r_cam1) = compare_ssim(r_without_pin_cam1, r_with_pin_cam1, gaussian_weights=False, full=True)
    # (score_cam2, diff_r_cam2) = compare_ssim(r_without_pin_cam2, r_with_pin_cam2, gaussian_weights=False, full=True)

    # cv2.imshow("Image1", g_without_pin_cam1)
    # cv2.imshow("Image2", g_without_pin_cam2)
    # cv2.imshow("Image3", r_without_pin_cam1)
    # cv2.imshow("Image4", r_without_pin_cam2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(data_dir + 'gray_image_with_pin_cam1.png', gray_image_with_pin_cam1)
    # cv2.imwrite(data_dir + 'image_hsv_with_pin_cam2.png', image_with_pin_cam2)
    # cv2.imwrite(data_dir + 'g_with_pin_cam1.png', g_with_pin_cam1)
    # cv2.imwrite(data_dir + 'r_with_pin_cam1.png', r_with_pin_cam1)
    # cv2.imwrite(data_dir + 'g_with_pin_cam2.png', g_with_pin_cam2)
    # cv2.imwrite(data_dir + 'r_with_pin_cam2.png', r_with_pin_cam2)
    exit(0)



    gray_image_without_pin_cam1 = cv2.cvtColor(image_without_pin_cam1, cv2.COLOR_BGR2GRAY)
    gray_image_with_pin_cam1 = cv2.cvtColor(image_with_pin_cam1, cv2.COLOR_BGR2GRAY)
    gray_image_without_pin_cam2 = cv2.cvtColor(image_without_pin_cam2, cv2.COLOR_BGR2GRAY)
    gray_image_with_pin_cam2 = cv2.cvtColor(image_with_pin_cam2, cv2.COLOR_BGR2GRAY)



    (score_cam1, diff_cam1) = compare_ssim(gray_image_without_pin_cam1, gray_image_with_pin_cam1, gaussian_weights=True, full=True)
    (score_cam2, diff_cam2) = compare_ssim(gray_image_without_pin_cam2, gray_image_with_pin_cam2, gaussian_weights=True, full=True)

    diff_cam1 = (diff_cam1 * 255).astype("uint8")
    diff_cam2 = (diff_cam2 * 255).astype("uint8")

    thresh_cam1 = cv2.threshold(diff_cam1, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh_cam2 = cv2.threshold(diff_cam2, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts_1 = cv2.findContours(thresh_cam1.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    cnts_1 = imutils.grab_contours(cnts_1)
    cnts_2 = cv2.findContours(thresh_cam2.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    cnts_2 = imutils.grab_contours(cnts_2)

    max_contour1 = cnts_1[0]
    max_contour2 = cnts_2[0]
    for idx, c in enumerate(cnts_1):
        if idx > 0 and np.shape(cnts_1[idx])[0] > np.shape(cnts_1[idx - 1])[0]:
            max_contour1 = c
    for idx, c in enumerate(cnts_2):
        if idx > 0 and np.shape(cnts_2[idx])[0] > np.shape(cnts_2[idx - 1])[0]:
            max_contour2 = c

    # (x1, y1, w1, h1) = cv2.boundingRect(max_contour1)
    rect1 = cv2.minAreaRect(max_contour1)
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)
    cv2.drawContours(image_with_pin_cam1, [box1], 0, (0, 0, 255), 2)
    rect2 = cv2.minAreaRect(max_contour2)
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)
    cv2.drawContours(image_with_pin_cam2, [box2], 0, (0, 0, 255), 2)

    print(box1)
    pt2_cam1 = ((rect1[0][0]+m.cos(rect1[2]*m.pi/180)*(rect1[1][0]/2)),
                (rect1[0][1]+m.sin(rect1[2]*m.pi/180)*(rect1[1][0]/2)))
    pt2_cam2 = ((rect2[0][0]+m.cos(rect2[2]*m.pi/180)*(rect2[1][0]/2)),
                (rect2[0][1]+m.sin(rect1[2]*m.pi/180)*(rect2[1][0]/2)))

    print(pt2_cam1, pt2_cam2)
    # # (x2, y2, w2, h2) = cv2.boundingRect(max_contour2)
    # (x2, y2, w2, h2) = cv2.minAreaRect(max_contour2)
    # pt1_cam1 = (int(x1 + w1 / 2), y1)
    # pt2_cam1 = (int(x1 + w1 / 2), y1 + h1)
    # pt1_cam2 = (int(x2 + w2 / 2), y2)
    # pt2_cam2 = (int(x2 + w2 / 2), y2 + h2)

    # pin_location_cam1 = 'Left' if pt2_cam1[0] < img_center_x - 2 else 'Right' if pt2_cam1[0] > img_center_x + 2 else 'Center'
    # pin_location_cam2 = 'Left' if pt2_cam2[0] < img_center_x - 2 else 'Right' if pt2_cam2[0] > img_center_x + 2 else 'Center'

    # print('Pin location cam1:', pin_location_cam1)
    # print('Pin location cam2:', pin_location_cam2)

    # print(pt2_cam1)
    # print(pt2_cam2)
    # cv2.line(image_with_pin_cam1, pt1_cam1, pt2_cam1, (0, 255, 0), 1)
    # cv2.line(image_with_pin_cam2, pt1_cam2, pt2_cam2, (0, 255, 0), 1)
    #
    cv2.imwrite(data_dir + 'threshold_img_cam1.png', thresh_cam1)
    cv2.imwrite(data_dir + 'diff_img_cam1.png', diff_cam1)
    cv2.imwrite(data_dir + 'pin_img_cam1.png', image_with_pin_cam1)
    cv2.imwrite(data_dir + 'threshold_img_cam2.png', thresh_cam2)
    cv2.imwrite(data_dir + 'diff_img_cam2.png', diff_cam2)
    cv2.imwrite(data_dir + 'pin_img_cam2.png', image_with_pin_cam2)
    #
    # cv2.imshow("Image1", image_with_pin_cam1)
    # cv2.imshow("Image2", image_with_pin_cam2)
    # cv2.imshow("Diff", diff_cam2)
    # cv2.imshow("Thresh", thresh_cam2)
    cv2.waitKey(0)



