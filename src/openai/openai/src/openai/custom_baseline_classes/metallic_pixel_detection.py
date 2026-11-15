import cv2
import numpy as np
from matplotlib import pyplot as plt
import datetime
#from yumi_openai.task_envs.yumi_tht_env import THTRLEnv


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


def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def find_pin_img_x(diff_b, diff_g, diff_r):
    diff_weighted = 0.34 * diff_b + 0.33 * diff_g + 0.33 * diff_r
    # Weight diff histogram with Gauss around the middle of the image
    l = len(diff_weighted)
    mid = l / 2
    sigma = 5
    for i in range(l):
        diff_weighted[i] = gaussian(x=i - mid, mu=0, sig=sigma) * diff_weighted[i]
    pin_location_x = np.argmax(diff_weighted)
    # save_rgb_hist(diff_b, diff_g, diff_r)
    # save_weighted_hist(diff_weighted)
    return diff_weighted, pin_location_x

def save_rgb_hist(b_counts, g_counts, r_counts, title='RGB bright pixel histogram'):
    fig = plt.figure()
    X_axis = np.arange(len(b_counts))
    plt.bar(X_axis - 0.3, b_counts, 0.2, color='b', label='B')
    plt.bar(X_axis, g_counts, 0.2, color='g', label='G')
    plt.bar(X_axis + 0.3, r_counts, 0.2, color='r', label='R')
    plt.xlabel('Image columns')
    plt.ylabel('Number of bright pixels')
    plt.legend()
    plt.title(title)
    # now = datetime.datetime.now()
    # now_str = now.strftime('%Y-%m-%dT%H%M%S-%f')
    # plt.savefig('/home/ips/Pictures/230725_mpd_histograms/' + now_str + '_rgb_hist.png')

def save_weighted_hist(weighted, title='Bright pixel weighted histogram'):
    fig = plt.figure()
    X_axis = np.arange(len(weighted))
    plt.bar(X_axis, weighted)
    plt.xlabel('Image columns')
    plt.ylabel('Number of bright pixels')
    plt.title(title)
    # now = datetime.datetime.now()
    # now_str = now.strftime('%Y-%m-%dT%H%M%S-%f')
    # plt.savefig('/home/ips/Pictures/230725_mpd_histograms/' + now_str + '_weighted_hist.png')

class MetallicPixelDetection:
    def __init__(self, init_cropped_image1, init_cropped_image2, center_tolerance_px, img_center_x):
        self.image_without_pin_cam1 = init_cropped_image1
        self.image_without_pin_cam2 = init_cropped_image2
        self.center_tolerance_px = center_tolerance_px
        self.img_center_x = img_center_x

    def mpd_diff(self, image1, image2):
        image_without_pin_cam1 = self.image_without_pin_cam1
        image_without_pin_cam2 = self.image_without_pin_cam2
        # - Get classification left/center/right for both images
        image_with_pin_cam1 = image1
        image_with_pin_cam2 = image2

        bright_counts_without_pin_cam1 = count_bright_pixels(image_without_pin_cam1)
        bright_counts_with_pin_cam1 = count_bright_pixels(image_with_pin_cam1)
        bright_counts_without_pin_cam2 = count_bright_pixels(image_without_pin_cam2)
        bright_counts_with_pin_cam2 = count_bright_pixels(image_with_pin_cam2)

        diff_g_cam1 = bright_counts_with_pin_cam1[1] - bright_counts_without_pin_cam1[1]
        diff_r_cam1 = bright_counts_with_pin_cam1[2] - bright_counts_without_pin_cam1[2]
        diff_b_cam1 = bright_counts_with_pin_cam1[0] - bright_counts_without_pin_cam1[0]
        diff_weighted_cam1, pin_x_cam1 = find_pin_img_x(diff_b_cam1, diff_g_cam1, diff_r_cam1)

        diff_g_cam2 = bright_counts_with_pin_cam2[1] - bright_counts_without_pin_cam2[1]
        diff_r_cam2 = bright_counts_with_pin_cam2[2] - bright_counts_without_pin_cam2[2]
        diff_b_cam2 = bright_counts_with_pin_cam2[0] - bright_counts_without_pin_cam2[0]
        diff_weighted_cam2, pin_x_cam2 = find_pin_img_x(diff_b_cam2, diff_g_cam2, diff_r_cam2)

        pin_dev_cam1 = pin_x_cam1 - self.img_center_x
        pin_dev_cam2 = pin_x_cam2 - self.img_center_x

        # self.center_tolerance_px = 1

        # 0: Left, 1: Right, 2: Center
        pin_location_cam1 = 0 if pin_dev_cam1 < -self.center_tolerance_px else 1 if pin_dev_cam1 > self.center_tolerance_px else 2
        pin_location_cam2 = 0 if pin_dev_cam2 < -self.center_tolerance_px else 1 if pin_dev_cam2 > self.center_tolerance_px else 2

        if pin_location_cam1 != 'Center' or pin_location_cam2 != 'Center':
            abs_error_x_cam1 = abs(pin_x_cam1 - self.img_center_x)
            abs_error_x_cam2 = abs(pin_x_cam2 - self.img_center_x)
            if abs_error_x_cam1 > abs_error_x_cam2:
                if pin_location_cam1 == 'Right':
                    recommended_action = 0  # Keyboard:S Yumi perspective: Front
                elif pin_location_cam1 == 'Left':
                    recommended_action = 1  # Keyboard:W Yumi perspective: Back
            else:
                if pin_location_cam2 == 'Right':
                    recommended_action = 2  # Keyboard:D Yumi perspective: Right
                elif pin_location_cam2 == 'Left':
                    recommended_action = 3

        return pin_location_cam1, pin_location_cam2, pin_dev_cam1, pin_dev_cam2

    def mpd(self, image1, image2):
        # - Get classification left/center/right for both images
        image_with_pin_cam1 = image1
        image_with_pin_cam2 = image2

        bright_counts_with_pin_cam1 = count_bright_pixels(image_with_pin_cam1)
        bright_counts_with_pin_cam2 = count_bright_pixels(image_with_pin_cam2)

        diff_weighted_cam1, pin_x_cam1 = find_pin_img_x(bright_counts_with_pin_cam1[0],
                                                        bright_counts_with_pin_cam1[1],
                                                        bright_counts_with_pin_cam1[2])

        diff_weighted_cam2, pin_x_cam2 = find_pin_img_x(bright_counts_with_pin_cam2[0],
                                                        bright_counts_with_pin_cam2[1],
                                                        bright_counts_with_pin_cam2[2])

        pin_dev_cam1 = pin_x_cam1 - self.img_center_x
        pin_dev_cam2 = pin_x_cam2 - self.img_center_x

        # self.center_tolerance_px = 1

        # 0: Left, 1: Right, 2: Center
        pin_location_cam1 = 0 if pin_dev_cam1 < -self.center_tolerance_px else 1 if pin_dev_cam1 > self.center_tolerance_px else 2
        pin_location_cam2 = 0 if pin_dev_cam2 < -self.center_tolerance_px else 1 if pin_dev_cam2 > self.center_tolerance_px else 2

        if pin_location_cam1 != 2 or pin_location_cam2 != 2:    # pin_location_cam1 and pin_location_cam2 != Center
            abs_error_x_cam1 = abs(pin_x_cam1 - self.img_center_x)
            abs_error_x_cam2 = abs(pin_x_cam2 - self.img_center_x)
            if abs_error_x_cam1 > abs_error_x_cam2:
                if pin_location_cam1 == 1:    # pin_location_cam1 = Right
                    recommended_action = 0  # Keyboard:S Yumi perspective: Front
                elif pin_location_cam1 == 0:    # pin_location_cam1 = Left
                    recommended_action = 1  # Keyboard:W Yumi perspective: Back
            else:
                if pin_location_cam2 == 1:    # pin_location_cam2 = Right
                    recommended_action = 2  # Keyboard:D Yumi perspective: Right
                elif pin_location_cam2 == 0:    # pin_location_cam2 = Left
                    recommended_action = 3
        else:
            recommended_action = 4

        return pin_location_cam1, pin_location_cam2, pin_dev_cam1, pin_dev_cam2

