#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt


def count_red_pixels(img):
    red_channel = img[:, :, 2]
    red_channel = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    red_counts = np.sum(red_channel>0.5, axis=0)
    return red_counts, red_channel


def display_hist(counts, title):
    plt.figure()
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Columns')
    plt.ylabel('No. of red pixels')
    plt.title(title)


def set_notgrey_pixel_values(modified_image, indices):

    height, width, _ = modified_image.shape
    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Check if the pixel index is not in the indices list
            if (x, y) not in indices:
                # Set the pixel value to 255 for red channel (red)
                modified_image[y, x] = (255, 0, 0)

    return modified_image


def get_grey_pixel_indices(img):
    indices = []
    height, width, _ = img.shape
    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Get the RGB channel values for the current pixel
            r, g, b = img[y, x]
            b=float(b)
            g=float(g)
            r=float(r)
            # todo weight the values b g r according to grey scale formular


            # Calculate the absolute differences between the channel values
            diff_rg = abs(r - g)
            diff_rb = abs(r - b)
            diff_gb = abs(g - b)

            # Check if the differences are within a threshold
            if diff_rg < 50 and diff_rb < 50 and diff_gb < 50 and r > 100:
                # Add the pixel index to the list
                indices.append((x, y))

    return indices


if __name__ == "__main__":
    data_dir = '/home/ips/smartassembly_ws/dataset_pin_detection/set16/'
    filename_without_pin_cam1 = 'without_pin_cam1.png'
    filename_with_pin_cam1 = 'with_pin_cam1.png'
    filename_without_pin_cam2 = 'without_pin_cam2.png'
    filename_with_pin_cam2 = 'with_pin_cam2.png'

    image_without_pin_cam1 = cv2.imread(data_dir + filename_without_pin_cam1)
    image_with_pin_cam1 = cv2.imread(data_dir + filename_with_pin_cam1)
    image_without_pin_cam2 = cv2.imread(data_dir + filename_without_pin_cam2)
    image_with_pin_cam2 = cv2.imread(data_dir + filename_with_pin_cam2)

    image_rgb_with_pin_cam1 = cv2.cvtColor(image_with_pin_cam1, cv2.COLOR_BGR2RGB)
    image_rgb_without_pin_cam1 = cv2.cvtColor(image_without_pin_cam1, cv2.COLOR_BGR2RGB)
    image_rgb_with_pin_cam2 = cv2.cvtColor(image_with_pin_cam2, cv2.COLOR_BGR2RGB)
    image_rgb_without_pin_cam2 = cv2.cvtColor(image_without_pin_cam2, cv2.COLOR_BGR2RGB)

    grey_only_image_rgb_with_pin_cam1 = image_rgb_with_pin_cam1.copy()
    grey_only_image_rgb_without_pin_cam1 = image_rgb_without_pin_cam1.copy()
    grey_only_image_rgb_with_pin_cam2 = image_rgb_with_pin_cam2.copy()
    grey_only_image_rgb_without_pin_cam2 = image_rgb_without_pin_cam2.copy()

    grey_only_image_rgb_with_pin_cam1 = set_notgrey_pixel_values(grey_only_image_rgb_with_pin_cam1, get_grey_pixel_indices(grey_only_image_rgb_with_pin_cam1))
    grey_only_image_rgb_without_pin_cam1 = set_notgrey_pixel_values(grey_only_image_rgb_without_pin_cam1, get_grey_pixel_indices(grey_only_image_rgb_without_pin_cam1))
    grey_only_image_rgb_with_pin_cam2 = set_notgrey_pixel_values(grey_only_image_rgb_with_pin_cam2, get_grey_pixel_indices(grey_only_image_rgb_with_pin_cam2))
    grey_only_image_rgb_without_pin_cam2 = set_notgrey_pixel_values(grey_only_image_rgb_without_pin_cam2, get_grey_pixel_indices(grey_only_image_rgb_without_pin_cam2))



    cv2.imwrite(data_dir + 'grey_only_image_rgb_with_pin_cam1.png', grey_only_image_rgb_with_pin_cam1)
    cv2.imwrite(data_dir + 'grey_only_image_rgb_without_pin_cam1.png', grey_only_image_rgb_without_pin_cam1)
    cv2.imwrite(data_dir + 'grey_only_image_rgb_with_pin_cam2.png', grey_only_image_rgb_with_pin_cam2)
    cv2.imwrite(data_dir + 'grey_only_image_rgb_without_pin_cam2.png', grey_only_image_rgb_without_pin_cam2)
