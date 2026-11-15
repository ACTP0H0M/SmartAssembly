#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import curve_fit
from tht_control.observer import Observer
from tht_control.common import *
from vision.click_goal import GoalPixelSetter

def collect_clicked_points(robot_type=None):
    # Return variable
    points = []

    # Read parameters of the Observer
    config_file_name, robot_type = load_config_file(robot_type)
    ros_param_path = load_ros_params(rospackage_name='tht_control',
                                     rel_path_from_package_to_file='config', yaml_file_name=config_file_name)
    h = rospy.get_param('/cropped_image_pixels/h')
    w = rospy.get_param('/cropped_image_pixels/w')
    eef_state_pub = rospy.get_param('/ros_param/eef_state_pub')
    obs = Observer(eef_state_pub, h, w)

    # Define points on PCB to be clicked
    point_definitions = []

    # First define coordinates w.r.t. PCB center (~H1)
    # point_definitions.append({'name': 'p1',
    #                           'tilda': (-45.0, 35.0)})
    # point_definitions.append({'name': 'p2',
    #                           'tilda': (-45.0, -35.0)})
    # point_definitions.append({'name': 'p3',
    #                           'tilda': (0.0, -35.0)})
    # point_definitions.append({'name': 'p4',
    #                           'tilda': (45.0, -35.0)})
    # point_definitions.append({'name': 'p5',
    #                           'tilda': (45.0, 35.0)})
    # point_definitions.append({'name': 'p6',
    #                           'tilda': (0.0, 35.0)})
    # point_definitions.append({'name': 'p7',
    #                           'tilda': (-20.32, 13.97)}) # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    # point_definitions.append({'name': 'p8',
    #                           'tilda': (-20.32, -13.97)}) # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    # point_definitions.append({'name': 'p9',
    #                           'tilda': (20.32, -13.97)}) # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    # point_definitions.append({'name': 'p10',
    #                           'tilda': (20.32, 13.97)}) # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    # point_definitions.append({'name': 'p11',
    #                           'tilda': (0.0, 1.27)}) # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    point_definitions.append({'name': 'p1',
                              'tilda': (-45.0, 30.0)})
    point_definitions.append({'name': 'p2',
                              'tilda': (-45.0, -30.0)})
    point_definitions.append({'name': 'p3',
                              'tilda': (0.0, -30.0)})
    point_definitions.append({'name': 'p4',
                              'tilda': (45.0, -30.0)})
    point_definitions.append({'name': 'p5',
                              'tilda': (45.0, 30.0)})
    point_definitions.append({'name': 'p6',
                              'tilda': (0.0, 30.0)})
    point_definitions.append({'name': 'p7',
                              'tilda': (-12.0, 12.0)})  # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    point_definitions.append({'name': 'p8',
                              'tilda': (-10.0, -12.0)})  # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    point_definitions.append({'name': 'p9',
                              'tilda': (9.0, -12.0)})  # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    point_definitions.append({'name': 'p10',
                              'tilda': (7.0, 12.0)})  # this one needs to be adjusted to an actual hole where a pin can be put in for clicking
    point_definitions.append({'name': 'p11',
                              'tilda': (0.0, 0.0)})  # this one needs to be adjusted to an actual hole where a pin can be put in for clicking

    # Add camera lens offset
    camera_lens_offset = (9.0, 9.0)
    for point in point_definitions:
        point['tilda'] = (point['tilda'][0] + camera_lens_offset[0], point['tilda'][1] - camera_lens_offset[1])

    # Ask user to click each of the points and save pixel coordinates from both cameras
    goal_pixel_setter = GoalPixelSetter()
    for point in point_definitions:
        print('Grabbing images from both cameras...')
        current_image1, current_image2 = obs.current_visualdata()
        print('Please click point ', point['name'], ' in both images!')
        goal_pixel_setter.click_set_goal_in_pixels(current_image1)
        goal_pixel_image1 = (int(goal_pixel_setter.x), int(goal_pixel_setter.y))
        goal_pixel_setter.click_set_goal_in_pixels(current_image2)
        goal_pixel_image2 = (int(goal_pixel_setter.x), int(goal_pixel_setter.y))
        # Point "database" is specified as ((x_tilda, y_tilda), (w1, h1), (w2, h2))
        points.append(((point['tilda'][0], point['tilda'][1]), goal_pixel_image1, goal_pixel_image2))
        print('Added point ', point['name'], ' with:')
        print('(x_tilda, y_tilda) = ', point['tilda'])
        print('(w1, h1) = \t', goal_pixel_image1)
        print('(w2, h2) = \t', goal_pixel_image2)
        print('---------------------------')

    # Return a list of point coordinates
    return points    

def func(WH, a, b, c, d, e, f):
    w,h = WH
    return a*w**2 + b*w + c*h**2 + d*h + e*w*h + f  # square function
    # return a*w + b*h + c

def fit_function_parameters(points):
    # Data to fit
    # for x_tilda regression
    w2 = np.array([p[2][0] for p in points])
    h2 = np.array([p[2][1] for p in points])
    x_tilda = np.array([p[0][0] for p in points])
    # for y_tilda_regression
    w1 = np.array([p[1][0] for p in points])
    h1 = np.array([p[1][1] for p in points])
    y_tilda = np.array([p[0][1] for p in points])

    print('Fitting function to get x_tilda from image 2 pixel coordinates...')
    popt_x_tilda, pcov_x_tilda = curve_fit(func, (w2,h2), x_tilda)
    print('Optimal parameters:')
    print(popt_x_tilda)
    print('Condition number of covariance matrix:')
    print(np.linalg.cond(pcov_x_tilda))

    print('\nFitting function to get y_tilda from image 1 pixel coordinates...')
    popt_y_tilda, pcov_y_tilda = curve_fit(func, (w1,h1), y_tilda)
    print('Optimal parameters:')
    print(popt_y_tilda)
    print('Condition number of covariance matrix:')
    print(np.linalg.cond(pcov_y_tilda))


if __name__ == '__main__':
    robot_type = 'hexaglide'
    rospy.init_node('extrinsic_calibration', anonymous=True)
    points = collect_clicked_points(robot_type)
    fit_function_parameters(points)