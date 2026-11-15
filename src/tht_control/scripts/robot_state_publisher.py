#!/usr/bin/env python3

import sys
if sys.version_info[0] < 3:
    from yumi_tht_control.arm import Arm

from hexaglide_tht_control.robot import Hexaglide
import rospy
from std_msgs.msg import Int64, Float64MultiArray
from tht_control.common import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_type', type=str, help='yumi/hexaglide', default=None, required=True)
    args = parser.parse_args()
    robot_type = args.robot_type
    config_file_name, robot_type = load_config_file(robot_type)
    ros_param_path = load_ros_params(rospackage_name='tht_control',
                                     rel_path_from_package_to_file='config', yaml_file_name=config_file_name)
    eef_state_pub = rospy.get_param('/ros_param/eef_state_pub')
    joint_pub = rospy.Publisher('/joint_state_left_arm', Float64MultiArray, queue_size=10)
    eef_pub = rospy.Publisher(eef_state_pub, Float64MultiArray, queue_size=10)
    if robot_type == 'yumi':
        rospy.init_node('move_group_python_interface', anonymous=True)
        robot = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    elif robot_type == 'hexaglide':
        robot = Hexaglide()
        reference_input = input('Reference the motors? y/n')
        if reference_input == 'y':
            reference_success = robot.doReference([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
            if not reference_success:
                print('Referencing unsuccessful. Exiting...')
                exit(0)
        else:
            usr_input = input('Continue without referencing? y/n')
            if not usr_input == 'y':
                print('Exiting...')
                exit(0)
    else:
        raise NameError('Invalid robot type')

    while not rospy.is_shutdown():
        eef_position, eef_orientation = robot.getEEFState()
        eef_pose = Float64MultiArray()
        eef_pose.data = eef_position + eef_orientation
        eef_pub.publish(eef_pose)


