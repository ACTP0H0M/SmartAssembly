#!/usr/bin/env python

from yumi_tht_control.arm import Arm
from tht_control.common import *
import rospy
from std_msgs.msg import Int64, Float64MultiArray


if __name__ == '__main__':
    rospy.init_node('move_group_python_interface', anonymous=True)
    yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                     yaml_file_name='config_yumi.yaml')
    translation_stepsize = rospy.get_param('/env_param/translation_step_mm')
    rotation_stepsize = rospy.get_param('/env_param/rotation_step_deg')
    while not rospy.is_shutdown():
        msg = rospy.wait_for_message('/action', Int64, timeout=None)
        action = msg.data
        # save new eef position
        if action == 0:
            estimated_new_eef = yumi.moveAlongTCPAxisCalculateNewEEF(translation_stepsize, axis='x')
        elif action == 1:
            estimated_new_eef = yumi.moveAlongTCPAxisCalculateNewEEF(-translation_stepsize, axis='x')
        elif action == 2:
            estimated_new_eef = yumi.moveAlongTCPAxisCalculateNewEEF(translation_stepsize, axis='y')
        elif action == 3:
            estimated_new_eef = yumi.moveAlongTCPAxisCalculateNewEEF(-translation_stepsize, axis='x')
        elif action == 4:
            estimated_new_eef = yumi.moveAlongTCPAxisCalculateNewEEF(translation_stepsize, axis='z')
        elif action == 5:
            estimated_new_eef = yumi.moveAlongTCPAxisCalculateNewEEF(-translation_stepsize, axis='z')
        elif action == 6:
            estimated_new_eef = yumi.rotateAlongTCPAxisCalculateNewEEF(rotation_stepsize, axis='x')
        elif action == 7:
            estimated_new_eef = yumi.rotateAlongTCPAxisCalculateNewEEF(-rotation_stepsize, axis='x')
        elif action == 8:
            estimated_new_eef = yumi.rotateAlongTCPAxisCalculateNewEEF(rotation_stepsize, axis='y')
        elif action == 9:
            estimated_new_eef = yumi.rotateAlongTCPAxisCalculateNewEEF(-rotation_stepsize, axis='y')
        elif action == 10:
            estimated_new_eef = yumi.rotateAlongTCPAxisCalculateNewEEF(rotation_stepsize, axis='z')
        elif action == 11:
            estimated_new_eef = yumi.rotateAlongTCPAxisCalculateNewEEF(-rotation_stepsize, axis='z')
        else:
            raise ValueError('Action is not known. Provide action equal or less 11')

