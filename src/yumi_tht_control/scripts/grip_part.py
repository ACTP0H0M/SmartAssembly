#!/usr/bin/env python

from yumi_tht_control.arm import Arm
import rospy
import argparse

'''
This script closes or opens the gripper with argument --gripper close/open
'''
if __name__ == '__main__':
    rospy.init_node('gripper_node', anonymous=True)
    # Initialize the left YuMi arm
    yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gripper', type=str, help='close/open', default=None, required=True)
    args = parser.parse_args()
    if args.gripper == 'open':
        # rospy.sleep(5)
        yumi.openGripper()
    elif args.gripper == 'close':
        # rospy.sleep(5)
        yumi.closeGripper()
    else:
        rospy.logerr('Invalid argument! close or open is allowed')
