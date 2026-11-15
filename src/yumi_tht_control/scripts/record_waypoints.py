#!/usr/bin/env python

from yumi_tht_control.arm import Arm
import rospkg
import os, datetime
import rospy
import pickle


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def recordWaypoint():
    yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    yumi.getJointState()
    waypoints = []
    rec_home = raw_input('Record current (home pose)? y/n')
    if rec_home == 'y':
        waypoints.append(yumi.getJointState())
    check = 'y'
    while check == 'y':
        raw_input('Move the robot and press Enter to record joint value')
        print('Joint value: ', yumi.getJointState())
        waypoints.append(yumi.getJointState())
        check = raw_input("Record further waypoints? y/n")
    return waypoints


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('yumi_tht_control')
    waypoints = recordWaypoint()
    waypoint_file_path = ('%s/waypoints/%s' % (pkg_path, timeStamped('waypoints')))
    with open(waypoint_file_path, 'wb') as fp:
        pickle.dump(waypoints, fp)
        print('Done writing list into a binary file')



