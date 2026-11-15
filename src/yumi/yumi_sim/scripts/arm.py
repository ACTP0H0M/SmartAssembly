#! /usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
from math import pi
from moveit_commander.conversions import pose_to_list
from scipy.spatial.transform import Rotation


class Arm:
    def __init__(self, arm_group='', gripper_pub=''):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander(arm_group)
        self.position = []
        self.orientation = []
        self.waypoints = []
        self.eefpos = []
        self.gripper_pub = rospy.Publisher(self.gripper_pub, std_msgs.msg.Float64, queue_size=10, latch=True)

    def getJointState(self):
        return self.arm_group.get_current_joint_values()

    def getEEFState(self):
        self.position = [self.arm_group.get_current_pose().pose.position.x,
                         self.arm_group.get_current_pose().pose.position.y,
                         self.arm_group.get_current_pose().pose.position.z]
        self.orientation = [self.arm_group.get_current_pose().pose.orientation.x,
                            self.arm_group.get_current_pose().pose.orientation.y,
                            self.arm_group.get_current_pose().pose.orientation.z,
                            self.arm_group.get_current_pose().pose.orientation.w]
        return self.position, self.orientation

    def setJointState(self, joint_pose):
        self.arm_group.go()

    def getWayPoint(self):
        self.waypoints.append(self.getJointState())

    def openGripper(self):
        self.gripper_pub.publish(std_msgs.msg.Float64(-5))
        rospy.sleep(2)

    def closeGripper(self):
        self.gripper_pub.publish(std_msgs.msg.Float64(5))
        rospy.sleep(2)
