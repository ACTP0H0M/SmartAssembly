#!/usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
from tht_control.utils import *


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """

    for index in range(len(actual)):
        if abs(actual[index] - goal[index]) > tolerance:
            return False
    return True


class Arm:
    def __init__(self, arm_group='', gripper_pub=''):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot_connection = False
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.collisionBoxPose = []
        self.arm_group = moveit_commander.MoveGroupCommander(arm_group)
        self.arm_group.set_planner_id('RRTConnect')
        self.position = []
        self.orientation = []
        self.eefpos = geometry_msgs.msg.Pose()
        self.gripper_pub = rospy.Publisher(gripper_pub, std_msgs.msg.Float64, queue_size=10, latch=True)

    def connectionEstablished(self):
        if self.arm_group.get_current_pose() is not None:
            self.robot_connection = True
        else:
            self.robot_connection = False
        return self.robot_connection

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

    def setJointPose(self, joint_pose):
        self.arm_group.go(joint_pose, wait=True)

    def setEEFPose(self, eef_pose):
        self.eefpos.position.x = eef_pose[0]
        self.eefpos.position.y = eef_pose[1]
        self.eefpos.position.z = eef_pose[2]
        self.eefpos.orientation.x = eef_pose[3]
        self.eefpos.orientation.y = eef_pose[4]
        self.eefpos.orientation.z = eef_pose[5]
        self.eefpos.orientation.w = eef_pose[6]
        self.arm_group.set_pose_target(self.eefpos)
        self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.getEEFState()
        return all_close(eef_pose[0:3], self.getEEFState()[0], 0.0005)

    def pauseMovement(self):
        self.arm_group.stop()
        raw_input('Movement Paused (arm.py): Press Enter to continue...')

    def moveAlongTCPAxisCalculateNewEEF(self, goal, axis=None):
        self.getEEFState()
        _current_eef_position, _current_eef_orientation = self.getEEFState()
        if axis is not None:
            if axis == 'x':
                _current_eef_position[0] = _current_eef_position[0] + goal
            elif axis == 'y':
                _current_eef_position[1] = _current_eef_position[1] + goal
            elif axis == 'z':
                _current_eef_position[2] = _current_eef_position[2] + goal
            else:
                rospy.logerr('Invalid axis argument! Use x, y or z string arguments')
        else:
            rospy.logerr('Axis argument not provided!')

        _eef_pose = _current_eef_position + _current_eef_orientation
        return _eef_pose

    def moveAlongTCPAxis(self, goal, axis=None):
        _eef_pose = self.moveAlongTCPAxisCalculateNewEEF(goal, axis)
        return self.setEEFPose(_eef_pose)

    def rotateAlongTCPAxisCalculateNewEEF(self, goal, axis=None):
        self.getEEFState()
        _current_eef_position, _current_eef_orientation = self.getEEFState()
        _roll_x, _pitch_y, _yaw_z = euler_from_quaternion(_current_eef_orientation[0], _current_eef_orientation[1],
                                                          _current_eef_orientation[2], _current_eef_orientation[3])
        if axis is not None:
            if axis == 'x':
                _roll_x = _roll_x + goal
            elif axis == 'y':
                _pitch_y = _pitch_y + goal
            elif axis == 'z':
                _yaw_z = _yaw_z + goal
            else:
                rospy.logerr('Invalid axis argument! Use x, y or z string arguments')
        else:
            rospy.logerr('Axis argument not provided!')

        _current_eef_orientation = quaternion_from_euler(_roll_x, _pitch_y, _yaw_z)
        _eef_pose = _current_eef_position + _current_eef_orientation
        return _eef_pose

    def rotateAlongTCPAxis(self, goal, axis=None):
        _eef_pose = self.rotateAlongTCPAxisCalculateNewEEF(goal, axis)
        return self.setEEFPose(_eef_pose)

    def followWaypoints(self, waypoints):
        for wp in waypoints:
            self.setJointPose(wp)  # sets the joint poses and executes them using moveit_commander
            rospy.sleep(1)

        # for wp in waypoints[1:]:
        #     self.setJointPose(wp)  # sets the joint poses and executes them using moveit_commander
        #     rospy.sleep(1)

    def openGripper(self):
        self.gripper_pub.publish(std_msgs.msg.Float64(-5))
        rospy.sleep(1.0)

    def closeGripper(self):
        self.gripper_pub.publish(std_msgs.msg.Float64(15))
        rospy.sleep(0.3)

    def addCollisionBox(self, plane_center, box_size, obj_name='collision_object'):
        # rospy.sleep(2)
        self.collisionBoxPose.append(PoseStamped())
        self.collisionBoxPose[-1].header.frame_id = self.robot.get_planning_frame()
        self.collisionBoxPose[-1].pose.position.x = plane_center[0]
        self.collisionBoxPose[-1].pose.position.y = plane_center[1]
        self.collisionBoxPose[-1].pose.position.z = plane_center[2]
        self.scene.add_box(obj_name, self.collisionBoxPose[-1], size=(box_size[0], box_size[1], box_size[2]))

    def removeCollisionBox(self, box_name='collision_object'):
        self.scene.remove_world_object(box_name)
