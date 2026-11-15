#! /usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
from math import pi
from std_msgs.msg import String

from moveit_commander.conversions import pose_to_list
from scipy.spatial.transform import Rotation

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

arm_group_l = moveit_commander.MoveGroupCommander("left_arm")
hand_group_l = moveit_commander.MoveGroupCommander("left_gripper")

# print("============ Printing robot state")
# print(arm_group_l.get_current_pose().pose)
# print(arm_group_l.get_current_pose().pose.position.x)
rot = Rotation.from_euler('xyz', [180, 0, 90], degrees=True)
rot_quat = rot.as_quat()

# pose_target = geometry_msgs.msg.Pose()
# pose_target.orientation.w = rot_quat[3]
# pose_target.orientation.x = rot_quat[0]
# pose_target.orientation.y = rot_quat[1]
# pose_target.orientation.z = rot_quat[2]
# pose_target.position.x = 0.35
# pose_target.position.y = 0
# pose_target.position.z = arm_group_l.get_current_pose().pose.position.z-0.001
#
# plan1 = arm_group_l.set_pose_target(pose_target)
# plan1 = arm_group_l.go()

pubname = '/yumi/gripper_l_effort_cmd'
pub = rospy.Publisher(pubname, std_msgs.msg.Float64, queue_size=10, latch=True)
pub.publish(std_msgs.msg.Float64(0.1))
rospy.sleep(5)
# scene.add_cylinder("cylinder",)
