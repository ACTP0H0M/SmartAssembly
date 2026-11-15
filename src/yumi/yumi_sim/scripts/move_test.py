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
arm_group_r = moveit_commander.MoveGroupCommander("right_arm")
hand_group_l = moveit_commander.MoveGroupCommander("left_gripper")
hand_group_r = moveit_commander.MoveGroupCommander("right_gripper")

print("Left arm pose:")
print(arm_group_l.get_current_pose().pose)

print('Left arm joint value')
print(arm_group_l.get_current_joint_values())
print(arm_group_l.get_joints())

# print("Right arm pose:")
# print(arm_group_r.get_current_pose().pose)

rot = Rotation.from_euler('xyz', [arm_group_l.get_current_pose().pose.position.x, 0, 90], degrees=True)
rot_quat = rot.as_quat()

#waypoint1 = [[0.370329, 0.209463, 0.304738], [0.658671, 0.752032, 0.019669, 0.0145717]]
waypoint1 = [[0.564597461108, -0.000648653529038, 0.417406962563], [0.875157106126, 0.0677439229123, 0.29747875404, 0.375522557779]]

pose_target = geometry_msgs.msg.Pose()

pose_target.position.x = waypoint1[0][0]
pose_target.position.y = waypoint1[0][1]
pose_target.position.z = waypoint1[0][2]

pose_target.orientation.x = waypoint1[1][0]
pose_target.orientation.y = waypoint1[1][1]
pose_target.orientation.z = waypoint1[1][2]
pose_target.orientation.w = waypoint1[1][3]



# pose_target.orientation.w  = arm_group_l.get_current_pose().pose.orientation.w
# pose_target.orientation.x = arm_group_l.get_current_pose().pose.orientation.x
# pose_target.orientation.y = arm_group_l.get_current_pose().pose.orientation.y
# pose_target.orientation.z = arm_group_l.get_current_pose().pose.orientation.z
# pose_target.position.x = arm_group_l.get_current_pose().pose.position.x
# pose_target.position.y = arm_group_l.get_current_pose().pose.position.y
# pose_target.position.z = arm_group_l.get_current_pose().pose.position.z-0.05
#
# plan1 = arm_group_l.set_pose_target(pose_target)
# plan1 = arm_group_l.go()

# rospy.sleep(5)
# pubname = '/yumi/gripper_l_effort_cmd'
# pub = rospy.Publisher(pubname, std_msgs.msg.Float64, queue_size=10, latch=True)
# pub.publish(std_msgs.msg.Float64(5))
# rospy.sleep(5)
