#!/usr/bin/env python

from yumi_tht_control.arm import Arm
import rospy
from std_msgs.msg import Int64, Float64MultiArray, Bool
from tht_control.common import *
import numpy as np
import math as m
import argparse
import warnings


class Visual_servo_cmd():
    def __init__(self, robot_type=None):
        config_file_name, self.robot_type = load_config_file(robot_type)
        ros_param_path = load_ros_params(rospackage_name="tht_control", rel_path_from_package_to_file='config',
                                         yaml_file_name=config_file_name)
        self.translation_stepsize = rospy.get_param('/env_param/translation_step_mm') / 1000.0
        self.rotation_stepsize = rospy.get_param('/env_param/rotation_step_deg') * m.pi / 180.0
        self.approach_success_pub = None
        self.action_dict = rospy.get_param('/env_param/action_dict')
        self.robot_workspace_x = rospy.get_param('/robot_workspace/x_limit')
        self.robot_workspace_y = rospy.get_param('/robot_workspace/y_limit')
        print('VISUAL SERVO SUBSCRIBER: ', self.robot_workspace_x)
        print('VISUAL SERVO SUBSCRIBER: ', type(self.robot_workspace_x))
        if robot_type == 'yumi':
            rospy.init_node('move_group_python_interface', anonymous=True)
            self.robot = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
        elif robot_type == 'hexaglide':
            '''
            Implement hexaglide controller handler for hexaglide
            '''
            raise NotImplementedError

    def visual_servo_cmd_callback(self, eef_cmd):
        """
        First 3 components of eef_cmd are treated as desired changes in x,y,z position coordinates
        Remaining 4 components of eef_cmd are the new desired quaternion (not its change!)
        """
        eef_cmd = eef_cmd.data
        print("Visual_servo_cmd: change in x y z : ", eef_cmd)
        if abs(eef_cmd[0]) > 0.06 or abs(eef_cmd[1]) > 0.06 or abs(eef_cmd[2]) > 0.06:
            rospy.logerr("Abort recommended, because change in x,y,z is too high (bigger than 6cm!)")
            # warnings.warn("Abort recommended, because change in x,y,z is too high (bigger than 6cm!)")
        self.robot.getEEFState()
        eef_position, eef_orientation = self.robot.getEEFState()
        print("Visual_servo_cmd: current eef pose : ", eef_position)
        # First 3 components are treated as required changes in x,y,z position coordinates
        eef_position[0] = eef_position[0] + eef_cmd[0]
        eef_position[1] = eef_position[1] + eef_cmd[1]
        eef_position[2] = eef_position[2] + eef_cmd[2]
        # Remaining 4 components are treated as new desired orientation (quaternion)
        if len(eef_cmd) > 3:
            eef_orientation[0] = eef_cmd[3]
            eef_orientation[1] = eef_cmd[4]
            eef_orientation[2] = eef_cmd[5]
            eef_orientation[3] = eef_cmd[6]

        if len(eef_cmd) == 8:
            if not bool(eef_cmd[7]):
                usr_input = 'y'
            else:
                usr_input = raw_input('Continue robot motion? y/n: ')
        else:
            usr_input = raw_input('Continue robot motion? y/n: ')

        eef_cmd = np.concatenate((eef_position, eef_orientation), axis=0)
        print("Expected new position: ", eef_cmd)

        approach_success_message = Bool()
        if usr_input == 'y':
            approach_success_message.data = self.robot.setEEFPose(eef_cmd)
            # approach_success_message.data = True
            self.approach_success_pub.publish(approach_success_message)
            print('')
            print("Approach successful! Switch to RL Training")
        else:
            approach_success_message.data = False
            self.approach_success_pub.publish(approach_success_message)
            print("Approach failed!")

    def action_callback(self, msg):
        action = msg.data
        # save new eef position
        self.robot.getEEFState()
        current_eef_pose, current_eef_orientaion = self.robot.getEEFState()
        estimated_new_eef = None
        print('Translation step size: ', self.translation_stepsize)
        print('Rotation step size: ', self.rotation_stepsize)
        print('RL ACTION: ', self.action_dict[str(action)])
        if action == 0:
            estimated_new_eef = self.robot.moveAlongTCPAxisCalculateNewEEF(self.translation_stepsize, axis='x')
        elif action == 1:
            estimated_new_eef = self.robot.moveAlongTCPAxisCalculateNewEEF(-self.translation_stepsize, axis='x')
        elif action == 2:
            estimated_new_eef = self.robot.moveAlongTCPAxisCalculateNewEEF(self.translation_stepsize, axis='y')
        elif action == 3:
            estimated_new_eef = self.robot.moveAlongTCPAxisCalculateNewEEF(-self.translation_stepsize, axis='y')
        elif action == 4:
            estimated_new_eef = self.robot.moveAlongTCPAxisCalculateNewEEF(self.translation_stepsize, axis='z')
        elif action == 5:
            estimated_new_eef = self.robot.moveAlongTCPAxisCalculateNewEEF(-self.translation_stepsize, axis='z')
        elif action == 6:
            estimated_new_eef = self.robot.rotateAlongTCPAxisCalculateNewEEF(-self.rotation_stepsize, axis='z')
        elif action == 7:
            estimated_new_eef = self.robot.rotateAlongTCPAxisCalculateNewEEF(self.rotation_stepsize, axis='z')
        elif action == 8:
            estimated_new_eef = self.robot.rotateAlongTCPAxisCalculateNewEEF(self.rotation_stepsize, axis='x')
        elif action == 9:
            estimated_new_eef = self.robot.rotateAlongTCPAxisCalculateNewEEF(-self.rotation_stepsize, axis='x')
        elif action == 10:
            estimated_new_eef = self.robot.rotateAlongTCPAxisCalculateNewEEF(self.rotation_stepsize, axis='y')
        elif action == 11:
            estimated_new_eef = self.robot.rotateAlongTCPAxisCalculateNewEEF(-self.rotation_stepsize, axis='y')
        elif action == 999:
            approach_success_message = Bool()
            approach_success_message.data = False
            self.approach_success_pub.publish(approach_success_message)
            # pass # this will skip the following code in this case
        else:
            raise ValueError('Action is not known. Provide action equal or less 11')
        action_success_message = Bool()
        workspace_x_violation = False if current_eef_pose[0] + self.robot_workspace_x[0] < estimated_new_eef[0] < \
                                         current_eef_pose[0] + self.robot_workspace_x[1] else True
        workspace_y_violation = False if current_eef_pose[1] + self.robot_workspace_y[0] < estimated_new_eef[1] < \
                                         current_eef_pose[1] + self.robot_workspace_y[1] else True
        if not workspace_x_violation and not workspace_y_violation:
            action_success_message.data = self.robot.setEEFPose(estimated_new_eef)
            print('action_success_message.data: ', action_success_message.data)
            self.approach_success_pub.publish(action_success_message)
            print('Action executed Successfully!')
        else:
            action_success_message.data = False
            self.approach_success_pub.publish(action_success_message)
            print('Action violated the boundary conditions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_type', type=str, help='yumi/hexaglide', default=None, required=True)
    args = parser.parse_args()
    robot_type = args.robot_type
    v = Visual_servo_cmd(robot_type)
    # how much x,y,z movement from eef to cmd_eff
    sub_visual_servo_cmd = rospy.Subscriber('/visual_servo_cmd', Float64MultiArray, v.visual_servo_cmd_callback)
    sub_action = rospy.Subscriber('/action', Int64, v.action_callback)
    v.approach_success_pub = rospy.Publisher('approach_success', Bool, queue_size=1)  # queue_size=10
    rospy.spin()
