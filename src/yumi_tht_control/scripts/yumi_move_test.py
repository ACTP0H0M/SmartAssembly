#!/usr/bin/env python
from yumi_tht_control.arm import Arm
from tht_control.common import *
import rospy
import argparse

'''
This script is used to control the left arm of YuMi
'''
if __name__ == '__main__':
    # Initialize the left YuMi arm and empty lists of waypoints
    rospy.init_node('move_group_python_interface', anonymous=True)
    yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
    # Navigate to the file with serialized (saved/recorded) waypoints and read them
    ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                     yaml_file_name='config_yumi.yaml')
    waypoint_list_filename = rospy.get_param('ros_param/insertion_waypoint_list_filename')
    DIP_waypoint_list_filename = rospy.get_param('ros_param/DIP_waypoint_list_filename')
    waypoints = get_waypoints(waypoint_list_filename)
    DIP_waypoints = get_waypoints(DIP_waypoint_list_filename)
    home_waypoints = []
    ros_param_path = load_ros_params(rospackage_name="tht_control", rel_path_from_package_to_file='config',
                                     yaml_file_name='tht_insertion_plan.yaml')
    tht_component_dict = rospy.get_param('/tht_components/component_dict')
    if rospy.get_param('/insertion_plan/insert_single_part'):
        part_ID = rospy.get_param('/insertion_plan/part_ID')
        component_package = tht_component_dict[part_ID][0]
        pin_separation = tht_component_dict[part_ID][1]
        n_pins = tht_component_dict[part_ID][2]
    else:
        THT_insertion_sequence = rospy.get_param('/insertion_plan/THT_insertion_sequence')
        component_package_list = [tht_component_dict[part_id][0] for part_id in
                                       THT_insertion_sequence]
        pin_separation_list = [tht_component_dict[part_id][1] for part_id in
                                    THT_insertion_sequence]
        n_pins_list = [tht_component_dict[part_id][2] for part_id in
                            THT_insertion_sequence]
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', type=str, help='forwards/backwards/cameras', default=None,
                        required=False)
    args = parser.parse_args()
    if args.direction == 'forwards':
        home_waypoints = list(waypoints)
        yumi.followWaypoints(home_waypoints)
        if component_package == 'DIP':
            dip_waypoints = list(DIP_waypoints)
            yumi.followWaypoints(dip_waypoints)

    elif args.direction == 'backwards':
        home_waypoints = list(reversed(waypoints))
        print(home_waypoints)
        yumi.followWaypoints(home_waypoints)

    elif args.direction == 'cameras':
        home_waypoints = waypoints[-1]
        yumi.setJointPose(home_waypoints)
        if component_package == 'DIP':
            dip_waypoints = list(DIP_waypoints)
            yumi.followWaypoints(dip_waypoints)
