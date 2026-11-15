#!/usr/bin/env python

import rospy
import rospkg
import rosparam
import pickle
import geometry_msgs.msg
import os


def load_config_file(robot_type):
    config_file_name = None
    if robot_type is not None:
        if robot_type == 'yumi':
            config_file_name = 'config_yumi.yaml'
        elif robot_type == 'hexaglide':
            config_file_name = 'config_hexaglide.yaml'
        else:
            raise NameError('Invalid robot type')
    else:
        raise ValueError('Robot type not defined')
    return config_file_name, robot_type


def action_dict():
    return {
        0: 'FRONT',
        1: 'BACK',
        2: 'RIGHT',
        3: 'LEFT',
        4: 'DOWN',
        5: 'UP',
        6: 'ROLL_P',
        7: 'ROLL_N',
        8: 'PITCH_P',
        9: 'PITCH_N',
        10: 'YAW_P',
        11: 'YAW_N'
    }


def relativeQuaternionRotation(q0, q1):
    a0 = q0[3]  # if it is indeed w - otherwise change indices
    b0 = q0[0]
    c0 = q0[1]
    d0 = q0[2]
    a1 = q1[3]  # if it is indeed w - otherwise change indices
    b1 = q1[0]
    c1 = q1[1]
    d1 = q1[2]
    w = a0 * a1 + b0 * b1 + c0 * c1 + d0 * d1
    x = -a0 * b1 - b0 * a1 - c0 * d1 + d0 * c1
    y = -a0 * c1 + b0 * d1 - c0 * a1 - d0 * b1
    z = -a0 * d1 - b0 * c1 + c0 * b1 - d0 * a1
    return [w, x, y, z]


def QuaternionMultiplication(q0, q1):
    # Quaternion representation = [x, y, z, w]
    a0 = q0[0]
    b0 = q0[1]
    c0 = q0[2]
    d0 = q0[3]
    a1 = q1[0]
    b1 = q1[1]
    c1 = q1[2]
    d1 = q1[3]
    w = -a0 * a1 - b0 * b1 - c0 * c1 + d0 * d1
    x = b0 * c1 - c0 * b1 + d0 * a1 + a0 * d1
    y = -a0 * c1 + c0 * a1 + d0 * b1 + b0 * d1
    z = a0 * b1 - b0 * a1 + d0 * c1 + c0 * d1
    return [w, x, y, z]


def load_ros_params(rospackage_name, rel_path_from_package_to_file,
                    yaml_file_name):
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file)
    path_config_file = os.path.join(config_dir, yaml_file_name)

    paramlist = rosparam.load_file(path_config_file)

    for params, ns in paramlist:
        rosparam.upload_params(ns, params)
    return path_config_file


def get_config_path(rospackage_name, rel_path_from_package_to_file, yaml_file_name):
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file)
    path_config_file = os.path.join(config_dir, yaml_file_name)
    return path_config_file


def get_waypoints(waypoint_list_filename=''):
    ros_pack = rospkg.RosPack()
    pkg_path = ros_pack.get_path('yumi_tht_control')
    waypoint_path = pkg_path + '/' + waypoint_list_filename
    waypoints = read_list(waypoint_path)
    return waypoints


def read_list(file_path=''):
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def eefpos_to_geo_msg(eef_pose):
    eefpos = geometry_msgs.msg.Pose()
    eefpos.position.x = eef_pose[0]
    eefpos.position.y = eef_pose[1]
    eefpos.position.z = eef_pose[2]
    eefpos.orientation.x = eef_pose[3]
    eefpos.orientation.y = eef_pose[4]
    eefpos.orientation.z = eef_pose[5]
    eefpos.orientation.w = eef_pose[6]
    return eefpos
