import os
import rosparam
import rospy
import rospkg
import pickle
import geometry_msgs.msg
import tf


def get_sensor_weights(weights_list_filename=''):
    ros_pack = rospkg.RosPack()
    pkg_path = ros_pack.get_path('phidget_sensor')
    weight_path = pkg_path + '/' + weights_list_filename
    weights = read_list(weight_path)
    return weights


def read_list(file_path=''):
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list