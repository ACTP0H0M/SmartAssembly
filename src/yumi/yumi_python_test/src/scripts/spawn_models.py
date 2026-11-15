#!/usr/bin/env python

import rospy, tf
from os.path import dirname, abspath
from scipy.spatial.transform import Rotation
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *

if __name__ == '__main__':
    print("Waiting for gazebo services...")
    rospy.init_node("spawn_products_in_bins")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    print("Got it.")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    

    with open(dirname(dirname(abspath(__file__)))+"/models/peg_board/model.sdf", "r") as f:
        product_xml = f.read()

    rot = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    rot_quat = rot.as_quat()
    print("Spawning model:%s", "peg_board")
    model_pose = Pose()
    model_pose.position.x = 0.3
    model_pose.position.y = 0
    model_pose.position.z = 0
    model_pose.orientation.x = rot_quat[0]
    model_pose.orientation.y = rot_quat[1]
    model_pose.orientation.z = rot_quat[2]
    model_pose.orientation.w = rot_quat[3]
    spawn_model("peg_board", product_xml, "", model_pose, "world")
