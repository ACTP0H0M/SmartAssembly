#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray

class RobotState:
    def __init__(self, eef_state_pub):
        if eef_state_pub is None:
            raise ValueError('eef_state_publisher not defined')
        print('RobotState instance created')
        self.eef_sub = rospy.Subscriber(eef_state_pub, Float64MultiArray, self.dataCallback)
        self.eef_state = None

    def get_eef_state(self):
        return self.eef_state

    def dataCallback(self, msg):
        if len(msg.data) == 0:
            print("ERROR: RobotState publishes empty ROS messages!")
            exit(0)
        elif len(msg.data) > 0:
            self.eef_state = msg.data
