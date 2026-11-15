import numpy as np
import rospy
from std_msgs.msg import Float32


class PressureSensor:
    def __init__(self):
        self.pressure_sub = []
        self.pressure_data = []
        self.pressure_data_array = None

    def dataCallback(self, msg):
        if len(msg.data) == 0:
            print("ERROR: Pressure sensor not initialized!")
            exit(0)
        elif len(msg.data) > 0:
            print("")
        self.pressure_data.append(msg.data)

    def sensorValue(self):
        msg = rospy.wait_for_message('/pressure', Float32, timeout=None)
        self.pressure_data.append(msg.data)
        return self.pressure_data[-1]
