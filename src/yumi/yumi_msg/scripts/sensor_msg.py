#!/usr/bin/env python

from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
import rospy
from yumi_msg.msg import sensor


def sensor_pub(ch=[]):
    pub = rospy.Publisher('sensor_data', sensor, queue_size=10)
    rospy.init_node('sensor_data', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    msg = sensor()
    while not rospy.is_shutdown():
        sensor_val = 0
        for i in range(4):
            sensor_val += ch[i].getVoltageRatio()
        msg.force_data = sensor_val
	pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    ch = []
    for i in range(4):
        ch.append(VoltageRatioInput())
        ch[i].setChannel(i)
        ch[i].openWaitForAttachment(1000)
    try:
        sensor_pub(ch)
    except rospy.ROSInterruptException:
        pass
