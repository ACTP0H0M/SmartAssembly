#!/usr/bin/env python

from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
from phidget_sensor.calibration import Calibration
from phidget_sensor.common import *
from tht_control.common import *
import rospy
from std_msgs.msg import Float32


def main():
    # rospy.init_node('move_group_python_interface', anonymous=True)
    ch = []
    for i in range(4):
        ch.append(VoltageRatioInput())
        ch[i].setHubPort(568392)
        ch[i].setChannel(i)
        ch[i].openWaitForAttachment(1000)

    calib = Calibration(channel=ch, plot=False)
    # slope, intercept = calib.doCalibration()
    ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                     yaml_file_name='config_yumi.yaml')
    weights_list_filename = rospy.get_param('ros_param/weights_list_filename')
    weights = get_sensor_weights(weights_list_filename)
    slope = weights[0]
    intercept = weights[1]
    # print weights
    raw_input("Put the weight on the platform to test. Press enter to continue..")
    while(1):
        temp = 0
        for i in range(4):
            temp += ch[i].getVoltageRatio()
        test_weight = slope * temp + intercept
        print('Test weight value: ', test_weight)

def publish_pressure():
    pub = rospy.Publisher('pressure', Float32, queue_size=10)
    rospy.init_node('pressure_sensor', anonymous=True)
    rate = rospy.Rate(25) # hz

    ch = []
    for i in range(4):
        ch.append(VoltageRatioInput())
        ch[i].setChannel(i)
        ch[i].openWaitForAttachment(1000)

    calib = Calibration(channel=ch, plot=False)
    # slope, intercept = calib.doCalibration()
    ros_param_path = load_ros_params(rospackage_name="vision", rel_path_from_package_to_file='config',
                                     yaml_file_name='config_yumi.yaml')
    weights_list_filename = rospy.get_param('ros_param/weights_list_filename')
    weights = get_sensor_weights(weights_list_filename)
    slope = weights[0]
    intercept = weights[1]

    raw_input("Press enter to calculate tare and to start publishing pressure values...")

    tare = []
    n_tare_measurements = 100
    for i in range(n_tare_measurements):
        temp = 0
        for j in range(4):
            temp += ch[j].getVoltageRatio()
        weight = slope * temp + intercept
        tare.append(weight)

    tare_grams = sum(tare)/n_tare_measurements
    print 'Tare weight: ', tare_grams, ' g'

    while not rospy.is_shutdown():
        temp = 0
        for i in range(4):
            temp += ch[i].getVoltageRatio()
        weight = slope * temp + intercept - tare_grams
        pub.publish(weight)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_pressure()
    except rospy.ROSInterruptException:
        pass
