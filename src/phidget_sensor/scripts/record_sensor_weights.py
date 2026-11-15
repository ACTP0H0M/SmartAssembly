#!/usr/bin/env python

import rospkg
import os, datetime
import rospy
import pickle
from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
from phidget_sensor.calibration import Calibration


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('phidget_sensor')
    ch = []
    for i in range(4):
        ch.append(VoltageRatioInput())
        ch[i].setChannel(i)
        ch[i].openWaitForAttachment(1000)
    calib = Calibration(channel=ch, plot=False)
    slope, intercept = calib.doCalibration()
    weights_file_path = ('%s/sensor_weights/%s' % (pkg_path, timeStamped('weights')))
    with open(weights_file_path, 'wb') as fp:
        pickle.dump([slope, intercept], fp)
        print('Done writing list into a binary file')





