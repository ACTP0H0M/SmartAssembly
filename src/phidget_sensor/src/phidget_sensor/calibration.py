# from Phidget22.Phidget import *
# from Phidget22.Devices.VoltageRatioInput import *
import matplotlib.pyplot as plt
from scipy import stats

class Calibration:
    def __init__(self, channel=[], plot=False, tare_weight=0.0):
        self.channel = channel
        self.plot = plot
        self.tare_weight = tare_weight
        self.voltage_train = []
        self.weight_train = []

    def doCalibration(self):
        self.setTare()
        self.weight_train.append(int(raw_input("Enter weight value for calibration")))
        self.voltage_train.append(self.voltageRatio())
        check = raw_input("Add more calibration points? y/n")
        while check == "y":
            self.weight_train.append(int(raw_input("Enter weight value for calibration")))
            self.voltage_train.append(self.voltageRatio())
            check = raw_input("Add more calibration points? y/n")

        if self.plot:
            plt.ticklabel_format(axis='y', style='sci', scilimits=(-5, -5))
            plt.scatter(self.weight_train, self.voltage_train)
            plt.xlabel("Load Cell Voltage ratio")
            plt.ylabel("Calibration Weight (grams)")
            plt.show()

        self.slope, self.intercept, r, p, std_err = stats.linregress(self.voltage_train, self.weight_train)
        print("Predictability score: ", r)
        return self.slope, self.intercept

    def setTare(self):
        raw_input("Remove all weights to record tare value. Press enter to continue..")
        self.weight_train.append(self.tare_weight)
        self.voltage_train.append(self.voltageRatio())

    def voltageRatio(self):
        temp = 0
        for i in range(len(self.channel)):
            temp += self.channel[i].getVoltageRatio()
        return temp
