from sensor_msgs.msg import Image
import rospy


class VisionSensor:
    def __init__(self, camera=None):
        self.camera = camera
        self.cam_img_raw = []
        self.cam_sub = []
        for cam in self.camera:
            self.cam_img_raw.append(Image())
            self.cam_sub.append(rospy.Subscriber('/%s/color/image_raw' % cam, Image, self.subCallback_cam))

    def subCallback_cam(self, msg):
        self.cam_img_raw = msg
        self.sub_cam_1.unregister()
        print("Camera 1 image saved")

    def subCallback_cam2(self, msg):
        self.cam_2_img_raw = msg
        self.sub_cam_2.unregister()
        print("Camera 2 image saved")
