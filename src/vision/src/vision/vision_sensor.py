import sys
from sensor_msgs.msg import Image, CameraInfo
import rospy
import message_filters
import numpy as np
import cv2
import time


def infoCallback(msg):
    print(msg.width)


def convertNumpyImgToGrayscale(numpy_img):
    return cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)


def calculateGrayscaleImageDiff(grayscale_img1, grayscale_img2):
    return grayscale_img1.astype("int16") - grayscale_img2.astype("int16")


def convert16SC1Torgb8(src):
    return cv2.convertScaleAbs(src)


class VisionSensor:
    def __init__(self, cameras=None):
        print('VisionSensor instant created')
        self.img_callback_registered = False
        self.img = []
        self.prev_img = []  # to store previous image
        self.cam_info = []
        self.cameras = cameras
        self.cam_img_raw = Image()
        self.cam_sub = []
        self.cam_info_sub = []
        self.name_to_dtypes = {
            "rgb8": (np.uint8, 3),
            "rgba8": (np.uint8, 4),
            "rgb16": (np.uint16, 3),
            "rgba16": (np.uint16, 4),
            "bgr8": (np.uint8, 3),
            "bgra8": (np.uint8, 4),
            "bgr16": (np.uint16, 3),
            "bgra16": (np.uint16, 4),
            "mono8": (np.uint8, 1),
            "mono16": (np.uint16, 1),

            # for bayer image (based on cv_bridge.cpp)
            "bayer_rggb8": (np.uint8, 1),
            "bayer_bggr8": (np.uint8, 1),
            "bayer_gbrg8": (np.uint8, 1),
            "bayer_grbg8": (np.uint8, 1),
            "bayer_rggb16": (np.uint16, 1),
            "bayer_bggr16": (np.uint16, 1),
            "bayer_gbrg16": (np.uint16, 1),
            "bayer_grbg16": (np.uint16, 1),

            # OpenCV CvMat types
            "8UC1": (np.uint8, 1),
            "8UC2": (np.uint8, 2),
            "8UC3": (np.uint8, 3),
            "8UC4": (np.uint8, 4),
            "8SC1": (np.int8, 1),
            "8SC2": (np.int8, 2),
            "8SC3": (np.int8, 3),
            "8SC4": (np.int8, 4),
            "16UC1": (np.uint16, 1),
            "16UC2": (np.uint16, 2),
            "16UC3": (np.uint16, 3),
            "16UC4": (np.uint16, 4),
            "16SC1": (np.int16, 1),
            "16SC2": (np.int16, 2),
            "16SC3": (np.int16, 3),
            "16SC4": (np.int16, 4),
            "32SC1": (np.int32, 1),
            "32SC2": (np.int32, 2),
            "32SC3": (np.int32, 3),
            "32SC4": (np.int32, 4),
            "32FC1": (np.float32, 1),
            "32FC2": (np.float32, 2),
            "32FC3": (np.float32, 3),
            "32FC4": (np.float32, 4),
            "64FC1": (np.float64, 1),
            "64FC2": (np.float64, 2),
            "64FC3": (np.float64, 3),
            "64FC4": (np.float64, 4)
        }

    def readCamera(self):
        print("VisionSensor: readCamera() started...")
        start = time.time()
        self.cam_sub = []
        for cam in self.cameras:
            self.cam_sub.append(message_filters.Subscriber('/%s/color/image_raw' % cam, Image))
        ts = message_filters.ApproximateTimeSynchronizer(self.cam_sub, queue_size=10, slop=0.1)
        if not self.img_callback_registered:
            ts.registerCallback(self.imageCallback)
            self.img_callback_registered = True
        rospy.sleep(0.1)
        end = time.time()
        print("VisionSensor: readCamera() finished in ", end - start)
        return self.img

    def imageCallback(self, *args):
        self.img = []
        # br = CvBridge()
        for x in range(len(args)):
            self.img.append(cv2.cvtColor(self.image_to_numpy(args[x]), cv2.COLOR_BGR2RGB))

    def getCameraInfo(self):
        self.cam_info_sub.append(
            rospy.Subscriber('/%s/color/camera_info' % self.cameras[0], CameraInfo, self.infoCallback))
        rospy.sleep(0.1)
        return self.cam_info[0]

    def infoCallback(self, msg):
        self.cam_info = []
        self.cam_info.append([msg.width, msg.height, (msg.K[0] + msg.K[4]) / 2])

    def image_to_numpy(self, msg):
        """
        Taken from:
        https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
        """
        dtype_class, channels = (np.uint8, 3)  # rgb8
        dtype = np.dtype(dtype_class)
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
        shape = (msg.height, msg.width, channels)

        data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
        data.strides = (
            msg.step,
            dtype.itemsize * channels,
            dtype.itemsize
        )

        if channels == 1:
            data = data[..., 0]
        return data

    def numpy_to_image(self, arr, encoding):
        if not encoding in self.name_to_dtypes:
            raise TypeError('Unrecognized encoding {}'.format(encoding))

        im = Image(encoding=encoding)

        # extract width, height, and channels
        dtype_class, exp_channels = self.name_to_dtypes[encoding]
        dtype = np.dtype(dtype_class)
        if len(arr.shape) == 2:
            im.height, im.width, channels = arr.shape + (1,)
        elif len(arr.shape) == 3:
            im.height, im.width, channels = arr.shape
        else:
            raise TypeError("Array must be two or three dimensional")

        # check type and channels
        if exp_channels != channels:
            raise TypeError("Array has {} channels, {} requires {}".format(
                channels, encoding, exp_channels
            ))
        if dtype_class != arr.dtype.type:
            raise TypeError("Array is {}, {} requires {}".format(
                arr.dtype.type, encoding, dtype_class
            ))

        # make the array contiguous in memory, as mostly required by the format
        contig = np.ascontiguousarray(arr)
        im.data = contig.tostring()
        im.step = contig.strides[0]
        im.is_bigendian = (
                arr.dtype.byteorder == '>' or
                arr.dtype.byteorder == '=' and sys.byteorder == 'big'
        )
        return im


class WebcamVisionSensor(VisionSensor):
    def readUSBCamera(self):
        print("WebcamImageSensor: readCamera() started...")
        usb_cam_image = rospy.wait_for_message('/usb_cam/image_raw', Image)
        self.imageCallback(usb_cam_image)
        return self.img
