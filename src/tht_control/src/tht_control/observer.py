import sys
import numpy as np
import rospy
from std_msgs.msg import Float32, String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from tht_control.robot_state_subscriber import RobotState
from vision.vision_sensor import VisionSensor
from phidget_sensor.pressure_sensor import PressureSensor

class Observer:
    """Contains latest observed values (images and pressures)"""
    def __init__(self, eef_state_pub, h, w=40):
        self.cam_img = None
        self.h = h
        self.w = w
        self.vert_offset = 5    # how many pixels the target hole is below the middle of cropped images
        self.current_image_1 = None
        self.current_image_2 = None
        self.current_cropped_image_1 = None
        self.current_cropped_image_2 = None
        self.is_initial_pose = True
        self.initial_position = None
        self.initial_orientation = None
        self.current_position = None
        self.current_orientation = None
        self.vision_sensor = VisionSensor(cameras=['cam_1', 'cam_2'])
        self.pressure_sensor = PressureSensor()
        self.robot_state = RobotState(eef_state_pub)
        self.current_pressure_value = None
        self.current_pressure_history = []
        self.name_to_dtypes = {
            "rgb8":    (np.uint8,  3),
            "rgba8":   (np.uint8,  4),
            "rgb16":   (np.uint16, 3),
            "rgba16":  (np.uint16, 4),
            "bgr8":    (np.uint8,  3),
            "bgra8":   (np.uint8,  4),
            "bgr16":   (np.uint16, 3),
            "bgra16":  (np.uint16, 4),
            "mono8":   (np.uint8,  1),
            "mono16":  (np.uint16, 1),

            # for bayer image (based on cv_bridge.cpp)
            "bayer_rggb8":	(np.uint8,  1),
            "bayer_bggr8":	(np.uint8,  1),
            "bayer_gbrg8":	(np.uint8,  1),
            "bayer_grbg8":	(np.uint8,  1),
            "bayer_rggb16":	(np.uint16, 1),
            "bayer_bggr16":	(np.uint16, 1),
            "bayer_gbrg16":	(np.uint16, 1),
            "bayer_grbg16":	(np.uint16, 1),

            # OpenCV CvMat types
            "8UC1":    (np.uint8,   1),
            "8UC2":    (np.uint8,   2),
            "8UC3":    (np.uint8,   3),
            "8UC4":    (np.uint8,   4),
            "8SC1":    (np.int8,    1),
            "8SC2":    (np.int8,    2),
            "8SC3":    (np.int8,    3),
            "8SC4":    (np.int8,    4),
            "16UC1":   (np.uint16,   1),
            "16UC2":   (np.uint16,   2),
            "16UC3":   (np.uint16,   3),
            "16UC4":   (np.uint16,   4),
            "16SC1":   (np.int16,  1),
            "16SC2":   (np.int16,  2),
            "16SC3":   (np.int16,  3),
            "16SC4":   (np.int16,  4),
            "32SC1":   (np.int32,   1),
            "32SC2":   (np.int32,   2),
            "32SC3":   (np.int32,   3),
            "32SC4":   (np.int32,   4),
            "32FC1":   (np.float32, 1),
            "32FC2":   (np.float32, 2),
            "32FC3":   (np.float32, 3),
            "32FC4":   (np.float32, 4),
            "64FC1":   (np.float64, 1),
            "64FC2":   (np.float64, 2),
            "64FC3":   (np.float64, 3),
            "64FC4":   (np.float64, 4)
        }

    def image_callback_1(self, image):
        # img = self.image_to_numpy(image)
        # Crop image to focus only on the area where THT is happening
        # img = image[self.y1:self.y1 + self.h, self.x1:self.x1 + self.h]
        self.current_image_1 = image
        # for idx in range(3):
        #     plt.subplot(1, 3, idx + 1)
        #     plt.imshow(img[:, :, idx])

    def image_callback_2(self, image):
        # img = self.image_to_numpy(image)
        # Crop image to focus only on the area where THT is happening
        # img = image[self.y2:self.y2 + self.h, self.x2:self.x2 + self.h]
        self.current_image_2 = image

    def image_cropped_callback_1(self, image, goal_x, goal_y):
        # img = self.image_to_numpy(image)
        # Crop image to focus only on the area where THT is happening
        cropped_image = np.copy(image[goal_y - self.h//2 - self.vert_offset: goal_y + self.h//2 - self.vert_offset,
                                goal_x - self.w//2 : goal_x + self.w//2])
        self.current_cropped_image_1 = cropped_image
        # if cropped_image.shape == (40, 40, 3):
        #     self.current_cropped_image_1 = cropped_image
        # else:
        #     temp = np.zeros((40, 40, 3), dtype=np.uint8)
        #     temp[:cropped_image.shape[0],:cropped_image.shape[1],:cropped_image.shape[2]] = cropped_image
        #     self.current_cropped_image_1 = temp

        # for idx in range(3):
        #     plt.subplot(1, 3, idx + 1)
        #     plt.imshow(img[:, :, idx])

    def image_cropped_callback_2(self, image, goal_x, goal_y):
        print('obs: ', np.shape(image))
        # img = self.image_to_numpy(image)
        # Crop image to focus only on the area where THT is happening
        cropped_image = np.copy(image[goal_y - self.h//2 - self.vert_offset: goal_y + self.h//2 - self.vert_offset,
                                goal_x - self.w//2 : goal_x + self.w//2])
        self.current_cropped_image_2 = cropped_image
        # if cropped_image.shape == (40, 40, 3):
        #     self.current_cropped_image_2 = cropped_image
        # else:
        #     temp = np.zeros((40, 40, 3), dtype=np.uint8)
        #     temp[:cropped_image.shape[0],:cropped_image.shape[1],:cropped_image.shape[2]] = cropped_image
        #     self.current_cropped_image_2 = temp

    def current_visualdata(self):
        # print("Observer current_visualdata() started...")
        self.cam_img = self.vision_sensor.readCamera()
        rospy.sleep(0.2)
        # print("Observer current_visualdata(): cam_img length is ", len(self.cam_img))
        self.cam_img = self.cam_img[-2:]
        self.image_callback_1(self.cam_img[0])
        self.image_callback_2(self.cam_img[1])
        # print("Observer current_visualdata() finished")
        return self.current_image_1, self.current_image_2

    def current_cropped_visualdata(self, list_tuples_goal_pixel_image1, list_tuples_goal_pixel_image2):
        # print("Observer current_cropped_visualdata() started...")
        self.cam_img = self.vision_sensor.readCamera()
        rospy.sleep(0.1)
        # print("Observer current_cropped_visualdata(): cam_img length is ", len(self.cam_img))
        self.cam_img = self.cam_img[-2:]
        self.image_cropped_callback_1(image=self.cam_img[0], goal_x=list_tuples_goal_pixel_image1[0][0], goal_y=list_tuples_goal_pixel_image1[0][1])
        self.image_cropped_callback_2(image=self.cam_img[1], goal_x=list_tuples_goal_pixel_image2[0][0], goal_y=list_tuples_goal_pixel_image2[0][1])
        # print("Observer current_cropped_visualdata() finished")
        return self.current_cropped_image_1, self.current_cropped_image_2


    def image_to_numpy(self, msg):
        """
        Taken from:
        https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
        """

        dtype_class, channels = (np.uint8, 3)    # rgb8
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

    def current_pressure(self):
        # Use pressure value history because the same value can be observed during phases
        # of increasing and decreasing contact force but the reaction should be different
        self.current_pressure_value = self.pressure_sensor.sensorValue()
        self.pressure_history(self.current_pressure_value)
        return self.current_pressure_value

    def pressure_history(self, pressure):
        self.current_pressure_history.append(pressure)
        return self.current_pressure_history

    def pose_callback(self, pose):
        # Use pose relative to initial pose because it is not depending on PCB location
        if self.is_initial_pose:
            self.initial_position = np.array([pose.position.x, pose.position.y, pose.position.z])
            self.initial_orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            self.is_initial_pose = False
        else:
            # ToDo
            # Maybe using Euler angles as observation is more 'representative' for the RL model?
            cur_position = np.array([pose.position.x, pose.position.y, pose.position.z])
            cur_orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            self.current_position = np.subtract(self.initial_position, cur_position) # Attention: moving down corresponds to a positive depth value!
            self.current_orientation = np.subtract(self.initial_orientation, cur_orientation)