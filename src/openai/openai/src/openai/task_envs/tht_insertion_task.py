import operator

from gym import spaces
from openai.initialize_logger import initialize_logger
from openai.robot_envs.tht_env import ManipulatorEnv
from openai.custom_baseline_classes.metallic_pixel_detection import MetallicPixelDetection
from openai.observer_cam3 import ObserverCam3
from tht_control.observer import Observer
from tht_control.common import *
from tht_control.utils import *
from openai.utils import *
from openai.common import action_dict, relativeQuaternionRotation
# from vision.common import load_ros_params
from vision.click_goal import GoalPixelSetter
from vision.visual_servo import VisualServo
from vision.vision_sensor import convertNumpyImgToGrayscale, calculateGrayscaleImageDiff, convert16SC1Torgb8
from gym.envs.registration import register
from std_msgs.msg import Int64, Float64MultiArray, Bool
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
import rospy
import datetime
import numpy as np
import math as m
import time

class THTRLEnv(ManipulatorEnv):

    def __init__(self, robot_type='hexaglide'): # ToDo: robot_type was None
        print("Started initializing THTRlEnv...")
        config_file_name, self.robot_type = load_config_file(robot_type)
        ros_param_path = load_ros_params(rospackage_name='tht_control',
                                         rel_path_from_package_to_file='config', yaml_file_name=config_file_name)
        ManipulatorEnv.__init__(self)
        # Initialize observation space and action space for RL Algorithm:
        self.h3 = rospy.get_param('/cropped_image_pixels/h3')
        self.w3 = rospy.get_param('/cropped_image_pixels/w3')
        self.observation_space = spaces.Dict(
            spaces={
                "current_image3": spaces.Box(low=0, high=255,
                                     shape=(self.h3, self.w3, 3),
                                     dtype=np.uint8),
                "previous_image3": spaces.Box(low=0, high=255,
                                             shape=(self.h3, self.w3, 3),
                                             dtype=np.uint8),
                "pressure": spaces.Box(low=-200, high=700, shape=(1,), dtype=np.float32),
                "pose": spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32) # provide in meters and quaternions!
            })
        self.number_actions = rospy.get_param('/env_param/n_actions')
        self.action_space = spaces.Discrete(self.number_actions)
        self.action_pub = rospy.Publisher(rospy.get_param('/ros_param/action_pub'), Int64, queue_size=10)
        self.eef_state_pub = rospy.get_param('/ros_param/eef_state_pub')
        self.cropped_img1_pub = rospy.Publisher(rospy.get_param('/ros_param/cropped_img1_pub'), Image, queue_size=10)
        self.cropped_img2_pub = rospy.Publisher(rospy.get_param('/ros_param/cropped_img2_pub'), Image, queue_size=10)
        self.cropped_img3_pub = rospy.Publisher(rospy.get_param('/ros_param/cropped_img3_pub'), Image, queue_size=10)
        self.previous_cropped_img3_pub = rospy.Publisher(rospy.get_param('/ros_param/previous_cropped_img3_pub'), Image, queue_size=10)
        self.visual_servo_cmd_pub = rospy.Publisher('visual_servo_cmd', Float64MultiArray, queue_size=10)
        self.img1_pub = rospy.get_param('/ros_param/img1_pub')
        self.img2_pub = rospy.get_param('/ros_param/img2_pub')
        self.move_to_pose_pub = rospy.Publisher(rospy.get_param('/ros_param/move_to_pose_pub'), Float64MultiArray, queue_size=10)
        self.move_to_approach_pose_pub = rospy.Publisher(rospy.get_param('/ros_param/move_to_joint_pose_pub'), Bool, queue_size=10)
        self.record_joint_pos_pub = rospy.Publisher(rospy.get_param('/ros_param/record_joint_pos_pub'), Bool, queue_size=10)
        self.rl_init_pose_pub = rospy.Publisher(rospy.get_param('/ros_param/rl_init_pose_pub'), Float64MultiArray, queue_size=10)
        self.translation_stepsize = rospy.get_param('/env_param/translation_step_mm')
        self.rotation_stepsize = rospy.get_param('/env_param/rotation_step_deg')
        self.output_dir = rospy.get_param('/log_param/output_dir')
        self.workspace_x = rospy.get_param('/env_param/x_punished_limit')
        self.workspace_y = rospy.get_param('/env_param/y_punished_limit')
        self.robot_reference_pose = None
        self.workspace_violation = None
        self.tildaorigin_to_pcb_z = rospy.get_param('/robot_workspace/tildaorigin_to_pcb_z')
        self.H1origin_to_tildaorigin = rospy.get_param('/robot_workspace/H1origin_to_tildaorigin')
        self.robot_init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.rotation_stepsize = (rospy.get_param('/env_param/rotation_step_deg')) * m.pi / 180
        self.orientation_as_euler = rospy.get_param('/env_param/orientation_as_euler')
        self.image_as_input_without_cv = rospy.get_param('/env_param/image_as_input_without_cv')
        self.diff_image_as_input = rospy.get_param('/env_param/diff_image_as_input')
        self.pressure_hist_as_obs = rospy.get_param('/env_param/pressure_hist_as_obs')
        self.max_step_count = rospy.get_param('/env_param/max_step_count')
        self.max_pressure = rospy.get_param('/env_param/max_pressure')
        self.max_depth = rospy.get_param('/env_param/max_depth')
        self.obs_logfile = 'testlogs'  # TODO:
        self.obs_per_step = []
        self.action_dict = rospy.get_param('/env_param/action_dict')
        h = rospy.get_param('/cropped_image_pixels/h')
        w = rospy.get_param('/cropped_image_pixels/w')
        # Where the center of cropped cam3 image is
        self.x3 = rospy.get_param('/cropped_image_pixels/x3')
        self.y3 = rospy.get_param('/cropped_image_pixels/y3')
        self.obs = Observer(rospy.get_param('/ros_param/eef_state_pub'), h, w)  # TODO not used: x1, y1, x2, y2. Remove these
        self.metallic_pixel_detection = False
        # ToDo: Does obs need ph on construction???
        self.obs3 = ObserverCam3(rospy.get_param('/ros_param/eef_state_pub'), x3=self.x3, y3=self.y3, w=self.w3, h=self.h3) # was: rospy.get_param('/ros_param/eef_state_pub'), h, w
        self.visual_servo = VisualServo(self.robot_type)
        self.init_step = True
        self.init_episode = True

        # Capture complete images from both cameras for goal hole clicking
        _current_image1, _current_image2 = self.obs.current_visualdata()
        _current_cropped_image1 = None
        _current_cropped_image2 = None

        # Setting goal hole by clicking it in both camera images or use default values
        self.goal_pose_pixel_image1 = []
        self.goal_pose_pixel_image2 = []
        self.click_goal_image = rospy.get_param('/env_param/click_goal_image')
        if self.click_goal_image:
            print("THTRLEnv: Please set the goal hole by clicking on it in both images!")
            self.goal_pixel_setter = GoalPixelSetter()
            self.goal_pixel_setter.click_set_goal_in_pixels(_current_image1)
            if self.goal_pixel_setter.x is None or self.goal_pixel_setter.y is None:
                raise RuntimeError("Goal pixel in image1 not clicked")
            self.goal_pose_pixel_image1.append((int(self.goal_pixel_setter.x), int(self.goal_pixel_setter.y)))
            print("Clicked goal pixel in image 1: ", self.goal_pose_pixel_image1)
            self.goal_pixel_setter = GoalPixelSetter()
            self.goal_pixel_setter.click_set_goal_in_pixels(_current_image2)
            if self.goal_pixel_setter.x is None or self.goal_pixel_setter.y is None:
                raise RuntimeError("Goal pixel in image2 not clicked")
            self.goal_pose_pixel_image2.append((int(self.goal_pixel_setter.x), int(self.goal_pixel_setter.y)))
            print("Clicked goal pixel in image 2: ", self.goal_pose_pixel_image2)
        else:
            self.goal_pose_pixel_image1.append(tuple(rospy.get_param('/env_param/goal_pixel_img1')))
            self.goal_pose_pixel_image2.append(tuple(rospy.get_param('/env_param/goal_pixel_img2')))
        print('Goal image1 (pixels): ', self.goal_pose_pixel_image1)
        print('Goal image2 (pixels): ', self.goal_pose_pixel_image2)
        self.goal_pixel_correction_img1 = rospy.get_param('/env_param/goal_pixel_correction_img1')
        self.goal_pixel_correction_img2 = rospy.get_param('/env_param/goal_pixel_correction_img2')
        # self.corrected_goal_pose_pixel_image1 = [
        #     (self.goal_pose_pixel_image1[0][0] + self.goal_pixel_correction_img1[0],
        #      self.goal_pose_pixel_image1[0][1] + self.goal_pixel_correction_img1[1])]
        # self.corrected_goal_pose_pixel_image2 = [
        #     (self.goal_pose_pixel_image2[0][0] + self.goal_pixel_correction_img2[0],
        #      self.goal_pose_pixel_image2[0][1] + self.goal_pixel_correction_img2[1])]
        # print('Corrected goal image1 (pixels): ', self.corrected_goal_pose_pixel_image1)
        # print('Corrected goal image2 (pixels): ', self.corrected_goal_pose_pixel_image2)
        self.corrected_goal_pose_tilda = None

        # Get initial cropped images (without THD or gripper) to work with image differences
        self.init_cropped_image1, self.init_cropped_image2 = self.obs.current_cropped_visualdata(
            self.goal_pose_pixel_image1,
            self.goal_pose_pixel_image2)
        self.img_center_x = self.init_cropped_image1.shape[1] / 2
        self.img_center_y = self.init_cropped_image1.shape[0] / 2
        self.center_tolerance_px = rospy.get_param('/env_param/center_tolerance_px')
        self.mpd = MetallicPixelDetection(self.init_cropped_image1, self.init_cropped_image2, self.center_tolerance_px,
                                          self.img_center_x)

        # MPD correction of the clicked goal pixels
        self.mpd_goal_correction_cases = []  # List of Strings: which side of the hole was selected?
        self.pin_thickness_px = 6  # ToDo: ROS Parameter or read from assembly plan
        if self.click_goal_image:

            image1_pin_location, image2_pin_location, pin_dev_cam1, pin_dev_cam2 = self.mpd.mpd(
                self.init_cropped_image1, self.init_cropped_image2)
            self.mpd_corrected_goal_pose_pixel_image1 = [
                (self.goal_pose_pixel_image1[0][0] + pin_dev_cam1, self.goal_pose_pixel_image1[0][1])]
            self.mpd_corrected_goal_pose_pixel_image2 = [
                (self.goal_pose_pixel_image2[0][0] + pin_dev_cam2, self.goal_pose_pixel_image2[0][1])]

            print('MPD-corrected goal pixels image1: ', self.mpd_corrected_goal_pose_pixel_image1)
            print('MPD-corrected goal pixels image2: ', self.mpd_corrected_goal_pose_pixel_image2)

            # clicked goal pixel on left side of the MPD pin detection, add half pin thickness pixels to offset goal pixel to the center of the hole
            # Cam1:
            if pin_dev_cam1 > self.pin_thickness_px / 4:
                print('In camera 1, the pin MPD maximum is on the right of the clicked pixel')
                self.mpd_corrected_goal_pose_pixel_image1[0] = list(self.mpd_corrected_goal_pose_pixel_image1[0])
                self.mpd_corrected_goal_pose_pixel_image1[0] = tuple(
                    [self.mpd_corrected_goal_pose_pixel_image1[0][0] + int(self.pin_thickness_px / 2),
                     self.mpd_corrected_goal_pose_pixel_image1[0][1]])
            # clicked goal pixel on right side of the MPD pin detection, subtract half pin thickness pixels to offset goal pixel to the center of the hole
            elif pin_dev_cam1 < -self.pin_thickness_px / 4:
                print('In camera 1, the pin MPD maximum is on the left of the clicked pixel')
                self.mpd_corrected_goal_pose_pixel_image1[0] = list(self.mpd_corrected_goal_pose_pixel_image1[0])
                self.mpd_corrected_goal_pose_pixel_image1[0] = tuple(
                    [self.mpd_corrected_goal_pose_pixel_image1[0][0] - int(self.pin_thickness_px / 2),
                     self.mpd_corrected_goal_pose_pixel_image1[0][1]])

            # Cam2:
            if pin_dev_cam2 > self.pin_thickness_px / 4:
                print('In camera 2, the pin maximum is on the right of the clicked pixel')
                self.mpd_corrected_goal_pose_pixel_image2[0] = list(self.mpd_corrected_goal_pose_pixel_image2[0])
                self.mpd_corrected_goal_pose_pixel_image2[0] = tuple(
                    [self.mpd_corrected_goal_pose_pixel_image2[0][0] + int(self.pin_thickness_px / 2),
                     self.mpd_corrected_goal_pose_pixel_image2[0][1]])
            elif pin_dev_cam2 < -self.pin_thickness_px / 4:
                print('In camera 2, the pin maximum is on the left of the clicked pixel')
                self.mpd_corrected_goal_pose_pixel_image2[0] = list(self.mpd_corrected_goal_pose_pixel_image2[0])
                self.mpd_corrected_goal_pose_pixel_image2[0] = tuple(
                    [self.mpd_corrected_goal_pose_pixel_image2[0][0] - int(self.pin_thickness_px / 2),
                     self.mpd_corrected_goal_pose_pixel_image2[0][1]])

            # Save cam1 deviation case (which side of the hole was selected?)
            # cam1:
            if pin_dev_cam1 >= self.pin_thickness_px / 2:
                self.mpd_goal_correction_cases.append('left')

            elif -self.pin_thickness_px / 2 <= pin_dev_cam1 < self.pin_thickness_px / 2:
                self.mpd_goal_correction_cases.append('center')

            elif pin_dev_cam1 < -self.pin_thickness_px / 2:
                self.mpd_goal_correction_cases.append('right')
            # cam2:
            if pin_dev_cam2 >= self.pin_thickness_px / 2:
                self.mpd_goal_correction_cases.append('left')

            elif -self.pin_thickness_px / 2 <= pin_dev_cam2 < self.pin_thickness_px / 2:
                self.mpd_goal_correction_cases.append('center')

            elif pin_dev_cam2 < -self.pin_thickness_px / 2:
                self.mpd_goal_correction_cases.append('right')
        else:
            self.mpd_corrected_goal_pose_pixel_image1 = self.goal_pose_pixel_image1
            self.mpd_corrected_goal_pose_pixel_image2 = self.goal_pose_pixel_image2

        # print('MPD goal correction cases: ', self.mpd_goal_correction_cases)
        # print('MPD-corrected goal pixels image1 (after half pin thickness pixels offset): ', self.mpd_corrected_goal_pose_pixel_image1)
        # print('MPD-corrected goal pixels image2 (after half pin thickness pixels offset): ', self.mpd_corrected_goal_pose_pixel_image2)

        if self.diff_image_as_input:
            # Convert both images to grayscale
            self.init_cropped_image1 = convertNumpyImgToGrayscale(self.init_cropped_image1)
            self.init_cropped_image2 = convertNumpyImgToGrayscale(self.init_cropped_image2)
        # Calculate tilda (w.r.t. cameras' optical axis intersection) coordinates of the goal hole
        if self.robot_type == 'yumi':
            self.goal_pose_tilda = self.visual_servo.calc_1_tip_coords(self.goal_pose_pixel_image1,
                                                                       self.goal_pose_pixel_image2,
                                                                       self.visual_servo.x_tilda_c1,
                                                                       self.visual_servo.y_tilda_c2)
            print('Goal pose tilda: ', self.goal_pose_tilda)

        elif self.robot_type == 'hexaglide':

            z_tilda_goal = self.tildaorigin_to_pcb_z
            ######## CALCULATED TILDA GOAL #########
            self.mpd_corrected_goal_pose_tilda = self.visual_servo.calc_1_tip_coords(
                self.mpd_corrected_goal_pose_pixel_image1,
                self.mpd_corrected_goal_pose_pixel_image2,
                self.visual_servo.x_tilda_c1,
                self.visual_servo.y_tilda_c2)
            self.mpd_corrected_goal_pose_tilda = list(self.mpd_corrected_goal_pose_tilda)
            self.mpd_corrected_goal_pose_tilda[2] = z_tilda_goal * 1000
            self.mpd_corrected_goal_pose_tilda = tuple(self.mpd_corrected_goal_pose_tilda)

            ###### ESTIMATED TILDA GOAL WITH FITTING FUNCTION ######
            # x_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image2[0][0],
            #                                       self.corrected_goal_pose_pixel_image2[0][1],
            #                                       self.visual_servo.x_tilda_params[0],
            #                                       self.visual_servo.x_tilda_params[1],
            #                                       self.visual_servo.x_tilda_params[2],
            #                                       self.visual_servo.x_tilda_params[3],
            #                                       self.visual_servo.x_tilda_params[4],
            #                                       self.visual_servo.x_tilda_params[5])
            #
            # y_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image1[0][0],
            #                                       self.corrected_goal_pose_pixel_image1[0][1],
            #                                       self.visual_servo.y_tilda_params[0],
            #                                       self.visual_servo.y_tilda_params[1],
            #                                       self.visual_servo.y_tilda_params[2],
            #                                       self.visual_servo.y_tilda_params[3],
            #                                       self.visual_servo.y_tilda_params[4],
            #                                       self.visual_servo.y_tilda_params[5])
            # self.goal_pose_tilda = (x_tilda_goal, y_tilda_goal, z_tilda_goal*1000)
            # print('Goal pose tilda by Fitting function: ', self.goal_pose_tilda)

        self.hole_position_H1 = self.H1origin_to_tildaorigin + np.dot(self.visual_servo.rot_tilda_to_base,
                                                                      (np.array(
                                                                          self.mpd_corrected_goal_pose_tilda)) / 1000)
        self.pixel_goal_error_hist_1 = []
        self.pixel_goal_error_hist_2 = []
        self.tilda_goal_error_hist = []
        self.old_z_relative_coord = [0.0]
        self.old_robot_pose_roll = []
        self.old_robot_pose_pitch = []
        self.old_robot_pose_yaw = []
        self.goal_error_coords = rospy.get_param('/env_param/goal_error_coords')
        # self._set_init_pose(self.robot_init_pose)  # TODO: Add init pose in config file
        ros_param_path = load_ros_params(rospackage_name="tht_control",
                                         rel_path_from_package_to_file='config',
                                         yaml_file_name='tht_insertion_plan.yaml')
        self.tht_component_dict = rospy.get_param('/tht_components/component_dict')
        if rospy.get_param('/insertion_plan/insert_single_part'):
            self.part_ID = rospy.get_param('/insertion_plan/part_ID')
            self.component_package = self.tht_component_dict[self.part_ID][0]
            self.n_pins = self.tht_component_dict[self.part_ID][1]
            self.pin_separation = self.tht_component_dict[self.part_ID][2]
        else:
            self.THT_insertion_sequence = rospy.get_param('/insertion_plan/THT_insertion_sequence')
            self.component_package_list = [self.tht_component_dict[part_id][0] for part_id in
                                           self.THT_insertion_sequence]
            self.n_pins_list = [self.tht_component_dict[part_id][1] for part_id in
                                self.THT_insertion_sequence]
            self.pin_separation_list = [self.tht_component_dict[part_id][2] for part_id in
                                        self.THT_insertion_sequence]
        # ToDo: Actually each component in an insertion plan can have its own reorientation angle
        self.part_reorientation_angle = rospy.get_param('/insertion_plan/part_reorientation_angle')
        self.part_reorientation_angle = self.part_reorientation_angle * m.pi / 180  # radians
        self.D_dev_tolerance_m = 0.002
        self.D_dev_advanced_search_trigger_m = self.D_dev_tolerance_m
        self.D_dev_m = 0.0  # D_dev = D_p - D_h (pin separation - hole separation)

        self.cam1_tip_window_px = 10
        self.cam1_tip_window_px = 10
        self._set_init_pose()
        self.step_count = 0
        self.success = False

        # Load THT Insertion Plan Config

        print("Finished initializing THTRLEnv \n")

    def _set_init_pose(self):
        """
        Asks the user to prepare the robot for a RL training episode.
        This includes bringing the robot in its init pose (in front of the cameras to do visual calibration)
        and placing the THT element in the robot gripper.
        """
        self.step_count = 0

        # ToDo: do we want to redo the approach after every episode? (e.g. to check whether pins are ok)
        # if self.init_episode:
        print("Started THTRLEnv._set_init_pose()...")
        print("Please prepare the robot for a new RL training episode!")
        print("1. Bring the robot gripper in front of the cameras by executing:")
        print("   rosrun tht_control yumi_move_test.py --direction cameras")
        print("2. Adjust or place the THT element in robot gripper if required:")
        print("   rosrun tht_control grip_part.py --gripper open")
        print("   rosrun tht_control grip_part.py --gripper close")
        print("   Hold the THT element still by hand between the closing grippers.")
        gripper_in_front_of_cameras = input(
            "   When the robot holds the THT element in front of the cameras and dark background is present, press y")
        print("3. Executing visual calibration and approach phase:")

        start_set_init_pose = time.time()

        pin_side_cam2 = None
        wait_for_input = False
        self.robot_reference_pose = self.obs3.robot_state.eef_state
        selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.moveToTildaOrigin(pin_side_cam2,
                                                                                           wait_for_input)

        # TODO: Get pin separation from planner config
        if self.component_package == 'SIP':
            self.graspingAngleErrorCompensation(pin_side_cam2, self.pin_separation, wait_for_input)
        # pin_separation = self.tht_component_dict['T'][2]
        #

        # Then execute final translation
        pin_side_cam2 = 'lowest'
        selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.moveToTildaOrigin(pin_side_cam2,
                                                                                           wait_for_input)

        # ToDo remove - just for visualization to check how accurate we are with the pin tip in origin
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_eef_pos = _current_eef_state_msg.data[:3]
        _current_eef_orientation = _current_eef_state_msg.data[3:]
        if self.component_package == 'SIP':
            eef_to_tilda, selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.visual_servo.autoSelectRightPinTip(
                _current_eef_pos, pin_side_cam2)
        elif self.component_package == 'DIP':
            eef_to_tilda, selected_pin_tip_tilda = self.visual_servo.autoSelectDIPPinTip(_current_eef_pos, self.n_pins,
                                                                                         self.cam1_tip_window_px)

        # Approach Goal Position
        self.approachGoal(selected_pin_tip_tilda, wait_for_input)

        # Correct Approach Pose
        self.approachCorrection(wait_for_input)

        if self.robot_type == 'hexaglide':
            _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
            self.robot_init_pose = _current_eef_state_msg.data
            print('THTRLEnv._set_init_pose(): THTRLEnv.robot_init_pose after approachCorrection =\n', self.robot_init_pose)

        record_joint_msg = Bool()
        record_joint_msg.data = True
        self.record_joint_pos_pub.publish(record_joint_msg)

        rl_init_pose_msg = Float64MultiArray()
        rl_init_pose_msg.data = self.robot_init_pose
        self.rl_init_pose_pub.publish(rl_init_pose_msg)

        self.workspace_violation = False

    def _set_init_approach_pose(self):
        """
        Used to bring the manipulator to the starting pose of an RL episode
        """
        print('THTRLEnv._set_init_approach_pose(): robot init pose =\n', self.robot_init_pose)
        print('THTRLEnv._set_init_approach_pose(): robot current pose =\n', self.obs3.robot_state.eef_state)
        delta_eef_pose = list(map(operator.sub, self.obs3.robot_state.eef_state, self.robot_init_pose))
        delta_eef_pose[-1] = 1 if delta_eef_pose[-1] == 0 else delta_eef_pose[-1]
        print('THTRLEnv._set_init_approach_pose(): robot delta pose =\n', delta_eef_pose)
        eef_cmd = Float64MultiArray()
        eef_cmd.data = delta_eef_pose
        self._set_visual_servo_cmd(eef_cmd)
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        if appr_success_msg.data == True:
            print("THTRLEnv._set_init_approach_pose(): successful")
            print('THTRLEnv._set_init_approach_pose(): after approach robot current pose =\n', self.obs3.robot_state.eef_state)
        else:
            print("THTRLEnv._set_init_approach_pose(): failed")
            exit(0)
        self.approachCorrection(wait_for_input=False)
        if self.robot_type == 'hexaglide':
            _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
            self.robot_init_pose = _current_eef_state_msg.data
            print('THTRLEnv._set_init_approach_pose(): THTRLEnv.robot_init_pose after approachCorrection =\n', self.robot_init_pose)

        print('THTRLEnv._set_init_approach_pose(): please check if the pins are ready for new episode. Enter y if pins are acceptable, else enter n.')
        user_input = input('In case pins are unacceptable (n), the Hexaglide will move to home position. y/n')
        if user_input == 'y':
            return
        else:
            self.goToReferencePose()
            self._set_init_pose()
            return

    def goToReferencePose(self):
        print('THTRLEnv.goToReferencePose(): robot reference pose =\n', self.robot_reference_pose)
        print('THTRLEnv.goToReferencePose(): robot current pose =\n', self.obs3.robot_state.eef_state)
        delta_eef_position = list(map(operator.sub, self.obs3.robot_state.eef_state[:3], self.robot_reference_pose[:3]))
        eef_current_orientation_euler = euler_from_quaternion(self.obs3.robot_state.eef_state[3],
                                                              self.obs3.robot_state.eef_state[4],
                                                              self.obs3.robot_state.eef_state[5],
                                                              self.obs3.robot_state.eef_state[6])
        delta_eef_orientation = quaternion_from_euler_theta([-angle for angle in eef_current_orientation_euler])
        delta_eef_pose = delta_eef_position + delta_eef_orientation
        # delta_eef_orientation = relativeQuaternionRotation(q0=self.obs3.robot_state.eef_state[3:],
        #                                                    q1=self.robot_reference_pose[3:])
        print('THTRLEnv.goToReferencePose(): robot delta pose =\n', delta_eef_pose)
        eef_cmd = Float64MultiArray()
        eef_cmd.data = delta_eef_pose
        self._set_visual_servo_cmd(eef_cmd)
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        if appr_success_msg.data == True:
            print("THTRLEnv.goToReferencePose(): successful")
            print('THTRLEnv.goToReferencePose(): after approach robot current pose =\n',
                  self.obs3.robot_state.eef_state)
        else:
            print("THTRLEnv.goToReferencePose(): failed")
            exit(0)
    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self._log()
        self.step_count = 0
        self.success = False
        self.old_z_relative_coord = [0.0]
        self.cumulated_episode_reward = 0.0

    def check_eef_in_working_space(self, current_relative_eef_pose):
        """"
            Checks if eef is located in the working space, which is definied in config.yaml
            True, if eef is in working space. False otherwise.
        """
        if self.workspace_x[0] > current_relative_eef_pose[0] or current_relative_eef_pose[0] > \
                self.workspace_x[1]:
            self.workspace_x_violation = True
        else:
            self.workspace_x_violation = False

        if self.workspace_y[0] > current_relative_eef_pose[1] or current_relative_eef_pose[1] > \
                self.workspace_y[1]:
            self.workspace_y_violation = True
        else:
            self.workspace_y_violation = False

        print('THTRLEnv.check_eef_in_working_space(): current_relative_eef_pose_x =', current_relative_eef_pose[0])
        print('THTRLEnv.check_eef_in_working_space(): current_relative_eef_pose_y =', current_relative_eef_pose[1])

        return self.workspace_x_violation, self.workspace_y_violation

    def _set_visual_servo_cmd(self, cmd):
        print('THTRLEnv._set_visual_servo_cmd(): publishing cmd =\n', cmd.data)
        self.visual_servo_cmd_pub.publish(cmd)  # Float64MultiArray

    def _set_action(self, action, wait_for_input=False):
        """
        Move the robot based on the action variable given
        """

        print("\n======== EXECUTING STEP ", self.step_count, " ========")
        print("THTRLEnv._set_action(): ACTION", str(action))

        self.step_count = self.step_count + 1
        if self.init_step:  # TODO done to display images without lag
            self.init_step = False
            self.init_episode = False

        if self.robot_type == 'hexaglide':
            cmd_vector = Float64MultiArray()
            cmd_vector.data = [0, 0, 0]
            H1origin_to_tildaorigin_updated = []  # Coordinates of Tilda origin w.r.t H1 frame after rotation of theta about H1 z-axis
            if action >= 6:     # For rotation actions, rotate about the goal hole
                if action == 6:  # rotation command about z_H1 axis
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[0] * m.cos(self.rotation_stepsize) - self.hole_position_H1[1] * m.sin(self.rotation_stepsize))
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[1] * m.cos(self.rotation_stepsize) + self.hole_position_H1[0] * m.sin(self.rotation_stepsize))
                    cmd_vector.data[0] = cmd_vector.data[0] + (H1origin_to_tildaorigin_updated[0] - self.hole_position_H1[0])
                    cmd_vector.data[1] = cmd_vector.data[1] + (H1origin_to_tildaorigin_updated[1] - self.hole_position_H1[1])
                    eef_orientation_cmd = [0, 0, self.rotation_stepsize]
                    eef_orientation_cmd = quaternion_from_euler_theta(eef_orientation_cmd)
                    cmd_vector.data = np.concatenate((cmd_vector.data, eef_orientation_cmd, np.array([int(wait_for_input)])), axis=0)
                elif action == 7:
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[0] * m.cos(-self.rotation_stepsize) - self.hole_position_H1[1] * m.sin(-self.rotation_stepsize))
                    H1origin_to_tildaorigin_updated.append(self.hole_position_H1[1] * m.cos(-self.rotation_stepsize) + self.hole_position_H1[0] * m.sin(-self.rotation_stepsize))
                    cmd_vector.data[0] = cmd_vector.data[0] + (H1origin_to_tildaorigin_updated[0] - self.hole_position_H1[0])
                    cmd_vector.data[1] = cmd_vector.data[1] + (H1origin_to_tildaorigin_updated[1] - self.hole_position_H1[1])
                    eef_orientation_cmd = [0, 0, -self.rotation_stepsize]
                    eef_orientation_cmd = quaternion_from_euler_theta(eef_orientation_cmd)
                    cmd_vector.data = np.concatenate((cmd_vector.data, eef_orientation_cmd, np.array([int(wait_for_input)])), axis=0)

                self._set_visual_servo_cmd(cmd_vector)
                appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
                if appr_success_msg.data == True:
                    appr_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
                    self.robot_init_pose = appr_eef_state_msg.data
                    print("THTRLEnv._set_action(): finished action "+str(action)+" with success")
                else:
                    print("THTRLEnv._set_action(): action "+str(action)+" execution failed")
                    exit(0)
            else:   # for translation actions: Move hexaglide UP, DOWN, LEFT or RIGHT
                self.action_pub.publish(action)
                appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)  # timeout = 5
                if appr_success_msg.data == True:
                    print("THTRLEnv._set_action(): finished action "+str(action)+" with success")
                else:
                    print("THTRLEnv._set_action(): action "+str(action)+" execution failed")
                    exit(0)

        elif self.robot_type == 'yumi':
            self.action_pub.publish(action)
            if self.init_step:  # TODO done to display images without lag
                self.init_step = False
            appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)  # timeout = 5
            if appr_success_msg.data == True:
                print("THTRLEnv._set_action(): finished action "+str(action)+" with success")
            else:
                print("THTRLEnv._set_action(): action "+str(action)+" execution failed")
                exit(0)

    # Attention: This should only be used for RL, so instead of obs we get data from obs3 in this def
    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have access to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """

        print("THTRLEnv._get_obs()")
        start = time.time()
        _current_pressure = self.obs3.current_pressure()
        print("THTRLEnv._get_obs(): got pressure in [s]:", time.time() - start)
        #_current_pressure_hist = self.obs.pressure_history(_current_pressure)

        # _current_image1, _current_image2 = self.obs.current_visualdata()
        start = time.time()
        _current_cropped_image3, _previous_cropped_image3 = self.obs3.current_cropped_visualdata(list_tuples_goal_pixel_image3=[(self.x3, self.y3)])
        if self.init_step:
            # After episode resetting the previous cropped image from camera 3 must be empty
            _previous_cropped_image3 = np.zeros((self.obs3.h, self.obs3.w, 3), dtype=np.uint8)
        print("THTRLEnv._get_obs(): got images in [s]:", time.time() - start)

        # Publish cropped image for visualization
        _cropped_img3 = None
        if self.diff_image_as_input:
            print("THTRLEnv._get_obs(): working with grayscale diff images")
            raise NotImplementedError
        else:
            print("THTRLEnv._get_obs(): working with raw image")
            _cropped_img3 = self.obs3.numpy_to_image(_current_cropped_image3, 'bgr8')
            _prev_cropped_img3 = self.obs3.numpy_to_image(_previous_cropped_image3, 'bgr8')
            print("THTRLEnv._get_obs(): type(_cropped_img3.data) =", type(_cropped_img3.data))
            self.cropped_img3_pub.publish(_cropped_img3)
            self.previous_cropped_img3_pub.publish(_prev_cropped_img3)

        _robot_pose = self.obs3.robot_state.eef_state
        _current_robot_position = _robot_pose[:3]
        _current_robot_orientation = _robot_pose[3:]

        if self.orientation_as_euler:  # TODO: changes in the observation space
            # CHECK FORMAT FOR RECEIVING ROBOT ORIENTATION (x,y,z,w) or (w,x,y,z)
            _current_robot_orientation = euler_from_quaternion(_current_robot_orientation[0],
                                                               _current_robot_orientation[1],
                                                               _current_robot_orientation[2],
                                                               _current_robot_orientation[3])

        # Use relative robot pose as observation (current pose - init pose after approach phase)
        _relative_robot_position = [_robot_pose[i] - self.robot_init_pose[i] for i in range(3)]
        _relative_robot_orientation = relativeQuaternionRotation(q0=_robot_pose[3:], q1=self.robot_init_pose[3:])
        # Debugging
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_robot_position = _current_eef_state_msg.data[:3]
        _current_robot_orientation = _current_eef_state_msg.data[3:]
        _current_robot_pose = _current_robot_position + _current_robot_orientation
        print("THTRLEnv._get_obs(): _robot_init_pose =\n", self.robot_init_pose)
        print("THTRLEnv._get_obs(): _current_robot_pose =\n", _current_robot_pose)
        _current_robot_pose = np.array(_current_robot_pose)
        _relative_robot_pose = _relative_robot_position + _relative_robot_orientation
        print("THTRLEnv._get_obs(): _relative_robot_pose =\n", _relative_robot_pose)
        _relative_robot_pose = np.array(_relative_robot_pose)
        _current_pressure = np.array([_current_pressure])
        print("THTRLEnv._get_obs(): _current_pressure =\n", _current_pressure)

        self.old_z_relative_coord.append(_relative_robot_position[2])

        observations = {"current_image3": _current_cropped_image3,
                        "previous_image3": _previous_cropped_image3,
                        "pressure": _current_pressure,
                        "pose": _relative_robot_pose}
        print("THTRLEnv._get_obs(): finished getting observations")
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        print("THTRLEnv._is_done()")
        pressure_exceeded = observations["pressure"][0].item() > self.max_pressure
        if self.robot_type == 'yumi':
            depth_exceeded = observations["pose"][2].item() < self.max_depth / 1000
        elif self.robot_type == 'hexaglide':
            print('observations["pose"][2].item(): ', observations["pose"][2].item())
            depth_exceeded = observations["pose"][2].item() >= self.max_depth / 1000
        time_up = self.step_count >= self.max_step_count  # TODO incorporate steps
        component_lost_from_gripper = False  # TODO from opencv: vision task
        # Only ask whether goal is reached when target depth is exceeded
        if depth_exceeded:
            success = input("Goal reached? Press y for success, any other key otherwise!")
            if success == "y":
                self.success = True
            elif success == "w" or success == "a" or success == "s" or success == "d" or success == "e":
                print("Repeat w/a/s/d/e input, because depth is reached")
                depth_exceeded = False
                self.success = False
            else:
                self.success = False

        # Check if workspace exceeded:
        x_limit_violation, y_limit_violation = self.check_eef_in_working_space(observations['pose'][:3])
        self.workspace_violation = x_limit_violation or y_limit_violation

        # Alternative to input:
        # goal_reached = True if distance_norm < 0.001 & orientation_difference < 0.001 & _current_robot_pose[2] < zlimit else False
        # eef_not_in_working_space = not self.check_eef_in_working_space(observations.get_eef)
        # component_not_in_working_space = not self.check_eef_in_working_space(
        #    observations.get_component_eef)  # maybe remove this, we dont have 3d coord, or pin detection is not working

        done = pressure_exceeded or depth_exceeded or time_up or component_lost_from_gripper or self.success or self.workspace_violation  # TODO add component_in_image
        if done:
            final_str = "pressure_exceeded: ", pressure_exceeded, "depth_exceeded: ", depth_exceeded, "time_up: ", time_up, \
                        ", component_lost_from_gripper: ", component_lost_from_gripper, "success: ", self.success, \
                        ", workspace_violation: ", self.workspace_violation
            print(final_str)
        #       eef_not_in_working_space  # | component_not_in_working_space
        print("THTRLEnv._is_done() =", done)
        return done

    def moveToTildaOrigin(self, pin_side_cam2=None, wait_for_input=True):
        '''
        Finds primary pin tip and moves it (only translation) to the intersection of cameras' main optical axes
        (so called 'tilda origin', since tilda coordinate system is based on this optical intersection)
        '''
        print('\nTHTRLEnv.moveToTildaOrigin()')
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        eef_position = _current_eef_state_msg.data[:3]
        eef_orientation = _current_eef_state_msg.data[3:]
        # Case for SIP and DIP:
        if self.component_package == 'DIP':
            free_pin_tip_tilda = None
            eef_to_tilda, selected_pin_tip_tilda = self.visual_servo.autoSelectDIPPinTip(eef_position, self.n_pins,
                                                                                         self.cam1_tip_window_px)
        elif self.component_package == 'SIP':
            eef_to_tilda, selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.visual_servo.autoSelectRightPinTip(
                eef_position, pin_side_cam2)
        eef_cmd = Float64MultiArray()
        '''
        Warning:
        YuMi controller interprets translation commands as relative but rotations as absolute!
        Hexaglide controller interprets all commands (both translation and rotation) as relative!
        Since moveToTildaOrigin only involves translations, quaternion of the command is set to no relative rotation
        in case of Hexaglide.
        '''
        if self.robot_type == 'yumi':
            eef_cmd.data = np.concatenate(
                (eef_to_tilda - eef_position, eef_orientation, np.array([int(wait_for_input)])), axis=0)
        elif self.robot_type == 'hexaglide':
            eef_cmd.data = np.concatenate(
                (eef_to_tilda - eef_position, [0, 0, 0, 1], np.array([int(wait_for_input)])), axis=0)
        self._set_visual_servo_cmd(eef_cmd)
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        if appr_success_msg.data == True:
            print("THTRLEnv.moveToTildaOrigin(): successfully finished visual calibration iteration 1")
        else:
            print("THTRLEnv.moveToTildaOrigin(): failed at visual calibration iteration 1")
            exit(0)
        return selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2

    def graspingAngleErrorCompensation(self, pin_side_cam2, pin_separation, wait_for_input=True):
        print('\nTHTRLEnv.graspingAngleCompensation()')
        # Compensate the THD rotation around vertical axis
        # For 2 short pins p1, p2 that means orienting them so that:
        # the line through p1 and p2 goes directly through the line of sight of camera 2
        # print("THTVisualServo: Getting current complete images from cameras...")
        _current_image1, _current_image2 = self.obs.current_visualdata()
        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_eef_pos = _current_eef_state_msg.data[:3]
        _current_eef_orientation = _current_eef_state_msg.data[3:]
        eef_to_tilda, selected_pin_tip_tilda, free_pin_tip_tilda, pin_side_cam2 = self.visual_servo.autoSelectRightPinTip(
            _current_eef_pos, pin_side_cam2)
        p1 = selected_pin_tip_tilda
        p2 = free_pin_tip_tilda
        print('THTRLEnv.graspingAngleErrorCompensation(): p1 =', p1)
        print('THTRLEnv.graspingAngleErrorCompensation(): p2 =', p2)
        pin_separation_measured = m.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
        p2_d = np.add(p1, np.array([0, pin_separation_measured, p2[2]]))  # d means 'desired'
        diff_p1_p2 = p2 - p1
        diff_p1_p2_d = p2_d - p1
        print('THTRLEnv.graspingAngleErrorCompensation(): measured pin separation =\n', pin_separation_measured)
        print('THTRLEnv.graspingAngleErrorCompensation(): nominal pin separation =\n', pin_separation)
        # TODO: Add the pin separation tolerance in the config
        if abs(pin_separation - pin_separation_measured) > 0.0015:
            print('THTRLEnv.graspingAngleErrorCompensation(): pin separation is too different from the nominal! Exiting...')
            exit(0)
            # input('Fix the pin separation and press Enter to continue...')
            # self.goToReferencePose()
            # self._set_init_pose()
            # return

        # Theta = Rotation of THD around vertical axis (z-axis Yumi)
        theta = m.acos(np.dot(diff_p1_p2_d, diff_p1_p2) / (np.linalg.norm(diff_p1_p2_d) * np.linalg.norm(diff_p1_p2)))
        theta_degrees = theta / m.pi * 180.0
        eef_to_tilda_right_desired = eef_to_tilda
        eef_to_tilda_right_desired[0] = eef_to_tilda[0] + pin_separation

        _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
        _current_eef_pos = _current_eef_state_msg.data[:3]
        _current_eef_orientation = _current_eef_state_msg.data[3:]
        sign = -1 if pin_side_cam2 == 'right' else 1
        theta = sign * theta
        print('THTRLEnv.graspingAngleErrorCompensation(): theta [deg] =\n', sign * theta_degrees)

        eef_cmd = Float64MultiArray()
        _updated_eef_pos = [0, 0, 0]

        if self.robot_type == 'yumi':
            roll, pitch, yaw = euler_from_quaternion_theta(_current_eef_orientation)
            yaw = yaw + theta
            _updated_eef_orientation_euler = [roll, pitch, yaw]
            _updated_eef_orientation = quaternion_from_euler_theta(_updated_eef_orientation_euler)
            eef_cmd.data = np.concatenate((_updated_eef_pos, _updated_eef_orientation, np.array([int(wait_for_input)])),
                                          axis=0)

        elif self.robot_type == 'hexaglide':
            roll = 0.0
            pitch = 0.0
            yaw = -theta
            _updated_eef_orientation_euler = [roll, pitch, yaw]

            H1origin_to_tildaorigin_updated = []  # Coordinates of Tilda origin w.r.t H1 frame after rotation of theta about H1 z-axis
            H1origin_to_tildaorigin_updated.append(
                self.H1origin_to_tildaorigin[0] * m.cos(yaw) - self.H1origin_to_tildaorigin[1] * m.sin(yaw))
            H1origin_to_tildaorigin_updated.append(
                self.H1origin_to_tildaorigin[1] * m.cos(yaw) + self.H1origin_to_tildaorigin[0] * m.sin(yaw))

            _updated_eef_pos[0] = H1origin_to_tildaorigin_updated[0] - self.H1origin_to_tildaorigin[0]
            _updated_eef_pos[1] = H1origin_to_tildaorigin_updated[1] - self.H1origin_to_tildaorigin[1]

            _updated_eef_orientation = quaternion_from_euler_theta(_updated_eef_orientation_euler)
            # print('H1origin_to_tildaorigin_updated: ', H1origin_to_tildaorigin_updated)
            print('THTRLEnv.graspingAngleErrorCompensation(): _updated_eef_pos =\n', _updated_eef_pos)

            eef_cmd.data = np.concatenate((_updated_eef_pos, _updated_eef_orientation, np.array([int(wait_for_input)])),
                                          axis=0)

        self._set_visual_servo_cmd(eef_cmd)
        start = time.time()
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        print('THTRLEnv.graspingAngleErrorCompensation(): APPROACH MESSAGE 2 IN: ', time.time() - start)
        if appr_success_msg.data == True:
            print("THTRLEnv.graspingAngleErrorCompensation(): part reorientation SUCCESSFUL")
        else:
            print("THTRLEnv.graspingAngleErrorCompensation(): part reorientation FAILED")
            exit(0)

    def approachGoal(self, selected_pin_tip_tilda, wait_for_input):
        print('\nTHTRLEnv.approachGoal()')
        cmd_vector = Float64MultiArray()
        cmd_vector.data = np.dot(self.visual_servo.rot_tilda_to_base,
                                 (np.array(self.mpd_corrected_goal_pose_tilda)) / 1000 - selected_pin_tip_tilda)
        cmd_vector.data[2] = self.tildaorigin_to_pcb_z if cmd_vector.data[2] < self.tildaorigin_to_pcb_z else \
            cmd_vector.data[2]
        if self.robot_type == 'yumi':
            _current_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
            _current_eef_pos = _current_eef_state_msg.data[:3]
            _current_eef_orientation = _current_eef_state_msg.data[3:]
            cmd_vector.data = np.concatenate((cmd_vector.data, _current_eef_orientation,
                                              np.array([int(wait_for_input)])), axis=0)
        elif self.robot_type == 'hexaglide':
            roll = 0.0
            pitch = 0.0
            if abs(self.D_dev_m) > self.D_dev_advanced_search_trigger_m:
                pitch = self.rotation_stepsize
            yaw = self.part_reorientation_angle
            _updated_eef_orientation_euler = [roll, pitch, yaw]

            # H1origin_to_tildaorigin_updated = []  # Coordinates of Tilda origin w.r.t H1 frame after rotation of theta about H1 z-axis
            # H1origin_to_tildaorigin_updated.append(
            #     self.hole_position_H1[0] * m.cos(yaw) - self.hole_position_H1[1] * m.sin(yaw))
            # H1origin_to_tildaorigin_updated.append(
            #     self.hole_position_H1[1] * m.cos(yaw) + self.hole_position_H1[0] * m.sin(yaw))

            # cmd_vector.data[0] = cmd_vector.data[0] + (H1origin_to_tildaorigin_updated[0] - self.hole_position_H1[0])
            # cmd_vector.data[1] = cmd_vector.data[1] + (H1origin_to_tildaorigin_updated[1] - self.hole_position_H1[1])

            # Calculate R_approach based on assembled component data
            rx = R.from_euler('x', roll, degrees=False)
            ry = R.from_euler('y', pitch, degrees=False)
            rz = R.from_euler('z', yaw, degrees=False)
            r = rz * ry * rx
            R_approach = r.as_matrix()
            cmd_vector.data = cmd_vector.data + np.dot(R_approach, self.hole_position_H1) - self.hole_position_H1
            _updated_eef_orientation = quaternion_from_euler_theta(_updated_eef_orientation_euler)
            # cmd_vector.data[2] = self.tildaorigin_to_pcb_z
            cmd_vector.data = np.concatenate(
                (cmd_vector.data, _updated_eef_orientation, np.array([int(wait_for_input)])), axis=0)
            # cmd_vector.data = np.concatenate((cmd_vector.data, [0, 0, 0, 1], np.array([int(wait_for_input)])),
            #                                  axis=0)
        # Send command to YuMi
        print('THTRLEnv.approachGoal(): cmd_vector.data =\n', cmd_vector.data)
        self._set_visual_servo_cmd(cmd_vector)
        appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)
        if appr_success_msg.data == True:
            appr_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
            self.robot_init_pose = appr_eef_state_msg.data
            print("THTRLEnv.approachGoal(): finished with success")
        else:
            print("THTRLEnv.approachGoal(): finished without success, exiting")
            exit(0)

    def approachCorrection(self, wait_for_input):
        # Approach correction phase
        print('\nTHTRLEnv.approachCorrection()')
        _current_cropped_image1, _current_cropped_image2 = self.obs.current_cropped_visualdata(
            self.mpd_corrected_goal_pose_pixel_image1,
            self.mpd_corrected_goal_pose_pixel_image2)

        _cropped_img1 = self.obs.numpy_to_image(_current_cropped_image1, 'bgr8')
        _cropped_img2 = self.obs.numpy_to_image(_current_cropped_image2, 'bgr8')
        self.cropped_img1_pub.publish(_cropped_img1)
        self.cropped_img2_pub.publish(_cropped_img2)

        image1_pin_location, image2_pin_location, pin_dev_cam1, pin_dev_cam2 = self.mpd.mpd(_current_cropped_image1,
                                                                                            _current_cropped_image2)
        # Determine approach side cases (e.g. 'left' means hole is on the left of the pin)
        approach_side_cases = []
        # cam1
        if pin_dev_cam1 >= self.pin_thickness_px / 2:
            approach_side_cases.append('left')
        elif -self.pin_thickness_px / 2 <= pin_dev_cam1 < self.pin_thickness_px / 2:
            approach_side_cases.append('center')
        elif pin_dev_cam1 < -self.pin_thickness_px / 2:
            approach_side_cases.append('right')
        # cam2
        if pin_dev_cam2 >= self.pin_thickness_px / 2:
            approach_side_cases.append('left')
        elif -self.pin_thickness_px / 2 <= pin_dev_cam2 < self.pin_thickness_px / 2:
            approach_side_cases.append('center')
        elif pin_dev_cam2 < -self.pin_thickness_px / 2:
            approach_side_cases.append('right')

        print('THTRLEnv.approachCorrection(): approach side cases (hole w.r.t. the pin): ', approach_side_cases)

        # If MPD detects pin as outside of the tolerance...
        if abs(pin_dev_cam1) > self.center_tolerance_px or abs(pin_dev_cam2) > self.center_tolerance_px:
            orientation_cmd = None
            if self.robot_type == 'yumi':
                corrected_goal_pose_pixel_image1 = self.mpd_corrected_goal_pose_pixel_image1 + pin_dev_cam1
                corrected_goal_pose_pixel_image2 = self.mpd_corrected_goal_pose_pixel_image2 + pin_dev_cam2
                self.corrected_goal_pose_tilda = self.visual_servo.calc_1_tip_coords(
                    corrected_goal_pose_pixel_image1,
                    corrected_goal_pose_pixel_image2,
                    self.visual_servo.x_tilda_c1,
                    self.visual_servo.y_tilda_c2)
                orientation_cmd = self.robot_init_pose[3:]
            elif self.robot_type == 'hexaglide':
                # CALCULATE DIFF GOAL POSE WITH GEOMETRY:
                corrected_goal_pose_pixels = []
                # Depending on the selected side of the hole and approach side...
                for c in range(2):
                    pin_dev = pin_dev_cam1 if c == 0 else pin_dev_cam2
                    mpd_corrected_goal_pose_pixel = self.mpd_corrected_goal_pose_pixel_image1 if c == 0 else self.mpd_corrected_goal_pose_pixel_image2
                    correction_px = 0
                    if self.click_goal_image:
                        if approach_side_cases[c] == 'left':
                            correction_px = pin_dev + self.pin_thickness_px / 2
                        elif approach_side_cases[c] == 'center':
                            correction_px = pin_dev
                        elif approach_side_cases[c] == 'right':
                            correction_px = pin_dev - self.pin_thickness_px / 2
                    else:
                        correction_px = pin_dev
                    print('Camera ', c+1)
                    print('Approach correction: applied correction_px = ', correction_px)
                    tup = [(mpd_corrected_goal_pose_pixel[0][0] + correction_px, mpd_corrected_goal_pose_pixel[0][1])]
                    corrected_goal_pose_pixels.append(tup)
                # corrected_goal_pose_pixel_image1 = [(self.mpd_corrected_goal_pose_pixel_image1[0][0] + pin_dev_cam1, self.mpd_corrected_goal_pose_pixel_image1[0][1])]
                # corrected_goal_pose_pixel_image2 = [(self.mpd_corrected_goal_pose_pixel_image2[0][0] + pin_dev_cam2, self.mpd_corrected_goal_pose_pixel_image2[0][1])]
                print('Corrected goal pixel value depending on approach case: ', corrected_goal_pose_pixels)
                self.corrected_goal_pose_tilda = self.visual_servo.calc_1_tip_coords(
                    corrected_goal_pose_pixels[0],
                    corrected_goal_pose_pixels[1],
                    self.visual_servo.x_tilda_c1,
                    self.visual_servo.y_tilda_c2)
                # ESTIMATE DIFF GOAL POSE WITH FITTING FUNCTION
                # self.corrected_goal_pose_pixel_image1 = [(self.corrected_goal_pose_pixel_image1[0][0] + pin_dev_cam1,
                #                                           self.corrected_goal_pose_pixel_image1[0][1])]
                # self.corrected_goal_pose_pixel_image2 = [(self.corrected_goal_pose_pixel_image2[0][0] + pin_dev_cam2,
                #                                           self.corrected_goal_pose_pixel_image2[0][1])]
                # corrected_x_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image2[0][0],
                #                                                        self.corrected_goal_pose_pixel_image2[0][1],
                #                                                        self.visual_servo.x_tilda_params[0],
                #                                                        self.visual_servo.x_tilda_params[1],
                #                                                        self.visual_servo.x_tilda_params[2],
                #                                                        self.visual_servo.x_tilda_params[3],
                #                                                        self.visual_servo.x_tilda_params[4],
                #                                                        self.visual_servo.x_tilda_params[5])
                #
                # corrected_y_tilda_goal = self.visual_servo.square_func(self.corrected_goal_pose_pixel_image1[0][0],
                #                                                        self.corrected_goal_pose_pixel_image1[0][1],
                #                                                        self.visual_servo.y_tilda_params[0],
                #                                                        self.visual_servo.y_tilda_params[1],
                #                                                        self.visual_servo.y_tilda_params[2],
                #                                                        self.visual_servo.y_tilda_params[3],
                #                                                        self.visual_servo.y_tilda_params[4],
                #                                                        self.visual_servo.y_tilda_params[5])
                # self.corrected_goal_pose_tilda = (
                # corrected_x_tilda_goal, corrected_y_tilda_goal, self.mpd_corrected_goal_pose_tilda[2])
                orientation_cmd = [0, 0, 0, 1]

            diff_goal_pose_tilda = list(map(operator.sub, self.mpd_corrected_goal_pose_tilda, self.corrected_goal_pose_tilda))
            print('image1_pin_location, image2_pin_location', image1_pin_location, image2_pin_location)
            print('pin_dev_cam1, pin_dev_cam2: ', pin_dev_cam1, pin_dev_cam2)
            diff_goal_pose_tilda[2] = 0
            print('diff_goal_pose_tilda: ', diff_goal_pose_tilda)
            eef_cmd = np.dot(self.visual_servo.rot_tilda_to_base, np.array(diff_goal_pose_tilda) / 1000)
            # eef_cmd = [diff_goal_pose_tilda[1] / 1000, -diff_goal_pose_tilda[0] / 1000, 0]

            cmd_vector = Float64MultiArray()
            cmd_vector.data = np.concatenate((eef_cmd, orientation_cmd,
                                              np.array([int(wait_for_input)])), axis=0)
            self._set_visual_servo_cmd(cmd_vector)
            appr_success_msg = rospy.wait_for_message('/approach_success', Bool, timeout=None)

            if appr_success_msg.data == True:
                appr_eef_state_msg = rospy.wait_for_message(self.eef_state_pub, Float64MultiArray, timeout=None)
                self.robot_init_pose = appr_eef_state_msg.data
                print("THTRLEnv.approachCorrection(): successful")
            else:
                print("THTRLEnv.approachCorrection(): fail, exiting script")
                exit(0)
        else:
            # If MPD detects pin within tolerance...
            self.corrected_goal_pose_tilda = self.visual_servo.calc_1_tip_coords(self.mpd_corrected_goal_pose_pixel_image1,
                                                                                 self.mpd_corrected_goal_pose_pixel_image2,
                                                                                 self.visual_servo.x_tilda_c1,
                                                                                 self.visual_servo.y_tilda_c2)
            print("THTRLEnv.approachCorrection(): pin within tolerance, skipping approach correction")

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given.
        """
        print("\nTHTRLEnv._compute_reward():")
        reward = 0.0
        # TODO add goal as an observation?
        # _current_robot_pose = observations._current_robot_pose
        p = observations["pressure"][0].item()
        p_max = 200
        reward_pressure = -0.05 * (p - p_max) if p > p_max else 0  # TODO remove hardcoded: Multiplication factor changed from -0.1 to -0.05
        # reward_pressure = -10 if observations["pressure"][0].item() > 190 else reward_pressure

        # x_limit_violation, y_limit_violation = self.check_eef_in_working_space(observations['pose'][:3])
        # reward_workspace_exceeded = -10 if x_limit_violation or y_limit_violation else 0
        reward_workspace_exceeded = -150 if done & self.workspace_violation else 0

        # z = pose[2]; incr z means a deeper z
        # ToDo: Check signs for Hexaglide
        # ToDo: why old_z_relative_coord[-2]?
        if observations["pose"][2].item() > self.old_z_relative_coord[-2]:
            reward_depth_incr = 20.0
        elif observations["pose"][2].item() < self.old_z_relative_coord[-2]:
            reward_depth_incr = -20.0
        else:
            reward_depth_incr = 0.0

        if self.image_as_input_without_cv or self.metallic_pixel_detection:
            if self.init_step:  # TODO done to display images without lag
                self.init_step = False
            reward_time = -10.0
            reward_success = 500 if done & self.success else 0
            reward = reward_pressure + reward_depth_incr + reward_time + reward_success + reward_workspace_exceeded
        else:
            if self.init_step:  # needed because old error between pin and goal in pixel coordinates not exists in init step
                self.init_step = False
                pose_reward = 0
                reward_depth_incr = 0
            else:
                # pose reward: nearer to goal gives positive reward for each image
                if self.goal_error_coords == "Pixels":
                    pose_reward1 = 0.1 if observations["goal_error_hist"][1] < observations["goal_error_hist"][0] \
                        else 0  # [0] previous goal error, [1] current goal error in image1 (Pixel coords)
                    pose_reward2 = 0.1 if observations["goal_error_hist"][-1] < observations["goal_error_hist"][-2] \
                        else 0  # [-2] previous goal error, [-1] current goal error in image2 (pixel coords)
                    pose_reward = pose_reward1 + pose_reward2
                elif self.goal_error_coords == "EEF":
                    pose_reward = 0.1 if observations["goal_error_hist"][1] < observations["goal_error_hist"][0] \
                        else 0  # [0] previous goal error, [1] current goal error in EEF coords

            # tilt should be nearly the same as old tilt
            reward_orientation_robot_pose = 0
            reward_time = -0.1
            # z limit is the smallest possible value for two center point
            reward_success = 100 if done & self.success else 0
            reward = pose_reward + reward_depth_incr + reward_time + reward_success + reward_pressure + \
                     reward_orientation_robot_pose + reward_workspace_exceeded

        print("+++ THTRLEnv _compute_reward +++")
        print("    reward_pressure = ", reward_pressure)
        print("    reward_depth_incr = ", reward_depth_incr)
        print("    reward_time = ", reward_time)
        print("    reward_success = ", reward_success)
        print("    reward_workspace_exceeded = ", reward_workspace_exceeded)
        print("+++ THTRLEnv: Computed reward = ", reward)
        print()
        return reward

    def _reset(self):
        print("\nTHTRLEnv._reset()")
        rl_init_pose_msg = Float64MultiArray()
        rl_init_pose_msg.data = self.robot_init_pose
        # print("THTRLEnv._reset(): publishing rl_init_pose =\n", self.robot_init_pose)
        # self.rl_init_pose_pub.publish(rl_init_pose_msg)
        self.init_step = True

    def _log(self):
        if self.obs_logfile is None:
            try:
                self.obs_logfile = self.output_dir + "/state_" + \
                                   datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
                print("obs_logfile", self.obs_logfile)
                self.logger = initialize_logger(filename=self.output_dir + "/console.log")
            except Exception:
                return

        if len(self.obs_per_step) == 0:
            return

        data = []
        # if self.test_mode:
        #     data.append(self.obs_per_step)
        # else:
        #     data.append([])
        data.append([])
        data.append(self.reward_per_step)
        data.append(self.reward_details_per_step)

        try:
            tmp = np.load(self.obs_logfile, allow_pickle=True).tolist()
            tmp.append(data)
            np.save(self.obs_logfile, tmp)
            tmp = None
        except IOError:
            np.save(self.obs_logfile, [data], allow_pickle=True)

        self.reward_per_step = []
        self.obs_per_step = []
        self.reward_details_per_step = []

    # Internal TaskEnv Methods

