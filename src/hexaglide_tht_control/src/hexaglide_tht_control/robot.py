#! /usr/env/python
import math
import operator

import numpy as np
from Phidget22.Phidget import *
from Phidget22.Devices.Stepper import *
from Phidget22.Devices.Encoder import *
from Phidget22.Devices.DigitalInput import *
from Phidget22.Net import *
import time
from waiting import wait
import yaml
import threading
import os
from scipy.spatial.transform import Rotation as R
from tht_control.utils import *
from tht_control.common import *
from hexaglide_tht_control.kinematic_solver_v2 import KinematicSolver

global movement_flag
movement_flag = [False, False, False, False, False, False]


def onStateChangeHandler(self, state):
    if self.linkedMotor.getIsMoving() and state == False:
        self.linkedMotor.setVelocityLimit(0.0)


def parameter_setting_thread(robot, idx, commanded_q, vel_cmd, acceleration_cmd):
    robot.stepper[idx].setVelocityLimit(vel_cmd)
    robot.stepper[idx].setAcceleration(acceleration_cmd)
    robot.stepper[idx].setTargetPosition(commanded_q)


def motor_control_thread(robot, idx, start_signal):
    robot.stepper[idx].setEngaged(False)
    movement_flag[idx] = True
    start_signal.wait()  # Wait for the start signal
    # engage_time = time.time()
    robot.stepper[idx].setEngaged(True)
    # print('Motor ' + str(idx) + ' Engage time: ' + str(time.time() - engage_time))
    robot.motion_flag[idx] = robot.stepper[idx].getIsMoving()
    while robot.motion_flag[idx]:
        robot.motion_flag[idx] = robot.stepper[idx].getIsMoving()
        continue
    # print('Thread ', idx, ' closing')


class Hexaglide(KinematicSolver):
    def __init__(self, control_mode=None):
        super().__init__()
        Net.enableServerDiscovery(PhidgetServerType.PHIDGETSERVER_DEVICEREMOTE)
        # os.chdir('..')
        # print(os.path.abspath(os.curdir))
        # with open(os.path.abspath(os.curdir) + '/config/config.yaml', 'r') as file:
        #     config = yaml.safe_load(file)
        config_path = get_config_path(rospackage_name='hexaglide_tht_control',
                                      rel_path_from_package_to_file='config', yaml_file_name='config_hexaglide.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.degrees_per_step = 1.8 / 4.25
        self.rotation_per_degrees = 1 / 360
        self.steps_per_movement = 1 / 16
        self.mm_per_rotation = 10
        # ID Format: [[Port, Serial number]]
        self.phidget_dict = self.config['phidget_param']['phidget_dict']
        self.motor_id = self.config['phidget_param']['motor_id']
        if control_mode is None:
            self.control_mode = self.config['phidget_param']['control_mode']
        else:
            self.control_mode = control_mode
        self.rescale_factor = self.config['phidget_param']['rescale_factor']
        self.encoder_id = self.config['phidget_param']['encoder_id']
        self.pos_limit_switch_id = self.config['phidget_param']['pos_limit_switch_id']
        self.neg_limit_switch_id = self.config['phidget_param']['neg_limit_switch_id']

        # Initialize Motor Controller:
        self.stepper = self.init_motor_controller()
        self.motion_flag = [False, False, False, False, False, False]

        # Joints Zero Error for Referencing:
        self.joints_zero_error = self.config['referencing_param']['joints_zero_error']
        self.reference_joint_position = self.config['referencing_param']['reference_joint_position']
        self.reference_eef_pose = self.config['referencing_param']['reference_eef_pose']
        self.joint_position = []

        # Initialize Encoders:
        self.encoder = self.init_motor_encoders()

        # Initialize Limit Switches:
        self.pos_limit_switch = self.init_limit_switch(self.pos_limit_switch_id, switch='pos')
        self.neg_limit_switch = self.init_limit_switch(self.neg_limit_switch_id, switch='neg')

        # Engage The Motors:
        for idx in range(len(self.stepper)):
            self.stepper[idx].setEngaged(True)

        self.eef_pose = None
        self.current_eef_pose = None
        self.eef_pose_hist = []

        # Hexaglide param:
        self.max_joint_velocity_constant = self.config['hexaglide_param']['max_velocity_constant']
        self.max_joint_acceleration_constant = self.config['hexaglide_param']['max_acceleration_constant']
        self.joint_max_velocity = self.config['hexaglide_param']['joint_max_velocity']
        self.joint_max_acceleration = self.config['hexaglide_param']['joint_max_acceleration']
        self.init_rotation_eef_base = self.config['hexaglide_param']['init_rotation_eef_base']
        rx = R.from_euler('x', self.init_rotation_eef_base[0], degrees=True)
        ry = R.from_euler('y', self.init_rotation_eef_base[1], degrees=True)
        rz = R.from_euler('z', self.init_rotation_eef_base[2], degrees=True)
        r = rz * ry * rx
        self.rotation_matrix_H1 = r.as_matrix()  # Rotation matrix between EEF to Base (use r.as_dcm for python 2.7)
        self.safe_eef_pose_change = self.config['workspace']['deltaX_safe_limit']
        self.start_position_0 = self.init_translation_vector
        self.start_rotation_0 = self.init_rotation_matrix
        # For episode resetting with move_to_joint_pose
        self.rl_start_position_0 = None
        self.rl_start_rotation_0 = None

    def init_motor_encoders(self):
        # Initialize motor encoders
        encoder_id = self.encoder_id
        start = time.time()
        encoder = []
        for idx, encoder_id in enumerate(encoder_id):
            encoder.append(Encoder())
            encoder[idx].setHubPort(encoder_id[0])
            encoder[idx].setDeviceSerialNumber(encoder_id[1])
            encoder[idx].setIsRemote(True)
            encoder[idx].open()
            wait(lambda: encoder[idx].getAttached(), sleep_seconds=0.01,
                 waiting_for='Waiting for motor encoder: ' + str(idx))
            encoder[idx].setDataInterval(encoder[idx].getMinDataInterval())
            encoder[idx].setDataRate(encoder[idx].getMaxDataRate())
        print('Encoders initialized in: ', time.time() - start)
        return encoder

    def init_limit_switch(self, limit_switch_id, switch):
        # Initialize limit switches as digital input
        start = time.time()
        limit_switch = []
        for idx, limit_switch_id in enumerate(limit_switch_id):
            limit_switch.append(DigitalInput())
            limit_switch[idx].setIsHubPortDevice(True)
            limit_switch[idx].setHubPort(limit_switch_id[0])
            limit_switch[idx].setDeviceSerialNumber(limit_switch_id[1])
            limit_switch[idx].linkedMotor = self.stepper[idx]
            limit_switch[idx].setOnStateChangeHandler(onStateChangeHandler)
            limit_switch[idx].setIsRemote(True)
            limit_switch[idx].open()
            wait(lambda: limit_switch[idx].getAttached(), sleep_seconds=0.01,
                 waiting_for='Waiting for limit switch: ' + str(idx))
        print('Limit switches initialized in: ', time.time() - start)
        return limit_switch

    def init_motor_controller(self):
        motor_id = self.motor_id
        control_mode = self.control_mode
        rescale_factor = self.rescale_factor
        # Initialize motor controllers
        if control_mode is not None:
            if control_mode == 'pos':
                control_mode = StepperControlMode.CONTROL_MODE_STEP
            elif control_mode == 'vel':
                control_mode = StepperControlMode.CONTROL_MODE_RUN
            else:
                print('ERROR: Invalid control mode')
        else:
            print('ERROR: Specify the control mode: pos or vel')

        if rescale_factor == 'deg':
            # degrees/movement
            rescale_factor = self.degrees_per_step * self.steps_per_movement
        elif rescale_factor == 'rot':
            # rotation/movement
            rescale_factor = (self.degrees_per_step * self.steps_per_movement) / 360
        elif rescale_factor == 'rad':
            # rad/movement
            rescale_factor = (self.degrees_per_step * math.pi / 180) * self.steps_per_movement
        elif rescale_factor == 'mm':
            # millimeters/movement
            rescale_factor = (self.degrees_per_step * self.steps_per_movement *
                              self.rotation_per_degrees * self.mm_per_rotation)
        else:
            rescale_factor = None
        start = time.time()
        stepper = []
        for idx, motor_id in enumerate(motor_id):
            stepper.append(Stepper())
            stepper[idx].setHubPort(motor_id[0])
            stepper[idx].setDeviceSerialNumber(motor_id[1])
            stepper[idx].setIsRemote(True)
            stepper[idx].open()
            wait(lambda: stepper[idx].getAttached(), sleep_seconds=0.01,
                 waiting_for='Waiting for motor: ' + str(idx))
            stepper[idx].setDataInterval(stepper[idx].getMinDataInterval())
            stepper[idx].setDataRate(stepper[idx].getMaxDataRate())
            stepper[idx].setControlMode(control_mode)
            stepper[idx].setRescaleFactor(rescale_factor)
        print('Motors initialized in: ', time.time() - start)
        return stepper

    def doReference(self, moveDownVel=None):
        init_control_mode = self.stepper[0].getControlMode()
        init_acceleration = self.stepper[0].getAcceleration()
        control_mode = StepperControlMode.CONTROL_MODE_RUN
        success = False
        velocity_limit_const = []
        if moveDownVel is not None:
            for idx in range(len(moveDownVel)):
                velocity_limit_const.append(moveDownVel[idx])
        else:
            velocity_limit_const = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        safe_to_move = [False, False, False, False, False, False]
        move_up = +1
        move_down = -1

        # MOVE ALL JOINTS DOWN UNTIL LOWER LIMIT SWITCHES ARE ON #
        for idx in range(len(self.neg_limit_switch)):
            # self.stepper[idx].setAcceleration(init_acceleration * 10)
            if self.neg_limit_switch[idx].getState():
                safe_to_move[idx] = True
            else:
                safe_to_move[idx] = False

        for idx, value in enumerate(safe_to_move):
            self.stepper[idx].setControlMode(control_mode)
            if value:
                print('Moving joint ' + str(idx + 1) + ' DOWN')
                print('Joint Velocity: ' + str(velocity_limit_const[idx]) + ' mm/s')

        if any(safe_to_move):
            print('*** ENSURE MOTION IS SAFE BEFORE CONTINUING MOTION ***')
            usr_input = input('Continue? y/n')
        else:
            usr_input = 'y'

        if usr_input == 'y':
            for idx in range(len(self.stepper)):
                if safe_to_move[idx]:
                    self.stepper[idx].setVelocityLimit(move_down * velocity_limit_const[idx])
                else:
                    self.stepper[idx].setVelocityLimit(0.0)
                self.motion_flag[idx] = self.stepper[idx].getIsMoving()
        else:
            print('REFERENCING CANCELLED!')
            for idx in range(len(self.stepper)):
                self.stepper[idx].setControlMode(init_control_mode)
            return success

        while any(self.motion_flag):
            for idx in range(len(self.stepper)):
                self.motion_flag[idx] = self.stepper[idx].getIsMoving()

        # MOVE UP TO RELEASE LIMIT SWITCHES #
        for idx in range(len(self.neg_limit_switch)):
            if not self.neg_limit_switch[idx].getState():
                self.stepper[idx].setVelocityLimit(move_up * 1.0)
                self.motion_flag[idx] = self.stepper[idx].getIsMoving()
                safe_to_move[idx] = True
            else:
                self.stepper[idx].setVelocityLimit(0.0)

        while any(self.motion_flag):
            for idx in range(len(self.neg_limit_switch)):
                if self.motion_flag[idx]:
                    if self.neg_limit_switch[idx].getState():
                        self.stepper[idx].setVelocityLimit(0.0)
                        self.motion_flag[idx] = False

        for idx in range(len(self.stepper)):
            self.stepper[idx].setVelocityLimit(move_down * 0.5)
            while self.neg_limit_switch[idx].getState():
                continue
            self.stepper[idx].setVelocityLimit(0.0)
            self.stepper[idx].addPositionOffset(-self.stepper[idx].getPosition())

        # MOVE ALL SPINDLES TO Q = 0
        control_mode = StepperControlMode.CONTROL_MODE_STEP
        for idx in range(len(self.stepper)):
            self.stepper[idx].setControlMode(control_mode)
            self.stepper[idx].setVelocityLimit(0.5)
            self.stepper[idx].setTargetPosition(self.joints_zero_error[idx])
            self.motion_flag[idx] = self.stepper[idx].getIsMoving()

        while any(self.motion_flag):
            self.motion_flag[0] = self.stepper[0].getIsMoving()
            self.motion_flag[1] = self.stepper[1].getIsMoving()
            self.motion_flag[2] = self.stepper[2].getIsMoving()
            self.motion_flag[3] = self.stepper[3].getIsMoving()
            self.motion_flag[4] = self.stepper[4].getIsMoving()
            self.motion_flag[5] = self.stepper[5].getIsMoving()
            continue

        for idx in range(len(self.stepper)):
            self.stepper[idx].addPositionOffset(-self.stepper[idx].getPosition())

        success = True
        print('REFERENCE COMPLETE')

        # Move EEF to in +z direction with z_offset (mm) to put THD in the ROI
        z_offset = 5
        user_input = input('Move ' + str(z_offset) + ' mm in +z from reference point? y/n')
        if user_input == 'y':
            goal_cmd = [0, 0, z_offset, 0, 0, 0]
            success = self.move_to_pose(goal_cmd, self.start_rotation_0, self.start_position_0, cnt=1)
            if success:
                rx = R.from_euler('x', 0, degrees=True)
                ry = R.from_euler('y', 0, degrees=True)
                rz = R.from_euler('z', 0, degrees=True)
                r = rz * ry * rx
                self.start_position_0 = np.add(self.start_position_0,
                                               np.matmul(self.start_rotation_0, np.array(goal_cmd[:3])))
                self.start_rotation_0 = np.matmul(self.start_rotation_0, r.as_matrix())
                self.current_eef_pose = goal_cmd
            self.recordJointPosition()
        else:
            self.current_eef_pose = [0, 0, 0, 0, 0, 0]

        self.reference_eef_pose = self.current_eef_pose
        return success

    def getEEFState(self):
        position = [self.current_eef_pose[0], self.current_eef_pose[1],
                    self.current_eef_pose[2]]  # Units: millimeters (mm)
        position = [i / 1000 for i in position]  # Units: meters (m)
        theta = [self.current_eef_pose[3], self.current_eef_pose[4], self.current_eef_pose[5]]  # Units: degrees (deg)
        theta = [i * (math.pi / 180) for i in theta]  # Units: radians (rad)
        orientation = quaternion_from_euler_theta(theta)  # Quaternion
        return position, orientation

    def moveAlongTCPAxisCalculateNewEEF(self, goal, axis=None):
        _current_eef_position = [0, 0, 0]
        _current_eef_orientation = [0, 0, 0, 1]
        # _current_eef_position, _current_eef_orientation = self.getEEFState() ???
        if axis is not None:
            if axis == 'x':
                _current_eef_position[0] = goal
            elif axis == 'y':
                _current_eef_position[1] = goal
            elif axis == 'z':
                _current_eef_position[2] = goal
            else:
                raise ValueError('Invalid axis argument! Use x, y or z string arguments')
        else:
            raise ValueError('Axis argument not provided!')
        _eef_pose = _current_eef_position + _current_eef_orientation
        return _eef_pose

    def rotateAlongTCPAxisCalculateNewEEF(self, goal, axis=None):
        _current_eef_position, _current_eef_orientation = self.getEEFState()
        _roll_x, _pitch_y, _yaw_z = euler_from_quaternion(_current_eef_orientation[0], _current_eef_orientation[1],
                                                          _current_eef_orientation[2], _current_eef_orientation[3])
        if axis is not None:
            if axis == 'x':
                _roll_x = _roll_x + goal
            elif axis == 'y':
                _pitch_y = _pitch_y + goal
            elif axis == 'z':
                _yaw_z = _yaw_z + goal
            else:
                raise ValueError('Invalid axis argument! Use x, y or z string arguments')
        else:
            raise ValueError('Axis argument not provided!')
        _current_eef_orientation = quaternion_from_euler(_roll_x, _pitch_y, _yaw_z)
        _eef_pose = _current_eef_position + _current_eef_orientation
        return _eef_pose

    # EEF command processing
    def setEEFPose(self, eef_pose, max_vel=None):
        # For Hexaglide, commands are interpreted as relative commands in H1 (driven coordinate system)
        # EEF Commands are reversed because the platform moves not the THD
        goal_eef_position = [-eef_pose[0], -eef_pose[1], -eef_pose[2]]  # Units: meter(m)
        goal_eef_position = [i * 1000 for i in goal_eef_position]  # Units: millimeters (mm)
        goal_eef_orientation = eef_pose[3:]  # Quaternion
        theta_x, theta_y, theta_z = euler_from_quaternion_theta(goal_eef_orientation)  # Units: radians
        goal_theta_x = theta_x * (180 / math.pi)
        goal_theta_y = theta_y * (180 / math.pi)
        goal_theta_z = theta_z * (180 / math.pi)  # Units: degrees
        rx = R.from_euler('x', goal_theta_x, degrees=True)
        ry = R.from_euler('y', goal_theta_y, degrees=True)
        rz = R.from_euler('z', goal_theta_z, degrees=True)
        r = rz * ry * rx
        goal_eef_rotation_matrix = r.as_matrix()
        goal_cmd = [goal_eef_position[0], goal_eef_position[1], goal_eef_position[2],
                    goal_theta_x, goal_theta_y, goal_theta_z]
        for X_cmd, X_safe_limit in zip(goal_cmd, self.safe_eef_pose_change):
            if abs(X_cmd) > X_safe_limit:
                raise ValueError('Change in EEF exceeded safety limits, exiting...')

        print('---> Hexaglide.setEEFPose(): working with goal_cmd =', goal_cmd)

        # Execute command
        wait_for_input = False
        if max_vel is not None:
            self.max_joint_velocity_constant = max_vel

        if all(abs(goal_cmd) <= 5.5 for goal_cmd in goal_cmd[:2]) and all(abs(goal_cmd) <= 2.0 for goal_cmd in goal_cmd[3:]):
            # For EEF movements smaller than 5mm and rotation less than 2deg, set lower velocity
            self.max_joint_velocity_constant = 1.0  # mm/s
        else:
            self.max_joint_velocity_constant = self.config['hexaglide_param']['max_velocity_constant']

        success = self.move_to_pose(goal_cmd, self.start_rotation_0, self.start_position_0, cnt=1, variable_vel=True,
                                    wait_for_input=wait_for_input)
        if success:
            # Update start position and rotation of the platform for next command
            self.start_position_0 = np.add(self.start_position_0,
                                           np.matmul(self.start_rotation_0, np.array(goal_cmd[:3])))
            self.start_rotation_0 = np.matmul(self.start_rotation_0, r.as_matrix())
            # print('DEBUG --- robot.py setEEFPose() --- start_rotation_0 after movement success:\n',
            #       self.start_rotation_0)
            self.current_eef_pose = list(map(operator.add, self.current_eef_pose, goal_cmd))
            print(self.current_eef_pose)

        print('---> Hexaglide.setEEFPose(): current_eef_pose =', self.current_eef_pose)
        return success

    def move_to_pose(robot, deltaX_cmd, start_rotation_0, start_position_0, cnt=None, variable_vel=False,
                     wait_for_input=True):

        # Set start signal for all threads:
        start_signal = threading.Event()

        # Target pose in end-effector frame (H1)
        q_offset = 0.0
        success = False
        vel_cmd = robot.joint_max_velocity
        acceleration_cmd = robot.joint_max_acceleration
        # Commanded position delta in H1
        position_delta_1 = np.array([deltaX_cmd[0], deltaX_cmd[1], deltaX_cmd[2]]).reshape(-1, 1)
        # print('Commanded position delta in H1: ', position_delta_1)
        # Commanded rotation delta in H1
        rx = R.from_euler('x', deltaX_cmd[3], degrees=True)
        ry = R.from_euler('y', deltaX_cmd[4], degrees=True)
        rz = R.from_euler('z', deltaX_cmd[5], degrees=True)
        r = rz * ry * rx
        rotation_delta_1 = r.as_matrix()
        # print('Commanded rotation delta in H1: ', rotation_delta_1)

        # Target pose in base frame (H0)
        position_delta_0 = np.matmul(start_rotation_0, position_delta_1)
        # print('Commanded position delta in H0: ', position_delta_0)
        # print('start_position_0: ', start_position_0.reshape(3, 1))
        # print('position_delta_0: ', position_delta_0)
        position_0 = np.add(start_position_0.reshape(3, 1), position_delta_0)
        # print('Commanded position in H0: ', position_0)
        # print('start_rotation_0: ', start_rotation_0)
        rotation_0 = np.matmul(start_rotation_0, rotation_delta_1)
        # print('Commanded rotation in H0: ', rotation_0)

        # IK to get joint variables for commanded pose
        P = robot.init_P_1
        _, q = robot.calc_q(position_0, rotation_0, P)
        print('---> Hexaglide.move_to_pose(): IK solution found in [s]:', robot.IK_Solver_Duration)

        # Order the sequence of starting the motor according to the ascending order of joint variables
        q_start_sequence = sorted(range(len(q)), key=lambda k: q[k])

        # q_start_sequence = [4, 5, 0, 3, 1, 2]   # Motor sequence for x-direction
        # q_start_sequence = [1, 2, 0, 3, 4, 5]   # Motor sequence for y-direction

        # Check commanded joint variables q within lower and upper joint limits
        safe_to_move = [False, False, False, False, False, False]
        robot.motion_flag = [False, False, False, False, False, False]
        commanded_q = []
        change_q_position = []

        for idx in range(len(robot.stepper)):
            # print('Q' + str(idx) + ' Current Position: ', robot.stepper[idx].getPosition())
            # print('Q' + str(idx) + ' Commanded Position: ' + str(q[idx][0]))
            current_q = robot.stepper[idx].getPosition()
            commanded_q.append(q[idx][0])
            change_q_position.append(abs(q[idx][0] - current_q))
            if robot.pos_limit_switch[idx].getState() and q[idx] < 150:
                safe_to_move[idx] = True
            elif robot.neg_limit_switch[idx].getState() and q[idx] >= 0:
                safe_to_move[idx] = True
            else:
                safe_to_move[idx] = False
        if variable_vel:
            vel_cmd = robot.getVariableJointVelocity(change_q_position)

        # print('Variable velocity command: ', vel_cmd)
        if wait_for_input:
            print('---> Hexaglide.move_to_pose(): commanded position delta in H1: \n', position_delta_1)
            print('---> Hexaglide.move_to_pose(): commanded rotation delta in H1: \n', rotation_delta_1)
            print('---> Hexaglide.move_to_pose(): commanded position delta in H0: \n', position_delta_0)
            print('---> Hexaglide.move_to_pose(): start_position_0: \n', start_position_0.reshape(3, 1))
            print('---> Hexaglide.move_to_pose(): position_delta_0: \n', position_delta_0)
            print('---> Hexaglide.move_to_pose(): commanded position in H0: \n', position_0)
            print('---> Hexaglide.move_to_pose(): start_rotation_0: \n', start_rotation_0)
            print('---> Hexaglide.move_to_pose(): commanded rotation in H0: \n', rotation_0)
            usr_input = input('---> Hexaglide.move_to_pose(): enter y to start the displayed motion or any other key to abort')
        else:
            usr_input = 'y'

        parameter_threads = []
        motor_threads = []
        for idx in range(len(robot.stepper)):
            thread_init_start = time.time()
            parameter_thread = threading.Thread(target=parameter_setting_thread, args=(
                robot, idx, commanded_q[idx], vel_cmd[idx], acceleration_cmd[idx]))
            motor_thread = threading.Thread(target=motor_control_thread, args=(robot, idx, start_signal))
            motor_threads.append(motor_thread)
            parameter_threads.append(parameter_thread)
            parameter_thread.start()
            motor_thread.start()
            # print('Thread initialization time for motor ' + str(idx) + ': ' + str(time.time() - thread_init_start))

        parameters_setting_time = time.time()
        for thread in parameter_threads:
            thread.join()
        print('---> Hexaglide.move_to_pose(): motors parameter setting time [s]:', time.time() - parameters_setting_time)

        if all(safe_to_move) and usr_input == 'y':
            start_signal.set()

        else:
            print('---> Hexaglide.move_to_pose(): controller command unsafe for execution, exiting...')
            success = False
            return success

        movement_time = time.time()
        for thread in motor_threads:
            thread.join()
        print('---> Hexaglide.move_to_pose(): movement completion time [s]:', time.time() - movement_time)

        success = True
        # print('All threads completed')
        return success

    def getVariableJointVelocity(self, delta_joint_position):
        # This function returns joint velocities based on the defined max velocity (max_vel) and the change in joint position
        vel_cmd = [0] * 6
        sorted_idx = sorted(range(len(delta_joint_position)), key=lambda k: delta_joint_position[k])
        sorted_idx = sorted_idx[::-1]  # indices of delta_joint_position based on largest distance first
        max_time = delta_joint_position[sorted_idx[0]] / self.max_joint_velocity_constant
        vel_cmd[sorted_idx[0]] = self.max_joint_velocity_constant
        for idx in range(1, len(vel_cmd)):
            if max_time != 0:
                vel_cmd[sorted_idx[idx]] = delta_joint_position[sorted_idx[idx]] / max_time
            else:
                vel_cmd[sorted_idx[idx]] = self.max_joint_velocity_constant
        return vel_cmd

    def recordJointPosition(self):
        joint_list = []
        for idx in range(len(self.stepper)):
            joint_list.append(self.stepper[idx].getPosition())
        self.joint_position.append(joint_list)
        print('---> Hexaglide.recordJointPosition(): joint position (q) list:', self.joint_position)
        # Also save rl_start_position_0 and rl_start_rotation_0
        self.rl_start_position_0 = self.start_position_0
        self.rl_start_rotation_0 = self.start_rotation_0
        print('---> Hexaglide.recordJointPosition(): rl_start_position_0 =\n', self.rl_start_position_0)
        print('---> Hexaglide.recordJointPosition(): rl_start_rotation_0 =\n', self.rl_start_rotation_0)

    def move_to_joint_pose(self, joint_pose_cmd=None, variable_vel=False, wait_for_input=True):
        if joint_pose_cmd is None:
            print('---> Hexaglide.move_to_joint_pose(): joint_pose_cmd (q) array not provided, exiting')
            exit(0)

        print('---> Hexaglide.move_to_joint_pose(): moving to q =', joint_pose_cmd)

        # Set start signal for all threads:
        start_signal = threading.Event()

        # Target pose in end-effector frame (H1)
        vel_cmd = self.joint_max_velocity
        acceleration_cmd = self.joint_max_acceleration
        safe_to_move = [False, False, False, False, False, False]
        change_q_position = []
        for idx in range(len(self.stepper)):
            # print('Q' + str(idx) + ' Current Position: ', self.stepper[idx].getPosition())
            # print('Q' + str(idx) + ' Commanded Position: ' + str(q[idx][0]))
            current_q = self.stepper[idx].getPosition()
            change_q_position.append(abs(joint_pose_cmd[idx] - current_q))
            if self.pos_limit_switch[idx].getState() and joint_pose_cmd[idx] < 150:
                safe_to_move[idx] = True
            elif self.neg_limit_switch[idx].getState() and joint_pose_cmd[idx] >= 0:
                safe_to_move[idx] = True
            else:
                safe_to_move[idx] = False
        if variable_vel:
            vel_cmd = self.getVariableJointVelocity(change_q_position)

        # print('Variable velocity command: ', vel_cmd)
        if wait_for_input:
            usr_input = input('---> Hexaglide.move_to_joint_pose(): enter y to start motion or any other key to abort')
        else:
            usr_input = 'y'

        parameter_threads = []
        motor_threads = []
        for idx in range(len(self.stepper)):
            thread_init_start = time.time()
            parameter_thread = threading.Thread(target=parameter_setting_thread, args=(
                self, idx, joint_pose_cmd[idx], vel_cmd[idx], acceleration_cmd[idx]))
            motor_thread = threading.Thread(target=motor_control_thread, args=(self, idx, start_signal))
            motor_threads.append(motor_thread)
            parameter_threads.append(parameter_thread)
            parameter_thread.start()
            motor_thread.start()
            # print('Thread initialization time for motor ' + str(idx) + ': ' + str(time.time() - thread_init_start))

        parameters_setting_time = time.time()
        for thread in parameter_threads:
            thread.join()
        print('---> Hexaglide.move_to_joint_pose(): motors parameter setting time [s]:', time.time() - parameters_setting_time)

        if all(safe_to_move) and usr_input == 'y':
            start_signal.set()

        else:
            print('---> Hexaglide.move_to_joint_pose(): controller command unsafe for execution, exiting')
            success = False
            return success

        movement_time = time.time()
        for thread in motor_threads:
            thread.join()
        print('---> Hexaglide.move_to_joint_pose(): movement completion time [s]:', time.time() - movement_time)

        # Reset start_position_0 and start_rotation_0 (because we don't do FK)
        self.start_position_0 = self.rl_start_position_0
        self.start_rotation_0 = self.rl_start_rotation_0
        print('---> Hexaglide.move_to_joint_pose(): start_position_0 = \n', self.start_position_0)
        print('---> Hexaglide.move_to_joint_pose(): start_rotation_0 = \n', self.start_rotation_0)

        success = True
        return success
