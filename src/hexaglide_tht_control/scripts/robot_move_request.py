#! /usr/env/python
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from hexaglide.robot import Hexaglide


def move_to_pose(robot, X, rotation_matrix_H1, vel_cmd=None, acceleration_cmd=None):
    # Target pose in end-effector frame (frame 1)
    success = False
    if vel_cmd is None:
        vel_cmd = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if acceleration_cmd is None:
        acceleration_cmd = [robot.stepper[0].getAcceleration()] * 6

    position_1 = np.array([X[0], X[1], X[2]]).reshape(-1, 1)
    print('position 1 shape: ', position_1.shape)
    rx = R.from_euler('x', X[3], degrees=True)
    ry = R.from_euler('y', X[4], degrees=True)
    rz = R.from_euler('z', X[5], degrees=True)
    r = rz * ry * rx
    rot_1 = r.as_matrix()       # For scipy 1.2.x: r.as_dcm()
    print('rotation_matrix_H1: ', rotation_matrix_H1)
    print('position_1 before rotation: ', position_1)
    position_1 = np.matmul(rotation_matrix_H1, position_1)
    print('position_1 after rotation: ', position_1)
    print('rot_1: ', rot_1)

    # Transform target pose from end-effector frame (1) to base frame (0)
    transform = robot.get_transform_matrix(rot_1, position_1)
    T_1_0 = robot.transform_1_0(robot.init_T_1_0, transform)
    position_0 = T_1_0[:, -1][:3]
    rot_0 = T_1_0[:, :-1][:3]
    print('T_1_0: ', T_1_0)
    print('position_0: ', position_0)
    print('rot_0: ', rot_0)
    P = robot.init_P_1
    # IK to get joint variables for commanded pose
    _, q = robot.calc_q(position_0, rot_0, P)

    # Check commanded joint variables q within lower and upper joint limits
    safe_to_move = [False, False, False, False, False, False]
    motion_flag = []
    for idx in range(len(robot.stepper)):
        print('Q' + str(idx) + ' Current Position: ', robot.stepper[idx].getPosition())
        print('Q' + str(idx) + ' Commanded Position: ' + str(q[idx][0]))
        if robot.pos_limit_switch[idx].getState() and q[idx] < 150:
            safe_to_move[idx] = True
        elif robot.neg_limit_switch[idx].getState() and q[idx] >= 0:
            safe_to_move[idx] = True
        else:
            safe_to_move[idx] = False
    usr_input = input('press y to continue motion or any other button to abort')

    if all(safe_to_move) and usr_input == 'y':
        for idx in range(len(robot.stepper)):
            robot.stepper[idx].setVelocityLimit(vel_cmd[idx])
            robot.stepper[idx].setAcceleration(acceleration_cmd[idx])
            robot.stepper[idx].setTargetPosition(q[idx])
    else:
        print('Controller command unsafe for execution, exiting...')
        success = False
        return success

    # Motion flag checks if motors are moving
    motion_flag.append(robot.stepper[0].getIsMoving())
    motion_flag.append(robot.stepper[1].getIsMoving())
    motion_flag.append(robot.stepper[2].getIsMoving())
    motion_flag.append(robot.stepper[3].getIsMoving())
    motion_flag.append(robot.stepper[4].getIsMoving())
    motion_flag.append(robot.stepper[5].getIsMoving())

    # Return after all the motors stopped
    while any(flag for flag in motion_flag):
        motion_flag[0] = robot.stepper[0].getIsMoving()
        motion_flag[1] = robot.stepper[1].getIsMoving()
        motion_flag[2] = robot.stepper[2].getIsMoving()
        motion_flag[3] = robot.stepper[3].getIsMoving()
        motion_flag[4] = robot.stepper[4].getIsMoving()
        motion_flag[5] = robot.stepper[5].getIsMoving()
        continue
    success = True
    return success


if __name__ == "__main__":
    # Initialize the robot and calibrate to zero pose
    robot = Hexaglide(control_mode='pos')
    reference_success = robot.doReference([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    usr_input = 'y'
    if not reference_success:
        usr_input = input('Referencing unsuccessful, continue? y/n')
    if not usr_input == 'y':
        print('Exiting...')
        exit(0)

    deltaX_hist = [[0, 0, 0, 0, 0, 0]]
    deltaX_safe_limit = [40, 40, 50, 15, 15, 40]
    continue_input = True
    
    rx = R.from_euler('x', 0, degrees=True)
    ry = R.from_euler('y', 0, degrees=True)
    rz = R.from_euler('z', -30, degrees=True)
    r = rz * ry * rx
    rotation_matrix_H1 = r.as_matrix()

    # Motor commanded velocity and acceleration
    vel_cmd_const = 1.0 * 10
    acceleration_cmd_const = robot.stepper[0].getAcceleration()
    vel_cmd = [vel_cmd_const] * 6
    acceleration_cmd = [acceleration_cmd_const * 10] * 6
    
    while continue_input:
        # Delta X History for checking whether the next pose within safety limits
        print('GIVE COMMAND WITHIN SAFE RANGE: [delta_x, delta_y, delta_z, th_x, th_y, th_z] = +- [10, 10, 10, 15, '
              '15, 30]', ' mm, deg')
        delta_x = float(input('Enter delta x (mm): '))
        delta_y = float(input('Enter delta y (mm): '))
        delta_z = float(input('Enter delta z (mm): '))
        delta_theta_x = float(input('Enter delta theta x (degrees): '))
        delta_theta_y = float(input('Enter delta theta y (degrees): '))
        delta_theta_z = float(input('Enter delta theta z (degrees): '))
        deltaX_cmd = [delta_x, delta_y, delta_z, delta_theta_x, delta_theta_y, delta_theta_z]
        deltaX_cmd = [sum(x) for x in zip(deltaX_cmd, deltaX_hist[-1])]
        for X_cmd, X_safe_limit in zip(deltaX_cmd, deltaX_safe_limit):
            if abs(X_cmd) > X_safe_limit:
                print('DeltaX exceeded safety limits, exiting...')
                exit(0)
        deltaX_hist.append(deltaX_cmd)

        # Execute command
        success = move_to_pose(robot, deltaX_cmd, rotation_matrix_H1, vel_cmd, acceleration_cmd)
        if success:
            print('Movement Completed...')
            rx = R.from_euler('x', delta_theta_x, degrees=True)
            ry = R.from_euler('y', delta_theta_y, degrees=True)
            rz = R.from_euler('z', delta_theta_z, degrees=True)
            r = rz * ry * rx
            rotation_matrix_H1 = np.matmul(rotation_matrix_H1, r.as_matrix())
            continue_input = input('Continue? y/n')
            if continue_input == 'y':
                continue_input = True
            else:
                continue_input = False
                exit(0)
        else:
            print('Movement Aborted...')
            continue_input = input('Continue? y/n')
            if continue_input == 'y':
                continue_input = True
            else:
                continue_input = False
                exit(0)
