#!/usr/bin/env python
import gym
from .task_envs.task_envs_list import register_environment, get_all_registered_envs
import rospy
import rospkg
import os


def load_environment(id, max_episode_steps=10000, register_only=False):
    """
    It Does all the stuff that the user would have to do to make it simpler
    for the user.
    This means:
    0) Registers the TaskEnvironment wanted, if it exists in the Task_Envs.
    2) Checks that the workspace of the user has all that is needed for launching this.
    Which means that it will check that the robot spawn launch is there and the worls spawn is there.
    4) Launches the world launch and the robot spawn.
    5) It will import the Gym Env and Make it.
    """
    rospy.logwarn("Env: {} will be imported".format(id))
    result = None
    if id not in get_all_registered_envs():
        result = register_environment(
            task_env=id,
            max_episode_steps=max_episode_steps)
    else:
        result = True
    
    if register_only:
        return

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..." +
                      str(id))
        env = gym.make(id)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env

def action_dict():
    return {
            0: 'FRONT',
            1: 'BACK',
            2: 'RIGHT',
            3: 'LEFT',
            4: 'DOWN',
            5: 'UP',
            6: 'ROLL_P',
            7: 'ROLL_N',
            8: 'PITCH_P',
            9: 'PITCH_N',
            10: 'YAW_P',
            11: 'YAW_N'
        }


def relativeQuaternionRotation(q0, q1):
    a0 = q0[3] # if it is indeed w - otherwise change indices
    b0 = q0[0]
    c0 = q0[1]
    d0 = q0[2]
    a1 = q1[3] # if it is indeed w - otherwise change indices
    b1 = q1[0]
    c1 = q1[1]
    d1 = q1[2]
    w = a0*a1 + b0*b1 + c0*c1 + d0*d1
    x = -a0*b1 - b0*a1 - c0*d1 + d0*c1
    y = -a0*c1 + b0*d1 - c0*a1 - d0*b1
    z = -a0*d1 - b0*c1 + c0*b1 - d0*a1
    return [x, y, z, w]


def QuaternionMultiplication(q0, q1):
    # Quaternion representation = [x, y, z, w]
    a0 = q0[0]
    b0 = q0[1]
    c0 = q0[2]
    d0 = q0[3]
    a1 = q1[0]
    b1 = q1[1]
    c1 = q1[2]
    d1 = q1[3]
    w = -a0 * a1 - b0 * b1 - c0 * c1 + d0 * d1
    x = b0 * c1 - c0 * b1 + d0 * a1 + a0 * d1
    y = -a0 * c1 + c0 * a1 + d0 * b1 + b0 * d1
    z = a0 * b1 - b0 * a1 + d0 * c1 + c0 * d1
    return [w, x, y, z]
