#!/usr/bin/env python3

# import gym
# from stable_baselines3 import PPO
# from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
import os
# import sys
# sys.path.append("C:/Users/polikarpov/Documents/TU Dortmund/08_Projekte/SmartAssembly/Gitlab/smartassembly/src/openai/openai/src/openai")
# from common import load_environment
# from task_envs.task_envs_list import register_environment, get_all_registered_envs
# from custom_baseline_classes.custom_stable_baseline import HexaglideCustomCombinedExtractor
# from stable_baselines3.common.evaluation import evaluate_policy
import json
import zipfile
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


def load_transition(path_expert_data_and_filename):
    transitions = np.load(path_expert_data_and_filename)
    obs = transitions["obs"]  # transform to dict and reshape images
    print('Observation shape: ', obs.shape)
    image1 = obs[:, :256 * 256 * 3]
    image1 = image1.reshape((obs.shape[0], 256, 256, 3))
    image1 = image1[..., ::-1].copy()  # BGR to RGB
    image1 = np.int32(image1)
    print(image1.shape)
    pressure = obs[:, 256 * 256 * 3: 256 * 256 * 3 + 1]
    pose = obs[:, 256 * 256 * 3 + 1:]
    expert_observations = [{'current_image3': image1[step_counter],
                            'pressure': pressure[step_counter],
                            'pose': pose[step_counter]} for step_counter in range(obs.shape[0])]
    return expert_observations, transitions["acts"]


if __name__ == '__main__':
    # Read MPDExperimentLogs

    exp_obs, trans = load_transition("/home/ips/smartassembly/MPDExperimentLogs/2024-03-26-14-00-21_session_1_experiment_1.npz")
    T = len(exp_obs)

    # Define a transform to convert
    # the image to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Display camera 3 images for the entire episode
    images = [transform(exp_obs[t]['current_image3']) for t in range(T)]
    plt.imshow(torchvision.utils.make_grid(images, nrow=5, padding=4, pad_value=255).permute(1, 2, 0))
    plt.show()

    # Show pressure history for the episode
    pressures = [exp_obs[t]['pressure'] for t in range(T)]
    time = np.arange(0, T, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(time, pressures, label='Pressure history', color='blue')
    plt.title('Pressure history')
    plt.xlabel('Timestep')
    plt.ylabel('Pressure [g]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Show pose history for the episode
    x = [exp_obs[t]['pose'][0] for t in range(T)]
    y = [exp_obs[t]['pose'][1] for t in range(T)]
    z = [exp_obs[t]['pose'][2] for t in range(T)]
    time = np.arange(0, T, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(time, x, label='x', color='red')
    plt.plot(time, y, label='y', color='green')
    plt.plot(time, z, label='z', color='blue')
    plt.title('Pose history')
    plt.xlabel('Timestep')
    plt.ylabel('Relative pose [m]')
    plt.legend()
    plt.grid(True)
    plt.show()