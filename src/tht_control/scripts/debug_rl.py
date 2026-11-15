#!/usr/bin/env python3

#import gym
#from stable_baselines3 import PPO
#from stable_baselines3 import DQN
#from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
import os
# import sys
# sys.path.append("C:/Users/polikarpov/Documents/TU Dortmund/08_Projekte/SmartAssembly/Gitlab/smartassembly/src/openai/openai/src/openai")
#from common import load_environment
#from task_envs.task_envs_list import register_environment, get_all_registered_envs
#from custom_baseline_classes.custom_stable_baseline import HexaglideCustomCombinedExtractor
#from stable_baselines3.common.evaluation import evaluate_policy
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
    image1 = image1[...,::-1].copy()  # BGR to RGB
    image1 = np.int32(image1)
    print(image1.shape)
    image2 = obs[:, 256 * 256 * 3:2 * 256 * 256 * 3]
    image2 = image2.reshape((obs.shape[0], 256, 256, 3))
    image2 = image2[...,::-1].copy()  # BGR to RGB
    image2 = np.int32(image2)
    pressure = obs[:, 2 * 256 * 256 * 3:2 * 256 * 256 * 3 + 1]
    pose = obs[:, 2 * 256 * 256 * 3 + 1:]
    expert_observations = [{'current_image3': image1[step_counter],
                            'previous_image3': image2[step_counter],
                            'pressure': pressure[step_counter],
                            'pose': pose[step_counter]} for step_counter in range(obs.shape[0])]
    return expert_observations, transitions["acts"]


if __name__=='__main__':
    LOG_DIR = './debug_rl_logs/'
    #env = load_environment('THT-Insertion-v0', max_episode_steps=1000)
    #env = gym.Env()
    #print('Finished load_environment()...')

    # For tensorboard visualization
    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('logs/')

    # tensorboard
    #writer.add_graph(model)
    #writer.close()

    # Read selflearningdata .npz
    
    exp_obs, trans = load_transition("C:/Users/polikarpov/Documents/TU Dortmund/08_Projekte/SmartAssembly/Gitlab/smartassembly/selflearningdata/session_1/2024-03-01-15-38-14_selflearning_0.1.npz")
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


    d = None  
    data = None  
    with zipfile.ZipFile("C:/Users/polikarpov/Documents/TU Dortmund/08_Projekte/SmartAssembly/Gitlab/smartassembly/240301_hexaglide_tht_dqn.zip", "r") as z:
        for filename in z.namelist():
            print('Found', filename)  
            if filename=='data':
                with z.open(filename) as f:  
                    data = f.read()  
                    d = json.loads(data)  # .decode("utf-8")
                    print(type(d))
                    for key, value in d.items():
                        print(key)
                        print(value)
            if filename=='policy.optimizer.pth':
                with z.open(filename) as f: 
                    model = torch.load(f)

            if filename=='policy.pth':
                with z.open(filename) as f: 
                    model = torch.load(f)
                    # print(model)
                    # make a tensor to export onnx model for netron
                    '''
                    _current_cropped_image3 = np.zeros((256, 256, 3), dtype=np.uint8)
                    _previous_cropped_image3 = np.zeros((256, 256, 3), dtype=np.uint8)
                    _current_pressure = np.array([0.0])
                    _relative_robot_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                    observations = {"current_image3": _current_cropped_image3,
                        "previous_image3": _previous_cropped_image3,
                        "pressure": _current_pressure,
                        "pose": _relative_robot_pose}
                    '''
                    # observations = np.zeros(72)
                    # torch.onnx.export(model, args=observations, f='debug_model.onnx')

                    # tensorboard
                    # writer.add_graph(model)
                    # writer.close()

                    print('type(model):', type(model))

                    print('Model details:')
                    for key, value in model.items():
                        print(key, type(value), value.size())

                    # print(model['q_net.q_net.0.weight'])
                        
                    # plot_kernels(model['q_net.features_extractor.extractors.current_image3.conv1.weight'])
                    kernels = model['q_net.features_extractor.extractors.current_image3.fc.0.weight']
                    
                    # normalize
                    #kernels -= kernels.min()
                    #kernels /= kernels.max()
                    
                    # make 4D tensor
                    #kernels = torch.unsqueeze(kernels, 0)
                    #kernels = torch.unsqueeze(kernels, 0)

                    print(kernels)

                    # Plotting the tensor (2D!)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(kernels, cmap='coolwarm', interpolation='nearest')
                    plt.colorbar(label='Value')  # Add color bar indicating values
                    plt.title('2D Torch Tensor Visualization')
                    #plt.xlabel('X-axis')
                    #plt.ylabel('Y-axis')
                    plt.show()
                    
                    # show
                    #plt.imshow(torchvision.utils.make_grid(kernels, pad_value=255).permute(1, 2, 0))
                    #plt.show()

    policy_kwargs = dict(
        features_extractor_class=HexaglideCustomCombinedExtractor,
        net_arch=[]
    )

    # This didn't work because the interpreter doesn't find openai.custom_baseline_classes
    # Maybe it's because of different path handling on Linux and on Windows.
    # model = DQN.load("./DQN_models/best_model_00.zip", env, verbose=2, tensorboard_log=LOG_DIR, learning_rate=0.001,
    #                      policy_kwargs=policy_kwargs, n_steps=16)
    # print('Model loaded')