#!/usr/bin/env python3
# from yumi_openai.task_envs.tht_insertion_task import THTRLEnv
import gym
import rospy
from sensor_msgs.msg import Image
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from openai.common import load_environment
import os
from openai.custom_baseline_classes.custom_stable_baseline import HexaglideCustomCombinedExtractor
from stable_baselines3.common.evaluation import evaluate_policy


class TrainAndLoggingCallback(BaseCallback):
    """Class for (backup) model saving"""

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            print("~~~~ Saving /train/best_model.zip ~~~~")
            model_path = os.path.join(self.save_path, 'best_model')
            self.model.save(model_path)

        return True


if __name__ == '__main__':
    CHECKPOINT_DIR = './train/'  # TODO check dirs
    LOG_DIR = './logs/'
    # python3 -m tensorboard.main --logdir "/home/ips/smartassembly_ws/logs/PPO_63"
    callback = TrainAndLoggingCallback(check_freq=30, save_path=CHECKPOINT_DIR)
    # Start an RL training node
    rospy.init_node('rl_training', anonymous=True)
    # Define the environment type and parameters
    # env = gym.make('THT-Insertion-v0')
    env = load_environment('THT-Insertion-v0', max_episode_steps=1000)
    print('Finished load_environment()...')

    # check env for compatibility with stable baseline
    # check_env(env) # no return value is correct
    # print('Finished check_env()...')
    # This is the AI model started

    # For Training PPO:
    policy_kwargs = dict(features_extractor_class=HexaglideCustomCombinedExtractor, net_arch=[64, 32], ortho_init=False)
    # model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
    load_ppo = True
    if not load_ppo:
        model = PPO('MultiInputPolicy', env, verbose=2, tensorboard_log=LOG_DIR, learning_rate=1e-4,
                    policy_kwargs=policy_kwargs, n_steps=128, ent_coef=0.01)
        print('PPO Initialized...')
    else:
        # model = PPO.load("v0.2_ppo_student_resnet18", env, verbose=2, tensorboard_log=LOG_DIR, learning_rate=0.001,
        #                  policy_kwargs=policy_kwargs, n_steps=16)
        model = PPO.load("./train/best_model_00.zip", env, verbose=2, tensorboard_log=LOG_DIR, learning_rate=2e-4,
                         policy_kwargs=policy_kwargs, n_steps=128, ent_coef=0.001)
    print('PPO loaded')

    # # For Training DQN:
    # policy_kwargs = dict(features_extractor_class=HexaglideCustomCombinedExtractor, net_arch=[])
    # load_dqn = True
    # if not load_dqn:
    #     model = DQN('MultiInputPolicy', env, verbose=2, tensorboard_log=LOG_DIR, learning_rate=0.001,
    #                 learning_starts=100, batch_size=16, buffer_size=100000, policy_kwargs=policy_kwargs)
    #     print('DQN Initialized...')
    # else:
    #     model = DQN.load("./train/best_model_00.zip", env, verbose=2, tensorboard_log=LOG_DIR, learning_rate=0.005,
    #                      policy_kwargs=policy_kwargs, n_steps=16)
    #     print('DQN loaded')

        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # print(f"Mean reward = {mean_reward} +/- {std_reward}")
        #
        # exit(0)
        # env.reset()

    # Train the AI model, this is where the AI model starts to learn
    # train_session is set in robot_env.py
    model.learn(total_timesteps=512, callback=callback, log_interval=1, reset_num_timesteps=False)
    print('Finished learning...')
    model.save('240320_HHMM_hexaglide_tht_ppo')  # this is done by SB3
    # model.save_replay_buffer('replay_buffer/240227_hexaglide_tht_dqn')  # save replay buffer for retraining
    print('Finished saving model and replay buffer..')
