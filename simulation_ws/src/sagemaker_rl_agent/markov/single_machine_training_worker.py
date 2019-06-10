"""
This is single machine training worker. It starts a local training and stores the model in S3.
"""

import sys
import argparse
import copy
import numpy as np
import tensorflow as tf

import imp

import markov
from markov import utils
import markov.environments
import gym
import os

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines import PPO2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

MARKOV_DIRECTORY = os.path.dirname(markov.__file__)
CUSTOM_FILES_PATH = "./custom_files"

if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)


def start_graph(graph_manager: 'GraphManager', task_parameters: 'TaskParameters'):
    graph_manager.create_graph(task_parameters)

    # save randomly initialized graph
    graph_manager.save_checkpoint()

    # Start the training
    graph_manager.improve()


def add_items_to_dict(target_dict, source_dict):
    updated_task_parameters = copy.copy(source_dict)
    updated_task_parameters.update(target_dict)
    return updated_task_parameters

def should_stop_training_based_on_evaluation():
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markov-preset-file',
                        help="(string) Name of a preset file to run in Markov's preset directory.",
                        type=str,
                        default=os.environ.get("MARKOV_PRESET_FILE", "object_tracker.py"))
    parser.add_argument('-c', '--local_model_directory',
                        help='(string) Path to a folder containing a checkpoint to restore the model from.',
                        type=str,
                        default=os.environ.get("LOCAL_MODEL_DIRECTORY", "./checkpoint"))
    parser.add_argument('-n', '--num_workers',
                        help="(int) Number of workers for multi-process based agents, e.g. A3C",
                        default=1,
                        type=int)
    parser.add_argument('--checkpoint-save-secs',
                        help="(int) Time period in second between 2 checkpoints",
                        type=int,
                        default=300)
    parser.add_argument('--save-frozen-graph',
                        help="(bool) True if we need to store the frozen graph",
                        type=bool,
                        default=True)

    args = parser.parse_args()

    env = gym.make('RoboMaker-DeepRacer-v0')
    env = DummyVecEnv([lambda: env])
    # env.reset()
    
    # for i in range(50):
    #     action = np.random.random_integers(0, 4)
    #     next_state, reward, _, _ = env.step(action)
    model = PPO2(CnnPolicy, env, learning_rate=0.0003, noptepochs=10, n_steps=64, tensorboard_log='/home/rishabh/work/BRL_Car/deepracer/simulation_ws/tb_logs/')
    model.learn(total_timesteps=10000000) #, callback=callback)
    model.save('/home/rishabh/work/BRL_Car/deepracer/simulation_ws/final_medium_policy')
        


if __name__ == '__main__':
    main()
