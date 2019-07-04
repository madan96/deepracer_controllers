import sys
import argparse
import copy
import imp
import math
import time
import markov
from markov import utils
import markov.environments
import gym
import os

from markov.vehicle_pure_pursuit import VehiclePPController

def main():
    env = gym.make('RoboMaker-DeepRacerPID-v0')
    env = env.unwrapped
    # time.sleep(5)

    # TODO: Add arg parser for controller parameters
    car_pid = VehiclePPController(env)
    car_pid._lat_controller.waypoints = env.waypoints
    tgt_waypoint_idx = 1
    max_steps = 10000
    steps = 0
    target_speed = 1.0
    env.reset()

    while steps < max_steps:
        # tgt_waypoint_idx = env.get_closest_waypoint() % len(env.waypoints)
        # if tgt_waypoint_idx == 0:
        #     tgt_waypoint_idx += 1
        steering_angle, tgt_waypoint_idx, throttle = car_pid.run_step(target_speed, tgt_waypoint_idx)
        env.send_action(steering_angle, 1)
        max_steps += 1

if __name__ == '__main__':
    main()
    