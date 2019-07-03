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

from markov.vehicle_pid import VehiclePIDController

def main():
    env = gym.make('RoboMaker-DeepRacerPID-v0')
    env = env.unwrapped
    time.sleep(5)
    car_pid = VehiclePIDController(env)
    car_pid._lat_controller.waypoints = env.waypoints
    tgt_waypoint_idx = 1
    max_steps = 10000
    steps = 0
    target_speed = 1.0
    env.reset()

    while steps < max_steps:
        tgt_waypoint_idx = env.get_closest_waypoint() % len(env.waypoints)
        if tgt_waypoint_idx == 0:
            tgt_waypoint_idx += 1
        env.wp_idx = tgt_waypoint_idx
        target_waypoint = env.waypoints[tgt_waypoint_idx]
        steering_angle, throttle = car_pid.run_step(target_speed, tgt_waypoint_idx)
        env.send_action(steering_angle, 1)
        max_steps += 1

if __name__ == '__main__':
    main()
    