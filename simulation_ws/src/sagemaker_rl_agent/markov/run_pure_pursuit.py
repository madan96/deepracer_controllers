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
        steering_angle, tgt_waypoint_idx = car_pid._lat_controller.run_step(tgt_waypoint_idx)
        if math.fabs(steering_angle) > 0.4:
            target_speed = 1.0
        if math.fabs(steering_angle) > 0.3:
            target_speed = 1.5
        elif math.fabs(steering_angle) > 0.2:
            target_speed = 3.0
        elif math.fabs(steering_angle) > 0.1:
            target_speed = 4.0
        else:
            target_speed = 3.5
        speed = car_pid._lon_controller.run_step(target_speed)
        print ("S: ", steering_angle, "T: ", speed)
        env.send_action(steering_angle, speed)
        max_steps += 1

if __name__ == '__main__':
    main()
    