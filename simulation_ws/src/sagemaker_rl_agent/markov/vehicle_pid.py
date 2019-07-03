"""
Adapted from https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/controller.py
"""

from collections import deque
import math

from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt
import numpy as np


class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        if not args_lateral:
            args_lateral = {'K_P': 1.0, 'K_D': 0.001, 'K_I': 0.001}
        if not args_longitudinal:
            args_longitudinal = {'K_P': 1.0, 'K_D': 0.00, 'K_I': 0.00}

        self._vehicle = vehicle
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)
        self.waypoints = None

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.
        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(waypoint)

        cur_steer = steering
        cur_throttle = throttle

        return cur_steer, cur_throttle


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)
        self.waypoints = None

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
        :param target_speed: target speed in Km/h
        :return: throttle control in the range [0, 1]
        """
        current_speed = self._vehicle.current_speed

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations
        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), 0.0, 1.0)


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)
        self.waypoints = None

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint)

    def _pid_control(self, waypoint_idx):
        """
        Estimate the steering angle of the vehicle based on the PID equations
        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """

        orientation_vec = [(self._vehicle.x - self.waypoints[waypoint_idx][0]), (self._vehicle.y - self.waypoints[waypoint_idx][1]), 0.0]
        heading_vec = [math.cos(self._vehicle.yaw), math.sin(self._vehicle.yaw), 0.0]
        _dot = math.acos(np.clip(np.dot(heading_vec, orientation_vec) /
                                 (np.linalg.norm(heading_vec) * np.linalg.norm(orientation_vec)), -1.0, 1.0))

        _cross = np.cross(orientation_vec, heading_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)

