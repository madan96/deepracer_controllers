from collections import deque
import math

import matplotlib.pyplot as plt
import numpy as np

from geometry_msgs.msg import Twist
from markov.vehicle_pid import PIDLongitudinalController



k = 0.1  # look forward gain
Lfc = 2.0  # look-ahead distance
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s]


class VehiclePPController():
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
            args_longitudinal = {'K_P': 1.0, 'K_D': 0.1, 'K_I': 0.1}

        self._vehicle = vehicle
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PurePursuitController(self._vehicle, **args_lateral)
        self.waypoints = None
    
    def run_step(self, target_speed, waypoint_idx):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.
        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle = self._lon_controller.run_step(target_speed)
        steering, tgt_idx = self._lat_controller.run_step(waypoint_idx)

        cur_steer = steering
        cur_throttle = throttle

        return cur_steer, tgt_idx, cur_throttle

class PurePursuitController():
    """
    PurePursuitController implements lateral control using Pure Pursuit.
    """
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03, lfc=0.2):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P # speed proportional gain
        self._K = 0.1  # look forward gain
        self._lfc = 0.1  # look-ahead distance
        self._dt = dt
        self._e_buffer = deque(maxlen=10)
        self.waypoints = None
        self.old_nearest_point_index = None
    
    def run_step(self, waypoint_idx):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pure_pursuit_control(waypoint_idx)

    def calc_distance(self, point_x, point_y):
        dx = self._vehicle.rear_x - point_x
        dy = self._vehicle.rear_y - point_y
        return math.sqrt(dx ** 2 + dy ** 2)

    def calc_target_index(self):
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [self._vehicle.rear_x - icx for icx in self._vehicle.waypoints[:,0]]
            dy = [self._vehicle.rear_y - icy for icy in self._vehicle.waypoints[:,1]]
            d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
            ind = d.index(min(d))
            old_nearest_point_index = ind
        else:
            ind = old_nearest_point_index
            distance_this_index = self.calc_distance(self._vehicle.waypoints[ind][0], self._vehicle.waypoints[ind][1])
            while True:
                ind = ind + 1 if (ind + 1) < len(self._vehicle.waypoints) else ind
                distance_next_index = self.calc_distance(self._vehicle.waypoints[ind][0], self._vehicle.waypoints[ind][1])
                if distance_this_index < distance_next_index:
                    break
                distance_this_index = distance_next_index
            old_nearest_point_index = ind

        L = 0.0

        # TODO: v?
        Lf = self._K * self._vehicle.current_speed + self._lfc # k * state.v + self._lfc

        # search look ahead target point index
        while Lf > L and (ind + 1) < len(self._vehicle.waypoints):
            dx = self._vehicle.waypoints[ind][0] - self._vehicle.rear_x
            dy = self._vehicle.waypoints[ind][1] - self._vehicle.rear_y
            L = math.sqrt(dx ** 2 + dy ** 2)
            ind += 1

        return ind

    def _pure_pursuit_control(self, waypoint_idx):
        """
        Estimate the steering angle of the vehicle
        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """

        ind = self.calc_target_index()

        if waypoint_idx >= ind:
            ind = waypoint_idx

        if ind < len(self._vehicle.waypoints) - 2: # Hack: Subtracting a small number to continue laps
            tx = self._vehicle.waypoints[ind][0]
            ty = self._vehicle.waypoints[ind][1]
        else:
            tx = self._vehicle.waypoints[0][0]
            ty = self._vehicle.waypoints[0][1]
            ind = 0

        alpha = math.atan2(ty - self._vehicle.rear_y, tx - self._vehicle.rear_x) - self._vehicle.yaw

        Lf = self._K * self._vehicle.current_speed + self._lfc

        delta = math.atan2(2.0 * self._vehicle.wheelbase * math.sin(alpha) / Lf, 1.0)
        steering_angle = (delta * 180 / math.pi)
        steering_angle = min(45, max(-45, steering_angle)) / 45.
        return steering_angle, ind