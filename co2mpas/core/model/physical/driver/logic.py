#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to define driver logic.
"""
import numpy as np
import schedula as sh
from ..defaults import dfl

dsp = sh.BlueDispatcher(name='Driver logic')


# noinspection PyMissingOrEmptyDocstring
class SimulationModel:
    def __init__(self, models, outputs, index=0):
        self.index = index
        self.models = models
        self.outputs = outputs

    def __call__(self, acceleration, next_time):
        i = self.index
        self.outputs['accelerations'][i] = acceleration
        try:
            self.outputs['times'][i + 1] = next_time
        except IndexError:
            pass
        for m in self.models:
            m(i)
        return self

    def select(self, *items, di=0):
        i = max(self.index + di, 0)
        res = sh.selector(items, self.outputs, output_type='list')
        res = [v[i] for v in res]
        if len(res) == 1:
            return res[0]
        return res


@sh.add_function(dsp, outputs=['previous_velocity', 'previous_time'])
def get_previous(simulation_model):
    """
    Returns previous velocity and time.

    :param simulation_model:
        Simulation model.
    :type simulation_model: SimulationModel

    :return:
        Previous velocity [km/h] and time [s].
    :rtype: float, float
    """
    return simulation_model.select('velocities', 'times', di=-1)


@sh.add_function(dsp, outputs=['distance', 'velocity', 'time', 'angle_slope'])
def get_current(simulation_model):
    """
    Returns current distance, velocity, time, and angle_slope.

    :param simulation_model:
        Simulation model.
    :type simulation_model: SimulationModel

    :return:
        Previous distance [m], velocity [km/h], time [s], and angle_slope [rad].
    :rtype: float, float
    """
    return simulation_model.select(
        'distances', 'velocities', 'times', 'angle_slopes'
    )


@sh.add_function(dsp, outputs=['desired_velocity', 'maximum_distance'])
def calculate_desired_velocity(path_distances, path_velocities, distance):
    """
    Returns the desired velocity [km/h] and maximum distance [m].

    :param path_distances:
        Cumulative distance vector [m].
    :type path_distances: numpy.array

    :param path_velocities:
        Desired velocity vector [km/h].
    :type path_velocities: numpy.array

    :param distance:
        Current travelled distance [m].
    :type distance: float

    :return:
        Desired velocity [km/h] and maximum distance [m].
    :rtype: float, float
    """
    i = np.searchsorted(path_distances, distance, side='right')
    d = path_distances.take(i + 1, mode='clip') + dfl.EPS
    return path_velocities.take(i, mode='clip'), d


@sh.add_function(dsp, outputs=['maximum_power'])
def calculate_maximum_power(
        simulation_model, time, full_load_curve, delta_time):
    """
    Calculate maximum engine power [kW].

    :param simulation_model:
        Simulation model.
    :type simulation_model: SimulationModel

    :param time:
        Time [s].
    :type time: float

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param delta_time:
        Time step [s].
    :type delta_time: float

    :return:
        Maximum engine power [kW].
    :rtype: float
    """
    simulation_model(1, time + delta_time)
    return full_load_curve(simulation_model.select('engine_speeds_out_hot'))


@sh.add_function(dsp, outputs=['max_acceleration_model'])
def define_max_acceleration_model(
        road_loads, vehicle_mass, inertial_factor, static_friction,
        wheel_drive_load_fraction):
    """
    Defines maximum acceleration model.

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple


    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param inertial_factor:
        Factor that considers the rotational inertia [%].
    :type inertial_factor: float

    :param static_friction:
        Static friction coefficient [-].
    :type static_friction: float

    :param wheel_drive_load_fraction:
        Repartition of the load on wheel drive axles [-].
    :type wheel_drive_load_fraction: float

    :return:
        Maximum acceleration model.
    :rtype: function
    """
    from ..vehicle import _compile_traction_acceleration_limits
    from numpy.polynomial.polynomial import polyroots
    f0, f1, f2 = road_loads
    _m = vehicle_mass * (1 + inertial_factor / 100)
    _b = vehicle_mass * 9.81
    acc_lim = _compile_traction_acceleration_limits(
        static_friction, wheel_drive_load_fraction
    )

    def _func(previous_velocity, next_time, previous_time, angle_slope,
              motive_power):
        dt = (next_time - previous_time) * 3.6
        m = _m / dt

        b = f0 * np.cos(angle_slope) + _b * np.sin(angle_slope)
        b -= m * previous_velocity

        vel = max(polyroots((-motive_power * 3600, b, f1 + m, f2)))
        return np.clip((vel - previous_velocity) / dt, *acc_lim(angle_slope))

    return _func


@sh.add_function(dsp, outputs=['maximum_acceleration'])
def calculate_maximum_acceleration(
        simulation_model, maximum_power, max_acceleration_model, angle_slope,
        previous_velocity, previous_time):
    """
    Calculates the maximum vehicle acceleration.

    :param simulation_model:
        Simulation model.
    :type simulation_model: SimulationModel

    :param maximum_power:
        Maximum engine power [kW].
    :type maximum_power: float

    :param max_acceleration_model:
        Maximum acceleration model.
    :type max_acceleration_model: function

    :param angle_slope:
        Angle slope [rad].
    :type angle_slope: float

    :param previous_velocity:
        Previous velocity [km/h].
    :type previous_velocity: float

    :param previous_time:
        Previous time [s].
    :type previous_time: float

    :return:
        Maximum vehicle acceleration [m/s2].
    :rtype: float
    """
    acc = max_acceleration_model(
        previous_velocity, simulation_model.select('times', di=1),
        previous_time, angle_slope, maximum_power
    )
    return acc


@sh.add_function(dsp, outputs=['acceleration_damping'])
def calculate_acceleration_damping(previous_velocity, desired_velocity):
    """
    Calculates the acceleration damping [-].

    :param previous_velocity:
        Previous velocity [km/h].
    :type previous_velocity: float

    :param desired_velocity:
        Desired velocity [km/h].
    :type desired_velocity: float

    :return:
        Acceleration damping factor [-].
    :rtype: float
    """
    r = previous_velocity / desired_velocity
    if r >= 1:
        return 10 * (1 - r)  # Deceleration.
    if r > 0.5:
        return 1 - np.power(r, 60)  # Acceleration.
    return 1 - 0.8 * np.power(1 - r, 60)  # Acceleration boost.


@sh.add_function(dsp, outputs=['acceleration'])
def calculate_desired_acceleration(
        maximum_acceleration, driver_style_ratio, acceleration_damping):
    """
    Calculate the desired acceleration [m/s2].

    :param maximum_acceleration:
        Maximum achievable acceleration [m/s2].
    :type maximum_acceleration: float

    :param driver_style_ratio:
        Driver style ratio [-].
    :type driver_style_ratio: float

    :param acceleration_damping:
        Acceleration damping factor [-].
    :type acceleration_damping: float

    :return:
        Desired acceleration [m/s2].
    :rtype: float
    """
    return maximum_acceleration * driver_style_ratio * acceleration_damping


@sh.add_function(dsp, outputs=['next_time'])
def calculate_next_time(
        delta_time, time, previous_time, velocity, acceleration,
        previous_velocity, maximum_distance, distance):
    """
    Calculate next time [s].

    :param delta_time:
        Time frequency [1/s].
    :type delta_time: float

    :param time:
        Time [s].
    :type time: float

    :param previous_time:
        Previous time [s].
    :type previous_time: float

    :param velocity:
        Velocity [km/h].
    :type velocity: float

    :param acceleration:
        Acceleration [m/s2].
    :type acceleration: float

    :param previous_velocity:
        Previous velocity [km/h].
    :type previous_velocity: float

    :param maximum_distance:
        Maximum distance [m].
    :type maximum_distance: float

    :param distance:
        Current travelled distance [m].
    :type distance: float

    :return:
        Next time [s].
    :rtype: float
    """
    from numpy.polynomial.polynomial import polyroots
    v, a = (velocity + previous_velocity) / 3.6, acceleration
    p = 2 * (distance - maximum_distance), a * (time - previous_time) + v, a
    return time + min(max(polyroots(p)), delta_time)
