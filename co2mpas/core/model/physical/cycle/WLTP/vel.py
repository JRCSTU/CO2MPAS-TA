#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to calculate the WLTP theoretical velocities.
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
from ...vehicle import dsp as _vehicle

dsp = sh.BlueDispatcher(
    name='WLTP velocities model',
    description='Returns the theoretical velocities of WLTP.'
)


@sh.add_function(
    dsp, outputs=['resistance_coeffs_regression_curves', 'wltc_data']
)
def get_dfl(base_model):
    """
    Gets default values from wltp base model.

    :param base_model:
        WLTP base model.
    :type base_model: dict

    :return:
        Default values from wltp base model.
    :rtype: list
    """
    params = base_model['params']
    keys = 'resistance_coeffs_regression_curves', 'wltc_data'
    return sh.selector(keys, params, output_type='list')


@sh.add_function(dsp, outputs=['road_loads'], weight=15)
def default_road_loads(
        vehicle_mass, resistance_coeffs_regression_curves):
    """
    Returns default road loads.

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param resistance_coeffs_regression_curves:
        Regression curve coefficient to calculate the default road loads.
    :type resistance_coeffs_regression_curves: list[list[float]]

    :return:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :rtype: list, tuple
    """
    from wltp.experiment import calc_default_resistance_coeffs as func
    return func(vehicle_mass, resistance_coeffs_regression_curves)


@sh.add_function(dsp, outputs=['max_speed_velocity_ratio'])
def calculate_max_speed_velocity_ratio(speed_velocity_ratios):
    """
    Calculates the maximum speed velocity ratio of the gear box [h*RPM/km].

    :param speed_velocity_ratios:
        Speed velocity ratios of the gear box [h*RPM/km].
    :type speed_velocity_ratios: dict[int | float]

    :return:
        Maximum speed velocity ratio of the gear box [h*RPM/km].
    :rtype: float
    """

    return speed_velocity_ratios[max(speed_velocity_ratios)]


@sh.add_function(dsp, outputs=['max_velocity'])
def calculate_max_velocity(engine_speed_at_max_power, max_speed_velocity_ratio):
    """
    Calculates max vehicle velocity [km/h].

    :param engine_speed_at_max_power:
        Rated engine speed [RPM].
    :type engine_speed_at_max_power: float

    :param max_speed_velocity_ratio:
        Maximum speed velocity ratio of the gear box [h*RPM/km].
    :type max_speed_velocity_ratio: float

    :return:
        Max vehicle velocity [km/h].
    :rtype: float
    """

    return engine_speed_at_max_power / max_speed_velocity_ratio


@sh.add_function(dsp, outputs=['wltp_class'])
def calculate_wltp_class(
        wltc_data, engine_max_power, unladen_mass, max_velocity):
    """
    Calculates the WLTP vehicle class.

    :param wltc_data:
        WLTC data.
    :type wltc_data: dict

    :param engine_max_power:
        Maximum power [kW].
    :type engine_max_power: float

    :param unladen_mass:
        Unladen mass [kg].
    :type unladen_mass: float

    :param max_velocity:
        Max vehicle velocity [km/h].
    :type max_velocity: float

    :return:
        WLTP vehicle class.
    :rtype: str
    """
    from wltp.experiment import decideClass
    ratio = 1000.0 * engine_max_power / unladen_mass
    return decideClass(wltc_data, ratio, max_velocity)


@sh.add_function(dsp, outputs=['class_data'])
def get_class_data(wltc_data, wltp_class):
    """
    Returns WLTP class data.

    :param wltc_data:
        WLTC data.
    :type wltc_data: dict

    :param wltp_class:
        WLTP vehicle class.
    :type wltp_class: str

    :return:
        WLTP class data.
    :rtype: dict
    """

    return wltc_data['classes'][wltp_class]


@sh.add_function(dsp, outputs=['class_times', 'class_velocities'], weight=25)
def get_class_velocities(class_data):
    """
    Returns time and velocity profiles according to WLTP class data [s, km/h].

    :param class_data:
        WLTP class data.
    :type class_data: dict

    :return:
        Class time and velocity vectors [s, km/h].
    :rtype: tuple[numpy.array]
    """
    vel = np.asarray(class_data['cycle'], dtype=float)
    return np.arange(vel.shape[0], dtype=float), vel


i = ['vehicle_mass', 'road_loads', 'inertial_factor']
calculate_class_powers = sh.SubDispatchPipe(
    _vehicle,
    function_id='calculate_class_powers',
    inputs=['times', 'velocities'] + i,
    outputs=['motive_powers']
)
dsp.add_function(
    function=calculate_class_powers,
    inputs=['class_times', 'class_velocities'] + i,
    outputs=['class_powers']
)

dsp.add_data(
    'downscale_factor_threshold', dfl.values.downscale_factor_threshold
)


@sh.add_function(dsp, outputs=['downscale_factor'])
def calculate_downscale_factor(
        class_data, downscale_factor_threshold, max_velocity, engine_max_power,
        class_powers):
    """
    Calculates velocity downscale factor [-].

    :param class_data:
        WLTP class data.
    :type class_data: dict

    :param downscale_factor_threshold:
        Velocity downscale factor threshold [-].
    :type downscale_factor_threshold: float

    :param max_velocity:
        Max vehicle velocity [km/h].
    :type max_velocity: float

    :param engine_max_power:
        Maximum power [kW].
    :type engine_max_power: float

    :param class_powers:
        Class motive power [kW].
    :type class_powers: numpy.array

    :return:
        Velocity downscale factor [-].
    :rtype: float
    """
    from wltp.experiment import calcDownscaleFactor
    dsc_data = class_data['downscale']
    p_max_values = dsc_data['p_max_values']
    downsc_coeffs = dsc_data['factor_coeffs']
    dsc_v_split = dsc_data.get('v_max_split', None)
    downscale_factor = calcDownscaleFactor(
        class_powers, p_max_values, downsc_coeffs, dsc_v_split,
        engine_max_power, max_velocity, downscale_factor_threshold
    )
    return downscale_factor


@sh.add_function(dsp, outputs=['downscale_phases'])
def get_downscale_phases(class_data):
    """
    Returns downscale phases [s].

    :param class_data:
        WLTP class data.
    :type class_data: dict

    :return:
        Downscale phases [s].
    :rtype: list
    """
    return class_data['downscale']['phases']


@sh.add_function(dsp, outputs=['theoretical_velocities'])
def wltp_velocities(
        downscale_factor, class_times, class_velocities, downscale_phases,
        times):
    """
    Returns the downscaled velocity profile [km/h].

    :param downscale_factor:
        Velocity downscale factor [-].
    :type downscale_factor: float

    :param class_times:
        Class time vector [s].
    :type class_times: numpy.array

    :param class_velocities:
        Class velocity vector [km/h].
    :type class_velocities: numpy.array

    :param downscale_phases:
        Downscale phases [s].
    :type downscale_phases: list

    :param times:
        Time vector [s].
    :type times: numpy.array

    :return:
        Theoretical velocity vector [km/h].
    :rtype: numpy.array
    """

    if downscale_factor > 0:
        from wltp.experiment import downscaleCycle
        downscale_phases = np.searchsorted(times, downscale_phases)
        v = downscaleCycle(class_velocities, downscale_factor, downscale_phases)
    else:
        v = class_velocities

    n = int(np.ceil(times[-1] / class_times[-1]))
    t = np.cumsum(np.tile(np.ediff1d(class_times, to_begin=[0]), (n,)))
    t += class_times[0]
    return np.interp(times, t, np.tile(v, (n,)))


dsp.add_function(
    function_id='calculate_theoretical_motive_powers',
    function=calculate_class_powers,
    inputs=['times', 'theoretical_velocities'] + i,
    outputs=['theoretical_motive_powers']
)
dsp.add_function(
    function=sh.bypass,
    inputs=['theoretical_velocities'],
    outputs=['velocities']
)
