#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides constants for the WLTP cycle.
"""


import wltp.experiment as wltp_exp
import wltp.model as wltp_mdl
import schedula as sh
import logging
import copy
import numpy as np
logging.getLogger('wltp.experiment').setLevel(logging.WARNING)

log = logging.getLogger(__name__)


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


def calculate_max_velocity(
        engine_speed_at_max_power, max_speed_velocity_ratio):
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

    ratio = 1000.0 * engine_max_power / unladen_mass

    return wltp_exp.decideClass(wltc_data, ratio, max_velocity)


def get_class_velocities(class_data, times):
    """
    Returns the velocity profile according to WLTP class data [km/h].

    :param class_data:
        WLTP class data.
    :type class_data: dict

    :param times:
        Time vector [s].
    :type times: numpy.array

    :return:
        Class velocity vector [km/h].
    :rtype: numpy.array
    """

    vel = np.asarray(class_data['cycle'], dtype=float)
    n = int(np.ceil(times[-1] / len(vel)))
    vel = np.tile(vel, (n,))
    return np.interp(times, np.arange(len(vel)), vel)


def calculate_downscale_factor(
        class_data, downscale_factor_threshold, max_velocity, engine_max_power,
        class_powers, times):
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

    :param times:
        Time vector [s].
    :type times: numpy.array

    :return:
        Velocity downscale factor [-].
    :rtype: float
    """

    dsc_data = class_data['downscale']
    p_max_values = dsc_data['p_max_values']
    p_max_values[0] = np.searchsorted(times, p_max_values[0])
    downsc_coeffs = dsc_data['factor_coeffs']
    dsc_v_split = dsc_data.get('v_max_split', None)
    downscale_factor = wltp_exp.calcDownscaleFactor(
        class_powers, p_max_values, downsc_coeffs, dsc_v_split,
        engine_max_power, max_velocity, downscale_factor_threshold
    )
    return downscale_factor


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


def wltp_velocities(
        downscale_factor, class_velocities, downscale_phases, times):
    """
    Returns the downscaled velocity profile [km/h].

    :param downscale_factor:
        Velocity downscale factor [-].
    :type downscale_factor: float

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
        Velocity vector [km/h].
    :rtype: numpy.array
    """

    if downscale_factor > 0:
        downscale_phases = np.searchsorted(times, downscale_phases)
        v = wltp_exp.downscaleCycle(
            class_velocities, downscale_factor, downscale_phases
        )
    else:
        v = class_velocities
    return v


def wltp_gears(
        full_load_curve, velocities, accelerations, motive_powers,
        speed_velocity_ratios, idle_engine_speed, engine_speed_at_max_power,
        engine_max_power, engine_max_speed, wltp_base_model,
        initial_gears=None):
    """
    Returns the gear shifting profile according to WLTP [-].

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param speed_velocity_ratios:
        Speed velocity ratios of the gear box [h*RPM/km].
    :type speed_velocity_ratios: dict[int | float]

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_speed_at_max_power:
        Rated engine speed [RPM].
    :type engine_speed_at_max_power: float

    :param engine_max_power:
        Maximum power [kW].
    :type engine_max_power: float
    
    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :param wltp_base_model:
        WLTP base model params.
    :type wltp_base_model: dict

    :param initial_gears:
        Initial gear vector [-].
    :type initial_gears: numpy.array

    :return:
        Gear vector [-].
    :rtype: numpy.array
    """

    n_min_drive = None
    svr = [v for k, v in sorted(speed_velocity_ratios.items()) if k]
    idle = idle_engine_speed[0]

    n_norm = np.arange(
        0.0,
        (engine_max_speed - idle) / (engine_speed_at_max_power - idle),
        0.01
    )

    load_curve = {
        'n_norm': n_norm,
        'p_norm': full_load_curve(
            idle + n_norm * (engine_speed_at_max_power - idle)
        ) / engine_max_power
    }

    b = velocities < 0
    if b.any():
        vel = velocities.copy()
        vel[b] = 0
    else:
        vel = velocities

    res = wltp_exp.run_cycle(
        vel, accelerations, motive_powers, svr, idle_engine_speed[0],
        n_min_drive, engine_speed_at_max_power, engine_max_power,
        load_curve, wltp_base_model)

    if initial_gears:
        gears = initial_gears.copy()
    else:
        # noinspection PyUnresolvedReferences
        gears = res[0]

    # Apply Driveability-rules.
    # noinspection PyUnresolvedReferences
    wltp_exp.applyDriveabilityRules(vel, accelerations, gears, res[1], res[-1])

    gears[gears < 0] = 0
    log.warning('The WLTP gear-shift profile generation is for engineering '
                'purposes and the results are by no means valid according to '
                'the legislation.\nActually they are calculated based on a pre '
                'phase-1a version of the GTR spec.\n '
                'Please provide the gear-shifting profile '
                'within `prediction.WLTP` sheet.')
    return gears


def get_dfl(wltp_base_model):
    """
    Gets default values from wltp base model.

    :param wltp_base_model:
        WLTP base model params.
    :type wltp_base_model: dict

    :return:
        Default values from wltp base model.
    :rtype: list
    """

    params = wltp_base_model['params']
    keys = 'resistance_coeffs_regression_curves', 'wltc_data'
    return sh.selector(keys, params, output_type='list')


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


def define_wltp_base_model(base_model):

    return sh.combine_dicts(wltp_mdl._get_model_base(), base_model)


def wltp_cycle():
    """
    Defines the wltp cycle model.

    .. dispatcher:: d

        >>> d = wltp_cycle()

    :return:
        The wltp cycle model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='WLTP cycle model',
        description='Returns the theoretical times, velocities, and gears of '
                    'WLTP.'
    )

    from ..defaults import dfl
    d.add_data(
        data_id='initial_temperature',
        default_value=dfl.values.initial_temperature_WLTP,
        description='Initial temperature of the test cell [°C].'
    )

    d.add_data(
        data_id='max_time',
        default_value=dfl.values.max_time_WLTP,
        description='Maximum time [s].',
        initial_dist=5
    )

    d.add_data(
        data_id='wltp_base_model',
        default_value=copy.deepcopy(dfl.values.wltp_base_model)
    )

    d.add_function(
        function=define_wltp_base_model,
        inputs=['wltp_base_model'],
        outputs=['base_model']
    )

    d.add_dispatcher(
        dsp=calculate_wltp_velocities(),
        inputs=(
            'times', 'base_model', 'velocities', 'speed_velocity_ratios',
            'inertial_factor', 'downscale_phases', 'climbing_force',
            'downscale_factor', 'downscale_factor_threshold', 'vehicle_mass',
            'unladen_mass', 'road_loads', 'engine_max_power',
            'engine_speed_at_max_power', 'max_velocity', 'wltp_class',
            'max_speed_velocity_ratio'),
        outputs=('velocities',)
    )

    d.add_function(
        function=sh.add_args(wltp_gears),
        inputs=['gear_box_type', 'full_load_curve', 'velocities',
                'accelerations', 'motive_powers', 'speed_velocity_ratios',
                'idle_engine_speed', 'engine_speed_at_max_power',
                'engine_max_power', 'engine_max_speed', 'base_model'],
        outputs=['gears'],
        input_domain=lambda *args: args[0] == 'manual',
        weight=10
    )

    return d


def calculate_wltp_velocities():
    """
    Defines the wltp cycle model.

    .. dispatcher:: d

        >>> d = calculate_wltp_velocities()

    :return:
        The wltp cycle model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='WLTP velocities model',
        description='Returns the theoretical velocities of WLTP.'
    )

    d.add_function(
        function=get_dfl,
        inputs=['base_model'],
        outputs=['resistance_coeffs_regression_curves', 'wltc_data']
    )

    d.add_function(
        function=wltp_exp.calc_default_resistance_coeffs,
        inputs=['vehicle_mass', 'resistance_coeffs_regression_curves'],
        outputs=['road_loads'],
        weight=15
    )

    d.add_function(
        function=calculate_max_speed_velocity_ratio,
        inputs=['speed_velocity_ratios'],
        outputs=['max_speed_velocity_ratio']
    )

    d.add_function(
        function=calculate_max_velocity,
        inputs=['engine_speed_at_max_power', 'max_speed_velocity_ratio'],
        outputs=['max_velocity']
    )

    d.add_function(
        function=calculate_wltp_class,
        inputs=['wltc_data', 'engine_max_power', 'unladen_mass',
                'max_velocity'],
        outputs=['wltp_class']
    )

    d.add_function(
        function=get_class_data,
        inputs=['wltc_data', 'wltp_class'],
        outputs=['class_data']
    )

    d.add_function(
        function=get_class_velocities,
        inputs=['class_data', 'times'],
        outputs=['class_velocities'],
        weight=25
    )

    from ..vehicle import vehicle
    func = sh.SubDispatchFunction(
        dsp=vehicle(),
        function_id='calculate_class_powers',
        inputs=['vehicle_mass', 'velocities', 'climbing_force', 'road_loads',
                'inertial_factor', 'times'],
        outputs=['motive_powers']
    )

    d.add_function(
        function=func,
        inputs=['vehicle_mass', 'class_velocities', 'climbing_force',
                'road_loads', 'inertial_factor', 'times'],
        outputs=['class_powers']
    )
    from ..defaults import dfl
    d.add_data(
        data_id='downscale_factor_threshold',
        default_value=dfl.values.downscale_factor_threshold
    )

    d.add_function(
        function=calculate_downscale_factor,
        inputs=['class_data', 'downscale_factor_threshold', 'max_velocity',
                'engine_max_power', 'class_powers', 'times'],
        outputs=['downscale_factor']
    )

    d.add_function(
        function=get_downscale_phases,
        inputs=['class_data'],
        outputs=['downscale_phases']
    )

    d.add_function(
        function=wltp_velocities,
        inputs=['downscale_factor', 'class_velocities', 'downscale_phases',
                'times'],
        outputs=['velocities']
    )

    return d
