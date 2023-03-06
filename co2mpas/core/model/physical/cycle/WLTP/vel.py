#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2023 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to calculate the WLTP theoretical velocities.
"""
import functools
import collections
import schedula as sh
from ...vehicle import dsp as _vehicle
from gearshift.core.model.scaleTrace import dsp as _scaleTrace

dsp = sh.BlueDispatcher(
    name='WLTP velocities model',
    description='Returns the theoretical velocities of WLTP.'
)


@sh.add_function(dsp, outputs=['road_loads'], weight=15)
def default_road_loads(vehicle_mass):
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
    import wltp.model as wltp_mdl
    from wltp.experiment import calc_default_resistance_coeffs as func
    # noinspection PyProtectedMember
    params = wltp_mdl._get_model_base()['params']
    return func(vehicle_mass, params['resistance_coeffs_regression_curves'])


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
def calculate_wltp_class(engine_max_power, unladen_mass, max_velocity):
    """
    Calculates the WLTP vehicle class.

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
    from wltp.model import _get_wltc_data
    ratio = 1000.0 * engine_max_power / unladen_mass
    return decideClass(_get_wltc_data(), ratio, max_velocity)


@functools.lru_cache(None)
def _get_speed_phase_data():
    from gearshift.core.load import _load_speed_phase_data
    return _load_speed_phase_data()


@functools.lru_cache(None)
def _get_class_traces():
    it = _get_speed_phase_data()["trace"].groupby('class')
    return {k: v.drop('class', axis=1).to_numpy() for k, v in it}


@functools.lru_cache(None)
def _get_class_defaults():
    it = _get_speed_phase_data()["scale"].groupby('class')
    return {k: v.drop('class', axis=1).to_dict('records')[0] for k, v in it}


@functools.lru_cache(None)
def _get_class_phases():
    it = _get_speed_phase_data()["phase"].groupby('class')
    return {k: v.drop('class', axis=1).to_dict('list') for k, v in it}


scaleTrace = _scaleTrace.register()
scaleTrace.add_function(
    'splitting', sh.bypass, ['road_loads'], ['f0', 'f1', 'f2']
)


@sh.add_function(scaleTrace, outputs=['Trace'])
def get_Trace(wltp_class):
    return _get_class_traces()[wltp_class]


@sh.add_function(scaleTrace, outputs=['PhaseLengths'])
def get_PhaseLengths(wltp_class):
    return _get_class_phases()[wltp_class]['length']


defaults_mapping = collections.OrderedDict([
    ("algo", "ScalingAlgorithms"),
    ("t_beg", "ScalingStartTimes"),
    ("t_max", "ScalingCorrectionTimes"),
    ("t_end", "ScalingEndTimes"),
    ("r0", "r0"),
    ("a1", "a1"),
    ("b1", "b1"),
])
scaleTrace.add_data('ApplyDownscaling', 1)
scaleTrace.add_data('ApplyDistanceCompensation', 1)
scaleTrace.add_data('ApplySpeedCap', 0)
scaleTrace.add_data('UseCalculatedDownscalingPercentage', 1)
scaleTrace.add_data('DownscalingPercentage', 0)


@sh.add_function(scaleTrace, outputs=defaults_mapping.values())
def get_defaults(wltp_class):
    data = _get_class_defaults()[wltp_class]
    return [data.get(k, sh.NONE) for k in defaults_mapping]


@sh.add_function(scaleTrace, True, True, outputs=['CappedSpeed'])
def get_CappedSpeed(maximum_velocity_range='>150'):
    if isinstance(maximum_velocity_range, str):
        return float(maximum_velocity_range[1:])
    return maximum_velocity_range


dsp.add_dispatcher(
    include_defaults=True,
    dsp=scaleTrace,
    inputs={
        'wltp_class': 'wltp_class',
        'vehicle_mass': 'VehicleTestMass',
        'road_loads': 'road_loads',
        'has_capped_velocity': 'ApplySpeedCap',
        'maximum_velocity_range': 'maximum_velocity_range',
        'max_velocity': 'MaximumVelocityDefined',
        'engine_max_power': 'RatedEnginePower',
    },
    outputs={
        'compensatedTraceTimes': 'theoretical_times',
        'compensatedVehicleSpeeds': 'theoretical_velocities'
    }
)


@sh.add_function(dsp, outputs=['max_time'])
def calculate_max_time(theoretical_times):
    return max(theoretical_times)


i = ['vehicle_mass', 'road_loads', 'inertial_factor']
calculate_class_powers = sh.SubDispatchPipe(
    _vehicle,
    function_id='calculate_class_powers',
    inputs=['times', 'velocities'] + i,
    outputs=['motive_powers']
)

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
