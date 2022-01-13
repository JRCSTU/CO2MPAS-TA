# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the vehicle control strategy.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.control

.. autosummary::
    :nosignatures:
    :toctree: control/

    conventional
    hybrid
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
from .conventional import dsp as _conventional
from .hybrid import dsp as _hybrid

dsp = sh.BlueDispatcher(
    name='Control', description='Models the vehicle control strategy.'
)

dsp.add_data(
    'min_time_engine_on_after_start', dfl.values.min_time_engine_on_after_start
)


@sh.add_function(dsp, outputs=['on_engine'])
def identify_on_engine(
        times, engine_speeds_out, idle_engine_speed,
        min_time_engine_on_after_start):
    """
    Identifies if the engine is on [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param min_time_engine_on_after_start:
        Minimum time of engine on after a start [s].
    :type min_time_engine_on_after_start: float

    :return:
        If the engine is on [-].
    :rtype: numpy.array
    """

    on_engine = engine_speeds_out > idle_engine_speed[0] - idle_engine_speed[1]
    mask = np.where(identify_engine_starts(on_engine))[0] + 1
    ts = np.asarray(times[mask], dtype=float)
    ts += min_time_engine_on_after_start + dfl.EPS
    for i, j in np.column_stack((mask, np.searchsorted(times, ts))):
        on_engine[i:j] = True

    return on_engine


@sh.add_function(dsp, outputs=['engine_starts'])
def identify_engine_starts(on_engine):
    """
    Identifies when the engine starts [-].

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :return:
        When the engine starts [-].
    :rtype: numpy.array
    """
    engine_starts = np.zeros_like(on_engine, dtype=bool)
    engine_starts[:-1] = on_engine[1:] & (on_engine[:-1] != on_engine[1:])
    return engine_starts


# noinspection PyMissingOrEmptyDocstring
def is_hybrid(kwargs):
    return kwargs.get('is_hybrid')


# noinspection PyMissingOrEmptyDocstring
def is_not_hybrid(kwargs):
    return not kwargs.get('is_hybrid', True)


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_conventional,
    dsp_id='conventional_control',
    inputs=(
        'gear_box_type', 'start_stop_activation_time', 'accelerations', 'times',
        'idle_engine_speed', 'min_time_engine_on_after_start', 'engine_starts',
        'has_start_stop', 'correct_start_stop_with_gears', 'start_stop_model',
        'gear_box_speeds_in', 'velocities', 'on_engine', 'gears',
        {'is_hybrid': sh.SINK}
    ),
    outputs=(
        'correct_start_stop_with_gears', 'engine_speeds_out_hot', 'on_engine',
        'start_stop_activation_time', 'start_stop_model', 'hybrid_modes',
    ),
    input_domain=is_not_hybrid
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_hybrid,
    dsp_id='hybrid_control',
    inputs=(
        'clutch_tc_mean_efficiency', 'idle_engine_speed', 'motor_p0_efficiency',
        'motor_p0_maximum_power', 'drive_battery_model', 'belt_mean_efficiency',
        'min_time_engine_on_after_start', 'start_stop_activation_time', 'times',
        'drive_battery_state_of_charges', 'motor_p2_maximum_powers', 'fuel_map',
        'engine_thermostat_temperature', 'motor_p2_efficiency', 'accelerations',
        'motor_p0_maximum_power_function', 'on_engine', 'engine_moment_inertia',
        'motor_p1_maximum_power_function', 'ecms_s', 'start_stop_hybrid_params',
        'motor_p1_speed_ratio', 'auxiliaries_power_loss', 'motor_p1_efficiency',
        'final_drive_mean_efficiency', 'motors_electric_powers', 'is_cycle_hot',
        'auxiliaries_torque_loss', 'dcdc_converter_efficiency', 'motive_powers',
        'motor_p2_electric_powers', 'motor_p0_speed_ratio', 'engine_speeds_out',
        'motor_p3_rear_efficiency', 'motor_p1_maximum_power', 'full_load_curve',
        'motor_p3_rear_maximum_powers', 'motor_p0_electric_powers', 'is_serial',
        'after_treatment_warm_up_duration', 'engine_powers_out', 'hybrid_modes',
        'motor_p2_maximum_power', 'engine_speeds_out_hot', 'gear_box_speeds_in',
        'gear_box_mean_efficiency', 'motor_p1_electric_powers', 'starter_model',
        'motor_p2_planetary_maximum_power_function', 'motor_p4_rear_efficiency',
        'motor_p3_front_efficiency', 'final_drive_speeds_in', 'planetary_ratio',
        'after_treatment_cooling_duration', 'motor_p2_planetary_maximum_power',
        'motor_p2_planetary_electric_powers', 'after_treatment_warm_up_phases',
        'gear_box_mean_efficiency_guess', 'motor_p3_front_electric_powers',
        'motor_p2_planetary_speed_ratio', 'motor_p4_front_electric_powers',
        'motor_p2_planetary_efficiency', 'motor_p4_rear_electric_powers',
        'motor_p3_front_maximum_powers', 'motor_p3_rear_electric_powers',
        'motor_p4_front_maximum_powers', 'motor_p4_rear_maximum_powers',
        'motor_p4_front_maximum_power', 'motor_p3_front_maximum_power',
        'engine_temperatures', 'motor_p3_rear_maximum_power',
        'motor_p4_rear_maximum_power', 'motor_p4_front_efficiency',
        'planetary_mean_efficiency', 'has_motor_p2_planetary',
        'min_engine_on_speed', {'is_hybrid': sh.SINK},
    ),
    outputs=(
        'motor_p3_front_electric_powers', 'motor_p2_electric_powers', 'ecms_s',
        'motor_p4_front_electric_powers', 'engine_speeds_base', 'hybrid_modes',
        'after_treatment_warm_up_phases', 'engine_speeds_out_hot', 'on_engine',
        'motor_p2_planetary_electric_powers', 'motor_p3_rear_electric_powers',
        'motor_p4_rear_electric_powers', 'start_stop_hybrid_params',
        'motor_p1_electric_powers', 'motor_p0_electric_powers',
        'force_on_engine', 'start_stop_activation_time'
    ),
    input_domain=is_hybrid
)
