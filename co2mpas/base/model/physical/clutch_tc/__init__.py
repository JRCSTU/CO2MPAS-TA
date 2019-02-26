# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions that model the basic mechanics of the clutch and torque
converter.

Sub-Modules:

.. currentmodule:: co2mpas.base.model.physical.clutch_tc

.. autosummary::
    :nosignatures:
    :toctree: clutch_tc/

    clutch
    torque_converter
"""

import schedula as sh
from .clutch import dsp as clutch
from .torque_converter import dsp as torque_converter

dsp = sh.BlueDispatcher(
    name='Clutch and torque-converter',
    description='Models the clutch and torque-converter.'
)


@sh.add_function(dsp, outputs=['has_torque_converter'])
def default_has_torque_converter(gear_box_type):
    """
    Returns the default has torque converter value [-].

    :param gear_box_type:
        Gear box type (manual or automatic or cvt).
    :type gear_box_type: str

    :return:
        Does the vehicle use torque converter? [-]
    :rtype: bool
    """

    return gear_box_type == 'automatic'


@sh.add_function(dsp, outputs=['clutch_tc_powers'])
def calculate_clutch_tc_powers(
        clutch_tc_speeds_delta, k_factor_curve, gear_box_speeds_in,
        gear_box_powers_in, engine_speeds_out):
    """
    Calculates the power that flows in the clutch or torque converter [kW].

    :param clutch_tc_speeds_delta:
        Engine speed delta due to the clutch or torque converter [RPM].
    :type clutch_tc_speeds_delta: numpy.array

    :param k_factor_curve:
        k factor curve.
    :type k_factor_curve: callable

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :return:
        Clutch or torque converter power [kW].
    :rtype: numpy.array
    """
    import numpy as np

    is_not_eng2gb = gear_box_speeds_in >= engine_speeds_out
    speed_out = np.where(is_not_eng2gb, engine_speeds_out, gear_box_speeds_in)
    speed_in = np.where(is_not_eng2gb, gear_box_speeds_in, engine_speeds_out)

    ratios = np.ones_like(gear_box_powers_in, dtype=float)
    b = (speed_in > 0) & (clutch_tc_speeds_delta != 0)
    ratios[b] = speed_out[b] / speed_in[b]

    eff = k_factor_curve(ratios) * ratios
    eff[is_not_eng2gb] = np.nan_to_num(1 / eff[is_not_eng2gb])

    powers = gear_box_powers_in.copy()
    b = eff > 0
    powers[b] = gear_box_powers_in[b] / eff[b]

    return powers


# noinspection PyMissingOrEmptyDocstring
def clutch_domain(kwargs):
    b = not kwargs.get('has_torque_converter')
    return b or kwargs.get('gear_box_type') == 'cvt'


dsp.add_dispatcher(
    include_defaults=True,
    input_domain=clutch_domain,
    dsp=clutch,
    dsp_id='clutch',
    inputs=(
        'accelerations', 'clutch_model', 'clutch_window', 'lockup_speed_ratio',
        'cold_start_speeds_delta', 'engine_speeds_out', 'engine_speeds_out_hot',
        'gear_box_speeds_in', 'gear_shifts', 'gears', 'times', 'velocities',
        'max_clutch_window_width', 'stand_still_torque_ratio', 'stop_velocity',
        {'clutch_tc_speeds_delta': 'clutch_speeds_delta',
         'gear_box_type': sh.SINK,
         'has_torque_converter': sh.SINK}
    ),
    outputs=(
        'clutch_model', 'clutch_phases', 'clutch_window', 'k_factor_curve',
        {'clutch_speeds_delta': 'clutch_tc_speeds_delta'}
    )
)


# noinspection PyMissingOrEmptyDocstring
def torque_converter_domain(kwargs):
    b = kwargs.get('has_torque_converter')
    return b and kwargs.get('gear_box_type') != 'cvt'


dsp.add_dispatcher(
    include_defaults=True,
    input_domain=torque_converter_domain,
    dsp=torque_converter,
    dsp_id='torque_converter',
    inputs=(
        'accelerations', 'calibration_tc_speed_threshold',
        'cold_start_speeds_delta', 'engine_speeds_out', 'gears',
        'lock_up_tc_limits', 'lockup_speed_ratio',
        'stand_still_torque_ratio', 'stop_velocity', 'times',
        'torque_converter_model', 'velocities',
        {'clutch_tc_speeds_delta': 'torque_converter_speeds_delta',
         'engine_speeds_out_hot':
             ('gear_box_speeds_in', 'engine_speeds_out_hot'),
         'gear_box_type': sh.SINK,
         'has_torque_converter': sh.SINK}),
    outputs=(
        'k_factor_curve', 'torque_converter_model',
        {'torque_converter_speeds_delta': 'clutch_tc_speeds_delta'})
)
