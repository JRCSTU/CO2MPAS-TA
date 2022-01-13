# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model clutch and torque converter.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.clutch_tc

.. autosummary::
    :nosignatures:
    :toctree: clutch_tc/

    clutch
    torque_converter
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
from .clutch import dsp as _clutch
from .torque_converter import dsp as _torque_converter

dsp = sh.BlueDispatcher(
    name='Clutch and torque-converter',
    description='Models the clutch and torque-converter.'
)

dsp.add_data('stop_velocity', dfl.values.stop_velocity)


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


@sh.add_function(dsp, outputs=['clutch_phases'])
def calculate_clutch_phases(
        times, velocities, gears, gear_shifts, stop_velocity, clutch_window):
    """
    Calculate when the clutch is active [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param gear_shifts:
        When there is a gear shifting [-].
    :type gear_shifts: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param clutch_window:
        Clutching time window [s].
    :type clutch_window: tuple

    :return:
        When the clutch is active [-].
    :rtype: numpy.array
    """

    dn, up = clutch_window
    b = np.zeros_like(times, dtype=bool)

    for t in times[gear_shifts]:
        b |= ((t + dn) <= times) & (times <= (t + up))
    b &= (gears > 0) & (velocities > stop_velocity)
    return b


@sh.add_function(dsp, outputs=['clutch_tc_speeds_delta'])
def identify_clutch_tc_speeds_delta(
        clutch_phases, engine_speeds_out, engine_speeds_base):
    """
    Identifies the engine speed delta due to the clutch [RPM].

    :param clutch_phases:
        When the clutch is active [-].
    :type clutch_phases: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_speeds_base:
        Base engine speed (i.e., without clutch/TC effect) [RPM].
    :type engine_speeds_base: numpy.array

    :return:
        Engine speed delta due to the clutch or torque converter [RPM].
    :rtype: numpy.array
    """
    return np.where(clutch_phases, engine_speeds_out - engine_speeds_base, 0)


@sh.add_function(dsp, outputs=['clutch_tc_speeds_delta'])
def predict_clutch_tc_speeds_delta(
        clutch_tc_speed_model, times, clutch_phases, accelerations,
        velocities, gear_box_speeds_in, gears, gear_box_torques_in):
    """
    Predicts engine speed delta due to the clutch or torque converter [RPM].

    :param clutch_tc_speed_model:
        Clutch or Torque converter speed model.
    :type clutch_tc_speed_model: callable

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param clutch_phases:
        When the clutch is active [-].
    :type clutch_phases: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array

    :return:
        Engine speed delta due to the clutch or torque converter [RPM].
    :rtype: numpy.array
    """
    b = dfl.functions.predict_clutch_tc_speeds_delta.ENABLE
    if b and clutch_phases.any():
        func, kwargs = clutch_tc_speed_model, dict(
            accelerations=accelerations,
            gear_box_torques_in=gear_box_torques_in,
            gear_box_speeds_in=gear_box_speeds_in, gears=gears,
            velocities=velocities
        )
        return np.where(clutch_phases, func(times, **kwargs), 0)
    return np.zeros_like(clutch_phases, float)


def _calculate_clutch_tc_powers(
        clutch_tc_speeds_delta, k_factor_curve, gear_box_speed_in,
        gear_box_power_in, engine_speed_out_hot):
    engine_speed_out = engine_speed_out_hot + clutch_tc_speeds_delta
    is_not_eng2gb = gear_box_speed_in >= engine_speed_out
    if clutch_tc_speeds_delta == 0:
        ratio = 1
    else:
        if is_not_eng2gb:
            speed_out, speed_in = engine_speed_out, gear_box_speed_in
        else:
            speed_out, speed_in = gear_box_speed_in, engine_speed_out

        if (speed_in > 0) and (clutch_tc_speeds_delta != 0):
            ratio = speed_out / speed_in
        else:
            ratio = 1

    eff = k_factor_curve(ratio) * ratio
    if is_not_eng2gb and eff != 0:
        eff = 1 / eff
    if eff > 0:
        return gear_box_power_in / eff
    return gear_box_power_in


@sh.add_function(dsp, outputs=['clutch_tc_mean_efficiency'])
def identify_clutch_tc_mean_efficiency(clutch_tc_efficiencies):
    """
    Identify clutch or torque converter mean efficiency [-].

    :param clutch_tc_efficiencies:
        Clutch or torque converter efficiency [-].
    :type clutch_tc_efficiencies: numpy.array

    :return:
        Clutch or torque converter mean efficiency [-].
    :rtype: float
    """
    from co2mpas.utils import reject_outliers
    return reject_outliers(clutch_tc_efficiencies)[0]


@sh.add_function(dsp, outputs=['clutch_tc_efficiencies'])
def calculate_clutch_tc_efficiencies(
        clutch_tc_speeds_delta, k_factor_curve, gear_box_speeds_in,
        engine_speeds_out):
    """
    Calculates the efficiency of the clutch or torque converter [-].

    :param clutch_tc_speeds_delta:
        Engine speed delta due to the clutch or torque converter [RPM].
    :type clutch_tc_speeds_delta: numpy.array

    :param k_factor_curve:
        k factor curve.
    :type k_factor_curve: callable

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :return:
        Clutch or torque converter efficiency [-].
    :rtype: numpy.array
    """
    is_not_eng2gb = gear_box_speeds_in >= engine_speeds_out
    speed_out = np.where(is_not_eng2gb, engine_speeds_out, gear_box_speeds_in)
    speed_in = np.where(is_not_eng2gb, gear_box_speeds_in, engine_speeds_out)

    ratios = np.ones_like(clutch_tc_speeds_delta, dtype=float)
    b = (speed_in > 0) & ~np.isclose(clutch_tc_speeds_delta, 0)
    ratios[b] = speed_out[b] / speed_in[b]
    return k_factor_curve(ratios) * ratios


# noinspection PyIncorrectDocstring
@sh.add_function(dsp, outputs=['clutch_tc_powers'])
def calculate_clutch_tc_powers(
        clutch_tc_efficiencies, gear_box_speeds_in, clutch_tc_powers_out,
        engine_speeds_out, _i=1):
    """
    Calculates the power that flows in the clutch or torque converter [kW].

    :param clutch_tc_efficiencies:
        Clutch or torque converter efficiency [-].
    :type clutch_tc_efficiencies: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param clutch_tc_powers_out:
        Clutch or torque converter power out [kW].
    :type clutch_tc_powers_out: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :return:
        Clutch or torque converter power [kW].
    :rtype: numpy.array
    """
    b = gear_box_speeds_in >= engine_speeds_out
    eff = np.where(clutch_tc_efficiencies == 0, 1, clutch_tc_efficiencies)
    return clutch_tc_powers_out * np.where(b, *(eff, 1 / eff)[::_i])


@sh.add_function(dsp, outputs=['clutch_tc_powers_out'])
def calculate_clutch_tc_powers_out(
        clutch_tc_efficiencies, gear_box_speeds_in, clutch_tc_powers,
        engine_speeds_out):
    """
    Calculates the clutch or torque converter power out [kW].

    :param clutch_tc_efficiencies:
        Clutch or torque converter efficiency [-].
    :type clutch_tc_efficiencies: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :return:
        Clutch or torque converter power out [kW].
    :rtype: numpy.array
    """
    return calculate_clutch_tc_powers(
        clutch_tc_efficiencies, gear_box_speeds_in, clutch_tc_powers,
        engine_speeds_out, _i=-1
    )


# noinspection PyMissingOrEmptyDocstring
def clutch_domain(kwargs):
    b = not kwargs.get('has_torque_converter', True)
    return b or kwargs.get('gear_box_type') == 'cvt'


dsp.add_dispatcher(
    include_defaults=True,
    input_domain=clutch_domain,
    dsp=_clutch,
    dsp_id='clutch',
    inputs=(
        'engine_speeds_out', 'stop_velocity', 'clutch_tc_speeds_delta', 'times',
        'stand_still_torque_ratio', 'gear_box_speeds_in', 'clutch_speed_model',
        'accelerations', 'clutch_window', 'lockup_speed_ratio', 'velocities',
        'gear_shifts', 'gears', 'clutch_phases', dict(
            gear_box_type=sh.SINK, has_torque_converter=sh.SINK
        )),
    outputs=(
        'clutch_speed_model', 'clutch_phases', 'clutch_window',
        'k_factor_curve', 'clutch_tc_speeds_delta'
    )
)


# noinspection PyMissingOrEmptyDocstring
def torque_converter_domain(kwargs):
    b = kwargs.get('has_torque_converter')
    return b and kwargs.get('gear_box_type') != 'cvt'


dsp.add_dispatcher(
    include_defaults=True,
    input_domain=torque_converter_domain,
    dsp=_torque_converter,
    dsp_id='torque_converter',
    inputs=(
        'lockup_speed_ratio', 'engine_max_speed', 'stand_still_torque_ratio',
        'torque_converter_speed_model', 'gear_box_torques_in', 'clutch_window',
        'full_load_curve', 'engine_speeds_out_hot', 'm1000_curve_norm_torques',
        'm1000_curve_factor', 'm1000_curve_ratios', 'clutch_tc_speeds_delta',
        'gear_box_speeds_in', 'idle_engine_speed', dict(
            gear_box_type=sh.SINK, has_torque_converter=sh.SINK
        )),
    outputs=(
        'k_factor_curve', 'torque_converter_speed_model', 'clutch_window',
        'm1000_curve_factor'
    )
)

dsp.add_function(
    function=sh.bypass,
    inputs=['torque_converter_speed_model'],
    outputs=['clutch_tc_speed_model']
)

dsp.add_function(
    function=sh.bypass,
    inputs=['clutch_speed_model'],
    outputs=['clutch_tc_speed_model']
)
