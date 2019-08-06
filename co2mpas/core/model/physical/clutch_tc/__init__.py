# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model clutch and torque converter.

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
from ..defaults import dfl
from .clutch import dsp as _clutch
from co2mpas.utils import BaseModel
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
        clutch_phases, engine_speeds_out, engine_speeds_out_hot,
        cold_start_speeds_delta):
    """
    Identifies the engine speed delta due to the clutch [RPM].

    :param clutch_phases:
        When the clutch is active [-].
    :type clutch_phases: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param cold_start_speeds_delta:
        Engine speed delta due to the cold start [RPM].
    :type cold_start_speeds_delta: numpy.array

    :return:
        Engine speed delta due to the clutch or torque converter [RPM].
    :rtype: numpy.array
    """
    ds = engine_speeds_out - engine_speeds_out_hot - cold_start_speeds_delta
    return np.where(clutch_phases, ds, 0)


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
def identify_clutch_tc_mean_efficiency(clutch_tc_powers, clutch_tc_powers_out):
    """
    Identify clutch or torque converter mean efficiency [-].

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param clutch_tc_powers_out:
        Clutch or torque converter power out [kW].
    :type clutch_tc_powers_out: numpy.array

    :return:
        Clutch or torque converter mean efficiency [-].
    :rtype: float
    """
    from ..gear_box import identify_gear_box_mean_efficiency as func
    return func(clutch_tc_powers, clutch_tc_powers_out)


@sh.add_function(dsp, outputs=['clutch_tc_speeds'])
def calculate_clutch_tc_speeds(engine_speeds_out_hot, clutch_tc_speeds_delta):
    """
    Calculate clutch or torque converter speed (no cold start) [RPM].

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param clutch_tc_speeds_delta:
        Engine speed delta due to the clutch or torque converter [RPM].
    :type clutch_tc_speeds_delta: numpy.array

    :return:
        Clutch or torque converter speed (no cold start) [RPM].
    :rtype: numpy.array
    """
    return engine_speeds_out_hot + clutch_tc_speeds_delta


@sh.add_function(dsp, outputs=['clutch_tc_powers'])
def calculate_clutch_tc_powers(
        clutch_tc_speeds_delta, k_factor_curve, gear_box_speeds_in,
        clutch_tc_powers_out, clutch_tc_speeds):
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

    :param clutch_tc_powers_out:
        Clutch or torque converter power out [kW].
    :type clutch_tc_powers_out: numpy.array

    :param clutch_tc_speeds:
        Clutch or torque converter speed (no cold start) [RPM].
    :type clutch_tc_speeds: numpy.array

    :return:
        Clutch or torque converter power [kW].
    :rtype: numpy.array
    """
    is_not_eng2gb = gear_box_speeds_in >= clutch_tc_speeds
    speed_out = np.where(is_not_eng2gb, clutch_tc_speeds, gear_box_speeds_in)
    speed_in = np.where(is_not_eng2gb, gear_box_speeds_in, clutch_tc_speeds)

    ratios = np.ones_like(clutch_tc_powers_out, dtype=float)
    b = (speed_in > 0) & ~np.isclose(clutch_tc_speeds_delta, 0)
    ratios[b] = speed_out[b] / speed_in[b]

    eff = k_factor_curve(ratios) * ratios
    eff[is_not_eng2gb] = np.nan_to_num(1 / eff[is_not_eng2gb])

    powers = clutch_tc_powers_out.copy()
    b = eff > 0
    powers[b] = clutch_tc_powers_out[b] / eff[b]

    return powers


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
        'accelerations', 'clutch_window', 'lockup_speed_ratio', 'velocities',
        'cold_start_speeds_delta', 'engine_speeds_out', 'engine_speeds_out_hot',
        'gear_box_speeds_in', 'gear_shifts', 'gears', 'clutch_speed_model',
        'max_clutch_window_width', 'stand_still_torque_ratio', 'stop_velocity',
        'clutch_tc_speeds_delta', 'times', 'clutch_phases', dict(
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


# noinspection PyMissingOrEmptyDocstring
class ClutchTCModel(BaseModel):
    key_outputs = 'clutch_phases', 'clutch_tc_speeds_delta', 'clutch_tc_powers'
    types = {float: set(key_outputs[1:]), bool: {key_outputs[0]}}

    def __init__(self, clutch_tc_speed_model=None, clutch_window=None,
                 k_factor_curve=None, outputs=None):
        self.clutch_tc_speed_model = clutch_tc_speed_model
        self.clutch_window = clutch_window
        self.k_factor_curve = k_factor_curve
        super(ClutchTCModel, self).__init__(outputs)

    def init_phase(self, times, gears):
        key = 'clutch_phases'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]
        dt = self.clutch_window[1]

        def _next(i):
            if i:
                t0, g1 = times[i] - dt, gears[i]
                for g, t in zip(gears[i - 1::-1][::-1], times[i - 1::-1]):
                    if t < t0:
                        break
                    elif g != g1:
                        return True
            return False

        return _next

    def init_speed(self, clutch_phases, accelerations, velocities,
                   gear_box_speeds_in, gears, times, gear_box_torques_in):
        key = 'clutch_tc_speeds_delta'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]
        enable = dfl.functions.predict_clutch_tc_speeds_delta.ENABLE

        def _next(i):
            if enable and clutch_phases[i]:
                kwargs = dict(
                    accelerations=accelerations[i],
                    gear_box_torques_in=gear_box_torques_in[i],
                    gear_box_speeds_in=gear_box_speeds_in[i],
                    gears=gears[i],
                    velocities=velocities[i]
                )
                return self.clutch_tc_speed_model(times, **kwargs)
            return 0

        return _next

    def init_power(self, clutch_tc_speeds_delta, k_factor_curve,
                   gear_box_speeds_in, clutch_tc_powers_out,
                   engine_speeds_out_hot):
        key = 'clutch_tc_powers'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]

        def _next(i):
            return _calculate_clutch_tc_powers(
                clutch_tc_speeds_delta[i], k_factor_curve,
                gear_box_speeds_in[i], clutch_tc_powers_out[i],
                engine_speeds_out_hot[i]
            )

        return _next

    def init_results(self, accelerations, velocities, gear_box_speeds_in, gears,
                     times, clutch_tc_powers_out, engine_speeds_out_hot,
                     gear_box_torques_in):
        out = self.outputs
        deltas, powers = out['clutch_tc_speeds_delta'], out['clutch_tc_powers']
        phases = out['clutch_phases']
        ph_gen = self.init_phase(times, gears)
        s_gen = self.init_speed(
            phases, accelerations, velocities,
            gear_box_speeds_in, gears, times, gear_box_torques_in
        )
        p_gen = self.init_power(
            deltas, self.k_factor_curve, gear_box_speeds_in,
            clutch_tc_powers_out, engine_speeds_out_hot
        )

        def _next(i):
            phases[i] = ph = ph_gen(i)
            deltas[i] = s = s_gen(i)
            powers[i] = p = p_gen(i)
            return ph, s, p

        return _next


@sh.add_function(dsp, outputs=['clutch_tc_prediction_model'])
def define_fake_clutch_tc_prediction_model(
        clutch_phases, clutch_tc_speeds_delta, clutch_tc_powers):
    """
    Defines a fake clutch or torque converter prediction model.

    :param clutch_phases:
        When the clutch is active [-].
    :type clutch_phases: numpy.array

    :param clutch_tc_speeds_delta:
        Engine speed delta due to the clutch or torque converter [RPM].
    :type clutch_tc_speeds_delta: numpy.array

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :return:
        Clutch or torque converter prediction model.
    :rtype: ClutchTCModel
    """
    model = ClutchTCModel(outputs={
        'clutch_phases': clutch_phases,
        'clutch_tc_speeds_delta': clutch_tc_speeds_delta,
        'clutch_tc_powers': clutch_tc_powers
    })
    return model


@sh.add_function(dsp, outputs=['clutch_tc_prediction_model'], weight=4000)
def define_clutch_tc_prediction_model(
        clutch_tc_speed_model, clutch_window, k_factor_curve):
    """
    Defines the clutch or torque converter prediction model.

    :param clutch_tc_speed_model:
        Clutch or Torque converter speed model.
    :type clutch_tc_speed_model: callable

    :param clutch_window:
        Clutching time window [s].
    :type clutch_window: tuple

    :param k_factor_curve:
        k factor curve.
    :type k_factor_curve: callable

    :return:
        Clutch or torque converter prediction model.
    :rtype: ClutchTCModel
    """
    return ClutchTCModel(clutch_tc_speed_model, clutch_window, k_factor_curve)
