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
import schedula as sh
from .clutch import dsp as _clutch
from .torque_converter import dsp as _torque_converter
from co2mpas.utils import BaseModel

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
    b = not kwargs.get('has_torque_converter', True)
    return b or kwargs.get('gear_box_type') == 'cvt'


dsp.add_dispatcher(
    include_defaults=True,
    input_domain=clutch_domain,
    dsp=_clutch,
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
        'init_clutch_tc_speed_prediction_model',
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
    dsp=_torque_converter,
    dsp_id='torque_converter',
    inputs=(
        'accelerations', 'calibration_tc_speed_threshold',
        'cold_start_speeds_delta', 'engine_speeds_out', 'gears',
        'lock_up_tc_limits', 'lockup_speed_ratio',
        'stand_still_torque_ratio', 'stop_velocity', 'times',
        'torque_converter_model', 'velocities',
        'm1000_curve_factor', 'm1000_curve_ratios', 'm1000_curve_norm_torques',
        'full_load_curve', 'gear_box_torques_in',
        {'clutch_tc_speeds_delta': 'torque_converter_speeds_delta',
         'engine_speeds_out_hot':
             ('gear_box_speeds_in', 'engine_speeds_out_hot'),
         'gear_box_type': sh.SINK,
         'has_torque_converter': sh.SINK}),
    outputs=(
        'k_factor_curve', 'torque_converter_model',
        'init_clutch_tc_speed_prediction_model','normalized_m1000_curve',
        'm1000_curve_factor',
        {'torque_converter_speeds_delta': 'clutch_tc_speeds_delta'})
)


# noinspection PyMissingOrEmptyDocstring
class ClutchTCModel(BaseModel):
    key_outputs = [
        'clutch_tc_speeds_delta',
        'clutch_tc_powers'
    ]
    types = {float: set(key_outputs)}

    def __init__(self, init_clutch_tc_speed_prediction_model=None,
                 k_factor_curve=None, outputs=None):
        self.init_clutch_tc_speed_prediction_model = \
            init_clutch_tc_speed_prediction_model
        self.k_factor_curve = k_factor_curve
        super(ClutchTCModel, self).__init__(outputs)

    def init_speed(self, accelerations, velocities, gear_box_speeds_in, gears,
                   times, clutch_speeds_delta):
        key = 'clutch_tc_speeds_delta'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]

        return self.init_clutch_tc_speed_prediction_model(
            accelerations, velocities, gear_box_speeds_in, gears, times,
            clutch_speeds_delta
        )

    def init_power(self, clutch_tc_speeds_delta, k_factor_curve,
                   gear_box_speeds_in, gear_box_powers_in,
                   engine_speeds_out_hot):
        key = 'clutch_tc_powers'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]

        def _next(i):
            return _calculate_clutch_tc_powers(
                clutch_tc_speeds_delta[i], k_factor_curve,
                gear_box_speeds_in[i], gear_box_powers_in[i],
                engine_speeds_out_hot[i]
            )

        return _next

    def init_results(self, accelerations, velocities, gear_box_speeds_in, gears,
                     times, gear_box_powers_in, engine_speeds_out_hot):
        out = self.outputs
        deltas, powers = out['clutch_tc_speeds_delta'], out['clutch_tc_powers']

        s_gen = self.init_speed(
            accelerations, velocities, gear_box_speeds_in, gears, times, deltas
        )
        p_gen = self.init_power(
            deltas, self.k_factor_curve, gear_box_speeds_in, gear_box_powers_in,
            engine_speeds_out_hot
        )

        def _next(i):
            deltas[i] = s = s_gen(i)
            powers[i] = p = p_gen(i)
            return s, p

        return _next


@sh.add_function(dsp, outputs=['clutch_tc_prediction_model'])
def define_fake_clutch_tc_prediction_model(
        clutch_tc_speeds_delta, clutch_tc_powers):
    """
    Defines a fake clutch or torque converter prediction model.

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
        'clutch_tc_speeds_delta': clutch_tc_speeds_delta,
        'clutch_tc_powers': clutch_tc_powers
    })
    return model


@sh.add_function(dsp, outputs=['clutch_tc_prediction_model'], weight=4000)
def define_wheels_prediction_model(
        init_clutch_tc_speed_prediction_model, k_factor_curve):
    """
    Defines the clutch or torque converter prediction model.

    :param init_clutch_tc_speed_prediction_model:
        Initialization function of the clutch tc speed prediction model.
    :type init_clutch_tc_speed_prediction_model: function

    :param k_factor_curve:
        k factor curve.
    :type k_factor_curve: callable

    :return:
        Clutch or torque converter prediction model.
    :rtype: ClutchTCModel
    """
    return ClutchTCModel(init_clutch_tc_speed_prediction_model, k_factor_curve)
