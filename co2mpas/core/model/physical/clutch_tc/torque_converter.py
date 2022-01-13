# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the mechanic of the torque converter.
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl

dsp = sh.BlueDispatcher(
    name='Torque_converter', description='Models the torque converter.'
)


@sh.add_function(dsp, outputs=['clutch_window'])
def default_clutch_window():
    """
    Returns a default clutching time window [s] for a generic clutch.

    :return:
        Clutching time window [s].
    :rtype: tuple
    """
    return dfl.functions.default_clutch_window.clutch_window


@sh.add_function(dsp, inputs_kwargs=True, outputs=['k_factor_curve'])
def define_k_factor_curve(stand_still_torque_ratio=1.0, lockup_speed_ratio=0.0):
    """
    Defines k factor curve.

    :param stand_still_torque_ratio:
        Torque ratio when speed ratio==0.

        .. note:: The ratios are defined as follows:

           - Torque ratio = `gear box torque` / `engine torque`.
           - Speed ratio = `gear box speed` / `engine speed`.
    :type stand_still_torque_ratio: float

    :param lockup_speed_ratio:
        Minimum speed ratio where torque ratio==1.

        ..note::
            torque ratio==1 for speed ratio > lockup_speed_ratio.
    :type lockup_speed_ratio: float

    :return:
        k factor curve.
    :rtype: callable
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    if lockup_speed_ratio == 0:
        x = [0, 1]
        y = [1, 1]
    elif lockup_speed_ratio == 1:
        x = [0, 1]
        y = [stand_still_torque_ratio, 1]
    else:
        x = [0, lockup_speed_ratio, 1]
        y = [stand_still_torque_ratio, 1, 1]

    return InterpolatedUnivariateSpline(x, y, k=1)


@sh.add_function(dsp, outputs=['k_factor_curve'], weight=2)
def default_tc_k_factor_curve():
    """
    Returns a default k factor curve for a generic torque converter.

    :return:
        k factor curve.
    :rtype: callable
    """
    from co2mpas.defaults import dfl
    par = dfl.functions.default_tc_k_factor_curve
    a = par.STAND_STILL_TORQUE_RATIO, par.LOCKUP_SPEED_RATIO
    return define_k_factor_curve(*a)


@sh.add_function(
    dsp, outputs=['m1000_curve_ratios', 'm1000_curve_norm_torques'], weight=2
)
def default_tc_normalized_m1000_curve():
    """
    Returns default `m1000_curve_ratios` and `m1000_curve_norm_torques`.

    :return:
        Speed ratios and normalized torques of m1000 curve.
    :rtype: tuple[numpy.array]
    """
    from co2mpas.defaults import dfl
    curve = dfl.functions.default_tc_normalized_m1000_curve.curve
    return np.array(curve['x']), np.array(curve['y'])


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['normalized_VDI253_model'], weight=2)
def define_normalized_VDI253_model(
        m1000_curve_ratios, m1000_curve_norm_torques, idle_engine_speed,
        engine_max_speed, k_factor_curve):
    """
    Defines normalized VDI253 model function.

    :param m1000_curve_ratios:
        Speed ratios of m1000 curve [-].
    :type m1000_curve_ratios: numpy.array

    :param m1000_curve_norm_torques:
        Normalized torques of m1000 curve [-].
    :type m1000_curve_norm_torques: numpy.array

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :param k_factor_curve:
        k factor curve.
    :type k_factor_curve: callable

    :return:
        Normalized VDI253 model function.
    :rtype: scipy.interpolate.LinearNDInterpolator
    """
    from scipy.interpolate import interp1d, LinearNDInterpolator
    maximum_ratio, idle = np.max(m1000_curve_ratios), idle_engine_speed[0]
    eng_s = np.linspace(idle, engine_max_speed, 100)
    gb_s = np.linspace(0, engine_max_speed * maximum_ratio, 250)
    x, z = np.meshgrid(gb_s, eng_s)
    r = x / z
    b = r <= maximum_ratio
    x, z, r = x[b], z[b], r[b]
    func = interp1d(m1000_curve_ratios, m1000_curve_norm_torques, kind='cubic')
    tout = func(r) * k_factor_curve(r) * z ** 2
    return LinearNDInterpolator((x, tout), z)


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['m1000_curve_factor'])
def calibrate_m1000_curve_factor(
        full_load_curve, normalized_VDI253_model, clutch_phases,
        engine_speeds_out_hot, gear_box_speeds_in, gear_box_torques_in,
        clutch_tc_speeds_delta):
    """
    Calibrate the rescaling factor of m1000 curve [N*m/1e6].

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param normalized_VDI253_model:
        Normalized VDI253 model function.
    :type normalized_VDI253_model: scipy.interpolate.LinearNDInterpolator

    :param clutch_phases:
        When the clutch is active [-].
    :type clutch_phases: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array

    :param clutch_tc_speeds_delta:
        Engine speed delta due to the clutch or torque converter [RPM].
    :type clutch_tc_speeds_delta: numpy.array

    :return:
        Rescaling factor of m1000 curve [N*m/1e6].
    :rtype: float
    """
    if clutch_phases.sum() <= 10:
        return sh.NONE
    from co2mpas.utils import mae
    from scipy.optimize import fmin

    # noinspection PyUnresolvedReferences
    es, gbs, gbt, predict, ds = (
        engine_speeds_out_hot[clutch_phases], gear_box_speeds_in[clutch_phases],
        gear_box_torques_in[clutch_phases], normalized_VDI253_model.predict,
        clutch_tc_speeds_delta[clutch_phases]

    )

    def _err(factor):
        e = mae(ds, np.nan_to_num(predict((gbs, gbt / factor)) - es))
        return np.float32(e)

    return fmin(_err, default_m1000_curve_factor(full_load_curve))


@sh.add_function(dsp, outputs=['m1000_curve_factor'], weight=1000)
def default_m1000_curve_factor(full_load_curve):
    """
    Returns the default value of the rescaling factor of m1000 curve [N*m].

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :return:
        Rescaling factor of m1000 curve [N*m/1e6].
    :rtype: float
    """
    from ..wheels import calculate_wheel_torques
    return calculate_wheel_torques(full_load_curve(1000), 1000) / 1e6


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['torque_converter_speed_model'])
def define_tc_speed_model(
        normalized_VDI253_model, m1000_curve_factor, idle_engine_speed):
    """
    Define torque converter speed model.

    :param normalized_VDI253_model:
        Normalized VDI253 model function.
    :type normalized_VDI253_model: scipy.interpolate.LinearNDInterpolator

    :param m1000_curve_factor:
        Rescaling factor of m1000 curve [N*m/1e6].
    :type m1000_curve_factor: float

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Torque converter speed model.
    :rtype: callable
    """

    # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
    def model(times, **kwargs):
        gbs, gbt = kwargs['gear_box_speeds_in'], kwargs['gear_box_torques_in']
        es = normalized_VDI253_model((gbs, gbt / m1000_curve_factor))
        return np.nan_to_num(es - np.maximum(gbs, idle_engine_speed[0]))

    return model
