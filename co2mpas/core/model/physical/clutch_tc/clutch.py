# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and model `dsp` to model the mechanic of the clutch.
"""
import functools
import numpy as np
import schedula as sh
from ..defaults import dfl
from .torque_converter import TorqueConverter, define_k_factor_curve

dsp = sh.BlueDispatcher(name='Clutch', description='Models the clutch.')
dsp.add_data('stop_velocity', dfl.values.stop_velocity)


@sh.add_function(dsp, outputs=['clutch_window'])
def default_clutch_window():
    """
    Returns a default clutching time window [s] for a generic clutch.

    :return:
        Clutching time window [s].
    :rtype: tuple
    """
    return dfl.functions.default_clutch_window.clutch_window


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


@sh.add_function(dsp, outputs=['clutch_speeds_delta'])
def identify_clutch_speeds_delta(
        clutch_phases, engine_speeds_out, engine_speeds_out_hot,
        cold_start_speeds_delta, accelerations):
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

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :return:
        Engine speed delta due to the clutch or torque converter [RPM].
    :rtype: numpy.array
    """
    delta = np.zeros_like(clutch_phases, dtype=float)
    s, h, c = engine_speeds_out, engine_speeds_out_hot, cold_start_speeds_delta
    b = clutch_phases
    delta[b] = s[b] - h[b] - c[b]

    delta[(delta > 0) & (accelerations < 0)] = 0
    return delta


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class ClutchModel(TorqueConverter):
    def __init__(self, prev_dt=1):
        super(ClutchModel, self).__init__()
        self.prev_dt = prev_dt

    def _fit_sub_set(self, clutch_phases, *args):
        return clutch_phases

    @staticmethod
    def _prediction_inputs(X):
        return X[:, :-1]

    def model(self, times, clutch_phases, X):
        d = np.zeros(X.shape[0])
        if clutch_phases.any():
            predict = self.regressor.predict
            delta = functools.partial(np.interp, xp=times, fp=d)
            t = times - self.prev_dt
            gbs = np.interp(t, xp=times, fp=X[:, 2])
            i = np.where(clutch_phases)[0]
            for i, a in zip(i, np.column_stack((gbs, t, X))[i]):
                v = predict([tuple(a[2:]) + (a[0] + delta(a[1]),)])[0]
                if not ((v >= 0) & (a[2] < 0)):
                    d[i] = v
        return d

    # noinspection PyMethodOverriding
    def next(self, clutch_window, accelerations, velocities, gear_box_speeds_in,
             gears, times, clutch_speeds_delta):
        predict, up = self.regressor.predict, clutch_window[1]

        def _next(i):
            t0, g0 = times[i] - up, gears[i]
            for t, g in zip(times[:i][::-1], gears[:i - 1][::-1]):
                if t < t0:
                    break
                elif g != g0:
                    n = len(gear_box_speeds_in)
                    t, a = times[i] - self.prev_dt, accelerations[i]
                    gbs = np.interp(t, xp=times[:n], fp=gear_box_speeds_in)
                    if clutch_speeds_delta:
                        gbs += np.interp(
                            t, xp=times[:len(clutch_speeds_delta)],
                            fp=clutch_speeds_delta
                        )
                    args = a, velocities[i], gear_box_speeds_in[i], g0, gbs
                    v = predict([args])[0]
                    if not ((v >= 0) & (a < 0)):
                        return v
                    break
            return 0

        return _next


@sh.add_function(dsp, outputs=['clutch_model'])
def calibrate_clutch_prediction_model(
        times, clutch_phases, accelerations, clutch_speeds_delta, velocities,
        gear_box_speeds_in, gears):
    """
    Calibrate clutch prediction model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param clutch_phases:
        When the clutch is active [-].
    :type clutch_phases: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param clutch_speeds_delta:
        Engine speed delta due to the clutch [RPM].
    :type clutch_speeds_delta: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        Clutch prediction model.
    :rtype: ClutchModel
    """
    from ..defaults import dfl
    prev_dt = dfl.functions.calibrate_clutch_prediction_model.prev_dt
    model = ClutchModel(prev_dt=prev_dt)
    es = np.interp(
        times - model.prev_dt, times, gear_box_speeds_in + clutch_speeds_delta
    )

    model.fit(
        times, clutch_phases, None, None, clutch_speeds_delta, accelerations,
        velocities, gear_box_speeds_in, gears, es
    )

    return model


@sh.add_function(dsp, outputs=['clutch_speeds_delta'])
def predict_clutch_speeds_delta(
        clutch_model, times, clutch_phases, accelerations, velocities,
        gear_box_speeds_in, gears):
    """
    Predicts engine speed delta due to the clutch [RPM].

    :param clutch_model:
        Clutch prediction model.
    :type clutch_model: ClutchModel

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

    :return:
        Engine speed delta due to the clutch [RPM].
    :rtype: numpy.array
    """
    x = np.column_stack((accelerations, velocities, gear_box_speeds_in, gears))
    return clutch_model(times, clutch_phases, x)


@sh.add_function(dsp, outputs=['init_clutch_tc_speed_prediction_model'])
def define_init_clutch_tc_speed_prediction_model(clutch_model, clutch_window):
    """
    Defines initialization function of the clutch tc speed prediction model.

    :param clutch_model:
        Clutch prediction model.
    :type clutch_model: ClutchModel

    :param clutch_window:
        Clutching time window [s].
    :type clutch_window: tuple

    :return:
        Initialization function of the clutch tc speed prediction model.
    :rtype: function
    """
    return functools.partial(clutch_model.next, clutch_window)


dsp.add_function(
    function=define_k_factor_curve,
    inputs=['stand_still_torque_ratio', 'lockup_speed_ratio'],
    outputs=['k_factor_curve']
)


@sh.add_function(dsp, outputs=['k_factor_curve'], weight=2)
def default_clutch_k_factor_curve():
    """
    Returns a default k factor curve for a generic clutch.

    :return:
        k factor curve.
    :rtype: callable
    """
    from ..defaults import dfl
    par = dfl.functions.default_clutch_k_factor_curve
    a = par.STAND_STILL_TORQUE_RATIO, par.LOCKUP_SPEED_RATIO
    from .torque_converter import define_k_factor_curve
    return define_k_factor_curve(*a)
