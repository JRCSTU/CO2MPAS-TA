# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the engine start stop strategy.
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl

dsp = sh.BlueDispatcher(
    name='start_stop', description='Models the engine start/stop strategy.'
)

dsp.add_data('has_start_stop', dfl.values.has_start_stop)


@sh.add_function(dsp, outputs=['start_stop_activation_time'])
def default_start_stop_activation_time(has_start_stop):
    """
    Returns the default start stop activation time threshold [s].

    :return:
        Start-stop activation time threshold [s].
    :rtype: float
    """
    if not has_start_stop or dfl.functions.ENABLE_ALL_FUNCTIONS or \
            dfl.functions.default_start_stop_activation_time.ENABLE:
        return dfl.functions.default_start_stop_activation_time.threshold
    return sh.NONE


# noinspection PyPep8Naming
def _start_stop_model(X):
    off_engine = X[:, 0] <= dfl.functions.StartStopModel.stop_velocity
    off_engine &= X[:, 1] <= dfl.functions.StartStopModel.plateau_acceleration
    return ~off_engine


@sh.add_function(dsp, outputs=['start_stop_model'])
def calibrate_start_stop_model(
        on_engine, velocities, accelerations, times,
        start_stop_activation_time):
    """
    Calibrates an start/stop model to predict if the engine is on.

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param start_stop_activation_time:
        Start-stop activation time threshold [s].
    :type start_stop_activation_time: float

    :return:
        Start/stop model.
    :rtype: sklearn.tree.DecisionTreeClassifier
    """
    from sklearn.tree import DecisionTreeClassifier
    i = np.searchsorted(times, start_stop_activation_time)
    model = DecisionTreeClassifier(random_state=0, max_depth=3)
    if i >= velocities.shape[0]:
        return _start_stop_model

    model.fit(
        np.column_stack((velocities[i:], accelerations[i:])), on_engine[i:]
    )
    return model.predict


@sh.add_function(dsp, outputs=['correct_start_stop_with_gears'])
def default_correct_start_stop_with_gears(gear_box_type):
    """
    Defines a flag that imposes the engine on when there is a gear > 0.

    :param gear_box_type:
        Gear box type (manual or automatic or cvt).
    :type gear_box_type: str

    :return:
        A flag to impose engine on when there is a gear > 0.
    :rtype: bool
    """

    return gear_box_type == 'manual'


dsp.add_data(
    'min_time_engine_on_after_start', dfl.values.min_time_engine_on_after_start
)


@sh.add_function(dsp, outputs=['on_engine'], weight=sh.inf(1, 0))
def predict_on_engine(
        times, velocities, gears, accelerations, start_stop_model,
        start_stop_activation_time, min_time_engine_on_after_start,
        correct_start_stop_with_gears, has_start_stop):
    """
    Predicts if the engine is on [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param start_stop_model:
        Start/stop model.
    :type start_stop_model: callable

    :param start_stop_activation_time:
        Start-stop activation time threshold [s].
    :type start_stop_activation_time: float

    :param min_time_engine_on_after_start:
        Minimum time of engine on after a start [s].
    :type min_time_engine_on_after_start: float

    :param correct_start_stop_with_gears:
        A flag to impose engine on when there is a gear > 0.
    :type correct_start_stop_with_gears: bool

    :param has_start_stop:
        Does the vehicle have start/stop system?
    :type has_start_stop: bool

    :return:
        If the engine is on [-].
    :rtype: numpy.array
    """
    if not has_start_stop:
        return np.ones_like(times, bool)
    on_engine = times <= start_stop_activation_time
    if correct_start_stop_with_gears:
        on_engine |= gears > 0

    b0 = velocities > dfl.functions.StartStopModel.stop_velocity
    b0 |= accelerations > dfl.functions.StartStopModel.plateau_acceleration
    b1 = start_stop_model(np.column_stack((velocities, accelerations)))

    ts = min_time_engine_on_after_start
    # noinspection PyTypeChecker
    for i, (t, on, on_b, on_m) in enumerate(zip(times, on_engine, b0, b1)):
        prev = i == 0 or on_engine[i - 1]
        on = on or ((prev or on_b) and on_m)
        if not on and t <= ts:
            on = True
        if not prev and on:
            ts = t + min_time_engine_on_after_start
        on_engine[i] = on
    return on_engine


@sh.add_function(dsp, outputs=['hybrid_modes'], weight=sh.inf(1, 0))
def default_hybrid_modes(on_engine, gear_box_speeds_in, idle_engine_speed):
    """
    Identify the hybrid mode status (0: EV, 1: Parallel, 2: Serial).

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array, bool

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :rtype: numpy.array
    """
    modes = on_engine.astype(int)
    modes[on_engine & (gear_box_speeds_in < -np.diff(idle_engine_speed))] = 2
    return modes


@sh.add_function(dsp, inputs_kwargs=True, outputs=['engine_speeds_out_hot'])
def calculate_engine_speeds_out_hot(
        gear_box_speeds_in, on_engine, idle_engine_speed):
    """
    Calculates the engine speed at hot condition [RPM].

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array, bool

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Engine speed at hot condition [RPM].
    :rtype: numpy.array, float
    """
    return np.where(
        on_engine, np.maximum(gear_box_speeds_in, idle_engine_speed[0]), 0
    )
