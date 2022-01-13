# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the after treatment.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl
from co2mpas.defaults import dfl

dsp = sh.BlueDispatcher(
    name='After treatment', description='Models the after treatment.'
)


@sh.add_function(dsp, inputs=[
    'times', 'velocities', 'engine_speeds_base', 'gear_box_speeds_in', 'gears',
    'stop_velocity', 'min_engine_on_speed', 'on_engine', 'idle_engine_speed'
], outputs=['on_idle'])
@sh.add_function(dsp, outputs=['on_idle'])
def identify_on_idle(
        times, velocities, engine_speeds_out_hot, gear_box_speeds_in, gears,
        stop_velocity, min_engine_on_speed, on_engine, idle_engine_speed):
    """
    Identifies when the engine is on idle [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        If the engine is on idle [-].
    :rtype: numpy.array
    """
    # noinspection PyProtectedMember
    from .gear_box.mechanical import _shift
    b = engine_speeds_out_hot > min_engine_on_speed
    b &= (gears == 0) | (velocities <= stop_velocity)

    on_idle = np.zeros_like(times, int)
    i = np.where(on_engine)[0]
    ds = np.abs(gear_box_speeds_in[i] - engine_speeds_out_hot[i])
    on_idle[i[ds > idle_engine_speed[1]]] = 1
    on_idle = co2_utl.median_filter(times, on_idle, 4)
    on_idle[b] = 1
    for i, j in co2_utl.pairwise(_shift(on_idle)):
        if not on_idle[i] and times[j - 1] - times[i] <= 2:
            on_idle[i:j] = 1
    return co2_utl.clear_fluctuations(times, on_idle, 4).astype(bool)


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['after_treatment_warm_up_phases']
)
def identify_after_treatment_warm_up_phases(
        times, engine_speeds_out, engine_speeds_out_hot, on_idle, on_engine,
        idle_engine_speed, velocities, engine_starts, stop_velocity,
        is_hybrid=False):
    """
    Identifies when engine speed is affected by the after treatment warm up [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param on_idle:
        If the engine is on idle [-].
    :type on_idle: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        Phases when engine speed is affected by the after treatment warm up [-].
    :rtype: numpy.array
    """
    from .control import identify_engine_starts
    i, phases = np.where(on_idle)[0], np.zeros_like(times, int)
    start = engine_starts.copy()
    if is_hybrid:
        with np.errstate(divide='ignore', invalid='ignore'):
            r = engine_speeds_out[i] / engine_speeds_out_hot[i]
            b = ~co2_utl.get_inliers(r, 2, np.nanmedian, co2_utl.mad)[0]
        phases[i[b]] = 1
    else:
        ds = np.abs(engine_speeds_out[i] - engine_speeds_out_hot[i])
        phases[i[ds > idle_engine_speed[1]]] = 1
        start |= identify_engine_starts(velocities > stop_velocity)
    for i, j in np.searchsorted(times, times[start, None] + [-2, 5 + dfl.EPS]):
        phases[i:j], start[i:j] = 0, True
    phases = co2_utl.median_filter(times, phases, 4)
    phases = co2_utl.clear_fluctuations(times, phases, 4).astype(bool)
    indices = co2_utl.index_phases(phases)
    if is_hybrid:
        indices = indices[(np.diff(times[indices], axis=1) > 10).ravel()][:1]
    else:
        b, f = identify_engine_starts(~on_engine), False
        for i, j in np.searchsorted(times, times[b, None] + [-5, 2 + dfl.EPS]):
            b[i:j] = f
            f = True
        b = (on_idle & ~start) | b | (times > np.min(times[on_engine]) + 200)
        indices = indices[:co2_utl.argmax(b[indices[:, 1]])]
    phases[:], n = False, len(times)
    for i, j in indices:
        while i and on_idle.take(i - 1, mode='clip'):
            i -= 1
        phases[i:j + 1] = True
    return phases


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['after_treatment_warm_up_duration']
)
def identify_after_treatment_warm_up_duration(
        times, after_treatment_warm_up_phases, is_hybrid=False):
    """
    Identify after treatment warm up duration [s].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        After treatment warm up duration [s].
    :rtype: float
    """
    i = co2_utl.index_phases(after_treatment_warm_up_phases)
    if i.shape[0]:
        if not is_hybrid:
            return float(np.diff(times[i.ravel()[[0, -1]]]))
        return float(np.mean(np.diff(times[i], axis=1)))
    return .0


@sh.add_function(dsp, outputs=['after_treatment_warm_up_duration'])
def default_after_treatment_warm_up_duration(is_hybrid):
    """
    Returns the default after treatment warm up duration [s].

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        After treatment warm up duration [s].
    :rtype: float
    """
    if is_hybrid:
        return sh.NONE
    return dfl.functions.default_after_treatment_warm_up_duration.duration


@sh.add_function(dsp, outputs=['after_treatment_cooling_duration'])
def default_after_treatment_cooling_duration(is_hybrid):
    """
    Returns the default after treatment cooling duration [s].

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        After treatment cooling duration [s].
    :rtype: float
    """
    if is_hybrid:
        return sh.NONE
    return dfl.functions.default_after_treatment_cooling_duration.duration


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['after_treatment_cooling_duration']
)
def identify_after_treatment_cooling_duration(
        times, after_treatment_warm_up_phases, is_hybrid=False):
    """
    Identify after treatment cooling duration [s].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        After treatment cooling duration [s].
    :rtype: float
    """
    i = co2_utl.index_phases(after_treatment_warm_up_phases)
    if is_hybrid and i.shape[0] > 2:
        return float(np.mean(np.diff(times[i].ravel())[1::2]))
    return float('inf')


@sh.add_function(dsp, outputs=['after_treatment_speeds_delta'])
def identify_after_treatment_speeds_delta(
        after_treatment_warm_up_phases, engine_speeds_out,
        engine_speeds_out_hot):
    """
    Identifies the Engine speed delta due to the after treatment warm up [RPM].

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :return:
        Engine speed delta due to the after treatment warm up [RPM].
    :rtype: numpy.array
    """
    speeds = np.zeros_like(engine_speeds_out, dtype=float)
    b = after_treatment_warm_up_phases
    speeds[b] = np.maximum(0, engine_speeds_out[b] - engine_speeds_out_hot[b])
    return speeds


@sh.add_function(dsp, outputs=['engine_speeds_base'])
def calculate_engine_speeds_base(
        engine_speeds_out_hot, after_treatment_speeds_delta):
    """
    Calculate base engine speed (i.e., without clutch/TC effect) [RPM].

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param after_treatment_speeds_delta:
        Engine speed delta due to the after treatment warm up [RPM].
    :type after_treatment_speeds_delta: numpy.array

    :return:
        Base engine speed (i.e., without clutch/TC effect) [RPM].
    :rtype: numpy.array
    """
    return engine_speeds_out_hot + after_treatment_speeds_delta


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['after_treatment_speed_model']
)
def calibrate_after_treatment_speed_model(
        times, after_treatment_warm_up_phases, after_treatment_speeds_delta,
        is_hybrid=False):
    """
    Calibrates the engine after treatment speed model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param after_treatment_speeds_delta:
        Engine speed delta due to the after treatment warm up [RPM].
    :type after_treatment_speeds_delta: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        After treatment speed model.
    :rtype: function
    """
    if after_treatment_warm_up_phases.any():
        from sklearn.isotonic import IsotonicRegression
        x, y, model = [], [], IsotonicRegression(increasing=False)
        for i, j in co2_utl.index_phases(after_treatment_warm_up_phases):
            x.extend(times[i:j + 1] - (times[i] if is_hybrid else 0.0))
            y.extend(after_treatment_speeds_delta[i:j + 1])
        # noinspection PyUnresolvedReferences
        return model.fit(x, y).predict


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['after_treatment_power_model']
)
def calibrate_after_treatment_power_model(
        times, after_treatment_warm_up_phases, engine_powers_out,
        is_hybrid=False):
    """
    Calibrates the engine after treatment speed model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        After treatment speed model.
    :rtype: function
    """
    if after_treatment_warm_up_phases.any():
        from sklearn.isotonic import IsotonicRegression
        x, y = [], []
        for i, j in co2_utl.index_phases(after_treatment_warm_up_phases):
            t = times[i:j + 1] - (times[i] if is_hybrid else 0.0)
            x.extend(t)
            y.extend(co2_utl.median_filter(t, engine_powers_out[i:j + 1], 4))
        # noinspection PyUnresolvedReferences
        return IsotonicRegression().fit(x, np.maximum(0, y)).predict


@sh.add_function(dsp, outputs=['after_treatment_warm_up_phases'])
def predict_after_treatment_warm_up_phases(
        after_treatment_warm_up_duration, after_treatment_cooling_duration,
        times, on_engine, is_cycle_hot):
    """
    Calculates the engine speed delta due to the after treatment warm up [RPM].

    :param after_treatment_warm_up_duration:
        After treatment warm up duration [s].
    :type after_treatment_warm_up_duration: float

    :param after_treatment_cooling_duration:
        After treatment cooling duration [s].
    :type after_treatment_cooling_duration: float

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool

    :return:
        Phases when engine speed is affected by the after treatment warm up [-].
    :rtype: numpy.array
    """
    phases = np.zeros_like(times, bool)
    if after_treatment_warm_up_duration and on_engine.any():
        indices = co2_utl.index_phases(on_engine)
        indices = indices[np.append(not is_cycle_hot, np.diff(
            times[indices].ravel()
        )[1::2] > after_treatment_cooling_duration)]
        for i, j in indices:
            t = times[i:j + 1] - times[i]
            phases[i:j + 1] = t < after_treatment_warm_up_duration
    return phases


@sh.add_function(dsp, outputs=['after_treatment_speeds_delta'])
def predict_after_treatment_speeds_delta(
        after_treatment_speed_model, times, after_treatment_warm_up_phases,
        on_idle, is_hybrid):
    """
    Predicts the engine speed delta due to the after treatment warm up [RPM].

    :param after_treatment_speed_model:
        After treatment speed model.
    :type after_treatment_speed_model: function

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param on_idle:
        If the engine is on idle [-].
    :type on_idle: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        Engine speed delta due to the after treatment warm up [RPM].
    :rtype: numpy.array
    """
    ds = np.zeros_like(times, float)
    if after_treatment_speed_model:
        for i, j in co2_utl.index_phases(after_treatment_warm_up_phases):
            ds[i:j + 1] = after_treatment_speed_model(times[i:j + 1] - times[i])
        if not is_hybrid:
            ds[~on_idle] = 0
    return np.nan_to_num(ds)
