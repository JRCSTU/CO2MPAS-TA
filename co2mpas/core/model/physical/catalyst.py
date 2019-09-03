# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the catalyst.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(name='Catalyst', description='Models the catalyst.')


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
    for i, j in sh.pairwise(_shift(on_idle)):
        if not on_idle[i] and times[j - 1] - times[i] <= 2:
            on_idle[i:j] = True
    return co2_utl.clear_fluctuations(times, on_idle, 4).astype(bool)


@sh.add_function(dsp, outputs=['catalyst_warm_up_phases'])
def identify_catalyst_warm_up_phases(
        times, engine_speeds_out, engine_speeds_out_hot, on_idle):
    """
    Identifies Phases when engine speed is affected by the catalyst warm up [-].

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

    :return:
        Phases when engine speed is affected by the catalyst warm up [-].
    :rtype: numpy.array
    """
    # noinspection PyProtectedMember
    i = np.where(on_idle)[0]
    phases = np.zeros_like(times, int)
    with np.errstate(divide='ignore', invalid='ignore'):
        # noinspection PyUnresolvedReferences
        phases[i[~co2_utl.get_inliers(
            engine_speeds_out[i] / engine_speeds_out_hot[i], 2, np.nanmedian,
            co2_utl.mad
        )[0]]] = 1
    phases = co2_utl.median_filter(times, phases, 5)
    phases = co2_utl.clear_fluctuations(times, phases, 5).astype(bool)
    i = co2_utl.index_phases(phases)
    b = np.diff(times[i], axis=1) > 5
    i = i[b.ravel()]
    phases[:] = False
    if i.shape[0]:
        i, j = i[0]
        while i and on_idle.take(i - 1, mode='clip'):
            i -= 1
        phases[i:j + 1] = True
    return phases


@sh.add_function(dsp, outputs=['catalyst_speeds_delta'])
def identify_catalyst_speeds_delta(
        catalyst_warm_up_phases, engine_speeds_out, engine_speeds_out_hot):
    """
    Identifies the Engine speed delta due to the catalyst warm up [RPM].

    :param catalyst_warm_up_phases:
        Phases when engine speed is affected by the catalyst warm up [-].
    :type catalyst_warm_up_phases: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :return:
        Engine speed delta due to the catalyst warm up [RPM].
    :rtype: numpy.array
    """
    speeds = np.zeros_like(engine_speeds_out, dtype=float)
    b = catalyst_warm_up_phases
    speeds[b] = np.maximum(0, engine_speeds_out[b] - engine_speeds_out_hot[b])
    return speeds


@sh.add_function(dsp, outputs=['engine_speeds_base'])
def calculate_engine_speeds_base(
        engine_speeds_out_hot, catalyst_speeds_delta):
    """
    Calculate base engine speed (i.e., without clutch/TC effect) [RPM].

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param catalyst_speeds_delta:
        Engine speed delta due to the catalyst warm up [RPM].
    :type catalyst_speeds_delta: numpy.array

    :return:
        Base engine speed (i.e., without clutch/TC effect) [RPM].
    :rtype: numpy.array
    """
    return engine_speeds_out_hot + catalyst_speeds_delta


# noinspection PyMissingOrEmptyDocstring, PyProtectedMember
class CatalystSpeedModel:
    def __init__(self, model=None, warming_time=0, cooling_time=float('inf')):
        self.model = model
        self.cooling_time = cooling_time
        self.warming_time = warming_time

    def fit(self, times, catalyst_speeds_delta, catalyst_warm_up_phases):
        if not catalyst_warm_up_phases.any():
            return self
        from sklearn.isotonic import IsotonicRegression
        indices = co2_utl.index_phases(catalyst_warm_up_phases)
        x, y, model = [], [], IsotonicRegression(increasing=False)
        indices = indices[np.diff(times[indices], axis=1)[:, 0] > 5]
        for i, j in indices:
            x.extend(times[i:j + 1] - times[i])
            y.extend(catalyst_speeds_delta[i:j + 1])
        self.warming_time = np.max(x)
        dt = np.diff(times[indices].ravel())[1::2]
        self.cooling_time = np.mean(dt) if dt.size else float('inf')
        # noinspection PyUnresolvedReferences
        self.model = model.fit(x, y).predict
        return self

    def __call__(self, times, on_engine, is_warm=False):
        speeds_delta, w_time = np.zeros_like(times, float), self.warming_time
        if w_time and self.model:
            indices = co2_utl.index_phases(on_engine)
            indices = indices[np.append(not is_warm, np.diff(
                times[indices].ravel()
            )[1::2] > self.cooling_time)]
            for i, j in indices:
                t = times[i:j + 1] - times[i]
                speeds_delta[i:j + 1] = np.where(t < w_time, self.model(t), 0)
        return speeds_delta


@sh.add_function(dsp, outputs=['catalyst_speed_model'])
def calibrate_catalyst_speed_model(
        times, catalyst_warm_up_phases, catalyst_speeds_delta):
    """
    Calibrates the engine catalyst speed model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param catalyst_warm_up_phases:
        Phases when engine speed is affected by the catalyst warm up [-].
    :type catalyst_warm_up_phases: numpy.array

    :param catalyst_speeds_delta:
        Engine speed delta due to the catalyst warm up [RPM].
    :type catalyst_speeds_delta: numpy.array

    :return:
        Catalyst speed model.
    :rtype: CatalystSpeedModel
    """
    return CatalystSpeedModel().fit(
        times, catalyst_speeds_delta, catalyst_warm_up_phases
    )


@sh.add_function(dsp, outputs=['catalyst_speeds_delta'])
def calculate_catalyst_speeds_delta(catalyst_speed_model, times, on_engine):
    """
    Calculates the engine speed delta due to the catalyst warm up [RPM].

    :param catalyst_speed_model:
        Catalyst speed model.
    :type catalyst_speed_model: CatalystSpeedModel

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :return:
        Engine speed delta due to the catalyst warm up [RPM].
    :rtype: numpy.array
    """
    return catalyst_speed_model(times, on_engine)
