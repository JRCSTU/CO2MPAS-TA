# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the idle engine speed.
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='calculate_idle_engine_speed',
    description='Identify idle engine speed median and std.'
)
dsp.add_data('idle_engine_speed_std', dfl.values.idle_engine_speed_std, 20)


@sh.add_function(dsp, outputs=['idle_model_detector'], weight=100)
def define_idle_model_detector(
        velocities, engine_speeds_out, stop_velocity, min_engine_on_speed):
    """
    Defines idle engine speed model detector.

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Idle engine speed model detector.
    :rtype: sklearn.cluster.DBSCAN
    """

    b = (velocities < stop_velocity) & (engine_speeds_out > min_engine_on_speed)
    if not b.any():
        return sh.NONE
    x = engine_speeds_out[b, None]
    from ._idle import _IdleDetector
    model = _IdleDetector(eps=dfl.functions.define_idle_model_detector.EPS)
    model.fit(x)
    return model


@sh.add_function(dsp, outputs=['idle_engine_speed_median'])
def identify_idle_engine_speed_median(idle_model_detector):
    """
    Identifies idle engine speed [RPM].

    :param idle_model_detector:
        Idle engine speed model detector.
    :type idle_model_detector: _IdleDetector

    :return:
        Idle engine speed [RPM].
    :rtype: float
    """
    imd = idle_model_detector
    return np.median(imd.cluster_centers_[imd.labels_])


@sh.add_function(dsp, outputs=['idle_engine_speed_std'])
def identify_idle_engine_speed_std(
        idle_model_detector, engine_speeds_out, idle_engine_speed_median,
        min_engine_on_speed):
    """
    Identifies standard deviation of idle engine speed [RPM].

    :param idle_model_detector:
        Idle engine speed model detector.
    :type idle_model_detector: _IdleDetector

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param idle_engine_speed_median:
        Idle engine speed [RPM].
    :type idle_engine_speed_median: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Standard deviation of idle engine speed [RPM].
    :rtype: float
    """
    b = idle_model_detector.predict([(idle_engine_speed_median,)],
                                    set_outliers=False)
    b = idle_model_detector.predict(engine_speeds_out[:, None]) == b
    b &= (engine_speeds_out > min_engine_on_speed)
    idle_std = dfl.functions.identify_idle_engine_speed_std.MIN_STD
    # noinspection PyUnresolvedReferences
    if not b.any():
        return idle_std

    s = np.sqrt(np.mean((engine_speeds_out[b] - idle_engine_speed_median) ** 2))

    p = dfl.functions.identify_idle_engine_speed_std.MAX_STD_PERC
    return min(max(s, idle_std), idle_engine_speed_median * p)


dsp.add_function(
    function=sh.bypass,
    inputs=['idle_engine_speed_median', 'idle_engine_speed_std'],
    outputs=['idle_engine_speed']
)

dsp.add_function(
    function=sh.bypass,
    inputs=['idle_engine_speed'],
    outputs=['idle_engine_speed_median', 'idle_engine_speed_std']
)
