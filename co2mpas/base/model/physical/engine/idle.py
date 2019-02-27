# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions that model the basic mechanics of the engine.
"""
import numpy as np
import schedula as sh
from sklearn.cluster import DBSCAN
from ..defaults import dfl

dsp = sh.BlueDispatcher(
    name='calculate_idle_engine_speed',
    description='Identify idle engine speed median and std.'
)
dsp.add_data('idle_engine_speed_std', dfl.values.idle_engine_speed_std, 20)


# noinspection PyPep8Naming,PyMissingOrEmptyDocstring
class _IdleDetector(DBSCAN):
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 algorithm='auto', leaf_size=30, p=None):
        super(_IdleDetector, self).__init__(
            eps=eps, min_samples=min_samples, metric=metric,
            algorithm=algorithm, leaf_size=leaf_size, p=p
        )
        self.cluster_centers_ = None
        self.min, self.max = None, None

    def fit(self, X, y=None, sample_weight=None):
        super(_IdleDetector, self).fit(X, y=y, sample_weight=sample_weight)

        c, lb = self.components_, self.labels_[self.core_sample_indices_]
        self.cluster_centers_ = np.array(
            [np.mean(c[lb == i]) for i in range(lb.max() + 1)]
        )
        self.min, self.max = c.min(), c.max()
        return self

    def predict(self, X, set_outliers=True):
        import sklearn.metrics as sk_met
        y = sk_met.pairwise_distances_argmin(X, self.cluster_centers_[:, None])
        if set_outliers:
            y[((X > self.max) | (X < self.min))[:, 0]] = -1
        return y


@sh.add_function(dsp, outputs=['idle_model_detector'], weigth=100)
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
