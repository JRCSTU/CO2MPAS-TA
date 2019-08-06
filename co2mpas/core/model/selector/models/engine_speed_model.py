# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and constants to define the engine_speed_model selector.
"""
import schedula as sh
from ._core import define_sub_model
from ...physical import dsp as _physical
from ...physical.clutch_tc import calculate_clutch_phases

#: Model name.
name = 'engine_speed_model'

#: Parameters that constitute the model.
models = [
    'final_drive_ratios', 'gear_box_ratios', 'idle_engine_speed_median',
    'idle_engine_speed_std', 'CVT', 'max_speed_velocity_ratio',
    'tyre_dynamic_rolling_coefficient'
]

#: Inputs required to run the model.
inputs = [
    'velocities', 'gears', 'times', 'on_engine', 'gear_box_type',
    'accelerations', 'tyre_code'
]

#: Relevant outputs of the model.
outputs = ['engine_speeds_out_hot']

#: Targets to compare the outputs of the model.
targets = ['engine_speeds_out']

#: Extra inputs for the metrics.
metrics_inputs = [
    'times', 'velocities', 'gear_shifts', 'on_engine', 'stop_velocity'
]


def metric_engine_speed_model(
        y_true, y_pred, times, velocities, gear_shifts, on_engine,
        stop_velocity):
    """
    Metric for the `engine_cold_start_speed_model`.

    :param y_true:
        Reference engine speed vector [RPM].
    :type y_true: numpy.array

    :param y_pred:
        Predicted engine speed vector [RPM].
    :type y_pred: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param gear_shifts:
        When there is a gear shifting [-].
    :type gear_shifts: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Error.
    :rtype: float
    """
    from co2mpas.utils import mae
    b = ~calculate_clutch_phases(times, 1, 1, gear_shifts, 0, (-4.0, 4.0))
    b &= (velocities > stop_velocity) & (times > 100) & on_engine
    return float(mae(y_true[b], y_pred[b]))


#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, metric_engine_speed_model)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 40)


def select_models(keys, data):
    """
    Select models from data.

    :param keys:
        Model keys.
    :type keys: list

    :param data:
        Cycle data.
    :type data: dict

    :return:
        Models.
    :rtype: dict
    """
    mdl = sh.selector(keys, data, allow_miss=True)
    if 'tyre_dynamic_rolling_coefficient' in mdl:
        mdl.pop('r_dynamic', None)
    return mdl


#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_physical, inputs, outputs, models)._set_cls(
    define_sub_model
)
