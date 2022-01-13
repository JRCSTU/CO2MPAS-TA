# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and constants to define the clutch_torque_converter_model selector.
"""
import copy
import schedula as sh
from ...physical.clutch_tc import dsp as _clutch_tc
from ...physical.engine import calculate_engine_speeds_out

#: Model name.
name = 'clutch_torque_converter_model'

#: Parameters that constitute the model.
models = ['clutch_window', 'clutch_speed_model', 'torque_converter_speed_model']

#: Inputs required to run the model.
inputs = [
    'gear_box_speeds_in', 'on_engine', 'idle_engine_speed', 'gear_box_type',
    'gears', 'accelerations', 'times', 'gear_shifts', 'engine_speeds_out_hot',
    'velocities', 'has_torque_converter', 'gear_box_torques_in'
]

#: Relevant outputs of the model.
outputs = ['engine_speeds_out']

#: Targets to compare the outputs of the model.
targets = outputs

#: Extra inputs for the metrics.
metrics_inputs = ['on_engine']


def metric_clutch_torque_converter_model(y_true, y_pred, on_engine):
    """
    Metric for the `clutch_torque_converter_model`.

    :param y_true:
        Reference engine speed vector [RPM].
    :type y_true: numpy.array

    :param y_pred:
        Predicted engine speed vector [RPM].
    :type y_pred: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :return:
        Error.
    :rtype: float
    """
    from co2mpas.utils import mae
    return float(mae(y_true[on_engine], y_pred[on_engine]))


#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, metric_clutch_torque_converter_model)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 100)

#: Prediction model.
dsp = copy.deepcopy(_clutch_tc).add_function(
    function=calculate_engine_speeds_out,
    inputs=['on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
            'clutch_tc_speeds_delta'],
    outputs=['engine_speeds_out']
)
