# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and constants to define the after_treatment_model selector.
"""
import schedula as sh
from ._core import define_sub_model
from ...physical.after_treat import dsp as _after_treat

#: Model name.
name = 'after_treatment_model'

#: Parameters that constitute the model.
models = ['after_treatment_speed_model', 'after_treatment_warm_up_duration',
          'after_treatment_cooling_duration']

#: Inputs required to run the model.
inputs = [
    'times', 'on_engine', 'engine_speeds_out_hot', 'is_cycle_hot', 'on_idle',
    'is_hybrid'
]

#: Relevant outputs of the model.
outputs = ['engine_speeds_base']

#: Targets to compare the outputs of the model.
targets = outputs

#: Extra inputs for the metrics.
metrics_inputs = ['after_treatment_warm_up_phases']


def metric_after_treatment_speed_model(
        y_true, y_pred, after_treatment_warm_up_phases):
    """
    Metric for the `after_treatment_speed_model`.

    :param y_true:
        Reference engine speed vector [RPM].
    :type y_true: numpy.array

    :param y_pred:
        Predicted engine speed vector [RPM].
    :type y_pred: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :return:
        Error.
    :rtype: float
    """
    b = after_treatment_warm_up_phases
    if b.any():
        from co2mpas.utils import mae
        return float(mae(y_true[b], y_pred[b]))
    else:
        return 0


#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, metric_after_treatment_speed_model)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 160)

#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_after_treat, inputs, outputs, models)._set_cls(
    define_sub_model
)
