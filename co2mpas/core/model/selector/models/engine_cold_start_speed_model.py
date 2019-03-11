# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions to compare/select the CO2MPAS calibrated models.

Docstrings should provide sufficient understanding for any individual function.

Modules:

.. currentmodule:: co2mpas.model.selector

.. autosummary::
    :nosignatures:
    :toctree: selector/

    co2_params
"""
import copy
import schedula as sh
from ._core import define_sub_model
from ...physical.engine.cold_start import dsp as _cold_start
from ...physical.engine import calculate_engine_speeds_out

#: Model name.
name = 'engine_cold_start_speed_model'

#: Parameters that constitute the model.
models = ['cold_start_speed_model']

#: Inputs required to run the model.
inputs = [
    'engine_speeds_out_hot', 'engine_coolant_temperatures', 'on_engine',
    'idle_engine_speed'
]

#: Relevant outputs of the model.
outputs = ['engine_speeds_out']

#: Targets to compare the outputs of the model.
targets = outputs

#: Extra inputs for the metrics.
metrics_inputs = ['cold_start_speeds_phases', 'engine_coolant_temperatures']


def metric_engine_cold_start_speed_model(
        y_true, y_pred, cold_start_speeds_phases, engine_coolant_temperatures):
    """
    Metric for the `engine_cold_start_speed_model`.

    :param y_true:
        Reference engine speed vector [RPM].
    :type y_true: numpy.array

    :param y_pred:
        Predicted engine speed vector [RPM].
    :type y_pred: numpy.array

    :param cold_start_speeds_phases:
        Phases when engine speed is affected by the cold start [-].
    :type cold_start_speeds_phases: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Error.
    :rtype: float
    """
    b = cold_start_speeds_phases
    if b.any():
        from co2mpas.utils import mae
        t = engine_coolant_temperatures
        w = (t.max() + 1) - t[b]
        return float(mae(y_true[b], y_pred[b], w))
    else:
        return 0


#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, metric_engine_cold_start_speed_model)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 160)

#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(copy.deepcopy(_cold_start).add_function(
    function=calculate_engine_speeds_out,
    inputs=['on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
            'cold_start_speeds_delta'],
    outputs=['engine_speeds_out']
), inputs, outputs, models)._set_cls(define_sub_model)
