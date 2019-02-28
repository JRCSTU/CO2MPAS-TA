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
import schedula as sh
import sklearn.metrics as sk_met
from co2mpas.base.model.physical.clutch_tc import dsp as _clutch_tc
from co2mpas.base.model.physical.engine import calculate_engine_speeds_out

#: Model name.
name = 'clutch_torque_converter_model'

#: Parameters that constitute the model.
models = ['clutch_window', 'clutch_model', 'torque_converter_model']

#: Inputs required to run the model.
inputs = [
    'gear_box_speeds_in', 'on_engine', 'idle_engine_speed', 'gear_box_type',
    'gears', 'accelerations', 'times', 'gear_shifts', 'engine_speeds_out_hot',
    'velocities', 'lock_up_tc_limits', 'has_torque_converter'
]

#: Relevant outputs of the model.
outputs = ['engine_speeds_out']

#: Targets to compare the outputs of the model.
targets = outputs

#: Extra inputs for the metrics.
metrics_inputs = ['on_engine']


def metric_clutch_torque_converter_model(y_true, y_pred, on_engine):
    return sk_met.mean_absolute_error(y_true[on_engine], y_pred[on_engine])


#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, metric_clutch_torque_converter_model)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 100)

#: Prediction model.
dsp = sh.SubDispatch(sh.BlueDispatcher().extend(_clutch_tc).add_function(
    function=calculate_engine_speeds_out,
    inputs=['on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
            'clutch_tc_speeds_delta'],
    outputs=['engine_speeds_out']
))
