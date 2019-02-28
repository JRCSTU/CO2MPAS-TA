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
from ._core import define_sub_model
from co2mpas.base.model.physical import dsp as _physical
from co2mpas.base.model.physical.clutch_tc.clutch import calculate_clutch_phases

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
    'accelerations', 'final_drive_powers_in', 'engine_thermostat_temperature',
    'tyre_code'
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
    b = ~calculate_clutch_phases(times, 1, 1, gear_shifts, 0, (-4.0, 4.0))
    b &= (velocities > stop_velocity) & (times > 100) & on_engine
    return sk_met.mean_absolute_error(y_true[b], y_pred[b])


#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, metric_engine_speed_model)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 40)


def select_models(data):
    mdl = sh.selector(models, data, allow_miss=True)
    if 'tyre_dynamic_rolling_coefficient' in mdl:
        mdl.pop('r_dynamic', None)
    return mdl


#: Prediction model.
dsp = sh.Blueprint(_physical, inputs, outputs, models)
dsp.cls = define_sub_model
