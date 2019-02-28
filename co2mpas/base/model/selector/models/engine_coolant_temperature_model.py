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
from co2mpas.base.model.physical.engine.thermal import dsp as _thermal

#: Model name.
name = 'engine_coolant_temperature_model'

#: Parameters that constitute the model.
models = [
    'engine_temperature_regression_model', 'max_engine_coolant_temperature'
]

#: Inputs required to run the model.
inputs = [
    'times', 'accelerations', 'final_drive_powers_in', 'engine_speeds_out_hot',
    'initial_engine_temperature'
]

#: Relevant outputs of the model.
outputs = ['engine_coolant_temperatures']

#: Targets to compare the outputs of the model.
targets = outputs

#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, sk_met.mean_absolute_error)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 3)

#: Prediction model.
dsp = sh.Blueprint(_thermal, inputs, outputs, models)
dsp.cls = define_sub_model
