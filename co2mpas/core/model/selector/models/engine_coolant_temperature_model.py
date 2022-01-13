# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and constants to define the engine_coolant_temperature_model selector.
"""
import schedula as sh
import co2mpas.utils as co2_utl
from ._core import define_sub_model
from ...physical.engine.thermal import dsp as _thermal

#: Model name.
name = 'engine_coolant_temperature_model'

#: Parameters that constitute the model.
models = [
    'engine_temperature_regression_model', 'max_engine_coolant_temperature',
    'engine_thermostat_temperature', 'temperature_shift'
]

#: Inputs required to run the model.
inputs = [
    'times', 'on_engine', 'velocities', 'engine_speeds_out',
    'accelerations', 'initial_engine_temperature',
    'after_treatment_warm_up_phases'
]

#: Relevant outputs of the model.
outputs = ['engine_coolant_temperatures']

#: Targets to compare the outputs of the model.
targets = outputs

#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, co2_utl.mae)

#: Upper score limits to raise the warnings.
up_limit = sh.map_list(targets, 4)

#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_thermal, inputs, outputs, models)._set_cls(define_sub_model)
