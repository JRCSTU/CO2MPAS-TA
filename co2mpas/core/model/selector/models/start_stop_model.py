# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and constants to define the start_stop_model selector.
"""
import schedula as sh
import sklearn.metrics as sk_met
from ._core import define_sub_model
from ...physical.engine.start_stop import dsp as _start_stop

#: Model name.
name = 'start_stop_model'

#: Parameters that constitute the model.
models = ['start_stop_model', 'use_basic_start_stop']

#: Inputs required to run the model.
inputs = [
    'times', 'velocities', 'accelerations', 'engine_coolant_temperatures',
    'state_of_charges', 'gears', 'correct_start_stop_with_gears',
    'start_stop_activation_time', 'min_time_engine_on_after_start',
    'has_start_stop'
]

#: Relevant outputs of the model.
outputs = ['on_engine', 'engine_starts']

#: Targets to compare the outputs of the model.
targets = outputs

#: Weights coefficients to compute the model score.
weights = sh.map_list(targets, -1, -1)

#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, *([sk_met.accuracy_score] * 2))

#: Bottom score limits to raise the warnings.
dn_limit = sh.map_list(targets, 0.7, 0.7)

#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_start_stop, inputs, outputs, models)._set_cls(
    define_sub_model
)
