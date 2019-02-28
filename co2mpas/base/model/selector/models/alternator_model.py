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
from co2mpas.base.model.physical.electrics import dsp as _electrics

#: Model name.
name = 'alternator_model'

#: Parameters that constitute the model.
models = [
    'alternator_status_model', 'electric_load', 'max_battery_charging_current',
    'alternator_current_model', 'start_demand', 'alternator_nominal_power',
    'alternator_initialization_time', 'alternator_nominal_voltage',
    'alternator_efficiency'
]

#: Inputs required to run the model.
inputs = [
    'battery_capacity', 'alternator_nominal_voltage', 'initial_state_of_charge',
    'times', 'gear_box_powers_in', 'on_engine', 'engine_starts', 'accelerations'
]

#: Relevant outputs of the model.
outputs = [
    'alternator_currents', 'battery_currents', 'state_of_charges',
    'alternator_statuses'
]
#: Targets to compare the outputs of the model.
targets = outputs

#: Weights coefficients to compute the model score.
weights = sh.map_list(targets, 1, 1, 0, 0)

#: Metrics to compare outputs with targets.
metrics = sh.map_list(
    targets, *([sk_met.mean_absolute_error] * 3 + [sk_met.accuracy_score])
)

#: Upper score limits to raise the warnings.
up_limit = dict.fromkeys(('alternator_currents', 'battery_currents'), 60)

#: Prediction model.
dsp = sh.Blueprint(_electrics, inputs, outputs, models)
dsp.cls = define_sub_model
