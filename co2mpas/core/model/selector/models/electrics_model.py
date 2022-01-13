# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and constants to define the electrics_model selector.
"""
import schedula as sh
import co2mpas.utils as co2_utl
from ._core import define_sub_model
from ...physical.electrics import dsp as _electrics

#: Model name.
name = 'electrics_model'

#: Parameters that constitute the model.
models = [
    'alternator_current_model', 'dcdc_current_model', 'has_energy_recuperation',
    'service_battery_status_model', 'engine_moment_inertia', 'drive_battery_r0',
    'delta_time_engine_starter', 'drive_battery_capacity', 'drive_battery_load',
    'alternator_efficiency', 'service_battery_capacity', 'service_battery_load',
    'drive_battery_n_parallel_cells', 'drive_battery_ocv', 'starter_efficiency',
    'service_battery_initialization_time', 'service_battery_nominal_voltage',
    'drive_battery_nominal_voltage', 'drive_battery_n_series_cells',
    'dcdc_converter_efficiency', 'starter_nominal_voltage',
    'initial_drive_battery_state_of_charge'
]

#: Inputs required to run the model.
inputs = [
    'drive_battery_electric_powers', 'times', 'motive_powers', 'accelerations',
    'on_engine', 'starter_currents', 'initial_service_battery_state_of_charge'
]

#: Relevant outputs of the model.
outputs = [
    'alternator_currents', 'service_battery_currents', 'drive_battery_currents',
    'dcdc_converter_currents', 'service_battery_state_of_charges',
    'drive_battery_state_of_charges'
]
#: Targets to compare the outputs of the model.
targets = outputs

#: Weights coefficients to compute the model score.
weights = sh.map_list(targets, 1, 1, 1, 1, 0, 0)

#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, *([co2_utl.mae] * 6))

#: Upper score limits to raise the warnings.
up_limit = dict.fromkeys(
    ('alternator_currents', 'service_battery_currents',
     'drive_battery_currents', 'dcdc_converter_currents'), 60
)

#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_electrics, inputs, outputs, models)._set_cls(
    define_sub_model
)
