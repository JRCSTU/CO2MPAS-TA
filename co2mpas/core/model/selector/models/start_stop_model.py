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
from ._core import define_sub_model, _accuracy_score
from ...physical import dsp as _physical

#: Model name.
name = 'start_stop_model'

#: Parameters that constitute the model.
models = [
    'start_stop_model', 'start_stop_hybrid_params', 'catalyst_warm_up_duration',
    'motor_p3_efficiency', 'motor_p4_efficiency', 'final_drive_mean_efficiency',
    'belt_mean_efficiency', 'clutch_tc_mean_efficiency', 'motor_p2_efficiency',
    'motor_p0_efficiency', 'motor_p1_efficiency', 'gear_box_mean_efficiency',
    'ecms_s', 'motor_p0_maximum_power', 'motor_p0_rated_speed',
    'motor_p1_maximum_power', 'motor_p1_rated_speed', 'motor_p2_maximum_power',
    'motor_p2_rated_speed', 'motor_p3_maximum_power', 'motor_p3_rated_speed',
    'motor_p4_maximum_power', 'motor_p4_rated_speed'
]

#: Inputs required to run the model.
inputs = [
    'times', 'velocities', 'accelerations', 'gears', 'motive_powers',
    'correct_start_stop_with_gears', 'start_stop_activation_time',
    'min_time_engine_on_after_start', 'has_start_stop', 'is_hybrid',
    'drive_battery_model', 'fuel_map', 'full_load_curve', 'is_cycle_hot',
    'motor_p1_maximum_power_function', 'motor_p0_maximum_power_function',
    'motor_p1_speed_ratio', 'motor_p0_speed_ratio', 'gear_box_speeds_in',
    'idle_engine_speed', 'motor_p4_maximum_powers', 'motor_p3_maximum_powers',
    'motor_p2_maximum_powers', 'starter_model', 'engine_moment_inertia',
    'auxiliaries_torque_loss', 'auxiliaries_power_loss',
    'dcdc_converter_efficiency'
]

#: Relevant outputs of the model.
outputs = ['on_engine', 'engine_starts']

#: Targets to compare the outputs of the model.
targets = outputs

#: Weights coefficients to compute the model score.
weights = sh.map_list(targets, -1, -1)

#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, *([_accuracy_score] * 2))

#: Bottom score limits to raise the warnings.
dn_limit = sh.map_list(targets, 0.7, 0.7)

#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_physical, inputs, outputs, models)._set_cls(
    define_sub_model
)
