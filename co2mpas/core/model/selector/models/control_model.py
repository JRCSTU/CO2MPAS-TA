# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
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
name = 'control_model'

#: Parameters that constitute the model.
models = [
    'motor_p3_front_efficiency', 'motor_p3_rear_efficiency', 'start_stop_model',
    'motor_p1_maximum_power', 'motor_p1_efficiency', 'gear_box_mean_efficiency',
    'belt_mean_efficiency', 'clutch_tc_mean_efficiency', 'motor_p1_rated_speed',
    'motor_p3_front_rated_speed', 'motor_p2_rated_speed', 'motor_p2_efficiency',
    'start_stop_hybrid_params', 'motor_p0_maximum_power', 'motor_p0_efficiency',
    'motor_p2_planetary_rated_speed', 'motor_p2_planetary_efficiency', 'ecms_s',
    'motor_p3_rear_rated_speed', 'motor_p4_rear_rated_speed', 'planetary_ratio',
    'motor_p2_planetary_maximum_power', 'motor_p4_rear_efficiency', 'is_serial',
    'motor_p3_front_maximum_power', 'motor_p4_front_maximum_power',
    'motor_p3_rear_maximum_power', 'motor_p4_rear_maximum_power',
    'final_drive_mean_efficiency', 'motor_p4_front_rated_speed',
    'planetary_mean_efficiency', 'motor_p4_front_efficiency',
    'motor_p2_maximum_power', 'motor_p0_rated_speed',
]

#: Inputs required to run the model.
inputs = [
    'correct_start_stop_with_gears', 'motor_p0_maximum_power_function', 'gears',
    'motor_p1_maximum_power_function', 'motor_p3_front_maximum_powers', 'times',
    'min_time_engine_on_after_start', 'start_stop_activation_time', 'is_hybrid',
    'motor_p3_rear_maximum_powers', 'motor_p2_maximum_powers', 'has_start_stop',
    'auxiliaries_torque_loss', 'auxiliaries_power_loss', 'motor_p1_speed_ratio',
    'engine_moment_inertia', 'drive_battery_model', 'accelerations', 'fuel_map',
    'motor_p0_speed_ratio', 'gear_box_speeds_in', 'starter_model', 'velocities',
    'motor_p4_rear_maximum_powers', 'is_cycle_hot', 'dcdc_converter_efficiency',
    'after_treatment_cooling_duration', 'idle_engine_speed', 'full_load_curve',
    'after_treatment_warm_up_duration', 'motor_p2_planetary_speed_ratio',
    'motor_p2_planetary_maximum_power_function', 'final_drive_speeds_in',
    'motor_p4_front_maximum_powers', 'motive_powers',
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
