# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the electrics of the vehicle.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics

.. autosummary::
    :nosignatures:
    :toctree: electrics/

    motors
    batteries
"""

import schedula as sh
from .motors import dsp as _motors
from .batteries import dsp as _batteries

dsp = sh.BlueDispatcher(
    name='Electrics', description='Models the vehicle electrics.'
)

dsp.add_dispatcher(
    dsp_id='motors',
    dsp=_motors,
    inputs=(
        'engine_speeds_out', 'motor_p0_efficiency', 'wheel_speeds',
        'motor_p0_electric_power_loss_function', 'motor_p0_electric_powers',
        'motor_p0_loss_param_a', 'motor_p0_loss_param_b', 'motor_p0_powers',
        'motor_p0_speed_ratio', 'motor_p0_speeds', 'motor_p0_torques',
        'motor_p1_efficiency', 'motor_p1_electric_power_loss_function',
        'motor_p1_electric_powers', 'motor_p1_loss_param_a',
        'motor_p1_loss_param_b', 'motor_p1_powers', 'motor_p1_speed_ratio',
        'motor_p1_speeds', 'motor_p1_torques', 'gear_box_speeds_in',
        'motor_p2_efficiency', 'motor_p2_electric_power_loss_function',
        'motor_p2_electric_powers', 'motor_p2_loss_param_a',
        'motor_p2_loss_param_b', 'motor_p2_powers', 'motor_p2_speed_ratio',
        'motor_p2_speeds', 'motor_p2_torques', 'final_drive_speeds_in',
        'motor_p3_efficiency', 'motor_p3_electric_power_loss_function',
        'motor_p3_electric_powers', 'motor_p3_loss_param_a',
        'motor_p3_loss_param_b', 'motor_p3_powers', 'motor_p3_speed_ratio',
        'motor_p3_speeds', 'motor_p3_torques', 'motor_p4_efficiency',
        'motor_p4_electric_power_loss_function', 'motor_p4_electric_powers',
        'motor_p4_loss_param_a', 'motor_p4_loss_param_b', 'motor_p4_powers',
        'motor_p4_speed_ratio', 'motor_p4_speeds', 'motor_p4_torques',
        'alternator_currents', 'alternator_nominal_voltage', 'stop_velocity',
        'alternator_electric_powers', 'alternator_efficiency',
        'alternator_off_threshold', 'velocities', 'on_engine', 'times',
        'engine_starts', 'alternator_current_threshold',
        'alternator_start_window_width', 'alternator_statuses',
        'clutch_tc_powers', 'alternator_status_model',
        'alternator_initialization_time', 'service_battery_state_of_charges',
        'accelerations', 'service_battery_state_of_charge_balance',
        'service_battery_state_of_charge_balance_window',
        'alternator_charging_currents', 'alternator_current_model',
        'engine_moment_inertia', 'starter_efficiency',
        'delta_time_engine_starter'
    ),
    outputs=(
        'motor_p0_electric_power_loss_function', 'motor_p0_electric_powers',
        'motor_p0_powers', 'motor_p0_speed_ratio', 'motor_p0_speeds',
        'motor_p0_torques', 'motor_p0_efficiency_ratios',
        'motor_p1_electric_power_loss_function', 'motor_p1_electric_powers',
        'motor_p1_powers', 'motor_p1_speed_ratio', 'motor_p1_speeds',
        'motor_p1_torques', 'motor_p1_efficiency_ratios',
        'motor_p2_electric_power_loss_function', 'motor_p2_electric_powers',
        'motor_p2_powers', 'motor_p2_speed_ratio', 'motor_p2_speeds',
        'motor_p2_torques', 'motor_p2_efficiency_ratios',
        'motor_p3_electric_power_loss_function', 'motor_p3_electric_powers',
        'motor_p3_powers', 'motor_p3_speed_ratio', 'motor_p3_speeds',
        'motor_p3_torques', 'motor_p3_efficiency_ratios',
        'motor_p4_electric_power_loss_function', 'motor_p4_electric_powers',
        'motor_p4_powers', 'motor_p4_speed_ratio', 'motor_p4_speeds',
        'motor_p4_torques', 'motor_p4_efficiency_ratios',
        'alternator_current_threshold', 'alternator_current_model',
        'alternator_initialization_time', 'alternator_status_model',
        'service_battery_state_of_charge_balance', 'alternator_currents',
        'service_battery_state_of_charge_balance_window', 'alternator_powers',
        'alternator_electric_powers', 'alternator_statuses',
        'starter_electric_powers', 'starter_powers', 'engine_speeds_out',
        'wheel_speeds', 'final_drive_speeds_in', 'gear_box_speeds_in'
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='batteries',
    dsp=_batteries,
    inputs=(
        'alternator_electric_powers', 'service_battery_state_of_charges',
        'dcdc_converter_electric_powers', 'engine_powers_out', 'on_engine',
        'initial_service_battery_state_of_charge', 'drive_battery_load',
        'service_battery_capacity', 'service_battery_currents', 'cycle_type',
        'service_battery_electric_powers', 'service_battery_load',
        'service_battery_loads', 'service_battery_nominal_voltage', 'times',
        'dcdc_converter_efficiency', 'drive_battery_capacity',
        'drive_battery_currents', 'drive_battery_electric_powers',
        'drive_battery_loads', 'drive_battery_n_parallel_cells',
        'drive_battery_n_series_cells', 'drive_battery_ocv', 'drive_battery_r0',
        'drive_battery_state_of_charges', 'drive_battery_voltages',
        'electrical_hybridization_degree', 'starter_electric_powers',
        'initial_drive_battery_state_of_charge', 'motor_p0_electric_powers',
        'motor_p1_electric_powers', 'motor_p2_electric_powers',
        'motor_p3_electric_powers', 'motor_p4_electric_powers',
        'dcdc_converter_currents'
    ),
    outputs=(
        'initial_service_battery_state_of_charge', 'service_battery_capacity',
        'service_battery_currents', 'service_battery_electric_powers',
        'service_battery_load', 'service_battery_loads', 'drive_battery_r0',
        'service_battery_state_of_charges', 'drive_battery_voltages',
        'service_battery_delta_state_of_charge', 'drive_battery_capacity',
        'drive_battery_currents', 'drive_battery_electric_powers',
        'drive_battery_load', 'initial_drive_battery_state_of_charge',
        'drive_battery_n_parallel_cells', 'drive_battery_n_series_cells',
        'drive_battery_state_of_charges', 'drive_battery_delta_state_of_charge',
        'minimum_drive_battery_electric_power', 'drive_battery_ocv',
        'drive_battery_loads', 'dcdc_converter_electric_powers_demand'
    ),
    include_defaults=True
)
