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
        'motor_p0_electric_powers', 'motor_p0_loss_param_a', 'motor_p0_torques',
        'motor_p0_efficiency', 'motor_p0_maximum_power', 'motor_p0_speed_ratio',
        'motor_p0_loss_param_b', 'motor_p0_rated_speed', 'motor_p0_powers',
        'motor_p0_electric_power_loss_function', 'motor_p0_maximum_torque',
        'motor_p1_efficiency', 'motor_p1_maximum_power', 'motor_p1_speed_ratio',
        'motor_p1_electric_powers', 'motor_p1_loss_param_a', 'motor_p1_torques',
        'motor_p1_loss_param_b', 'motor_p1_rated_speed', 'motor_p1_powers',
        'motor_p1_electric_power_loss_function', 'motor_p1_maximum_torque',
        'motor_p2_efficiency', 'motor_p2_maximum_power', 'motor_p2_speed_ratio',
        'motor_p2_electric_powers', 'motor_p2_loss_param_a', 'motor_p2_torques',
        'motor_p2_loss_param_b', 'motor_p2_rated_speed', 'motor_p2_powers',
        'motor_p2_electric_power_loss_function', 'motor_p2_maximum_torque',
        'motor_p3_efficiency', 'motor_p3_maximum_power', 'motor_p3_speed_ratio',
        'motor_p3_electric_powers', 'motor_p3_loss_param_a', 'motor_p3_torques',
        'motor_p3_loss_param_b', 'motor_p3_rated_speed', 'motor_p3_powers',
        'motor_p3_electric_power_loss_function', 'motor_p3_maximum_torque',
        'motor_p4_efficiency', 'motor_p4_maximum_power', 'motor_p4_speed_ratio',
        'motor_p4_electric_powers', 'motor_p4_loss_param_a', 'motor_p4_torques',
        'motor_p4_loss_param_b', 'motor_p4_rated_speed', 'motor_p4_powers',
        'motor_p4_electric_power_loss_function', 'motor_p4_maximum_torque',
        'motor_p0_speeds', 'motor_p1_speeds', 'alternator_charging_currents',
        'motor_p2_speeds', 'alternator_current_model', 'engine_moment_inertia',
        'motor_p3_speeds', 'motor_p4_speeds', 'alternator_electric_powers',
        'service_battery_state_of_charges', 'alternator_currents', 'on_engine',
        'final_drive_speeds_in', 'engine_speeds_out', 'alternator_efficiency',
        'service_battery_charging_statuses', 'motive_powers', 'wheel_speeds',
        'engine_starts', 'accelerations', 'alternator_nominal_voltage', 'times',
        'starter_efficiency', 'delta_time_engine_starter', 'gear_box_speeds_in',
        'service_battery_initialization_time'
    ),
    outputs=(
        'motor_p0_speeds', 'motor_p0_speed_ratio', 'motor_p0_efficiency_ratios',
        'motor_p0_maximum_torque', 'motor_p0_maximum_powers', 'motor_p0_powers',
        'motor_p0_maximum_power', 'motor_p0_rated_speed', 'motor_p0_torques',
        'motor_p0_electric_power_loss_function', 'motor_p0_electric_powers',
        'motor_p0_maximum_power_function', 'wheel_speeds', 'gear_box_speeds_in',
        'motor_p1_speeds', 'motor_p1_speed_ratio', 'motor_p1_efficiency_ratios',
        'motor_p1_maximum_torque', 'motor_p1_maximum_powers', 'motor_p1_powers',
        'motor_p1_maximum_power', 'motor_p1_rated_speed', 'motor_p1_torques',
        'motor_p1_electric_power_loss_function', 'motor_p1_electric_powers',
        'motor_p1_maximum_power_function', 'alternator_electric_powers',
        'motor_p2_speeds', 'motor_p2_speed_ratio', 'motor_p2_efficiency_ratios',
        'motor_p2_maximum_torque', 'motor_p2_maximum_powers', 'motor_p2_powers',
        'motor_p2_maximum_power', 'motor_p2_rated_speed', 'motor_p2_torques',
        'motor_p2_electric_power_loss_function', 'motor_p2_electric_powers',
        'motor_p3_speeds', 'motor_p3_speed_ratio', 'motor_p3_efficiency_ratios',
        'motor_p3_maximum_torque', 'motor_p3_maximum_powers', 'motor_p3_powers',
        'motor_p3_maximum_power', 'motor_p3_rated_speed', 'motor_p3_torques',
        'motor_p3_electric_power_loss_function', 'motor_p3_electric_powers',
        'motor_p4_speeds', 'motor_p4_speed_ratio', 'motor_p4_efficiency_ratios',
        'motor_p4_maximum_torque', 'motor_p4_maximum_powers', 'motor_p4_powers',
        'motor_p4_maximum_power', 'motor_p4_rated_speed', 'motor_p4_torques',
        'motor_p4_electric_power_loss_function', 'motor_p4_electric_powers',
        'alternator_current_model', 'starter_powers', 'starter_electric_powers',
        'delta_time_engine_starter', 'alternator_currents', 'engine_speeds_out',
        'alternator_powers', 'final_drive_speeds_in', 'start_demand_function',
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='batteries',
    dsp=_batteries,
    inputs=(
        'drive_battery_ocv', 'drive_battery_r0', 'drive_battery_n_series_cells',
        'initial_service_battery_state_of_charge', 'alternator_electric_powers',
        'service_battery_start_window_width', 'service_battery_nominal_voltage',
        'cycle_type', 'service_battery_electric_powers', 'service_battery_load',
        'initial_drive_battery_state_of_charge', 'service_battery_status_model',
        'motor_p2_electric_powers', 'motor_p3_electric_powers', 'accelerations',
        'service_battery_loads', 'dcdc_charging_currents', 'dcdc_current_model',
        'service_battery_electric_powers_supply_threshold', 'engine_powers_out',
        'motor_p0_electric_powers', 'alternator_current_model', 'motive_powers',
        'service_battery_state_of_charge_balance_window', 'drive_battery_loads',
        'drive_battery_state_of_charges', 'drive_battery_currents', 'on_engine',
        'service_battery_state_of_charges', 'motor_p1_electric_powers', 'times',
        'service_battery_currents', 'motor_p4_electric_powers', 'engine_starts',
        'service_battery_initialization_time', 'dcdc_converter_electric_powers',
        'service_battery_state_of_charge_balance', 'dcdc_converter_efficiency',
        'drive_battery_n_parallel_cells', 'electrical_hybridization_degree',
        'service_battery_capacity', 'drive_battery_electric_powers',
        'dcdc_converter_currents', 'starter_electric_powers',
        'drive_battery_capacity', 'drive_battery_voltages',
        'has_energy_recuperation', 'drive_battery_load',
    ),
    outputs=(
        'service_battery_electric_powers_supply_threshold', 'drive_battery_ocv',
        'service_battery_state_of_charge_balance_window', 'drive_battery_model',
        'drive_battery_state_of_charges', 'drive_battery_delta_state_of_charge',
        'initial_drive_battery_state_of_charge', 'drive_battery_n_series_cells',
        'service_battery_load', 'service_battery_loads', 'drive_battery_loads',
        'initial_service_battery_state_of_charge', 'service_battery_capacity',
        'service_battery_initialization_time', 'service_battery_status_model',
        'service_battery_electric_powers', 'service_battery_state_of_charges',
        'service_battery_charging_statuses', 'drive_battery_electric_powers',
        'service_battery_state_of_charge_balance', 'drive_battery_currents',
        'service_battery_delta_state_of_charge', 'service_battery_currents',
        'dcdc_converter_electric_powers_demand', 'drive_battery_voltages',
        'drive_battery_n_parallel_cells', 'drive_battery_capacity',
        'drive_battery_load', 'drive_battery_r0', 'dcdc_current_model'
    ),
    include_defaults=True
)
