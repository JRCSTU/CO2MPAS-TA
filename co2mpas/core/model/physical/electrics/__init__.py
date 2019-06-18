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
        'engine_speeds_out', 'motor_p0_efficiency',
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
        'wheel_speeds',
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
        'motors_electric_powers'
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='batteries',
    dsp=_batteries,
    inputs=(
        'cycle_type', 'dcdc_converter_electric_powers', 'engine_powers_out',
        'initial_service_battery_state_of_charge', 'on_engine',
        'service_battery_capacity', 'service_battery_currents',
        'service_battery_electric_powers', 'service_battery_load',
        'service_battery_loads', 'service_battery_nominal_voltage',
        'service_battery_state_of_charges', 'times',
        'dcdc_converter_efficiency', 'dcdc_converter_electric_powers_demand',
        'drive_battery_capacity', 'drive_battery_currents',
        'drive_battery_electric_powers', 'drive_battery_load',
        'drive_battery_loads', 'drive_battery_n_parallel_cells',
        'drive_battery_n_series_cells', 'drive_battery_ocv', 'drive_battery_r0',
        'drive_battery_state_of_charges', 'drive_battery_voltages',
        'electrical_hybridization_degree',
        'initial_drive_battery_state_of_charge', 'motors_electric_powers',
        'maximum_drive_battery_electric_power'
    ),
    outputs=(
        'alternator_electric_powers', 'dcdc_converter_electric_powers',
        'service_battery_currents', 'service_battery_electric_powers',
        'service_battery_load', 'service_battery_loads',
        'service_battery_nominal_voltage', 'service_battery_state_of_charges',
        'dcdc_converter_efficiency', 'dcdc_converter_electric_powers_demand',
        'drive_battery_currents', 'drive_battery_electric_powers',
        'drive_battery_load', 'drive_battery_loads',
        'drive_battery_n_parallel_cells', 'drive_battery_n_series_cells',
        'drive_battery_ocv', 'drive_battery_r0',
        'drive_battery_state_of_charges', 'drive_battery_voltages',
        'maximum_drive_battery_electric_power'
    ),
    include_defaults=True
)
