# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It provides CO2MPAS model `dsp` to predict light-vehicles' CO2 emissions.

Docstrings should provide sufficient understanding for any individual function.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical

.. autosummary::
    :nosignatures:
    :toctree: physical/

    vehicle
    wheels
    final_drive
    gear_box
    clutch_tc
    cycle
    electrics
    engine
    control
    defaults
"""

import schedula as sh
from .driver import dsp as _driver
from .vehicle import dsp as _vehicle
from .wheels import dsp as _wheels
from .final_drive import dsp as _final_drive
from .gear_box import dsp as _gear_box
from .clutch_tc import dsp as _clutch_torque_converter
from .electrics import dsp as _electrics
from .engine import dsp as _engine
from .control import dsp as _control

dsp = sh.BlueDispatcher(
    name='CO2MPAS physical model',
    description='Wraps all functions needed to calibrate and predict '
                'light-vehicles\' CO2 emissions.'
)

dsp.add_data('path_velocities', wildcard=True)
dsp.add_data('path_distances', wildcard=True)
dsp.add_data('path_elevations', wildcard=True)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='driver_model',
    dsp=_driver,
    inputs=(
        'max_speed_velocity_ratio', 'engine_speed_at_max_power', 'unladen_mass',
        'downscale_factor_threshold', 'speed_velocity_ratios', 'path_distances',
        'time_sample_frequency', 'driver_style_ratio', 'downscale_factor', 'k1',
        'idle_engine_speed', 'downscale_phases', 'engine_max_power', 'max_gear',
        'path_velocities', 'wltp_base_model', 'inertial_factor', 'vehicle_mass',
        'engine_max_speed', 'full_load_curve', 'accelerations', 'motive_powers',
        'gear_box_type', 'road_loads', 'cycle_type', 'use_driver', 'wltp_class',
        'max_velocity', 'velocities', 'distances', 'driver_style', 'bag_phases',
        'max_time', 'times', 'gears', 'k2', 'k5',
    ),
    outputs=(
        'initial_temperature', 'phases_integration_times', 'desired_velocities',
        'times', 'velocities', 'gears',
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='vehicle_model',
    dsp=_vehicle,
    inputs=(
        'traction_acceleration_limit', 'traction_deceleration_limit', 'n_wheel',
        'aerodynamic_drag_coefficient', 'rolling_resistance_coeff', 'fuel_mass',
        'wheel_drive_load_fraction', 'n_wheel_drive', 'obd_velocities', 'times',
        'static_friction', 'initial_velocity', 'tyre_category', 'vehicle_width',
        'vehicle_body', 'road_state', 'cargo_mass', 'angle_slope', 'tyre_class',
        'n_passengers', 'passenger_mass', 'air_density', 'f0_uncorrected', 'f1',
        'n_dyno_axes', 'velocities', 'angle_slopes', 'correct_f0', 'cycle_type',
        'curb_mass', 'elevations', 'frontal_area', 'has_roof_box', 'road_loads',
        'unladen_mass', 'inertial_factor', 'vehicle_height', 'vehicle_category',
        'tyre_state', 'vehicle_mass', 'f0', 'f2',
    ),
    outputs=(
        'wheel_drive_load_fraction', 'traction_acceleration_limit', 'curb_mass',
        'traction_deceleration_limit', 'inertial_factor', 'motive_powers', 'f0',
        'static_friction', 'accelerations', 'angle_slopes', 'n_dyno_axes', 'f1',
        'unladen_mass', 'vehicle_mass', 'velocities', 'road_loads', 'distances',
        'f2',
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='wheels_model',
    dsp=_wheels,
    inputs=(
        'accelerations', 'change_gear_window_width', 'engine_speeds_out',
        'final_drive_ratios', 'gear_box_ratios', 'gears', 'idle_engine_speed',
        'motive_powers', 'plateau_acceleration', 'r_dynamic', 'r_wheels',
        'gear_box_speeds_in', 'on_engine', 'tyre_dimensions', 'stop_velocity',
        'tyre_dynamic_rolling_coefficient', 'tyre_code', 'velocities', 'times',
        'velocity_speed_ratios'
    ),
    outputs=(
        'r_dynamic', 'r_wheels', 'tyre_code', 'wheel_powers', 'wheel_speeds',
        'tyre_dynamic_rolling_coefficient', 'wheel_torques',
    ),
    inp_weight={'r_dynamic': 3}
)


@sh.add_function(dsp, outputs=['final_drive_powers_out'])
def calculate_final_drive_powers_out(wheel_powers, motor_p4_powers):
    """
    Calculate final drive power out [kW].

    :param wheel_powers:
        Power at the wheels [kW].
    :type wheel_powers: numpy.array | float

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array | float

    :return:
        Final drive power out [kW].
    :rtype: numpy.array | float
    """
    return wheel_powers - motor_p4_powers


dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='final_drive_model',
    dsp=_final_drive,
    inputs=(
        'final_drive_efficiency', 'final_drive_ratio', 'final_drive_ratios',
        'gear_box_type', 'gears', 'n_wheel_drive', 'n_gears',
        'final_drive_powers_out', {'wheel_speeds': 'final_drive_speeds_out'}
    ),
    outputs=(
        'final_drive_powers_in', 'final_drive_ratios', 'final_drive_speeds_in',
        'final_drive_torques_in', 'final_drive_mean_efficiency'
    )
)


@sh.add_function(dsp, outputs=['gear_box_powers_out'])
def calculate_gear_box_powers_out(
        final_drive_powers_in, motor_p3_front_powers, motor_p3_rear_powers):
    """
    Calculate gear box power vector [kW].

    :param final_drive_powers_in:
        Final drive power in [kW].
    :type final_drive_powers_in: numpy.array | float

    :param motor_p3_front_powers:
        Power at motor P3 front [kW].
    :type motor_p3_front_powers: numpy.array | float

    :param motor_p3_rear_powers:
        Power at motor P3 rear [kW].
    :type motor_p3_rear_powers: numpy.array | float

    :return:
        Gear box power vector [kW].
    :rtype: numpy.array | float
    """
    return final_drive_powers_in - motor_p3_front_powers - motor_p3_rear_powers


dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='gear_box_model',
    dsp=_gear_box,
    inputs=(
        'has_gear_box_thermal_management', 'has_torque_converter', 'velocities',
        'maximum_vehicle_laden_mass', 'maximum_velocity', 'min_engine_on_speed',
        'engine_coolant_temperatures', 'engine_mass', 'engine_max_power', 'MVL',
        'initial_gear_box_temperature', 'gear_box_temperature_references', 'f0',
        'change_gear_window_width', 'idle_engine_speed', 'cycle_type', 'times',
        'full_load_speeds', 'gear_box_efficiency_constants', 'gear_box_ratios',
        'CMV', 'CMV_Cold_Hot', 'CVT', 'DTGS', 'GSPV', 'GSPV_Cold_Hot', 'gears',
        'engine_thermostat_temperature', 'final_drive_ratios', 'accelerations',
        'r_dynamic', 'road_loads', 'specific_gear_shifting', 'stop_velocity',
        'engine_max_speed', 'engine_max_torque', 'engine_speed_at_max_power',
        'first_gear_box_ratio', 'fuel_saving_at_strategy', 'full_load_curve',
        'engine_speed_at_max_velocity', 'engine_speeds_out', 'gear_box_type',
        'gear_box_efficiency_parameters_cold_hot', 'use_dt_gear_shifting',
        'max_velocity_full_load_correction', 'motive_powers', 'n_gears',
        'on_engine', 'plateau_acceleration', 'time_cold_hot_transition',
        'last_gear_box_ratio', 'final_drive_mean_efficiency',
        'gear_box_speeds_in', 'velocity_speed_ratios', 'gear_box_powers_out', {
            'final_drive_speeds_in': 'gear_box_speeds_out',
            'initial_engine_temperature': 'initial_gear_box_temperature',
            'initial_temperature': 'initial_gear_box_temperature'
        }
    ),
    outputs=(
        'CMV', 'CMV_Cold_Hot', 'CVT', 'DTGS', 'GSPV', 'GSPV_Cold_Hot', 'MVL',
        'engine_speed_at_max_velocity', 'equivalent_gear_box_heat_capacity',
        'first_gear_box_ratio', 'gear_box_efficiencies', 'gear_box_powers_in',
        'gear_box_ratios', 'gear_box_speeds_in', 'gear_box_temperatures',
        'gear_box_torque_losses', 'gear_box_torques_in', 'gear_shifts', 'gears',
        'last_gear_box_ratio', 'max_gear', 'max_speed_velocity_ratio',
        'maximum_velocity', 'specific_gear_shifting', 'velocity_speed_ratios',
        'speed_velocity_ratios', 'n_gears', 'gear_box_mean_efficiency',
        'gear_box_mean_efficiency_guess'
    ),
    inp_weight={'initial_temperature': 5}
)


@sh.add_function(dsp, outputs=['clutch_tc_powers_out'])
def calculate_clutch_tc_powers_out(gear_box_powers_in, motor_p2_powers):
    """
    Calculate gear box power vector [kW].

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array | float

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array | float

    :return:
        Clutch or torque converter power out [kW].
    :rtype: numpy.array | float
    """
    return gear_box_powers_in - motor_p2_powers


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_clutch_torque_converter,
    dsp_id='clutch_torque_converter_model',
    inputs=(
        'accelerations', 'clutch_window', 'cold_start_speeds_delta', 'gears',
        'engine_speeds_out', 'clutch_speed_model', 'has_torque_converter',
        'engine_speeds_out_hot', 'clutch_tc_powers_out', 'gear_box_speeds_in',
        'gear_box_type', 'gear_shifts', 'full_load_curve', 'idle_engine_speed',
        'lockup_speed_ratio', 'stand_still_torque_ratio', 'engine_max_speed',
        'stop_velocity', 'times', 'torque_converter_speed_model', 'velocities',
        'm1000_curve_factor', 'm1000_curve_ratios', 'm1000_curve_norm_torques',
        'gear_box_torques_in', 'clutch_tc_speeds'
    ),
    outputs=(
        'clutch_speed_model', 'clutch_phases', 'clutch_tc_mean_efficiency',
        'clutch_window', 'clutch_tc_speeds_delta', 'has_torque_converter',
        'lockup_speed_ratio', 'stand_still_torque_ratio', 'm1000_curve_factor',
        'torque_converter_speed_model', 'clutch_tc_powers', 'clutch_tc_speeds'
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='electric_model',
    dsp=_electrics,
    inputs=(
        'motor_p0_efficiency', 'motor_p0_maximum_power', 'motor_p0_speed_ratio',
        'motor_p0_electric_powers', 'motor_p0_torques', 'motor_p0_rated_speed',
        'motor_p0_powers', 'motor_p0_maximum_torque', 'motor_p0_speeds',
        'motor_p1_efficiency', 'motor_p1_maximum_power', 'motor_p1_speed_ratio',
        'motor_p1_electric_powers', 'motor_p1_torques', 'motor_p1_rated_speed',
        'motor_p1_powers', 'motor_p1_maximum_torque', 'motor_p1_speeds',
        'motor_p2_efficiency', 'motor_p2_maximum_power', 'motor_p2_speed_ratio',
        'motor_p2_electric_powers', 'motor_p2_torques', 'motor_p2_rated_speed',
        'motor_p2_powers', 'motor_p2_maximum_torque', 'motor_p2_speeds',
        'motor_p3_front_efficiency', 'motor_p3_front_maximum_power',
        'motor_p3_front_speed_ratio', 'motor_p3_front_electric_powers',
        'motor_p3_front_torques', 'motor_p3_front_rated_speed',
        'motor_p3_front_powers', 'motor_p3_front_maximum_torque',
        'motor_p3_front_speeds',
        'motor_p3_rear_efficiency', 'motor_p3_rear_maximum_power',
        'motor_p3_rear_speed_ratio', 'motor_p3_rear_electric_powers',
        'motor_p3_rear_torques', 'motor_p3_rear_rated_speed',
        'motor_p3_rear_powers', 'motor_p3_rear_maximum_torque',
        'motor_p3_rear_speeds',
        'motor_p4_efficiency', 'motor_p4_maximum_power', 'motor_p4_speed_ratio',
        'motor_p4_electric_powers', 'motor_p4_torques', 'motor_p4_rated_speed',
        'motor_p4_powers', 'motor_p4_maximum_torque', 'motor_p4_speeds',
        'has_motor_p0', 'has_motor_p1', 'has_motor_p2', 'has_motor_p3_front',
        'has_motor_p3_rear', 'has_motor_p4',
        'service_battery_initialization_time', 'engine_starts', 'accelerations',
        'starter_efficiency', 'delta_time_engine_starter', 'gear_box_speeds_in',
        'service_battery_charging_statuses', 'alternator_currents', 'on_engine',
        'starter_nominal_voltage', 'engine_speeds_out', 'alternator_efficiency',
        'service_battery_state_of_charges', 'alternator_current_model', 'times',
        'alternator_charging_currents', 'engine_moment_inertia', 'wheel_speeds',
        'drive_battery_ocv', 'drive_battery_r0', 'drive_battery_n_series_cells',
        'initial_service_battery_state_of_charge', 'alternator_electric_powers',
        'service_battery_start_window_width', 'service_battery_nominal_voltage',
        'cycle_type', 'service_battery_electric_powers', 'service_battery_load',
        'initial_drive_battery_state_of_charge', 'service_battery_status_model',
        'final_drive_speeds_in', 'dcdc_charging_currents', 'dcdc_current_model',
        'service_battery_electric_powers_supply_threshold', 'engine_powers_out',
        'service_battery_state_of_charge_balance_window', 'drive_battery_loads',
        'drive_battery_currents', 'service_battery_loads', 'drive_battery_load',
        'service_battery_state_of_charge_balance', 'alternator_nominal_voltage',
        'dcdc_converter_efficiency', 'has_energy_recuperation', 'motive_powers',
        'dcdc_converter_currents', 'drive_battery_voltages', 'clutch_tc_speeds',
        'dcdc_converter_electric_powers', 'electrical_hybridization_degree',
        'drive_battery_state_of_charges', 'drive_battery_n_parallel_cells',
        'drive_battery_electric_powers', 'service_battery_capacity',
        'starter_electric_powers', 'service_battery_currents',
        'drive_battery_capacity',
    ),
    outputs=(
        'motor_p0_electric_powers', 'motor_p0_maximum_power', 'motor_p0_powers',
        'motor_p0_maximum_torque', 'motor_p0_speed_ratio', 'motor_p0_torques',
        'motor_p0_maximum_powers', 'motor_p0_rated_speed', 'motor_p0_speeds',
        'motor_p0_maximum_power_function', 'motor_p1_maximum_power_function',
        'motor_p1_electric_powers', 'motor_p1_maximum_power', 'motor_p1_powers',
        'motor_p1_maximum_torque', 'motor_p1_speed_ratio', 'motor_p1_torques',
        'motor_p1_maximum_powers', 'motor_p1_rated_speed', 'motor_p1_speeds',
        'motor_p2_electric_powers', 'motor_p2_maximum_power', 'motor_p2_powers',
        'motor_p2_maximum_torque', 'motor_p2_speed_ratio', 'motor_p2_torques',
        'motor_p2_maximum_powers', 'motor_p2_rated_speed', 'motor_p2_speeds',
        'motor_p3_front_electric_powers', 'motor_p3_front_maximum_power',
        'motor_p3_front_powers', 'motor_p3_front_maximum_torque',
        'motor_p3_front_speed_ratio', 'motor_p3_front_torques',
        'motor_p3_front_maximum_powers', 'motor_p3_front_rated_speed',
        'motor_p3_front_speeds',
        'motor_p3_rear_electric_powers', 'motor_p3_rear_maximum_power',
        'motor_p3_rear_powers', 'motor_p3_rear_maximum_torque',
        'motor_p3_rear_speed_ratio', 'motor_p3_rear_torques',
        'motor_p3_rear_maximum_powers', 'motor_p3_rear_rated_speed',
        'motor_p3_rear_speeds',
        'motor_p4_electric_powers', 'motor_p4_maximum_power', 'motor_p4_powers',
        'motor_p4_maximum_torque', 'motor_p4_speed_ratio', 'motor_p4_torques',
        'motor_p4_maximum_powers', 'motor_p4_rated_speed', 'motor_p4_speeds',
        'has_motor_p0', 'has_motor_p1', 'has_motor_p2', 'has_motor_p3_front',
        'has_motor_p3_rear', 'has_motor_p4', 'is_hybrid',
        'motors_electric_powers',
        'starter_electric_powers', 'final_drive_speeds_in', 'alternator_powers',
        'alternator_electric_powers', 'engine_speeds_out', 'gear_box_speeds_in',
        'delta_time_engine_starter', 'alternator_current_model', 'wheel_speeds',
        'alternator_currents', 'starter_powers', 'starter_model',
        'service_battery_electric_powers_supply_threshold', 'drive_battery_ocv',
        'service_battery_state_of_charge_balance_window', 'drive_battery_model',
        'drive_battery_state_of_charges', 'drive_battery_delta_state_of_charge',
        'initial_drive_battery_state_of_charge', 'drive_battery_n_series_cells',
        'dcdc_converter_currents', 'drive_battery_capacity', 'drive_battery_r0',
        'service_battery_delta_state_of_charge', 'service_battery_status_model',
        'service_battery_state_of_charges', 'service_battery_charging_statuses',
        'dcdc_current_model', 'drive_battery_loads', 'service_battery_currents',
        'starter_currents', 'service_battery_load', 'dcdc_converter_efficiency',
        'drive_battery_voltages', 'service_battery_loads', 'drive_battery_load',
        'service_battery_state_of_charge_balance', 'alternator_nominal_voltage',
        'service_battery_initialization_time', 'drive_battery_electric_powers',
        'initial_service_battery_state_of_charge', 'service_battery_capacity',
        'service_battery_electric_powers', 'service_battery_nominal_voltage',
        'dcdc_converter_electric_powers_demand', 'starter_nominal_voltage',
        'drive_battery_n_parallel_cells', 'drive_battery_currents',
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='engine_model',
    dsp=_engine,
    inputs=(
        'accelerations', 'active_cylinder_ratios', 'alternator_powers', 'gears',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_powers_out', 'engine_speeds_out', 'stop_velocity', 'velocities',
        'has_exhausted_gas_recirculation', 'gear_box_speeds_in', 'is_cycle_hot',
        'has_lean_burn', 'has_periodically_regenerating_systems', 'ki_additive',
        'initial_friction_params', 'min_engine_on_speed', 'fuel_carbon_content',
        'motor_p0_powers', 'motor_p1_powers', 'auxiliaries_torque_loss_factors',
        'idle_engine_speed', 'engine_coolant_temperatures', 'full_load_torques',
        'has_selective_catalytic_reduction', 'engine_has_cylinder_deactivation',
        'engine_has_variable_valve_actuation', 'max_engine_coolant_temperature',
        'engine_fuel_lower_heating_value', 'fuel_consumptions', 'gear_box_type',
        'engine_capacity', 'clutch_tc_speeds_delta', 'phases_integration_times',
        'engine_speeds_out_hot', 'enable_phases_willans', 'co2_emission_medium',
        'engine_idle_fuel_consumption', 'idle_engine_speed_std', 'angle_slopes',
        'initial_engine_temperature', 'fuel_carbon_content_percentage', 'times',
        'engine_temperature_regression_model', 'ki_multiplicative', 'on_engine',
        'engine_max_speed', 'engine_max_torque', 'motive_powers', 'engine_type',
        'engine_speed_at_max_power', 'obd_fuel_type_code', 'calibration_status',
        'co2_normalization_references', 'auxiliaries_torque_loss', 'co2_params',
        'idle_engine_speed_median', 'auxiliaries_power_loss', 'engine_is_turbo',
        'co2_params_calibrated', 'co2_emission_extra_high', 'co2_emission_high',
        'cold_start_speed_model', 'full_load_powers', 'fuel_density', 'on_idle',
        'enable_willans', 'engine_n_cylinders', 'fuel_type', 'co2_emission_low',
        'engine_max_power', 'clutch_tc_powers', 'full_load_speeds', 'is_hybrid',
        'engine_mass', 'ignition_type', 'engine_stroke', 'gear_box_powers_out',
        'belt_efficiency', {'initial_temperature': 'initial_engine_temperature'}
    ),
    outputs=(
        'has_exhausted_gas_recirculation', 'full_load_curve', 'willans_factors',
        'identified_co2_emissions', 'idle_engine_speed', 'has_sufficient_power',
        'idle_engine_speed_median', 'idle_engine_speed_std', 'full_load_powers',
        'ki_multiplicative', 'max_engine_coolant_temperature', 'missing_powers',
        'fuel_map', 'optimal_efficiency', 'phases_co2_emissions', 'ki_additive',
        'co2_emission_value', 'active_variable_valves', 'engine_speeds_out_hot',
        'engine_temperature_regression_model', 'engine_speeds_out', 'fuel_type',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'phases_willans_factors', 'engine_idle_fuel_consumption', 'engine_mass',
        'engine_heat_capacity', 'cold_start_speed_model', 'fuel_carbon_content',
        'auxiliaries_power_losses', 'engine_temperature_derivatives', 'on_idle',
        'after_treatment_temperature_threshold', 'engine_type', 'ignition_type',
        'auxiliaries_torque_losses', 'cold_start_speeds_delta', 'co2_emissions',
        'co2_params_calibrated', 'co2_params_initial_guess', 'active_cylinders',
        'initial_friction_params', 'engine_moment_inertia', 'engine_powers_out',
        'declared_co2_emission_value', 'active_lean_burns', 'engine_max_torque',
        'extended_phases_co2_emissions', 'gross_engine_powers_out', 'is_hybrid',
        'engine_inertia_powers_losses', 'fuel_consumptions', 'engine_max_speed',
        'fuel_carbon_content_percentage', 'active_exhausted_gas_recirculations',
        'co2_error_function_on_emissions', 'calibration_status', 'brake_powers',
        'engine_speed_at_max_power', 'belt_mean_efficiency', 'engine_max_power',
        'initial_engine_temperature', 'auxiliaries_torque_loss', 'fuel_density',
        'co2_rescaling_scores', 'auxiliaries_power_loss', 'co2_emissions_model',
        'extended_phases_integration_times', 'engine_fuel_lower_heating_value',
        'co2_error_function_on_phases', 'engine_coolant_temperatures',
        'cold_start_speeds_phases', 'phases_fuel_consumptions',
        'full_load_speeds',
    ),
    inp_weight={'initial_temperature': 5}
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='control_model',
    dsp=_control,
    inputs=(
        'full_load_curve', 'gear_box_speeds_in', 'motor_p0_efficiency', 'gears',
        'drive_battery_model', 'idle_engine_speed', 'catalyst_warm_up_duration',
        'motor_p3_front_maximum_power', 'motor_p3_rear_maximum_power', 'ecms_s',
        'motor_p4_maximum_powers', 'motor_p1_maximum_power', 'start_stop_model',
        'engine_coolant_temperatures', 'engine_moment_inertia', 'gear_box_type',
        'final_drive_mean_efficiency', 'gear_box_mean_efficiency', 'velocities',
        'start_stop_hybrid_params', 'motor_p2_maximum_powers', 'has_start_stop',
        'motor_p4_maximum_power', 'motor_p0_maximum_power_function', 'fuel_map',
        'drive_battery_state_of_charges', 'is_hybrid', 'motor_p2_maximum_power',
        'motor_p0_speed_ratio', 'motor_p2_efficiency', 'auxiliaries_power_loss',
        'motor_p3_rear_electric_powers', 'motive_powers', 'motor_p4_efficiency',
        'start_stop_activation_time', 'motor_p0_maximum_power', 'starter_model',
        'motor_p3_rear_maximum_powers', 'engine_powers_out', 'catalyst_warm_up',
        'is_cycle_hot', 'motor_p2_electric_powers', 'clutch_tc_mean_efficiency',
        'motor_p3_rear_efficiency', 'motor_p0_electric_powers', 'accelerations',
        'motor_p1_efficiency', 'motor_p1_speed_ratio', 'engine_speeds_out_hot',
        'motor_p1_maximum_power_function', 'motor_p4_electric_powers', 'times',
        'engine_speeds_out', 'auxiliaries_torque_loss', 'belt_mean_efficiency',
        'motor_p3_front_efficiency', 'dcdc_converter_efficiency', 'on_engine',
        'motor_p1_electric_powers', 'motors_electric_powers', 'hybrid_modes',
        'min_time_engine_on_after_start', 'motor_p3_front_electric_powers',
        'motor_p3_front_maximum_powers', 'gear_box_mean_efficiency_guess',
        'engine_thermostat_temperature', 'correct_start_stop_with_gears',
    ),
    outputs=(
        'correct_start_stop_with_gears', 'start_stop_activation_time', 'ecms_s',
        'motor_p0_electric_powers', 'motor_p1_electric_powers', 'engine_starts',
        'engine_speeds_out_hot', 'start_stop_hybrid_params', 'start_stop_model',
        'catalyst_warm_up_duration', 'motor_p2_electric_powers', 'hybrid_modes',
        'motor_p3_front_electric_powers', 'catalyst_warm_up', 'on_engine',
        'motor_p3_rear_electric_powers', 'motor_p4_electric_powers',
    )
)
