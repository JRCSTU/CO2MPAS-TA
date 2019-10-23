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
    after_treat
    co2
"""

import schedula as sh
from .cycle import dsp as _cycle
from .vehicle import dsp as _vehicle
from .wheels import dsp as _wheels
from .final_drive import dsp as _final_drive
from .gear_box import dsp as _gear_box
from .clutch_tc import dsp as _clutch_torque_converter
from .electrics import dsp as _electrics
from .engine import dsp as _engine
from .control import dsp as _control
from .after_treat import dsp as _after_treat
from .co2 import dsp as _co2

dsp = sh.BlueDispatcher(
    name='CO2MPAS physical model',
    description='Wraps all functions needed to calibrate and predict '
                'light-vehicles\' CO2 emissions.'
)

dsp.add_data('path_elevations', wildcard=True)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='cycle_model',
    dsp=_cycle,
    inputs=(
        'max_speed_velocity_ratio', 'engine_speed_at_max_power', 'unladen_mass',
        'downscale_factor_threshold', 'speed_velocity_ratios', 'k1', 'k2', 'k5',
        'idle_engine_speed', 'downscale_phases', 'engine_max_power', 'max_gear',
        'engine_max_speed', 'full_load_curve', 'accelerations', 'motive_powers',
        'gear_box_type', 'road_loads', 'cycle_type', 'wltp_class', 'bag_phases',
        'downscale_factor', 'inertial_factor', 'vehicle_mass', 'times', 'gears',
        'time_sample_frequency', 'wltp_base_model', 'max_velocity', 'max_time',
        'velocities',
    ),
    outputs=(
        'initial_temperature', 'phases_integration_times', 'times', 'gears',
        'velocities', 'theoretical_motive_powers', 'theoretical_velocities'
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
        'tyre_state', 'vehicle_mass', 'f0', 'path_distances', 'path_elevations',
        'f2',
    ),
    outputs=(
        'wheel_drive_load_fraction', 'traction_acceleration_limit', 'curb_mass',
        'traction_deceleration_limit', 'inertial_factor', 'motive_powers', 'f0',
        'static_friction', 'accelerations', 'angle_slopes', 'n_dyno_axes', 'f1',
        'unladen_mass', 'vehicle_mass', 'velocities', 'road_loads', 'distances',
        'f2'
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
def calculate_final_drive_powers_out(
        wheel_powers, motor_p4_front_powers, motor_p4_rear_powers):
    """
    Calculate final drive power out [kW].

    :param wheel_powers:
        Power at the wheels [kW].
    :type wheel_powers: numpy.array | float

    :param motor_p4_front_powers:
        Power at motor P4 front [kW].
    :type motor_p4_front_powers: numpy.array | float

    :param motor_p4_rear_powers:
        Power at motor P4 rear [kW].
    :type motor_p4_rear_powers: numpy.array | float

    :return:
        Final drive power out [kW].
    :rtype: numpy.array | float
    """
    return wheel_powers - motor_p4_front_powers - motor_p4_rear_powers


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
        'engine_speeds_base', 'gear_box_torques_in', 'm1000_curve_norm_torques',
        'engine_speeds_out', 'stand_still_torque_ratio', 'has_torque_converter',
        'engine_speeds_out_hot', 'm1000_curve_ratios', 'clutch_window', 'gears',
        'torque_converter_speed_model', 'm1000_curve_factor', 'full_load_curve',
        'engine_max_speed', 'idle_engine_speed', 'accelerations', 'gear_shifts',
        'clutch_tc_powers_out', 'gear_box_speeds_in', 'stop_velocity', 'times',
        'lockup_speed_ratio', 'clutch_speed_model', 'gear_box_type',
        'velocities',
    ),
    outputs=(
        'clutch_speed_model', 'clutch_phases', 'clutch_tc_mean_efficiency',
        'clutch_window', 'clutch_tc_speeds_delta', 'has_torque_converter',
        'lockup_speed_ratio', 'stand_still_torque_ratio', 'm1000_curve_factor',
        'torque_converter_speed_model', 'clutch_tc_powers'
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
        'motor_p2_planetary_efficiency', 'motor_p2_planetary_maximum_power',
        'motor_p2_planetary_speed_ratio', 'motor_p2_planetary_electric_powers',
        'motor_p2_planetary_torques', 'motor_p2_planetary_rated_speed',
        'motor_p2_planetary_powers', 'motor_p2_planetary_maximum_torque',
        'motor_p2_planetary_speeds',
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
        'motor_p4_front_efficiency', 'motor_p4_front_maximum_power',
        'motor_p4_front_speed_ratio', 'motor_p4_front_electric_powers',
        'motor_p4_front_torques', 'motor_p4_front_rated_speed',
        'motor_p4_front_powers', 'motor_p4_front_maximum_torque',
        'motor_p4_front_speeds',
        'motor_p4_rear_efficiency', 'motor_p4_rear_maximum_power',
        'motor_p4_rear_speed_ratio', 'motor_p4_rear_electric_powers',
        'motor_p4_rear_torques', 'motor_p4_rear_rated_speed',
        'motor_p4_rear_powers', 'motor_p4_rear_maximum_torque',
        'motor_p4_rear_speeds',
        'has_motor_p0', 'has_motor_p1', 'has_motor_p2_planetary',
        'has_motor_p2', 'has_motor_p3_front', 'has_motor_p3_rear',
        'has_motor_p4_front', 'has_motor_p4_rear',
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
        'starter_electric_powers', 'dcdc_converter_currents', 'planetary_ratio',
        'dcdc_converter_electric_powers_demand', 'planetary_mean_efficiency',
        'electrical_hybridization_degree', 'drive_battery_n_parallel_cells',
        'drive_battery_state_of_charges', 'drive_battery_electric_powers',
        'dcdc_converter_electric_powers', 'drive_battery_nominal_voltage',
        'service_battery_currents', 'drive_battery_technology',
        'drive_battery_capacity', 'service_battery_capacity',
        'drive_battery_n_cells', 'drive_battery_voltages',
    ),
    outputs=(
        'motor_p0_electric_powers', 'motor_p0_maximum_power', 'motor_p0_powers',
        'motor_p0_maximum_torque', 'motor_p0_speed_ratio', 'motor_p0_torques',
        'motor_p0_maximum_powers', 'motor_p0_rated_speed', 'motor_p0_speeds',
        'motor_p0_maximum_power_function', 'motor_p1_maximum_power_function',
        'motor_p1_electric_powers', 'motor_p1_maximum_power', 'motor_p1_powers',
        'motor_p1_maximum_torque', 'motor_p1_speed_ratio', 'motor_p1_torques',
        'motor_p1_maximum_powers', 'motor_p1_rated_speed', 'motor_p1_speeds',
        'motor_p2_planetary_maximum_power_function',
        'motor_p2_planetary_electric_powers',
        'motor_p2_planetary_maximum_power', 'motor_p2_planetary_powers',
        'motor_p2_planetary_maximum_torque', 'motor_p2_planetary_speed_ratio',
        'motor_p2_planetary_torques', 'motor_p2_planetary_maximum_powers',
        'motor_p2_planetary_rated_speed', 'motor_p2_planetary_speeds',
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
        'motor_p4_front_electric_powers', 'motor_p4_front_maximum_power',
        'motor_p4_front_powers', 'motor_p4_front_maximum_torque',
        'motor_p4_front_speed_ratio', 'motor_p4_front_torques',
        'motor_p4_front_maximum_powers', 'motor_p4_front_rated_speed',
        'motor_p4_front_speeds',
        'motor_p4_rear_electric_powers', 'motor_p4_rear_maximum_power',
        'motor_p4_rear_powers', 'motor_p4_rear_maximum_torque',
        'motor_p4_rear_speed_ratio', 'motor_p4_rear_torques',
        'motor_p4_rear_maximum_powers', 'motor_p4_rear_rated_speed',
        'motor_p4_rear_speeds',
        'has_motor_p0', 'has_motor_p1', 'has_motor_p2_planetary',
        'has_motor_p2', 'has_motor_p3_front', 'has_motor_p3_rear',
        'has_motor_p4_front', 'has_motor_p4_rear', 'is_hybrid',
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
        'planetary_mean_efficiency', 'drive_battery_n_cells', 'planetary_ratio',
        'service_battery_initialization_time', 'drive_battery_n_parallel_cells',
        'initial_service_battery_state_of_charge', 'service_battery_capacity',
        'service_battery_electric_powers', 'service_battery_nominal_voltage',
        'dcdc_converter_electric_powers_demand', 'starter_nominal_voltage',
        'drive_battery_electric_powers', 'drive_battery_currents',
        'drive_battery_nominal_voltage', 'motors_electric_powers',
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='engine_model',
    dsp=_engine,
    inputs=(
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_powers_out', 'engine_speeds_out', 'stop_velocity', 'velocities',
        'initial_friction_params', 'min_engine_on_speed', 'fuel_carbon_content',
        'motor_p0_powers', 'motor_p1_powers', 'auxiliaries_torque_loss_factors',
        'idle_engine_speed', 'engine_coolant_temperatures', 'full_load_torques',
        'has_selective_catalytic_reduction', 'engine_has_cylinder_deactivation',
        'engine_has_variable_valve_actuation', 'max_engine_coolant_temperature',
        'engine_fuel_lower_heating_value', 'fuel_consumptions', 'gear_box_type',
        'engine_capacity', 'clutch_tc_speeds_delta', 'phases_integration_times',
        'engine_speed_at_max_power', 'obd_fuel_type_code', 'calibration_status',
        'co2_normalization_references', 'auxiliaries_torque_loss', 'co2_params',
        'idle_engine_speed_median', 'auxiliaries_power_loss', 'engine_is_turbo',
        'engine_max_power', 'clutch_tc_powers', 'full_load_speeds', 'is_hybrid',
        'engine_temperature_regression_model', 'has_lean_burn', 'accelerations',
        'full_load_powers', 'belt_efficiency', 'engine_stroke', 'ignition_type',
        'engine_speeds_out_hot', 'idle_engine_speed_std', 'gear_box_powers_out',
        'engine_n_cylinders', 'engine_max_speed', 'is_cycle_hot', 'engine_type',
        'has_exhausted_gas_recirculation', 'co2_params_calibrated', 'on_engine',
        'engine_idle_fuel_consumption', 'after_treatment_speeds_delta', 'times',
        'active_cylinder_ratios', 'alternator_powers', 'engine_mass', 'on_idle',
        'initial_engine_temperature', 'motor_p2_planetary_powers', 'fuel_type',
        'phases_co2_emissions', 'engine_max_torque', 'phases_distances', {
            'initial_temperature': 'initial_engine_temperature'
        }
    ),
    outputs=(
        'identified_co2_emissions', 'idle_engine_speed', 'has_sufficient_power',
        'idle_engine_speed_median', 'idle_engine_speed_std', 'full_load_powers',
        'engine_temperature_regression_model', 'engine_speeds_out', 'fuel_type',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'after_treatment_temperature_threshold', 'engine_type', 'ignition_type',
        'co2_params_calibrated', 'co2_params_initial_guess', 'active_cylinders',
        'initial_friction_params', 'engine_moment_inertia', 'engine_powers_out',
        'extended_phases_co2_emissions', 'gross_engine_powers_out', 'is_hybrid',
        'engine_inertia_powers_losses', 'fuel_consumptions', 'engine_max_speed',
        'engine_speed_at_max_power', 'belt_mean_efficiency', 'engine_max_power',
        'co2_rescaling_scores', 'auxiliaries_power_loss', 'co2_emissions_model',
        'auxiliaries_torque_losses', 'engine_heat_capacity', 'full_load_speeds',
        'active_exhausted_gas_recirculations', 'engine_temperature_derivatives',
        'initial_engine_temperature', 'engine_coolant_temperatures', 'fuel_map',
        'auxiliaries_torque_loss', 'auxiliaries_power_losses', 'missing_powers',
        'extended_phases_integration_times', 'mean_piston_speeds', 'fmep_model',
        'has_exhausted_gas_recirculation', 'active_lean_burns', 'co2_emissions',
        'engine_fuel_lower_heating_value', 'calibration_status', 'brake_powers',
        'engine_idle_fuel_consumption', 'active_variable_valves', 'engine_mass',
        'engine_speeds_out_hot', 'fuel_carbon_content', 'engine_max_torque',
        'max_engine_coolant_temperature', 'full_load_curve',
    ),
    inp_weight={'initial_temperature': 5}
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='control_model',
    dsp=_control,
    inputs=(
        'full_load_curve', 'gear_box_speeds_in', 'motor_p0_efficiency', 'gears',
        'motor_p3_front_maximum_power', 'motor_p3_rear_maximum_power', 'ecms_s',
        'engine_coolant_temperatures', 'engine_moment_inertia', 'gear_box_type',
        'final_drive_mean_efficiency', 'gear_box_mean_efficiency', 'velocities',
        'start_stop_hybrid_params', 'motor_p2_maximum_powers', 'has_start_stop',
        'drive_battery_state_of_charges', 'is_hybrid', 'motor_p2_maximum_power',
        'motor_p0_speed_ratio', 'motor_p2_efficiency', 'auxiliaries_power_loss',
        'start_stop_activation_time', 'motor_p0_maximum_power', 'starter_model',
        'is_cycle_hot', 'motor_p2_electric_powers', 'clutch_tc_mean_efficiency',
        'motor_p3_rear_efficiency', 'motor_p0_electric_powers', 'accelerations',
        'min_time_engine_on_after_start', 'motors_electric_powers', 'is_serial',
        'motor_p1_electric_powers', 'dcdc_converter_efficiency', 'hybrid_modes',
        'motor_p1_maximum_power_function', 'motor_p3_front_efficiency', 'times',
        'motor_p3_front_maximum_powers', 'auxiliaries_torque_loss', 'on_engine',
        'motor_p1_maximum_power', 'belt_mean_efficiency', 'drive_battery_model',
        'motor_p4_front_maximum_power', 'motor_p1_speed_ratio', 'motive_powers',
        'motor_p3_rear_maximum_powers', 'motor_p4_front_efficiency', 'fuel_map',
        'motor_p4_rear_efficiency', 'engine_speeds_out_hot', 'start_stop_model',
        'motor_p4_rear_maximum_power', 'engine_powers_out', 'idle_engine_speed',
        'correct_start_stop_with_gears', 'engine_speeds_out', 'planetary_ratio',
        'motor_p2_planetary_electric_powers', 'motor_p2_planetary_speed_ratio',
        'after_treatment_cooling_duration', 'after_treatment_warm_up_duration',
        'motor_p2_planetary_maximum_power_function', 'final_drive_speeds_in',
        'motor_p2_planetary_maximum_power', 'motor_p4_front_electric_powers',
        'motor_p0_maximum_power_function', 'motor_p3_front_electric_powers',
        'after_treatment_warm_up_phases', 'motor_p2_planetary_efficiency',
        'gear_box_mean_efficiency_guess', 'motor_p3_rear_electric_powers',
        'motor_p4_rear_electric_powers', 'engine_thermostat_temperature',
        'motor_p4_front_maximum_powers', 'motor_p4_rear_maximum_powers',
        'planetary_mean_efficiency', 'has_motor_p2_planetary',
        'motor_p1_efficiency',

    ),
    outputs=(
        'correct_start_stop_with_gears', 'start_stop_activation_time', 'ecms_s',
        'motor_p0_electric_powers', 'motor_p1_electric_powers', 'engine_starts',
        'engine_speeds_out_hot', 'start_stop_hybrid_params', 'start_stop_model',
        'motor_p3_front_electric_powers', 'engine_speeds_base', 'hybrid_modes',
        'motor_p2_planetary_electric_powers', 'motor_p3_rear_electric_powers',
        'motor_p4_front_electric_powers', 'after_treatment_warm_up_phases',
        'motor_p4_rear_electric_powers', 'motor_p2_electric_powers',
        'force_on_engine', 'on_engine',
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='after_treat_model',
    dsp=_after_treat,
    inputs=(
        'after_treatment_warm_up_phases', 'engine_speeds_out_hot', 'velocities',
        'after_treatment_speed_model', 'engine_speeds_out', 'engine_powers_out',
        'min_engine_on_speed', 'gear_box_speeds_in', 'stop_velocity', 'on_idle',
        'after_treatment_cooling_duration', 'after_treatment_warm_up_duration',
        'after_treatment_speeds_delta', 'after_treatment_power_model', 'times',
        'engine_speeds_base', 'idle_engine_speed', 'is_cycle_hot', 'on_engine',
        'gears', 'is_hybrid', 'engine_starts'
    ),
    outputs=(
        'after_treatment_power_model', 'after_treatment_speed_model', 'on_idle',
        'after_treatment_warm_up_duration', 'after_treatment_cooling_duration',
        'after_treatment_warm_up_phases', 'after_treatment_speeds_delta',
        'engine_speeds_base',
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='co2_model',
    dsp=_co2,
    inputs=(
        'co2_emission_extra_high', 'phases_co2_emissions', 'mean_piston_speeds',
        'co2_params_calibrated', 'enable_phases_willans', 'co2_emission_medium',
        'engine_fuel_lower_heating_value', 'co2_emission_low', 'missing_powers',
        'co2_emission_value', 'ki_multiplicative', 'co2_emission_high', 'times',
        'engine_coolant_temperatures', 'phases_integration_times', 'fmep_model',
        'engine_speeds_out', 'co2_emission_UDC', 'fuel_density', 'angle_slopes',
        'has_periodically_regenerating_systems', 'motive_powers', 'ki_additive',
        'min_engine_on_speed', 'fuel_carbon_content', 'velocities', 'fuel_type',
        'fuel_carbon_content_percentage', 'engine_powers_out', 'enable_willans',
        'speed_distance_correction', 'theoretical_velocities', 'rcb_correction',
        'theoretical_motive_powers', 'alternator_efficiency', 'engine_capacity',
        'co2_emission_EUDC', 'engine_max_power', 'engine_stroke', 'engine_type',
        'atct_family_correction_factor', 'cycle_type', 'distances', 'is_hybrid',
        'drive_battery_delta_state_of_charge', 'after_treatment_warm_up_phases',
        'drive_battery_nominal_voltage', 'drive_battery_capacity', 'is_plugin',
        'service_battery_electric_powers', 'force_on_engine', 'co2_emissions',
        'drive_battery_electric_powers', 'kco2_wltp_correction_factor',
        'kco2_nedc_correction_factor', 'accelerations',
    ),
    outputs=(
        'phases_willans_factors', 'phases_co2_emissions', 'fuel_carbon_content',
        'fuel_carbon_content_percentage', 'phases_distances', 'willans_factors',
        'declared_sustaining_co2_emission_value', 'kco2_wltp_correction_factor',
        'corrected_co2_emission_value', 'rcb_correction', 'fuel_heating_value',
        'declared_co2_emission_value', 'co2_emission_value', 'phases_indices',
        'phases_fuel_consumptions', 'optimal_efficiency', 'ki_multiplicative',
        'kco2_nedc_correction_factor', 'corrected_sustaining_co2_emission_value'
    )
)

try:
    from co2mpas_driver.co2mpas import plugin_physycal

    dsp = plugin_physycal(dsp)
except ImportError:
    pass
