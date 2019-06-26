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
        'accelerations', 'bag_phases', 'cycle_type', 'downscale_factor',
        'downscale_factor_threshold', 'downscale_phases', 'engine_max_power',
        'engine_max_speed', 'engine_speed_at_max_power', 'full_load_curve',
        'gear_box_type', 'gears', 'idle_engine_speed', 'inertial_factor', 'k1',
        'k2', 'k5', 'max_gear', 'wltp_class', 'max_speed_velocity_ratio',
        'max_time', 'max_velocity', 'motive_powers', 'road_loads',
        'speed_velocity_ratios', 'time_sample_frequency', 'times',
        'unladen_mass', 'vehicle_mass', 'velocities', 'wltp_base_model',
        'use_driver', 'path_velocities', 'path_distances', 'static_friction',
        'wheel_drive_load_fraction', 'distances', 'auxiliaries_power_loss',
        'auxiliaries_torque_loss', 'maximum_velocity', 'engine_moment_inertia',
        'driver_style_ratio', 'driver_style'
    ),
    outputs=(
        'gears', 'initial_temperature', 'phases_integration_times', 'times',
        'velocities', 'driver_prediction_model', 'desired_velocities'
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='vehicle_model',
    dsp=_vehicle,
    inputs=(
        'aerodynamic_drag_coefficient', 'air_density', 'angle_slope',
        'angle_slopes', 'cargo_mass', 'correct_f0', 'cycle_type', 'curb_mass',
        'elevations', 'f0', 'f0_uncorrected', 'f1', 'f2', 'frontal_area',
        'fuel_mass', 'has_roof_box', 'inertial_factor', 'n_dyno_axes',
        'n_passengers', 'n_wheel_drive', 'obd_velocities', 'passenger_mass',
        'road_loads', 'rolling_resistance_coeff', 'times', 'tyre_category',
        'tyre_class', 'unladen_mass', 'vehicle_body', 'vehicle_category',
        'vehicle_height', 'vehicle_mass', 'vehicle_width', 'velocities',
        'traction_acceleration_limit', 'traction_deceleration_limit',
        'wheel_drive_load_fraction', 'n_wheel', 'tyre_state', 'road_state',
        'static_friction', 'initial_velocity'
    ),
    outputs=(
        'accelerations', 'angle_slopes', 'curb_mass', 'distances', 'f0', 'f1',
        'f2', 'inertial_factor', 'motive_powers', 'n_dyno_axes', 'road_loads',
        'unladen_mass', 'vehicle_mass', 'velocities', 'static_friction',
        'vehicle_prediction_model', 'wheel_drive_load_fraction',
        'traction_acceleration_limit', 'traction_deceleration_limit'
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
        'wheels_prediction_model'
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
        'final_drive_torques_in', 'final_drive_prediction_model'
    )
)


@sh.add_function(dsp, outputs=['gear_box_powers_out'])
def calculate_gear_box_powers_out(final_drive_powers_in, motor_p3_powers):
    """
    Calculate gear box power vector [kW].

    :param final_drive_powers_in:
        Final drive power in [kW].
    :type final_drive_powers_in: numpy.array | float

    :param motor_p3_powers:
        Power at motor P3 [kW].
    :type motor_p3_powers: numpy.array | float

    :return:
        Gear box power vector [kW].
    :rtype: numpy.array | float
    """
    return final_drive_powers_in - motor_p3_powers


dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='gear_box_model',
    dsp=_gear_box,
    inputs=(
        'CMV', 'CMV_Cold_Hot', 'CVT', 'DTGS', 'GSPV', 'GSPV_Cold_Hot', 'MVL',
        'accelerations', 'change_gear_window_width', 'cycle_type',
        'engine_coolant_temperatures', 'engine_mass', 'engine_max_power',
        'engine_max_speed', 'engine_max_torque', 'engine_speed_at_max_power',
        'engine_speed_at_max_velocity', 'engine_speeds_out',
        'engine_thermostat_temperature', 'f0', 'final_drive_ratios',
        'first_gear_box_ratio', 'fuel_saving_at_strategy', 'full_load_curve',
        'full_load_speeds', 'gear_box_efficiency_constants',
        'gear_box_efficiency_parameters_cold_hot', 'gear_box_ratios',
        'gear_box_temperature_references', 'gear_box_type', 'gears',
        'has_gear_box_thermal_management', 'has_torque_converter', 'velocities',
        'idle_engine_speed', 'initial_gear_box_temperature',
        'last_gear_box_ratio', 'max_velocity_full_load_correction',
        'maximum_vehicle_laden_mass', 'maximum_velocity', 'min_engine_on_speed',
        'motive_powers', 'n_gears', 'on_engine', 'plateau_acceleration',
        'r_dynamic', 'road_loads', 'specific_gear_shifting', 'stop_velocity',
        'time_cold_hot_transition', 'times', 'use_dt_gear_shifting',
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
        'speed_velocity_ratios', 'gear_box_prediction_model', 'n_gears'
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
        'accelerations', 'clutch_speed_model',
        'clutch_window', 'cold_start_speeds_delta', 'engine_speeds_out',
        'engine_speeds_out_hot', 'clutch_tc_powers_out', 'gear_box_speeds_in',
        'gear_box_type', 'gear_shifts', 'gears', 'has_torque_converter',
        'lockup_speed_ratio', 'stand_still_torque_ratio', 'engine_max_speed',
        'stop_velocity', 'times', 'torque_converter_speed_model', 'velocities',
        'm1000_curve_factor', 'm1000_curve_ratios', 'm1000_curve_norm_torques',
        'full_load_curve', 'gear_box_torques_in', 'idle_engine_speed',
    ),
    outputs=(
        'clutch_speed_model', 'clutch_phases', 'clutch_tc_powers',
        'clutch_window', 'clutch_tc_speeds_delta', 'has_torque_converter',
        'lockup_speed_ratio', 'stand_still_torque_ratio', 'm1000_curve_factor',
        'torque_converter_speed_model', 'clutch_tc_prediction_model',
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='electric_model',
    dsp=_electrics,
    inputs=(
        'accelerations', 'alternator_charging_currents', 'alternator_currents',
        'alternator_efficiency', 'alternator_electric_powers', 'engine_starts',
        'alternator_current_model', 'alternator_current_threshold',
        'alternator_initialization_time', 'alternator_nominal_voltage',
        'alternator_off_threshold', 'alternator_start_window_width',
        'alternator_status_model', 'alternator_statuses', 'engine_speeds_out',
        'delta_time_engine_starter', 'engine_moment_inertia', 'on_engine',
        'final_drive_speeds_in', 'gear_box_powers_in', 'gear_box_speeds_in',
        'motor_p0_efficiency', 'motor_p0_electric_power_loss_function',
        'motor_p0_electric_powers', 'motor_p0_loss_param_a', 'stop_velocity',
        'motor_p0_loss_param_b', 'motor_p0_powers', 'motor_p0_speed_ratio',
        'motor_p0_speeds', 'motor_p0_torques', 'motor_p1_efficiency',
        'motor_p1_electric_power_loss_function', 'motor_p1_electric_powers',
        'motor_p1_loss_param_a', 'motor_p1_loss_param_b', 'motor_p1_powers',
        'motor_p1_speed_ratio', 'motor_p1_speeds', 'motor_p1_torques',
        'motor_p2_efficiency', 'motor_p2_electric_power_loss_function',
        'motor_p2_electric_powers', 'motor_p2_loss_param_a', 'motor_p4_speeds',
        'motor_p2_loss_param_b', 'motor_p2_powers', 'motor_p2_speed_ratio',
        'motor_p2_speeds', 'motor_p2_torques', 'motor_p3_efficiency',
        'motor_p3_electric_power_loss_function', 'motor_p3_electric_powers',
        'motor_p3_loss_param_a', 'motor_p3_loss_param_b', 'motor_p3_powers',
        'motor_p3_speed_ratio', 'motor_p3_speeds', 'motor_p3_torques', 'times',
        'motor_p4_efficiency', 'motor_p4_electric_power_loss_function',
        'motor_p4_electric_powers', 'motor_p4_loss_param_a', 'motor_p4_torques',
        'motor_p4_loss_param_b', 'motor_p4_powers', 'motor_p4_speed_ratio',
        'service_battery_state_of_charge_balance', 'velocities', 'wheel_speeds',
        'service_battery_state_of_charge_balance_window', 'starter_efficiency',
        'service_battery_state_of_charges', 'starter_electric_powers',
        'dcdc_converter_efficiency', 'dcdc_converter_electric_powers',
        'drive_battery_capacity', 'drive_battery_currents', 'cycle_type',
        'drive_battery_electric_powers', 'drive_battery_load',
        'drive_battery_loads', 'drive_battery_n_parallel_cells',
        'drive_battery_n_series_cells', 'drive_battery_ocv', 'drive_battery_r0',
        'drive_battery_state_of_charges', 'drive_battery_voltages',
        'electrical_hybridization_degree', 'engine_powers_out',
        'service_battery_capacity', 'initial_drive_battery_state_of_charge',
        'service_battery_electric_powers', 'service_battery_nominal_voltage',
        'service_battery_load', 'initial_service_battery_state_of_charge',
        'service_battery_loads', 'service_battery_currents',
        'dcdc_converter_currents'
    ),
    outputs=(
        'alternator_current_model', 'alternator_current_threshold',
        'alternator_currents', 'alternator_electric_powers', 'motor_p2_torques',
        'alternator_initialization_time', 'alternator_status_model',
        'alternator_statuses', 'motor_p0_electric_power_loss_function',
        'motor_p0_electric_powers', 'motor_p0_powers', 'motor_p0_speed_ratio',
        'motor_p1_electric_power_loss_function', 'motor_p1_electric_powers',
        'motor_p1_powers', 'motor_p1_speed_ratio', 'motor_p1_speeds',
        'motor_p1_torques', 'motor_p2_electric_power_loss_function',
        'motor_p2_electric_powers', 'motor_p2_powers', 'motor_p2_speed_ratio',
        'motor_p3_electric_power_loss_function', 'motor_p3_electric_powers',
        'motor_p3_powers', 'motor_p3_speed_ratio', 'motor_p3_speeds',
        'motor_p3_torques', 'motor_p4_electric_power_loss_function',
        'motor_p4_electric_powers', 'motor_p4_powers', 'motor_p4_speed_ratio',
        'service_battery_state_of_charge_balance', 'motor_p2_speeds',
        'service_battery_state_of_charge_balance_window', 'motor_p0_torques',
        'service_battery_state_of_charges', 'alternator_powers',
        'motor_p0_efficiency_ratios', 'motor_p1_efficiency_ratios',
        'motor_p2_efficiency_ratios', 'motor_p3_efficiency_ratios',
        'motor_p4_efficiency_ratios', 'starter_electric_powers',
        'starter_powers', 'drive_battery_capacity', 'drive_battery_currents',
        'drive_battery_electric_powers', 'drive_battery_load',
        'drive_battery_loads', 'drive_battery_n_parallel_cells',
        'drive_battery_n_series_cells', 'drive_battery_ocv', 'drive_battery_r0',
        'drive_battery_state_of_charges', 'drive_battery_voltages',
        'initial_drive_battery_state_of_charge', 'motor_p4_torques',
        'initial_service_battery_state_of_charge', 'service_battery_capacity',
        'service_battery_currents', 'service_battery_electric_powers',
        'dcdc_converter_electric_powers_demand', 'service_battery_loads',
        'drive_battery_delta_state_of_charge', 'motor_p4_speeds',
        'minimum_drive_battery_electric_power', 'service_battery_load',
        'service_battery_delta_state_of_charge', 'motor_p0_speeds',
        'engine_speeds_out', 'wheel_speeds', 'final_drive_speeds_in',
        'gear_box_speeds_in'
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='engine_model',
    dsp=_engine,
    inputs=(
        'accelerations', 'active_cylinder_ratios', 'alternator_powers',
        'angle_slopes', 'auxiliaries_power_loss', 'auxiliaries_torque_loss',
        'calibration_status', 'clutch_tc_powers', 'clutch_tc_speeds_delta',
        'co2_emission_extra_high', 'co2_emission_high', 'co2_emission_low',
        'co2_emission_medium', 'co2_normalization_references', 'co2_params',
        'co2_params_calibrated', 'cold_start_speed_model', 'enable_willans',
        'engine_capacity', 'engine_coolant_temperatures', 'full_load_powers',
        'engine_fuel_lower_heating_value', 'engine_has_cylinder_deactivation',
        'engine_has_variable_valve_actuation', 'engine_idle_fuel_consumption',
        'engine_is_turbo', 'engine_mass', 'engine_max_power', 'gear_box_type',
        'engine_max_speed', 'engine_speed_at_max_power', 'engine_max_torque',
        'engine_powers_out', 'engine_speeds_out', 'stop_velocity', 'velocities',
        'engine_stroke', 'engine_temperature_regression_model', 'on_engine',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_type', 'final_drive_powers_in', 'fuel_carbon_content', 'gears',
        'fuel_carbon_content_percentage', 'fuel_consumptions', 'fuel_density',
        'fuel_type', 'engine_speeds_out_hot', 'enable_phases_willans',
        'has_exhausted_gas_recirculation', 'gear_box_speeds_in', 'is_cycle_hot',
        'has_lean_burn', 'has_periodically_regenerating_systems', 'ki_additive',
        'has_selective_catalytic_reduction', 'max_engine_coolant_temperature',
        'idle_engine_speed', 'idle_engine_speed_median', 'full_load_torques',
        'idle_engine_speed_std', 'ignition_type', 'initial_engine_temperature',
        'initial_friction_params', 'ki_multiplicative', 'obd_fuel_type_code',
        'min_engine_on_speed', 'phases_integration_times', 'motive_powers',
        'motor_p0_powers', 'motor_p1_powers', 'auxiliaries_torque_loss_factors',
        'engine_n_cylinders', 'times', 'on_idle', 'full_load_speeds', {
            'initial_temperature': 'initial_engine_temperature'
        }
    ),
    outputs=(
        'active_exhausted_gas_recirculations', 'initial_friction_params',
        'active_lean_burns', 'active_variable_valves', 'engine_speeds_out_hot',
        'after_treatment_temperature_threshold', 'auxiliaries_power_losses',
        'auxiliaries_torque_losses', 'brake_powers', 'calibration_status',
        'co2_emission_value', 'co2_emissions_model', 'co2_rescaling_scores',
        'co2_error_function_on_emissions', 'co2_error_function_on_phases',
        'co2_params_calibrated', 'co2_params_initial_guess', 'ignition_type',
        'cold_start_speed_model', 'cold_start_speeds_delta', 'co2_emissions',
        'cold_start_speeds_phases', 'declared_co2_emission_value', 'is_hybrid',
        'engine_coolant_temperatures', 'engine_fuel_lower_heating_value',
        'engine_heat_capacity', 'engine_idle_fuel_consumption', 'engine_mass',
        'engine_max_power', 'engine_max_speed', 'engine_speed_at_max_power',
        'engine_max_torque', 'engine_moment_inertia', 'engine_powers_out',
        'engine_speeds_out', 'engine_temperature_derivatives', 'fuel_type',
        'engine_temperature_regression_model', 'auxiliaries_torque_loss',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_type', 'extended_phases_co2_emissions', 'fuel_carbon_content',
        'extended_phases_integration_times', 'initial_engine_temperature',
        'fuel_carbon_content_percentage', 'fuel_consumptions', 'fuel_density',
        'has_exhausted_gas_recirculation', 'full_load_curve', 'willans_factors',
        'identified_co2_emissions', 'idle_engine_speed', 'has_sufficient_power',
        'idle_engine_speed_median', 'idle_engine_speed_std', 'full_load_powers',
        'ki_multiplicative', 'max_engine_coolant_temperature', 'missing_powers',
        'on_idle', 'optimal_efficiency', 'phases_co2_emissions', 'ki_additive',
        'engine_prediction_model', 'full_load_speeds', 'auxiliaries_power_loss',
        'phases_fuel_consumptions', 'phases_willans_factors', 'active_cylinders'
    ),
    inp_weight={'initial_temperature': 5}
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='control_model',
    dsp=_control,
    inputs=(
        'times', 'engine_speeds_out', 'idle_engine_speed', 'velocities',
        'min_time_engine_on_after_start', 'accelerations', 'has_start_stop',
        'correct_start_stop_with_gears', 'start_stop_model', 'on_engine',
        'engine_coolant_temperatures', 'use_basic_start_stop', 'is_hybrid',
        'gears', 'gear_box_type', 'start_stop_activation_time'
    ),
    outputs=('on_engine', 'engine_starts')
)

OUTPUTS_PREDICTION_LOOP = [
    'times',
    'accelerations',

    'velocities',
    'distances',
    'angle_slopes',
    'motive_powers',

    'wheel_torques',
    'wheel_speeds',
    'wheel_powers',

    'final_drive_powers_in',
    'final_drive_torques_in',
    'final_drive_speeds_in',

    'gears',
    'gear_box_torques_in',
    'gear_box_temperatures',
    'gear_box_speeds_in',
    'gear_box_powers_in',
    'gear_box_efficiencies',

    'engine_starts',
    'on_engine',
    'engine_speeds_out_hot',
    'engine_coolant_temperatures',

    'alternator_currents',
    'alternator_statuses',
    'alternator_powers',
    'battery_currents',
    'state_of_charges',

    'clutch_phases',
    'clutch_tc_speeds_delta',
    'clutch_tc_powers'
]


@sh.add_function(dsp, outputs=OUTPUTS_PREDICTION_LOOP, weight=10)
def prediction_loop(
        driver_prediction_model, vehicle_prediction_model,
        wheels_prediction_model, final_drive_prediction_model,
        gear_box_prediction_model, engine_prediction_model,
        electrics_prediction_model, clutch_tc_prediction_model):
    """
    Predicts vehicle time-series.

    :param driver_prediction_model:
        Driver prediction model.
    :type driver_prediction_model: .driver.DriverModel

    :param vehicle_prediction_model:
        Vehicle prediction model.
    :type vehicle_prediction_model: .vehicle.VehicleModel

    :param wheels_prediction_model:
        Wheels prediction model.
    :type wheels_prediction_model: .wheels.WheelsModel

    :param final_drive_prediction_model:
        Final drive prediction model.
    :type final_drive_prediction_model: .final_drive.FinalDriveModel

    :param gear_box_prediction_model:
        Gear box prediction model.
    :type gear_box_prediction_model: .gear_box.GearBoxModel

    :param engine_prediction_model:
        Engine prediction model.
    :type engine_prediction_model: .engine.EngineModel

    :param electrics_prediction_model:
        Electrics prediction model.
    :type electrics_prediction_model: .electrics.ElectricModel

    :param clutch_tc_prediction_model:
        Clutch or torque converter prediction model.
    :type clutch_tc_prediction_model: .clutch_tc.ClutchTCModel

    :return:
        Vehicle time-series
    :rtype: tuple[numpy.array]
    """
    outputs = {}
    driver_prediction_model.set_outputs(outputs)
    vehicle_prediction_model.set_outputs(outputs)
    wheels_prediction_model.set_outputs(outputs)
    final_drive_prediction_model.set_outputs(outputs)
    gear_box_prediction_model.set_outputs(outputs)
    engine_prediction_model.set_outputs(outputs)
    electrics_prediction_model.set_outputs(outputs)
    clutch_tc_prediction_model.set_outputs(outputs)

    vhl = vehicle_prediction_model.init_results(
        outputs['times'], outputs['accelerations']
    )

    whl = wheels_prediction_model.init_results(
        outputs['velocities'], outputs['motive_powers']
    )

    fd = final_drive_prediction_model.init_results(
        outputs['gears'], outputs['wheel_speeds'], outputs['wheel_torques'],
        outputs['wheel_powers']
    )

    gb = gear_box_prediction_model.init_results(
        outputs['times'], outputs['velocities'], outputs['accelerations'],
        outputs['motive_powers'], outputs['engine_coolant_temperatures'],
        outputs['final_drive_speeds_in'], outputs['final_drive_powers_in']
    )

    eng = engine_prediction_model.init_results(
        outputs['times'], outputs['velocities'], outputs['accelerations'],
        outputs['state_of_charges'], outputs['final_drive_powers_in'],
        outputs['gears'], outputs['gear_box_speeds_in']
    )

    ele = electrics_prediction_model.init_results(
        outputs['times'], outputs['accelerations'], outputs['on_engine'],
        outputs['engine_starts'], outputs['gear_box_powers_in']
    )

    ctc = clutch_tc_prediction_model.init_results(
        outputs['accelerations'], outputs['velocities'],
        outputs['gear_box_speeds_in'], outputs['gears'], outputs['times'],
        outputs['gear_box_powers_in'], outputs['engine_speeds_out_hot'],
        outputs['gear_box_torques_in']
    )

    for _ in driver_prediction_model.yield_results(
            vhl, whl, fd, gb, eng, ele, ctc):
        pass

    driver_prediction_model.format_results()
    vehicle_prediction_model.format_results()
    wheels_prediction_model.format_results()
    final_drive_prediction_model.format_results()
    gear_box_prediction_model.format_results()
    engine_prediction_model.format_results()
    electrics_prediction_model.format_results()
    clutch_tc_prediction_model.format_results()

    return sh.selector(OUTPUTS_PREDICTION_LOOP, outputs, output_type='list')
