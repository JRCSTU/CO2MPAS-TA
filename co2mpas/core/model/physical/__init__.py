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
    defaults
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

dsp = sh.BlueDispatcher(
    name='CO2MPAS physical model',
    description='Wraps all functions needed to calibrate and predict '
                'light-vehicles\' CO2 emissions.'
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='cycle_model',
    dsp=_cycle,
    inputs=(
        'accelerations', 'bag_phases', 'climbing_force', 'cycle_type',
        'downscale_factor', 'downscale_factor_threshold', 'downscale_phases',
        'engine_max_power', 'engine_max_speed', 'engine_speed_at_max_power',
        'full_load_curve', 'gear_box_type', 'gears', 'idle_engine_speed',
        'inertial_factor', 'k1', 'k2', 'k5', 'max_gear', 'wltp_class',
        'max_speed_velocity_ratio', 'max_time', 'max_velocity', 'motive_powers',
        'road_loads', 'speed_velocity_ratios', 'time_sample_frequency', 'times',
        'unladen_mass', 'vehicle_mass', 'velocities', 'wltp_base_model',
    ),
    outputs=(
        'gears', 'initial_temperature', 'phases_integration_times', 'times',
        'velocities'
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
        'vehicle_height', 'vehicle_mass', 'vehicle_width', 'velocities'
    ),
    outputs=(
        'accelerations', 'angle_slopes', 'climbing_force', 'curb_mass',
        'distances', 'f0', 'f1', 'f2', 'inertial_factor', 'motive_powers',
        'n_dyno_axes', 'road_loads', 'unladen_mass', 'vehicle_mass',
        'velocities'
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
        'stop_velocity', 'times', 'velocities', 'tyre_dimensions',
        'tyre_dynamic_rolling_coefficient', 'tyre_code', 'velocity_speed_ratios'
    ),
    outputs=(
        'r_dynamic', 'r_wheels', 'tyre_code', 'wheel_powers', 'wheel_speeds',
        'tyre_dynamic_rolling_coefficient', 'wheel_torques',
        'wheels_prediction_model'
    ),
    inp_weight={'r_dynamic': 3}
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='final_drive_model',
    dsp=_final_drive,
    inputs=(
        'final_drive_efficiency', 'final_drive_ratio', 'final_drive_ratios',
        'final_drive_torque_loss', 'gear_box_type', 'gears', 'n_dyno_axes',
        'n_wheel_drive', 'n_gears', {
            'wheel_powers': 'final_drive_powers_out',
            'wheel_speeds': 'final_drive_speeds_out',
            'wheel_torques': 'final_drive_torques_out'
        }
    ),
    outputs=(
        'final_drive_powers_in', 'final_drive_ratios', 'final_drive_speeds_in',
        'final_drive_torques_in', 'final_drive_prediction_model'
    )
)

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
        'has_gear_box_thermal_management', 'has_torque_converter',
        'idle_engine_speed', 'initial_gear_box_temperature',
        'last_gear_box_ratio', 'max_velocity_full_load_correction',
        'maximum_vehicle_laden_mass', 'maximum_velocity', 'min_engine_on_speed',
        'motive_powers', 'n_gears', 'on_engine', 'plateau_acceleration',
        'r_dynamic', 'road_loads', 'specific_gear_shifting', 'stop_velocity',
        'time_cold_hot_transition', 'times', 'use_dt_gear_shifting',
        'velocities', 'velocity_speed_ratios', {
            'final_drive_powers_in': 'gear_box_powers_out',
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

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_clutch_torque_converter,
    dsp_id='clutch_torque_converter_model',
    inputs=(
        'accelerations', 'calibration_tc_speed_threshold', 'clutch_model',
        'clutch_window', 'cold_start_speeds_delta', 'engine_speeds_out',
        'engine_speeds_out_hot', 'gear_box_powers_in', 'gear_box_speeds_in',
        'gear_box_type', 'gear_shifts', 'gears', 'has_torque_converter',
        'lock_up_tc_limits', 'lockup_speed_ratio', 'stand_still_torque_ratio',
        'stop_velocity', 'times', 'torque_converter_model', 'velocities'
    ),
    outputs=(
        'clutch_model', 'clutch_phases', 'clutch_tc_powers', 'clutch_window',
        'clutch_tc_speeds_delta', 'has_torque_converter', 'lockup_speed_ratio',
        'stand_still_torque_ratio', 'torque_converter_model'
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='electric_model',
    dsp=_electrics,
    inputs=(
        'accelerations', 'alternator_charging_currents', 'stop_velocity',
        'alternator_current_model', 'alternator_currents', 'state_of_charges',
        'alternator_efficiency', 'alternator_initialization_time',
        'alternator_nominal_power', 'alternator_nominal_voltage', 'velocities',
        'alternator_off_threshold', 'alternator_start_window_width',
        'alternator_status_model', 'alternator_statuses', 'battery_capacity',
        'battery_currents', 'delta_time_engine_starter', 'start_demand',
        'electric_load', 'engine_moment_inertia', 'engine_starts', 'cycle_type',
        'gear_box_powers_in', 'has_energy_recuperation', 'idle_engine_speed',
        'initial_state_of_charge', 'max_battery_charging_current', 'times',
        'state_of_charge_balance', 'state_of_charge_balance_window', 'on_engine'
    ),
    outputs=(
        'alternator_current_model', 'alternator_currents', 'start_demand',
        'alternator_initialization_time', 'alternator_nominal_power',
        'alternator_powers_demand', 'alternator_status_model', 'electric_load',
        'alternator_statuses', 'battery_currents', 'initial_state_of_charge',
        'max_battery_charging_current', 'state_of_charge_balance_window',
        'state_of_charge_balance', 'state_of_charges', 'delta_state_of_charge',
        'electrics_prediction_model'
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp_id='engine_model',
    dsp=_engine,
    inputs=(
        'accelerations', 'active_cylinder_ratios', 'alternator_powers_demand',
        'angle_slopes', 'auxiliaries_power_loss', 'auxiliaries_torque_loss',
        'calibration_status', 'clutch_tc_powers', 'clutch_tc_speeds_delta',
        'co2_emission_extra_high', 'co2_emission_high', 'co2_emission_low',
        'co2_emission_medium', 'co2_normalization_references', 'co2_params',
        'co2_params_calibrated', 'cold_start_speed_model', 'has_start_stop',
        'correct_start_stop_with_gears', 'enable_phases_willans', 'on_engine',
        'enable_willans', 'engine_capacity', 'engine_coolant_temperatures',
        'engine_fuel_lower_heating_value', 'engine_has_cylinder_deactivation',
        'engine_has_variable_valve_actuation', 'engine_idle_fuel_consumption',
        'engine_is_turbo', 'engine_mass', 'engine_max_power', 'gear_box_type',
        'engine_max_speed', 'engine_speed_at_max_power', 'engine_max_torque',
        'engine_powers_out', 'engine_speeds_out', 'stop_velocity', 'velocities',
        'engine_starts', 'engine_stroke', 'engine_temperature_regression_model',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_type', 'final_drive_powers_in', 'fuel_carbon_content', 'gears',
        'fuel_carbon_content_percentage', 'fuel_consumptions', 'fuel_density',
        'fuel_type', 'full_load_powers', 'is_hybrid', 'engine_speeds_out_hot',
        'has_exhausted_gas_recirculation', 'gear_box_speeds_in', 'is_cycle_hot',
        'has_lean_burn', 'has_periodically_regenerating_systems', 'ki_additive',
        'has_selective_catalytic_reduction', 'max_engine_coolant_temperature',
        'idle_engine_speed', 'idle_engine_speed_median', 'full_load_torques',
        'idle_engine_speed_std', 'ignition_type', 'initial_engine_temperature',
        'initial_friction_params', 'ki_multiplicative', 'use_basic_start_stop',
        'obd_fuel_type_code', 'min_engine_on_speed', 'phases_integration_times',
        'min_time_engine_on_after_start', 'motive_powers', 'engine_n_cylinders',
        'times', 'on_idle', 'full_load_speeds', 'start_stop_activation_time',
        'start_stop_model', 'state_of_charges', {
            'initial_temperature': 'initial_engine_temperature'
        }),
    outputs=(
        'active_cylinders', 'active_exhausted_gas_recirculations',
        'active_lean_burns', 'active_variable_valves', 'engine_speeds_out_hot',
        'after_treatment_temperature_threshold', 'auxiliaries_power_losses',
        'auxiliaries_torque_losses', 'brake_powers', 'calibration_status',
        'co2_emission_value', 'co2_emissions', 'co2_emissions_model',
        'co2_error_function_on_emissions', 'co2_error_function_on_phases',
        'co2_params_calibrated', 'co2_params_initial_guess', 'ignition_type',
        'cold_start_speed_model', 'cold_start_speeds_delta', 'engine_starts',
        'cold_start_speeds_phases', 'correct_start_stop_with_gears',
        'declared_co2_emission_value', 'engine_coolant_temperatures',
        'engine_fuel_lower_heating_value', 'engine_heat_capacity',
        'engine_idle_fuel_consumption', 'engine_mass', 'engine_max_power',
        'engine_max_speed', 'engine_speed_at_max_power', 'engine_max_torque',
        'engine_moment_inertia', 'engine_powers_out', 'engine_speeds_out',
        'engine_temperature_derivatives', 'start_stop_activation_time',
        'engine_temperature_regression_model', 'fuel_type', 'is_hybrid',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_type', 'extended_phases_co2_emissions', 'fuel_carbon_content',
        'extended_phases_integration_times', 'initial_engine_temperature',
        'fuel_carbon_content_percentage', 'fuel_consumptions', 'fuel_density',
        'has_exhausted_gas_recirculation', 'full_load_curve', 'willans_factors',
        'identified_co2_emissions', 'idle_engine_speed', 'has_sufficient_power',
        'idle_engine_speed_median', 'idle_engine_speed_std', 'full_load_powers',
        'ki_multiplicative', 'max_engine_coolant_temperature', 'missing_powers',
        'on_engine', 'on_idle', 'optimal_efficiency', 'phases_co2_emissions',
        'phases_fuel_consumptions', 'phases_willans_factors', 'ki_additive',
        'start_stop_model', 'use_basic_start_stop', 'initial_friction_params',
        'co2_rescaling_scores', 'engine_prediction_model', 'full_load_speeds'
    ),
    inp_weight={'initial_temperature': 5}
)

OUTPUTS_PREDICTION_LOOP = [
    'alternator_currents',
    'alternator_statuses',
    'battery_currents',
    'state_of_charges',

    'engine_coolant_temperatures',
    'engine_speeds_out_hot',
    'on_engine',
    'engine_starts',

    'gear_box_efficiencies',
    'gear_box_powers_in',
    'gear_box_speeds_in',
    'gear_box_temperatures',
    'gear_box_torques_in',
    'gears',

    'final_drive_speeds_in',
    'final_drive_torques_in',
    'final_drive_powers_in',

    'wheel_powers',
    'wheel_speeds',
    'wheel_torques'
]


@sh.add_function(dsp, outputs=OUTPUTS_PREDICTION_LOOP, weight=10)
def prediction_loop(
        wheels_prediction_model, final_drive_prediction_model,
        gear_box_prediction_model, engine_prediction_model,
        electrics_prediction_model, times, velocities, accelerations,
        motive_powers):
    """
    Predicts vehicle time-series.

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

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Gear vector [-].
    :type motive_powers: numpy.array

    :return:
        Vehicle time-series
    :rtype: tuple[numpy.array]
    """
    outputs = {}
    n = times.shape[0]
    wheels_prediction_model.set_outputs(n, outputs)
    final_drive_prediction_model.set_outputs(n, outputs)
    gear_box_prediction_model.set_outputs(n, outputs)
    engine_prediction_model.set_outputs(n, outputs)
    electrics_prediction_model.set_outputs(n, outputs)

    whl = wheels_prediction_model.yield_results(velocities, motive_powers)

    fd = final_drive_prediction_model.yield_results(
        outputs['gears'], outputs['wheel_speeds'], outputs['wheel_torques'],
        outputs['wheel_powers']
    )

    gb = gear_box_prediction_model.yield_results(
        times, velocities, accelerations, motive_powers,
        outputs['final_drive_speeds_in'], outputs['final_drive_powers_in']
    )

    eng = engine_prediction_model.yield_results(
        times, velocities, accelerations, outputs['final_drive_powers_in'],
        outputs['gears'], outputs['gear_box_speeds_in']
    )

    ele = electrics_prediction_model.yield_results(
        times, accelerations, outputs['on_engine'], outputs['engine_starts'],
        outputs['gear_box_powers_in']
    )

    for _ in zip(whl, fd, gb, eng, ele):
        pass

    return sh.selector(OUTPUTS_PREDICTION_LOOP, outputs, output_type='list')
