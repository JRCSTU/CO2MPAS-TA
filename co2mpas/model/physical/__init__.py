# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It provides CO2MPAS model to predict light-vehicles' CO2 emissions.

Docstrings should provide sufficient understanding for any individual function.

Modules:

.. currentmodule:: co2mpas.model.physical

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
import numpy as np
import functools
import co2mpas.utils as co2_utl


def predict_vehicle_electrics_and_engine_behavior(
        electrics_model, start_stop_model, engine_temperature_regression_model,
        initial_engine_temperature, initial_state_of_charge, idle_engine_speed,
        times, final_drive_powers_in, gear_box_speeds_in, gear_box_powers_in,
        velocities, accelerations, gears, start_stop_activation_time,
        correct_start_stop_with_gears, min_time_engine_on_after_start,
        has_start_stop, use_basic_start_stop, max_engine_coolant_temperature):
    """
    Predicts alternator and battery currents, state of charge, alternator
    status, if the engine is on and when the engine starts, the engine speed at
    hot condition, and the engine coolant temperature.

    :param electrics_model:
        Electrics model.
    :type electrics_model: callable

    :param start_stop_model:
        Start/stop model.
    :type start_stop_model: StartStopModel

    :param engine_temperature_regression_model:
        The calibrated engine temperature regression model.
    :type engine_temperature_regression_model: ThermalModel

    :param initial_engine_temperature:
        Engine initial temperature [°C]
    :type initial_engine_temperature: float

    :param initial_state_of_charge:
        Initial state of charge of the battery [%].

        .. note::

            `initial_state_of_charge` = 99 is equivalent to 99%.
    :type initial_state_of_charge: float

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param final_drive_powers_in:
        Final drive power in [kW].
    :type final_drive_powers_in: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param start_stop_activation_time:
        Start-stop activation time threshold [s].
    :type start_stop_activation_time: float

    :param correct_start_stop_with_gears:
        A flag to impose engine on when there is a gear > 0.
    :type correct_start_stop_with_gears: bool

    :param min_time_engine_on_after_start:
        Minimum time of engine on after a start [s].
    :type min_time_engine_on_after_start: float

    :param has_start_stop:
        Does the vehicle have start/stop system?
    :type has_start_stop: bool

    :param use_basic_start_stop:
        If True the basic start stop model is applied, otherwise complex one.

        ..note:: The basic start stop model is function of velocity and
          acceleration. While, the complex model is function of velocity,
          acceleration, temperature, and battery state of charge.
    :type use_basic_start_stop: bool

    :param max_engine_coolant_temperature:
        Maximum engine coolant temperature [°C].
    :type max_engine_coolant_temperature: float

    :return:
        Alternator and battery currents, state of charge, alternator status,
        if the engine is on and when the engine starts, the engine speed at hot
        condition, and the engine coolant temperature.
        [A, A, %, -, -, -, RPM, °C].
    :rtype: tuple[numpy.array]
    """
    n = len(times)
    soc, temp = np.zeros((2, n), dtype=float)
    soc[0], temp[0] = initial_state_of_charge, initial_engine_temperature

    gen = start_stop_model.yield_on_start(
        times, velocities, accelerations, temp, soc,
        gears=gears, start_stop_activation_time=start_stop_activation_time,
        correct_start_stop_with_gears=correct_start_stop_with_gears,
        min_time_engine_on_after_start=min_time_engine_on_after_start,
        has_start_stop=has_start_stop, use_basic_start_stop=use_basic_start_stop
    )

    args = (np.ediff1d(times, to_begin=[0]), gear_box_powers_in, accelerations,
            gear_box_speeds_in, final_drive_powers_in, times)

    thermal_model = functools.partial(engine_temperature_regression_model.delta,
                                      max_temp=max_engine_coolant_temperature)
    from .engine import calculate_engine_speeds_out_hot as eng_speed_hot

    def _func():
        eng, T, e = (True, False), temp[0], (0, 0, None, soc[0])
        # min_soc = electrics_model.alternator_status_model.min
        for i, (on_eng, dt, p, a, s, fdp, t) in enumerate(zip(gen, *args), 1):
            # if e[-1] < min_soc and not on_eng[0]:
            #    on_eng[0], on_eng[1] = True, not eng[0]

            eng_s = eng_speed_hot(s, on_eng[0], idle_engine_speed)

            T += thermal_model(dt, fdp, eng_s, a, prev_temperature=T)

            eng = tuple(on_eng)
            e = tuple(electrics_model(dt, p, a, t, *(eng + e[1:])))
            try:
                temp[i], soc[i] = T, e[-1]
            except IndexError:
                pass
            yield e + (eng_s, T) + eng

    dtype = [('alt_c', 'f'), ('alt_sts', 'f'), ('bat_c', 'f'), ('soc', 'f'),
             ('eng_s', 'f'), ('tmp', 'f'), ('on_eng', '?'), ('eng_st', '?')]
    k = ('alt_c', 'bat_c', 'soc', 'alt_sts', 'on_eng', 'eng_st', 'eng_s', 'tmp')
    return co2_utl.fromiter(_func(), dtype, k, n)


def physical():
    """
    Defines the CO2MPAS physical model.

    .. dispatcher:: d

        >>> d = physical()

    :return:
        The CO2MPAS physical model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='CO2MPAS physical model',
        description='Wraps all functions needed to calibrate and predict '
                    'light-vehicles\' CO2 emissions.'
    )

    from .cycle import cycle
    d.add_dispatcher(
        include_defaults=True,
        dsp_id='cycle_model',
        dsp=cycle(),
        inputs=(
            'accelerations', 'bag_phases', 'climbing_force', 'cycle_type',
            'downscale_factor', 'downscale_factor_threshold',
            'downscale_phases', 'driver_mass', 'engine_max_power',
            'engine_max_speed_at_max_power', 'full_load_curve', 'gear_box_type',
            'gears', 'idle_engine_speed', 'inertial_factor', 'k1', 'k2', 'k5',
            'max_gear', 'max_speed_velocity_ratio', 'max_time', 'max_velocity',
            'motive_powers', 'road_loads', 'speed_velocity_ratios',
            'time_sample_frequency', 'times', 'vehicle_mass', 'velocities',
            'wltp_base_model', 'wltp_class'),
        outputs=(
            'gears', 'initial_temperature', 'phases_integration_times', 'times',
            'velocities')
    )

    from .vehicle import vehicle
    d.add_dispatcher(
        include_defaults=True,
        dsp_id='vehicle_model',
        dsp=vehicle(),
        inputs=(
            'aerodynamic_drag_coefficient', 'air_density', 'angle_slope',
            'angle_slopes', 'correct_f0', 'cycle_type', 'f0', 'f0_uncorrected',
            'f1', 'f2', 'frontal_area', 'has_roof_box', 'inertial_factor',
            'n_dyno_axes', 'obd_velocities', 'road_loads',
            'rolling_resistance_coeff', 'times', 'tyre_category', 'tyre_class',
            'vehicle_body', 'vehicle_category', 'vehicle_height',
            'vehicle_mass', 'vehicle_width', 'velocities'),
        outputs=(
            'accelerations', 'angle_slopes', 'climbing_force', 'f0', 'f1', 'f2',
            'inertial_factor', 'motive_powers', 'n_dyno_axes', 'road_loads',
            'velocities'),
    )

    from .wheels import wheels
    d.add_dispatcher(
        include_defaults=True,
        dsp_id='wheels_model',
        dsp=wheels(),
        inputs=(
            'accelerations', 'change_gear_window_width', 'engine_speeds_out',
            'final_drive_ratios', 'gear_box_ratios', 'gears',
            'idle_engine_speed', 'motive_powers', 'plateau_acceleration',
            'r_dynamic', 'r_wheels', 'stop_velocity', 'times', 'tyre_code',
            'tyre_dimensions', 'tyre_dynamic_rolling_coefficient', 'velocities',
            'velocity_speed_ratios'),
        outputs=(
            'r_dynamic', 'r_wheels', 'tyre_code',
            'tyre_dynamic_rolling_coefficient', 'wheel_powers', 'wheel_speeds',
            'wheel_torques'),
        inp_weight={'r_dynamic': 3}
    )

    from .final_drive import final_drive
    d.add_dispatcher(
        include_defaults=True,
        dsp_id='final_drive_model',
        dsp=final_drive(),
        inputs=(
            'final_drive_efficiency', 'final_drive_ratio', 'final_drive_ratios',
            'final_drive_torque_loss', 'gear_box_ratios', 'gear_box_type',
            'gears', 'n_dyno_axes', 'n_wheel_drive', 'velocity_speed_ratios',
            {'wheel_powers': 'final_drive_powers_out',
             'wheel_speeds': 'final_drive_speeds_out',
             'wheel_torques': 'final_drive_torques_out'}),
        outputs=(
            'final_drive_powers_in', 'final_drive_ratios',
            'final_drive_speeds_in', 'final_drive_torques_in')
    )

    from .gear_box import gear_box
    d.add_dispatcher(
        include_defaults=True,
        dsp_id='gear_box_model',
        dsp=gear_box(),
        inputs=(
            'CMV', 'CMV_Cold_Hot', 'CVT', 'DT_VA', 'DT_VAP', 'DT_VAT',
            'DT_VATP', 'GSPV', 'GSPV_Cold_Hot', 'MVL', 'accelerations',
            'change_gear_window_width', 'cycle_type',
            'engine_coolant_temperatures', 'engine_mass', 'engine_max_power',
            'engine_max_speed_at_max_power', 'engine_max_torque',
            'engine_speeds_out', 'engine_thermostat_temperature',
            'final_drive_ratios', 'fuel_saving_at_strategy', 'full_load_curve',
            'gear_box_efficiency_constants',
            'gear_box_efficiency_parameters_cold_hot', 'gear_box_ratios',
            'gear_box_temperature_references', 'gear_box_type', 'gears',
            'has_gear_box_thermal_management', 'has_torque_converter',
            'idle_engine_speed', 'initial_gear_box_temperature',
            'max_velocity_full_load_correction', 'min_engine_on_speed',
            'motive_powers', 'n_gears', 'on_engine', 'plateau_acceleration',
            'r_dynamic', 'road_loads', 'specific_gear_shifting',
            'stop_velocity', 'time_cold_hot_transition', 'times',
            'use_dt_gear_shifting', 'vehicle_mass', 'velocities',
            'velocity_speed_ratios',
            {'final_drive_powers_in': 'gear_box_powers_out',
             'final_drive_speeds_in': 'gear_box_speeds_out',
             'initial_engine_temperature': 'initial_gear_box_temperature',
             'initial_temperature': 'initial_gear_box_temperature'}),
        outputs=(
            'CMV', 'CMV_Cold_Hot', 'CVT', 'DT_VA', 'DT_VAP', 'DT_VAT',
            'DT_VATP', 'GSPV', 'GSPV_Cold_Hot', 'MVL',
            'equivalent_gear_box_heat_capacity', 'gear_box_efficiencies',
            'gear_box_powers_in', 'gear_box_ratios', 'gear_box_speeds_in',
            'gear_box_temperatures', 'gear_box_torque_losses',
            'gear_box_torques_in', 'gear_shifts', 'gears', 'max_gear',
            'max_speed_velocity_ratio', 'n_gears', 'specific_gear_shifting',
            'speed_velocity_ratios', 'velocity_speed_ratios'),
        inp_weight={'initial_temperature': 5}
    )

    from .clutch_tc import clutch_torque_converter
    d.add_dispatcher(
        include_defaults=True,
        dsp=clutch_torque_converter(),
        dsp_id='clutch_torque_converter_model',
        inputs=(
            'accelerations', 'calibration_tc_speed_threshold', 'clutch_model',
            'clutch_window', 'cold_start_speeds_delta', 'engine_speeds_out',
            'engine_speeds_out_hot', 'gear_box_powers_in', 'gear_box_speeds_in',
            'gear_box_type', 'gear_shifts', 'gears', 'has_torque_converter',
            'lock_up_tc_limits', 'lockup_speed_ratio',
            'stand_still_torque_ratio', 'stop_velocity', 'times',
            'torque_converter_model', 'velocities'),
        outputs=(
            'clutch_model', 'clutch_phases', 'clutch_tc_powers',
            'clutch_tc_speeds_delta', 'clutch_window', 'has_torque_converter',
            'lockup_speed_ratio', 'stand_still_torque_ratio',
            'torque_converter_model')
    )

    from .electrics import electrics
    d.add_dispatcher(
        include_defaults=True,
        dsp_id='electric_model',
        dsp=electrics(),
        inputs=(
            'accelerations', 'alternator_charging_currents',
            'alternator_current_model', 'alternator_currents',
            'alternator_efficiency', 'alternator_initialization_time',
            'alternator_nominal_power', 'alternator_nominal_voltage',
            'alternator_off_threshold', 'alternator_start_window_width',
            'alternator_status_model', 'alternator_statuses',
            'battery_capacity', 'battery_currents', 'cycle_type',
            'delta_time_engine_starter', 'electric_load',
            'engine_moment_inertia', 'engine_starts', 'gear_box_powers_in',
            'has_energy_recuperation', 'idle_engine_speed',
            'initial_state_of_charge', 'max_battery_charging_current',
            'on_engine', 'start_demand', 'state_of_charge_balance',
            'state_of_charge_balance_window', 'state_of_charges',
            'stop_velocity', 'times', 'velocities'),
        outputs=(
            'alternator_current_model', 'alternator_currents',
            'alternator_initialization_time', 'alternator_nominal_power',
            'alternator_powers_demand', 'alternator_status_model',
            'alternator_statuses', 'battery_currents', 'electric_load',
            'electrics_model', 'initial_state_of_charge',
            'max_battery_charging_current', 'start_demand',
            'state_of_charge_balance', 'state_of_charge_balance_window',
            'state_of_charges')
    )

    from .engine import engine
    d.add_dispatcher(
        include_defaults=True,
        dsp_id='engine_model',
        dsp=engine(),
        inputs=(
            'accelerations', 'active_cylinder_ratios',
            'alternator_powers_demand', 'angle_slopes',
            'auxiliaries_power_loss', 'auxiliaries_torque_loss',
            'calibration_status', 'clutch_tc_powers', 'clutch_tc_speeds_delta',
            'co2_emission_extra_high', 'co2_emission_high', 'co2_emission_low',
            'co2_emission_medium', 'co2_normalization_references', 'co2_params',
            'co2_params_calibrated', 'cold_start_speed_model',
            'correct_start_stop_with_gears', 'enable_phases_willans',
            'enable_willans', 'engine_capacity', 'engine_coolant_temperatures',
            'engine_fuel_lower_heating_value',
            'engine_has_cylinder_deactivation',
            'engine_has_variable_valve_actuation',
            'engine_idle_fuel_consumption', 'engine_is_turbo', 'engine_mass',
            'engine_max_power', 'engine_max_speed',
            'engine_max_speed_at_max_power', 'engine_max_torque',
            'engine_powers_out', 'engine_speeds_out', 'engine_speeds_out_hot',
            'engine_starts', 'engine_stroke',
            'engine_temperature_regression_model',
            'engine_thermostat_temperature',
            'engine_thermostat_temperature_window', 'engine_type',
            'final_drive_powers_in', 'fuel_carbon_content',
            'fuel_carbon_content_percentage', 'fuel_consumptions',
            'fuel_density', 'fuel_type', 'full_load_powers', 'full_load_speeds',
            'full_load_torques', 'gear_box_speeds_in', 'gear_box_type', 'gears',
            'has_exhausted_gas_recirculation', 'has_lean_burn',
            'has_periodically_regenerating_systems',
            'has_selective_catalytic_reduction', 'has_start_stop',
            'idle_engine_speed', 'idle_engine_speed_median',
            'idle_engine_speed_std', 'ignition_type',
            'initial_engine_temperature', 'initial_friction_params',
            'is_cycle_hot', 'is_hybrid', 'ki_factor',
            'max_engine_coolant_temperature', 'min_engine_on_speed',
            'min_time_engine_on_after_start', 'motive_powers', 'on_engine',
            'on_idle', 'phases_integration_times', 'plateau_acceleration',
            'start_stop_activation_time', 'start_stop_model',
            'state_of_charges', 'stop_velocity', 'times',
            'use_basic_start_stop', 'velocities',
            {'initial_temperature': 'initial_engine_temperature'}),
        outputs=(
            'active_cylinders', 'active_exhausted_gas_recirculations',
            'active_lean_burns', 'active_variable_valves',
            'after_treatment_temperature_threshold', 'auxiliaries_power_losses',
            'auxiliaries_torque_losses', 'brake_powers', 'calibration_status',
            'co2_emission_value', 'co2_emissions', 'co2_emissions_model',
            'co2_error_function_on_emissions', 'co2_error_function_on_phases',
            'co2_params_calibrated', 'co2_params_initial_guess',
            'cold_start_speed_model', 'cold_start_speeds_delta',
            'cold_start_speeds_phases', 'correct_start_stop_with_gears',
            'declared_co2_emission_value', 'engine_coolant_temperatures',
            'engine_fuel_lower_heating_value', 'engine_heat_capacity',
            'engine_idle_fuel_consumption', 'engine_mass', 'engine_max_power',
            'engine_max_speed', 'engine_max_speed_at_max_power',
            'engine_max_torque', 'engine_moment_inertia', 'engine_powers_out',
            'engine_speeds_out', 'engine_speeds_out_hot', 'engine_starts',
            'engine_temperature_derivatives',
            'engine_temperature_regression_model',
            'engine_thermostat_temperature',
            'engine_thermostat_temperature_window', 'engine_type',
            'extended_phases_co2_emissions',
            'extended_phases_integration_times',
            'fuel_carbon_content',
            'fuel_carbon_content_percentage', 'fuel_consumptions',
            'fuel_density', 'full_load_curve', 'full_load_powers',
            'full_load_speeds', 'has_exhausted_gas_recirculation',
            'has_sufficient_power', 'identified_co2_emissions',
            'idle_engine_speed', 'idle_engine_speed_median',
            'idle_engine_speed_std', 'ignition_type',
            'initial_engine_temperature', 'initial_friction_params',
            'ki_factor', 'max_engine_coolant_temperature', 'missing_powers',
            'on_engine', 'on_idle', 'optimal_efficiency',
            'phases_co2_emissions', 'phases_fuel_consumptions',
            'phases_willans_factors', 'start_stop_model',
            'use_basic_start_stop', 'willans_factors', 'co2_rescaling_scores'),
        inp_weight={'initial_temperature': 5}
    )

    d.add_function(
        function=predict_vehicle_electrics_and_engine_behavior,
        inputs=['electrics_model', 'start_stop_model',
                'engine_temperature_regression_model',
                'initial_engine_temperature', 'initial_state_of_charge',
                'idle_engine_speed', 'times', 'final_drive_powers_in',
                'gear_box_speeds_in', 'gear_box_powers_in', 'velocities',
                'accelerations', 'gears', 'start_stop_activation_time',
                'correct_start_stop_with_gears',
                'min_time_engine_on_after_start', 'has_start_stop',
                'use_basic_start_stop', 'max_engine_coolant_temperature'],
        outputs=['alternator_currents', 'battery_currents', 'state_of_charges',
                 'alternator_statuses', 'on_engine', 'engine_starts',
                 'engine_speeds_out_hot', 'engine_coolant_temperatures'],
        weight=10
    )

    return d
