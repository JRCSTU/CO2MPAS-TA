# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the electrics of the vehicle.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics

.. autosummary::
    :nosignatures:
    :toctree: electrics/

    motors
    batteries
"""
import numpy as np
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
        'motor_p0_efficiency', 'motor_p0_maximum_power', 'motor_p0_speed_ratio',
        'motor_p0_electric_powers', 'motor_p0_torques', 'motor_p0_rated_speed',
        'motor_p0_powers', 'motor_p0_maximum_torque', 'motor_p0_speeds',
        'motor_p1_efficiency', 'motor_p1_maximum_power', 'motor_p1_speed_ratio',
        'motor_p1_electric_powers', 'motor_p1_torques', 'motor_p1_rated_speed',
        'motor_p1_powers', 'motor_p1_maximum_torque', 'motor_p1_speeds',
        'motor_p2_planetary_efficiency', 'motor_p2_planetary_maximum_power',
        'motor_p2_planetary_speed_ratio', 'planetary_ratio',
        'motor_p2_planetary_electric_powers', 'motor_p2_planetary_torques',
        'motor_p2_planetary_powers', 'motor_p2_planetary_maximum_torque',
        'motor_p2_planetary_rated_speed', 'motor_p2_planetary_speeds',
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
        'has_motor_p0', 'has_motor_p1', 'has_motor_p2', 'has_motor_p3_front',
        'has_motor_p3_rear', 'has_motor_p4_front', 'has_motor_p4_rear',
        'service_battery_initialization_time', 'engine_starts', 'accelerations',
        'starter_efficiency', 'delta_time_engine_starter', 'gear_box_speeds_in',
        'service_battery_charging_statuses', 'alternator_currents', 'on_engine',
        'starter_nominal_voltage', 'engine_speeds_out', 'alternator_efficiency',
        'service_battery_state_of_charges', 'alternator_current_model', 'times',
        'alternator_charging_currents', 'engine_moment_inertia', 'wheel_speeds',
        'alternator_electric_powers', 'final_drive_speeds_in', 'motive_powers',
        'alternator_nominal_voltage', 'planetary_mean_efficiency',
        'planetary_speeds_in',
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
        'motor_p2_planetary_electric_powers', 'planetary_ratio',
        'motor_p2_planetary_maximum_power', 'motor_p2_planetary_powers',
        'motor_p2_planetary_maximum_torque', 'motor_p2_planetary_speed_ratio',
        'motor_p2_planetary_maximum_powers', 'motor_p2_planetary_rated_speed',
        'motor_p2_planetary_torques', 'motor_p2_planetary_speeds',
        'motor_p2_electric_powers', 'motor_p2_maximum_power', 'motor_p2_powers',
        'motor_p2_maximum_torque', 'motor_p2_speed_ratio', 'motor_p2_torques',
        'motor_p2_maximum_powers', 'motor_p2_rated_speed', 'motor_p2_speeds',
        'motor_p3_front_electric_powers', 'motor_p3_front_maximum_power',
        'motor_p3_front_maximum_powers', 'motor_p3_front_rated_speed',
        'motor_p3_front_powers', 'motor_p3_front_maximum_torque',
        'motor_p3_front_speed_ratio', 'motor_p3_front_torques',
        'motor_p3_front_speeds',
        'motor_p3_rear_electric_powers', 'motor_p3_rear_maximum_power',
        'motor_p3_rear_maximum_powers', 'motor_p3_rear_rated_speed',
        'motor_p3_rear_powers', 'motor_p3_rear_maximum_torque',
        'motor_p3_rear_speed_ratio', 'motor_p3_rear_torques',
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
        'starter_electric_powers', 'final_drive_speeds_in', 'alternator_powers',
        'alternator_electric_powers', 'engine_speeds_out', 'gear_box_speeds_in',
        'delta_time_engine_starter', 'alternator_currents', 'starter_currents',
        'alternator_current_model', 'starter_powers', 'starter_model',
        'wheel_speeds', 'planetary_speeds_in', 'has_motor_p0', 'has_motor_p1',
        'has_motor_p2_planetary', 'has_motor_p2', 'has_motor_p3_front',
        'has_motor_p3_rear', 'has_motor_p4_front', 'has_motor_p4_rear',
        'is_hybrid', 'planetary_mean_efficiency',
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
        'motor_p2_electric_powers', 'service_battery_capacity', 'accelerations',
        'service_battery_loads', 'dcdc_charging_currents', 'dcdc_current_model',
        'service_battery_electric_powers_supply_threshold', 'engine_powers_out',
        'motor_p0_electric_powers', 'alternator_current_model', 'motive_powers',
        'service_battery_state_of_charge_balance_window', 'drive_battery_loads',
        'drive_battery_state_of_charges', 'drive_battery_currents', 'on_engine',
        'service_battery_state_of_charges', 'motor_p1_electric_powers', 'times',
        'service_battery_initialization_time', 'dcdc_converter_electric_powers',
        'has_energy_recuperation', 'drive_battery_voltages', 'starter_currents',
        'dcdc_converter_currents', 'drive_battery_electric_powers', 'is_hybrid',
        'service_battery_state_of_charge_balance', 'dcdc_converter_efficiency',
        'motor_p2_planetary_electric_powers', 'drive_battery_nominal_voltage',
        'drive_battery_technology', 'drive_battery_capacity', 'engine_starts',
        'drive_battery_n_parallel_cells', 'electrical_hybridization_degree',
        'dcdc_converter_electric_powers_demand', 'starter_electric_powers',
        'motor_p3_front_electric_powers', 'motor_p3_rear_electric_powers',
        'motor_p4_front_electric_powers', 'motor_p4_rear_electric_powers',
        'service_battery_capacity_kwh', 'drive_battery_capacity_kwh',
        'service_battery_currents', 'drive_battery_n_cells',
        'drive_battery_load',
    ),
    outputs=(
        'service_battery_electric_powers_supply_threshold', 'drive_battery_ocv',
        'service_battery_state_of_charge_balance_window', 'drive_battery_model',
        'drive_battery_state_of_charges', 'drive_battery_delta_state_of_charge',
        'initial_drive_battery_state_of_charge', 'drive_battery_n_series_cells',
        'dcdc_converter_currents', 'drive_battery_capacity', 'drive_battery_r0',
        'service_battery_delta_state_of_charge', 'service_battery_status_model',
        'service_battery_state_of_charges', 'service_battery_charging_statuses',
        'drive_battery_n_cells', 'service_battery_loads', 'drive_battery_loads',
        'drive_battery_voltages', 'service_battery_model', 'drive_battery_load',
        'dcdc_converter_electric_powers_demand', 'service_battery_capacity_kwh',
        'service_battery_initialization_time', 'drive_battery_electric_powers',
        'service_battery_state_of_charge_balance', 'dcdc_converter_efficiency',
        'motors_electric_powers', 'starter_currents', 'drive_battery_currents',
        'initial_service_battery_state_of_charge', 'service_battery_capacity',
        'service_battery_electric_powers', 'drive_battery_n_parallel_cells',
        'drive_battery_nominal_voltage', 'drive_battery_capacity_kwh',
        'service_battery_currents', 'service_battery_load',
        'dcdc_current_model',
    ),
    include_defaults=True
)

dsp.add_function(
    function=sh.bypass,
    inputs=['service_battery_nominal_voltage'],
    outputs=['alternator_nominal_voltage'],
)

dsp.add_function(
    function=sh.bypass,
    inputs=['alternator_nominal_voltage'],
    outputs=['starter_nominal_voltage'],
)

dsp.add_function(
    function=sh.bypass,
    inputs=['starter_nominal_voltage'],
    outputs=['service_battery_nominal_voltage'],
)


@sh.add_function(dsp, outputs=[
    'service_battery_state_of_charges', 'service_battery_charging_statuses',
    'dcdc_converter_currents', 'alternator_currents'
])
def predict_service_battery_flows(
        service_battery_model, times, motive_powers, accelerations, on_engine,
        starter_currents):
    """
    Predict the service battery currents flows.

    :param service_battery_model:
         Service battery model.
    :type service_battery_model: ServiceBatteryModel

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param accelerations:
        Acceleration [m/s2].
    :type accelerations: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param starter_currents:
        Starter currents [A].
    :type starter_currents: numpy.array

    :return:
        - State of charge of the service battery [%].
        - Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
          3: Initialization) [-].
        - DC/DC converter currents [A].
        - Alternator currents [A].
    :rtype: numpy.array
    """
    service_battery_model.reset()
    it = zip(times, motive_powers, accelerations, on_engine, starter_currents)
    return np.array([service_battery_model(*a) for a in it], float).T
