# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the drive battery (high voltage).
"""

import numpy as np
import schedula as sh
from ...defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Drive Battery',
    description='Models the drive battery (e.g., high voltage).'
)


@sh.add_function(dsp, outputs=['drive_battery_voltages'])
def calculate_drive_battery_voltages(
        drive_battery_electric_powers, drive_battery_currents):
    """
    Calculate the drive battery voltage [V].

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param drive_battery_currents:
        Drive battery current vector [A].
    :type drive_battery_currents: numpy.array

    :return:
        Drive battery voltage [V].
    :rtype: numpy.array
    """
    from .service import calculate_service_battery_currents as func
    return func(drive_battery_electric_powers, drive_battery_currents)


@sh.add_function(dsp, outputs=['drive_battery_currents'])
def calculate_drive_battery_currents(
        drive_battery_electric_powers, drive_battery_voltages):
    """
    Calculate the drive battery current vector [A].

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param drive_battery_voltages:
        Drive battery voltage [V].
    :type drive_battery_voltages: numpy.array

    :return:
        Drive battery current vector [A].
    :rtype: numpy.array
    """
    from .service import calculate_service_battery_currents as func
    return func(drive_battery_electric_powers, drive_battery_voltages)


@sh.add_function(dsp, outputs=['drive_battery_currents'], weight=1)
def calculate_drive_battery_currents_v1(
        drive_battery_capacity, times, drive_battery_state_of_charges):
    """
    Calculate the drive battery current vector [A].

    :param drive_battery_capacity:
        Drive battery capacity [Ah].
    :type drive_battery_capacity: float

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param drive_battery_state_of_charges:
        State of charge of the drive battery [%].
    :type drive_battery_state_of_charges: numpy.array

    :return:
        Drive battery current vector [A].
    :rtype: numpy.array
    """
    from .service import calculate_service_battery_currents_v1 as func
    return func(drive_battery_capacity, times, drive_battery_state_of_charges)


@sh.add_function(dsp, outputs=['drive_battery_capacity'])
def identify_drive_battery_capacity(
        times, drive_battery_currents, drive_battery_state_of_charges):
    """
    Identify drive battery capacity [Ah].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param drive_battery_currents:
        Drive battery current vector [A].
    :type drive_battery_currents: numpy.array

    :param drive_battery_state_of_charges:
        State of charge of the drive battery [%].
    :type drive_battery_state_of_charges: numpy.array

    :return:
        Drive battery capacity [Ah].
    :rtype: float
    """
    from .service import identify_service_battery_capacity as func
    return func(times, drive_battery_currents, drive_battery_state_of_charges)


@sh.add_function(dsp, outputs=['drive_battery_electric_powers'])
def calculate_drive_battery_electric_powers(
        drive_battery_currents, drive_battery_voltages):
    """
    Calculate the drive battery electric power [kW].

    :param drive_battery_currents:
        Drive battery current vector [A].
    :type drive_battery_currents: numpy.array

    :param drive_battery_voltages:
        Drive battery voltage [V].
    :type drive_battery_voltages: float

    :return:
        Drive battery electric power [kW].
    :rtype: numpy.array
    """
    from .service import calculate_service_battery_electric_powers as func
    return func(drive_battery_currents, drive_battery_voltages)


@sh.add_function(dsp, outputs=['drive_battery_electric_powers'], weight=1)
def calculate_drive_battery_electric_powers_v1(
        drive_battery_loads, motors_electric_powers,
        dcdc_converter_electric_powers_demand):
    """
    Calculate the drive battery electric power [kW].

    :param drive_battery_loads:
        Drive battery load vector [kW].
    :type drive_battery_loads: numpy.array

    :param motors_electric_powers:
        Motors electric power [kW].
    :type motors_electric_powers: numpy.array

    :param dcdc_converter_electric_powers_demand:
        DC/DC converter electric power demand [kW].
    :type dcdc_converter_electric_powers_demand: numpy.array

    :return:
        Drive battery electric power [kW].
    :rtype: numpy.array
    """
    p = drive_battery_loads - motors_electric_powers
    p += dcdc_converter_electric_powers_demand
    return p


@sh.add_function(dsp, outputs=['initial_drive_battery_state_of_charge'])
def identify_initial_drive_battery_state_of_charge(
        drive_battery_state_of_charges):
    """
    Identify the initial state of charge of drive battery [%].

    :param drive_battery_state_of_charges:
        State of charge of the drive battery [%].
    :type drive_battery_state_of_charges: numpy.array

    :return:
        Initial state of charge of the drive battery [%].
    :rtype: float
    """
    return drive_battery_state_of_charges[0]


@sh.add_function(
    dsp, outputs=['initial_drive_battery_state_of_charge'], weight=10
)
def default_initial_drive_battery_state_of_charge(
        electrical_hybridization_degree):
    """
    Return the default initial state of charge of drive battery [%].

    :param electrical_hybridization_degree:
        Electrical hybridization degree (i.e., mild, full, plugin, electric).
    :type electrical_hybridization_degree: str

    :return:
        Initial state of charge of the drive battery [%].
    :rtype: float
    """
    d = dfl.functions.default_initial_drive_battery_state_of_charge
    return d.initial_state_of_charge[electrical_hybridization_degree]


@sh.add_function(dsp, outputs=['drive_battery_state_of_charges'])
def calculate_drive_battery_state_of_charges(
        drive_battery_capacity, initial_drive_battery_state_of_charge,
        times, drive_battery_currents):
    """
    Calculates the state of charge of the drive battery [%].

    :param drive_battery_capacity:
        Drive battery capacity [Ah].
    :type drive_battery_capacity: float

    :param initial_drive_battery_state_of_charge:
        Initial state of charge of the drive battery [%].
    :type initial_drive_battery_state_of_charge: float

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param drive_battery_currents:
        Drive battery current vector [A].
    :type drive_battery_currents: numpy.array

    :return:
        State of charge of the drive battery [%].
    :rtype: numpy.array
    """
    from .service import calculate_service_battery_state_of_charges as func
    return func(
        drive_battery_capacity, initial_drive_battery_state_of_charge, times,
        drive_battery_currents
    )


@sh.add_function(dsp, outputs=['drive_battery_loads'])
def calculate_drive_battery_loads(
        drive_battery_electric_powers, motors_electric_powers,
        dcdc_converter_electric_powers_demand):
    """
    Calculates drive battery load vector [kW].

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param motors_electric_powers:
        Motors electric power [kW].
    :type motors_electric_powers: numpy.array

    :param dcdc_converter_electric_powers_demand:
        DC/DC converter electric power demand [kW].
    :type dcdc_converter_electric_powers_demand: numpy.array

    :return:
        Drive battery load vector [kW].
    :rtype: numpy.array
    """
    p = drive_battery_electric_powers + motors_electric_powers
    p -= dcdc_converter_electric_powers_demand
    return p


@sh.add_function(dsp, outputs=['drive_battery_loads'])
def calculate_drive_battery_loads_v1(times, drive_battery_load):
    """
    Calculates drive battery load vector [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param drive_battery_load:
        Drive electric load [kW].
    :type drive_battery_load: float

    :return:
        Drive battery load vector [kW].
    :rtype: numpy.array
    """
    return np.tile(drive_battery_load, times.shape)


@sh.add_function(dsp, outputs=['drive_battery_load'])
def identify_drive_battery_load(drive_battery_loads):
    """
    Identifies drive electric load [kW].

    :param drive_battery_loads:
        Drive battery load vector [kW].
    :type drive_battery_loads: numpy.array

    :return:
        Drive electric load [kW].
    :rtype: float
    """
    return co2_utl.reject_outliers(drive_battery_loads)[0]


@sh.add_function(dsp, outputs=['drive_battery_delta_state_of_charge'])
def calculate_drive_battery_delta_state_of_charge(
        drive_battery_state_of_charges):
    """
    Calculates the overall delta state of charge of the drive battery [%].

    :param drive_battery_state_of_charges:
        State of charge of the drive battery [%].
    :type drive_battery_state_of_charges: numpy.array

    :return:
        Overall delta state of charge of the drive battery [%].
    :rtype: float
    """
    from .service import calculate_service_battery_delta_state_of_charge as func
    return func(drive_battery_state_of_charges)


@sh.add_function(dsp, outputs=['drive_battery_r0', 'drive_battery_ocv'])
def calibrate_drive_battery_r0_and_ocv(
        drive_battery_n_parallel_cells, drive_battery_n_series_cells,
        drive_battery_currents, drive_battery_voltages):
    """
    Calibrate drive battery resistance and open circuit voltage [ohm, V].

    :param drive_battery_n_parallel_cells:
        Number of battery cells in parallel [-].
    :type drive_battery_n_parallel_cells: int

    :param drive_battery_n_series_cells:
        Number of battery cells in series [-].
    :type drive_battery_n_series_cells: int

    :param drive_battery_currents:
        Drive battery current vector [A].
    :type drive_battery_currents: numpy.array

    :param drive_battery_voltages:
        Drive battery voltage [V].
    :type drive_battery_voltages: numpy.array

    :return:
        Driver battery resistance and open circuit voltage [ohm, V].
    :rtype: float, float
    """
    from sklearn.linear_model import RANSACRegressor
    m = RANSACRegressor()
    m.fit(drive_battery_currents[:, None], drive_battery_voltages)
    r0, ocv = float(m.estimator_.coef_), float(m.estimator_.intercept_)
    r0 *= -drive_battery_n_parallel_cells / drive_battery_n_series_cells
    ocv /= drive_battery_n_series_cells
    return r0, ocv


dsp.add_data('drive_battery_n_parallel_cells', 1)
dsp.add_data('drive_battery_n_series_cells', 1)


@sh.add_function(dsp, outputs=['maximum_drive_battery_electric_power'])
def calculate_maximum_drive_battery_electric_power(
        drive_battery_r0, drive_battery_ocv, drive_battery_n_parallel_cells,
        drive_battery_n_series_cells):
    """
    Calculate the maximum admissible electric power of drive battery [kW].

    :param drive_battery_r0:
        Driver battery resistance [ohm].
    :type drive_battery_r0: float

    :param drive_battery_ocv:
        Driver battery open circuit voltage [V].
    :type drive_battery_ocv: float

    :param drive_battery_n_parallel_cells:
        Number of battery cells in parallel [-].
    :type drive_battery_n_parallel_cells: int

    :param drive_battery_n_series_cells:
        Number of battery cells in series [-].
    :type drive_battery_n_series_cells: int

    :return:
        Maximum admissible electric power of drive battery [kW].
    :rtype: float
    """
    if not drive_battery_r0:
        return float('inf')
    n_p, n_s = drive_battery_n_parallel_cells, drive_battery_n_series_cells
    r0, ocv = drive_battery_r0, drive_battery_ocv
    return ocv ** 2 / (4e3 * r0 / (n_s * n_p))


@sh.add_function(dsp, outputs=['drive_battery_currents'])
def calculate_drive_battery_currents_v2(
        drive_battery_electric_powers, drive_battery_r0, drive_battery_ocv,
        drive_battery_n_parallel_cells, drive_battery_n_series_cells):
    """
    Calculate the drive battery current vector [A].

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param drive_battery_r0:
        Driver battery resistance [ohm].
    :type drive_battery_r0: float

    :param drive_battery_ocv:
        Driver battery open circuit voltage [V].
    :type drive_battery_ocv: float

    :param drive_battery_n_parallel_cells:
        Number of battery cells in parallel [-].
    :type drive_battery_n_parallel_cells: int

    :param drive_battery_n_series_cells:
        Number of battery cells in series [-].
    :type drive_battery_n_series_cells: int

    :return:
        Drive battery current vector [A].
    :rtype: numpy.array
    """
    n_p, n_s = drive_battery_n_parallel_cells, drive_battery_n_series_cells
    p = drive_battery_electric_powers
    r0, ocv = drive_battery_r0, drive_battery_ocv
    x = ocv - np.nan_to_num(np.sqrt(ocv ** 2 - (4e3 * r0 / (n_s * n_p)) * p))
    x *= n_p / (2 * r0)
    return x


@sh.add_function(dsp, outputs=['motors_electric_powers'])
def calculate_motors_electric_powers(
        motor_p0_electric_powers, motor_p1_electric_powers,
        motor_p2_electric_powers, motor_p3_electric_powers,
        motor_p4_electric_powers):
    """
    Calculate motors electric power [kW].

    :param motor_p0_electric_powers:
        Electric power of motor P0 [kW].
    :type motor_p0_electric_powers: numpy.array | float

    :param motor_p1_electric_powers:
        Electric power of motor P1 [kW].
    :type motor_p1_electric_powers: numpy.array | float

    :param motor_p2_electric_powers:
        Electric power of motor P2 [kW].
    :type motor_p2_electric_powers: numpy.array | float

    :param motor_p3_electric_powers:
        Electric power of motor P3 [kW].
    :type motor_p3_electric_powers: numpy.array | float

    :param motor_p4_electric_powers:
        Electric power of motor P4 [kW].
    :type motor_p4_electric_powers: numpy.array | float

    :return:
        Motors electric power [kW].
    :rtype: numpy.array | float
    """
    p = motor_p0_electric_powers + motor_p1_electric_powers
    p += motor_p2_electric_powers
    p += motor_p3_electric_powers
    p += motor_p4_electric_powers
    return p
