# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the service battery (low voltage).
"""

import numpy as np
import schedula as sh
from ...defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Service Battery',
    description='Models the service battery (e.g., low voltage).'
)


@sh.add_function(dsp, outputs=['service_battery_currents'])
def calculate_service_battery_currents(
        service_battery_electric_powers, service_battery_nominal_voltage):
    """
    Calculate the service battery current vector [A].

    :param service_battery_electric_powers:
        Service battery electric power [kW].
    :type service_battery_electric_powers: numpy.array

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :return:
        Service battery current vector [A].
    :rtype: numpy.array
    """
    c = 1000.0 / service_battery_nominal_voltage
    return service_battery_electric_powers * c


@sh.add_function(dsp, outputs=['service_battery_currents'], weight=1)
def calculate_service_battery_currents_v1(
        service_battery_capacity, times, service_battery_state_of_charges):
    """
    Calculate the service battery current vector [A].

    :param service_battery_capacity:
        Service battery capacity [Ah].
    :type service_battery_capacity: float

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :return:
        Service battery current vector [A].
    :rtype: numpy.array
    """
    from scipy.interpolate import UnivariateSpline
    soc = service_battery_state_of_charges
    ib = UnivariateSpline(times, soc).derivative()(times)
    ib *= service_battery_capacity * 36.0
    return ib


@sh.add_function(dsp, outputs=['service_battery_capacity'])
def identify_service_battery_capacity(
        times, service_battery_currents, service_battery_state_of_charges):
    """
    Identify service battery capacity [Ah].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_currents:
        Service battery current vector [A].
    :type service_battery_currents: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :return:
        Service battery capacity [Ah].
    :rtype: float
    """
    d = calculate_service_battery_currents_v1(
        1, times, service_battery_state_of_charges
    )
    b = (d < -dfl.EPS) | (d > dfl.EPS)
    return co2_utl.reject_outliers(service_battery_currents[b] / d[b])[0]


@sh.add_function(dsp, outputs=['service_battery_electric_powers'])
def calculate_service_battery_electric_powers(
        service_battery_currents, service_battery_nominal_voltage):
    """
    Calculate the service battery electric power [kW].

    :param service_battery_currents:
        Service battery current vector [A].
    :type service_battery_currents: numpy.array

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :return:
        Service battery electric power [kW].
    :rtype: numpy.array
    """
    return service_battery_currents * (service_battery_nominal_voltage / 1000.0)


@sh.add_function(dsp, outputs=['service_battery_electric_powers'], weight=1)
def calculate_service_battery_electric_powers_v1(
        service_battery_loads, alternator_electric_powers,
        dcdc_converter_electric_powers):
    """
    Calculate the service battery electric power [kW].

    :param service_battery_loads:
        Service battery load vector [kW].
    :type service_battery_loads: numpy.array

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array

    :return:
        Service battery electric power [kW].
    :rtype: numpy.array
    """
    p = service_battery_loads - alternator_electric_powers
    p -= dcdc_converter_electric_powers
    return p


@sh.add_function(dsp, outputs=['initial_service_battery_state_of_charge'])
def identify_initial_service_battery_state_of_charge(
        service_battery_state_of_charges):
    """
    Identify the initial state of charge of service battery [%].

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :return:
        Initial state of charge of the service battery [%].
    :rtype: float
    """
    return service_battery_state_of_charges[0]


@sh.add_function(
    dsp, outputs=['initial_service_battery_state_of_charge'], weight=10
)
def default_initial_service_battery_state_of_charge(cycle_type):
    """
    Return the default initial state of charge of service battery [%].

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :return:
        Initial state of charge of the service battery [%].
    :rtype: float
    """
    d = dfl.functions.default_initial_service_battery_state_of_charge
    return d.initial_state_of_charge[cycle_type]


@sh.add_function(dsp, outputs=['service_battery_state_of_charges'])
def calculate_service_battery_state_of_charges(
        service_battery_capacity, initial_service_battery_state_of_charge,
        times, service_battery_currents):
    """
    Calculates the state of charge of the service battery [%].

    :param service_battery_capacity:
        Service battery capacity [Ah].
    :type service_battery_capacity: float

    :param initial_service_battery_state_of_charge:
        Initial state of charge of the service battery [%].
    :type initial_service_battery_state_of_charge: float

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_currents:
        Service battery current vector [A].
    :type service_battery_currents: numpy.array

    :return:
        State of charge of the service battery [%].
    :rtype: numpy.array
    """

    soc = np.empty_like(times, float)
    soc[0] = initial_service_battery_state_of_charge
    bc = (service_battery_currents[:-1] + service_battery_currents[1:])
    bc *= np.diff(times)
    bc /= 2.0 * service_battery_capacity * 36.0

    for i, v in enumerate(bc, 1):
        soc[i] = min(soc[i - 1] + v, 100.0)

    return soc


@sh.add_function(dsp, outputs=['service_battery_loads'])
def calculate_service_battery_loads(
        service_battery_electric_powers, alternator_electric_powers,
        dcdc_converter_electric_powers):
    """
    Calculates service battery load vector [kW].

    :param service_battery_electric_powers:
        Service battery electric power [kW].
    :type service_battery_electric_powers: numpy.array

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array

    :return:
        Service battery load vector [kW].
    :rtype: numpy.array
    """
    p = service_battery_electric_powers - alternator_electric_powers
    p -= dcdc_converter_electric_powers
    return p


@sh.add_function(dsp, outputs=['service_battery_loads'])
def calculate_service_battery_loads_v1(on_engine, service_battery_load):
    """
    Calculates service battery load vector [kW].

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param service_battery_load:
        Service electric load (engine off and on) [kW].
    :type service_battery_load: float, float

    :return:
        Service battery load vector [kW].
    :rtype: numpy.array
    """
    return np.where(on_engine, *service_battery_load[::-1])


@sh.add_function(dsp, outputs=['service_battery_load'])
def identify_service_battery_load(
        service_battery_loads, engine_powers_out, on_engine):
    """
    Identifies service electric load (engine off and on) [kW].

    :param service_battery_loads:
        Service battery load vector [kW].
    :type service_battery_loads: numpy.array

    ::param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :return:
        Service electric load (engine off and on) [kW].
    :rtype: float, float
    """
    rjo, mae = co2_utl.reject_outliers, co2_utl.mae
    p, b = service_battery_loads, engine_powers_out >= -dfl.EPS
    on = min(0.0, co2_utl.reject_outliers(p[on_engine & b], med=np.mean)[0])
    off, b_off = on, b & ~on_engine & (p < 0)
    if b_off.any():
        off = rjo(p[b_off], med=np.mean)[0]
        if on > off:
            p = p[b]
            if mae(p, on) > mae(p, off):
                on = off
            else:
                off = on
    return off, on


@sh.add_function(dsp, outputs=['service_battery_delta_state_of_charge'])
def calculate_service_battery_delta_state_of_charge(
        service_battery_state_of_charges):
    """
    Calculates the overall delta state of charge of the service battery [%].

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :return:
        Overall delta state of charge of the service battery [%].
    :rtype: float
    """
    soc = service_battery_state_of_charges
    return soc[-1] - soc[0]
