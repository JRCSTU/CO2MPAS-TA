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


@sh.add_function(dsp, outputs=['initial_service_battery_state_of_charge'])
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
    d = dfl.functions.initial_service_battery_state_of_charge
    return d.initial_state_of_charge[cycle_type]


@sh.add_function(dsp, outputs=['maximum_service_battery_charging_current'])
def identify_max_battery_charging_current(service_battery_currents):
    """
    Identifies the maximum charging current of the service battery [A].

    :param service_battery_currents:
        Service battery current vector [A].
    :type service_battery_currents: numpy.array

    :return:
         Maximum charging current of the service battery [A].
    :rtype: float
    """
    return service_battery_currents.max()


@sh.add_function(dsp, outputs=['service_battery_state_of_charges'])
def calculate_service_battery_state_of_charges(
        service_battery_capacity, times,
        initial_service_battery_state_of_charge,
        service_battery_currents, maximum_service_battery_charging_current):
    """
    Calculates the state of charge of the service battery [%].

    :param service_battery_capacity:
        Service battery capacity [Ah].
    :type service_battery_capacity: float

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param initial_service_battery_state_of_charge:
        Initial state of charge of the service battery [%].
    :type initial_service_battery_state_of_charge: float

    :param service_battery_currents:
        Service battery current vector [A].
    :type service_battery_currents: numpy.array

    :param maximum_service_battery_charging_current:
        Maximum charging current of the service battery [A].
    :type maximum_service_battery_charging_current: float

    :return:
        State of charge of the service battery [%].
    :rtype: numpy.array
    """

    soc = np.empty_like(times, float)
    soc[0] = initial_service_battery_state_of_charge
    bc = np.minimum(
        service_battery_currents, maximum_service_battery_charging_current
    )
    bc = (bc[:-1] + bc[1:]) * np.diff(times)
    bc /= 2.0 * service_battery_capacity * 36.0

    for i, v in enumerate(bc, 1):
        soc[i] = min(soc[i - 1] + v, 100.0)

    return soc


@sh.add_function(dsp, outputs=['service_battery_currents'])
def calculate_service_battery_currents(
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


# noinspection PyPep8Naming
@sh.add_function(
    dsp,
    outputs=['service_battery_state_of_charge_balance',
             'service_battery_state_of_charge_balance_window']
)
def identify_service_battery_state_of_charge_balance_and_window(
        alternator_status_model):
    """
    Identify the service battery state of charge balance and window [%].

    :param alternator_status_model:
        A function that predicts the alternator status.
    :type alternator_status_model: AlternatorStatusModel

    :return:
        Service battery state of charge balance and window [%].
    :rtype: float, float
    """
    model = alternator_status_model
    min_soc, max_soc = model.min, model.max
    X = np.column_stack((np.ones(100), np.linspace(min_soc, max_soc, 100)))
    s = np.where(model.charge(X))[0]
    if s.shape[0]:
        min_soc, max_soc = max(min_soc, X[s[0], 1]), min(max_soc, X[s[-1], 1])

    state_of_charge_balance_window = max_soc - min_soc
    state_of_charge_balance = min_soc + state_of_charge_balance_window / 2
    return state_of_charge_balance, state_of_charge_balance_window


def _starts_windows(times, engine_starts, dt):
    ts = times[engine_starts]
    return np.searchsorted(times, np.column_stack((ts - dt, ts + dt + dfl.EPS)))


# noinspection PyPep8Naming,PyPep8
@sh.add_function(dsp, outputs=['service_battery_loads', 'start_demand'])
def identify_service_battery_loads(
        alternator_nominal_voltage, service_battery_currents, alternator_currents,
        gear_box_powers_in, times, on_engine, engine_starts,
        alternator_start_window_width):
    """
    Identifies vehicle electric load and engine start demand [kW].

    :param alternator_nominal_voltage:
        Alternator nominal voltage [V].
    :type alternator_nominal_voltage: float

    :param lv_battery_currents:
        Low voltage battery current vector [A].
    :type lv_battery_currents: numpy.array

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

    :param alternator_start_window_width:
        Alternator start window width [s].
    :type alternator_start_window_width: float

    :return:
        Vehicle electric load (engine off and on) [kW] and energy required to
        start engine [kJ].
    :rtype: ((float, float), float)
    """

    rjo, mae = co2_utl.reject_outliers, co2_utl.mae
    b_c, a_c = service_battery_currents, alternator_currents
    c, b = alternator_nominal_voltage / 1000.0, gear_box_powers_in >= 0

    bH = b & on_engine
    bH = b_c[bH] + a_c[bH]
    on = off = min(0.0, c * rjo(bH, med=np.mean)[0])

    bL = b & ~on_engine & (b_c < 0)
    if bL.any():
        bL = b_c[bL]
        off = min(0.0, c * rjo(bL, med=np.mean)[0])
        if on > off:
            curr = np.append(bL, bH)
            if mae(curr, on / c) > mae(curr, off / c):
                on = off
            else:
                off = on

    loads = [off, on]
    start_demand = []
    dt = alternator_start_window_width / 2
    for i, j in _starts_windows(times, engine_starts, dt):
        p = b_c[i:j] * c
        # noinspection PyUnresolvedReferences
        p[p > 0] = 0.0
        # noinspection PyTypeChecker
        p = np.trapz(p, x=times[i:j])

        if p < 0:
            ld = np.trapz(np.choose(on_engine[i:j], loads), x=times[i:j])
            if p < ld:
                start_demand.append(p - ld)

    start_demand = -rjo(start_demand)[0] if start_demand else 0.0

    return (off, on), start_demand


def calculate_service_battery_currents_v1(
        service_battery_loads, on_engine, alternator_powers,
        dcdc_converter_powers, alternator_nominal_voltage):
    """
    Calculate the service battery current vector [A].

    :param service_battery_loads:
    :type service_battery_loads: float, float

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param alternator_powers:
    :param dcdc_converter_powers:

    :param alternator_nominal_voltage:
        Alternator nominal voltage [V].
    :type alternator_nominal_voltage: float

    :return:
    """
    p = np.where(on_engine, *service_battery_loads[::-1])
    p += alternator_powers + dcdc_converter_powers
    p /= alternator_nominal_voltage
    return p
