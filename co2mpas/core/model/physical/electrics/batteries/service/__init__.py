# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the service battery (low voltage).

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics.batteries.service

.. autosummary::
    :nosignatures:
    :toctree: service/

    status
"""
import functools
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
import co2mpas.utils as co2_utl
from .status import dsp as _status

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


@sh.add_function(dsp, outputs=['service_battery_capacity_kwh'])
def calculate_service_battery_capacity_kwh(
        service_battery_capacity, service_battery_nominal_voltage):
    """
    Calculate service battery capacity [kWh].

    :param service_battery_capacity:
        Service battery capacity [Ah].
    :type service_battery_capacity: float

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :return:
        Service battery capacity [kWh].
    :rtype: float
    """
    return service_battery_capacity * service_battery_nominal_voltage / 1000.0


@sh.add_function(dsp, outputs=['service_battery_capacity'])
def calculate_service_battery_capacity(
        service_battery_capacity_kwh, service_battery_nominal_voltage):
    """
    Calculate service battery capacity [Ah].

    :param service_battery_capacity_kwh:
        Service battery capacity [kWh].
    :type service_battery_capacity_kwh: float

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :return:
        Service battery capacity [Ah].
    :rtype: float
    """
    return service_battery_capacity_kwh * 1000 / service_battery_nominal_voltage


@sh.add_function(dsp, outputs=['service_battery_nominal_voltage'])
def calculate_service_battery_nominal_voltage(
        service_battery_capacity_kwh, service_battery_capacity):
    """
    Calculate service battery nominal voltage [V].

    :param service_battery_capacity_kwh:
        Service battery capacity [kWh].
    :type service_battery_capacity_kwh: float

    :param service_battery_capacity:
        Service battery capacity [Ah].
    :type service_battery_capacity: float

    :return:
        Service battery nominal voltage [V].
    :rtype: float
    """
    return service_battery_capacity_kwh * 1000 / service_battery_capacity


@sh.add_function(dsp, outputs=['service_battery_currents'], weight=sh.inf(1, 0))
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
    from scipy.interpolate import UnivariateSpline as Spline
    soc = service_battery_state_of_charges
    ib = Spline(times, soc, w=np.tile(10, times.shape[0])).derivative()(times)
    return ib * (service_battery_capacity * 36.0)


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
    soc = service_battery_state_of_charges
    ib = calculate_service_battery_currents_v1(1, times, soc)
    b = (ib < -dfl.EPS) | (ib > dfl.EPS)
    return co2_utl.reject_outliers(service_battery_currents[b] / ib[b])[0]


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


@sh.add_function(dsp, outputs=['service_battery_electric_powers_supply'])
def calculate_service_battery_electric_powers_supply(
        alternator_electric_powers, dcdc_converter_electric_powers,
        starter_electric_powers):
    """
    Calculate the service battery electric power supply [kW].

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array

    :param starter_electric_powers:
        Starter electric power [kW].
    :type starter_electric_powers: numpy.array

    :return:
        Service battery electric power supply [kW].
    :rtype: numpy.array
    """
    p = alternator_electric_powers + dcdc_converter_electric_powers
    p += starter_electric_powers
    return p


@sh.add_function(dsp, outputs=['service_battery_electric_powers'], weight=1)
def calculate_service_battery_electric_powers_v1(
        service_battery_loads, service_battery_electric_powers_supply):
    """
    Calculate the service battery electric power [kW].

    :param service_battery_loads:
        Service battery load vector [kW].
    :type service_battery_loads: numpy.array

    :param service_battery_electric_powers_supply:
        Service battery electric power supply [kW].
    :type service_battery_electric_powers_supply: numpy.array

    :return:
        Service battery electric power [kW].
    :rtype: numpy.array
    """
    return service_battery_loads - service_battery_electric_powers_supply


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
        soc[i] = max(0.0, min(soc[i - 1] + v, 100.0))

    return soc


@sh.add_function(dsp, outputs=['service_battery_loads'])
def calculate_service_battery_loads(
        service_battery_electric_powers,
        service_battery_electric_powers_supply):
    """
    Calculates service battery load vector [kW].

    :param service_battery_electric_powers:
        Service battery electric power [kW].
    :type service_battery_electric_powers: numpy.array

    :param service_battery_electric_powers_supply:
        Service battery electric power supply [kW].
    :type service_battery_electric_powers_supply: numpy.array

    :return:
        Service battery load vector [kW].
    :rtype: numpy.array
    """
    p = service_battery_electric_powers + service_battery_electric_powers_supply
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

    :param engine_powers_out:
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


dsp.add_dispatcher(
    dsp_id='status_model',
    dsp=_status,
    inputs=(
        'service_battery_start_window_width', 'service_battery_nominal_voltage',
        'service_battery_state_of_charge_balance', 'on_engine', 'accelerations',
        'alternator_electric_powers', 'dcdc_converter_electric_powers', 'times',
        'service_battery_initialization_time', 'service_battery_status_model',
        'service_battery_electric_powers_supply_threshold', 'motive_powers',
        'service_battery_state_of_charges', 'engine_starts', 'is_hybrid',
        'service_battery_state_of_charge_balance_window',
        'service_battery_capacity',
    ),
    outputs=(
        'service_battery_initialization_time', 'service_battery_status_model',
        'service_battery_electric_powers_supply_threshold',
        'service_battery_state_of_charge_balance_window',
        'service_battery_state_of_charge_balance',
        'service_battery_charging_statuses',
    )
)


# noinspection PyMissingOrEmptyDocstring
class ServiceBatteryModel:
    def __init__(self, service_battery_status_model, dcdc_current_model,
                 alternator_current_model, has_energy_recuperation,
                 service_battery_initialization_time, service_battery_load,
                 initial_service_battery_state_of_charge,
                 service_battery_nominal_voltage, service_battery_capacity):
        self.status = functools.partial(
            service_battery_status_model.predict, has_energy_recuperation,
            service_battery_initialization_time
        )
        self.nominal_voltage = service_battery_nominal_voltage
        self.dcdc = dcdc_current_model
        self.alternator = alternator_current_model
        self.current_load = np.divide(
            service_battery_load, service_battery_nominal_voltage / 1e3
        )
        self._d_soc = service_battery_capacity * 36.0 * 2
        self.init_soc = initial_service_battery_state_of_charge
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self._prev_status = 0
        self._prev_time = 0
        self._prev_current = 0
        self._prev_soc = self.init_soc

    def __call__(self, time, motive_power, acceleration, on_engine,
                 starter_current=0, prev_soc=None, prev_status=None,
                 update=True):
        if prev_status is None:
            prev_status = self._prev_status
        if prev_soc is None:
            prev_soc = self._prev_soc
        alt_c = dcdc_c = .0
        status_ = self.status(time, prev_status, prev_soc, motive_power)
        if status_:
            if status_ == 1:
                dcdc_c = self.dcdc(time, prev_soc, status_)
            if on_engine:
                alt_c = self.alternator(
                    time, prev_soc, status_, motive_power, acceleration
                )
        c = self.current_load[int(on_engine)] - alt_c - dcdc_c - starter_current
        dsoc = (c + self._prev_current) * (time - self._prev_time) / self._d_soc
        soc = max(0.0, min(prev_soc + dsoc, 100.0))

        if update:
            self._prev_status, self._prev_soc = status_, soc
            self._prev_time, self._prev_current = time, c

        return soc, status_, dcdc_c, alt_c


dsp.add_data('has_energy_recuperation', dfl.values.has_energy_recuperation)


@sh.add_function(dsp, outputs=['service_battery_model'])
def define_service_battery_model(
        service_battery_status_model, dcdc_current_model,
        alternator_current_model, has_energy_recuperation,
        service_battery_initialization_time, service_battery_load,
        initial_service_battery_state_of_charge,
        service_battery_nominal_voltage, service_battery_capacity):
    """
    Define a service battery model.

    :param service_battery_status_model:
        A function that predicts the service battery charging status.
    :type service_battery_status_model: BatteryStatusModel

    :param dcdc_current_model:
        DC/DC converter current model.
    :type dcdc_current_model: callable

    :param alternator_current_model:
        Alternator current model.
    :type alternator_current_model: callable

    :param has_energy_recuperation:
        Is the vehicle equipped with any brake energy recuperation technology?
    :type has_energy_recuperation: bool

    :param service_battery_initialization_time:
        Service battery initialization time delta [s].
    :type service_battery_initialization_time: float

    :param service_battery_load:
        Service electric load (engine off and on) [kW].
    :type service_battery_load: float, float

    :param initial_service_battery_state_of_charge:
        Initial state of charge of the service battery [%].
    :type initial_service_battery_state_of_charge: float

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :param service_battery_capacity:
        Service battery capacity [Ah].
    :type service_battery_capacity: float

    :return:
        Service battery model.
    :rtype: ServiceBatteryModel
    """
    return ServiceBatteryModel(
        service_battery_status_model, dcdc_current_model,
        alternator_current_model, has_energy_recuperation,
        service_battery_initialization_time, service_battery_load,
        initial_service_battery_state_of_charge,
        service_battery_nominal_voltage, service_battery_capacity
    )
