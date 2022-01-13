# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the drive battery (high voltage).
"""

import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
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


@sh.add_function(dsp, outputs=['drive_battery_capacity_kwh'])
def calculate_drive_battery_capacity_kwh(
        drive_battery_capacity, drive_battery_nominal_voltage):
    """
    Calculate drive battery capacity [kWh].

    :param drive_battery_capacity:
        Drive battery capacity [Ah].
    :type drive_battery_capacity: float

    :param drive_battery_nominal_voltage:
        Drive battery nominal voltage [V].
    :type drive_battery_nominal_voltage: float

    :return:
        Drive battery capacity [kWh].
    :rtype: float
    """
    from .service import calculate_service_battery_capacity_kwh as func
    return func(drive_battery_capacity, drive_battery_nominal_voltage)


@sh.add_function(dsp, outputs=['drive_battery_capacity'])
def calculate_drive_battery_capacity(
        drive_battery_capacity_kwh, drive_battery_nominal_voltage):
    """
    Calculate drive battery capacity [Ah].

    :param drive_battery_capacity_kwh:
        Drive battery capacity [kWh].
    :type drive_battery_capacity_kwh: float

    :param drive_battery_nominal_voltage:
        Drive battery nominal voltage [V].
    :type drive_battery_nominal_voltage: float

    :return:
        Drive battery capacity [Ah].
    :rtype: float
    """
    from .service import calculate_service_battery_capacity as func
    return func(drive_battery_capacity_kwh, drive_battery_nominal_voltage)


@sh.add_function(dsp, outputs=['drive_battery_nominal_voltage'])
def calculate_drive_battery_nominal_voltage(
        drive_battery_capacity_kwh, drive_battery_capacity):
    """
    Calculate drive battery nominal voltage [V].

    :param drive_battery_capacity_kwh:
        Drive battery capacity [kWh].
    :type drive_battery_capacity_kwh: float

    :param drive_battery_capacity:
        Drive battery capacity [Ah].
    :type drive_battery_capacity: float

    :return:
        Drive battery nominal voltage [V].
    :rtype: float
    """
    from .service import calculate_service_battery_nominal_voltage as func
    return func(drive_battery_capacity_kwh, drive_battery_capacity)


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
        Cumulative motors electric power [kW].
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
        Cumulative motors electric power [kW].
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


dsp.add_data('drive_battery_load', 0, sh.inf(11, 0))


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


# noinspection PyMissingOrEmptyDocstring,PyProtectedMember
class DriveBatteryModel:
    def __init__(self, service_battery_model, drive_battery_load,
                 initial_drive_battery_state_of_charge, drive_battery_capacity,
                 dcdc_converter_efficiency=.95, r0=None, ocv=None,
                 n_parallel_cells=1, n_series_cells=1):
        self.r0, self.ocv = r0, ocv
        self.np, self.ns = n_parallel_cells, n_series_cells
        r0 is not None and ocv is not None and self.compile()
        self.service = service_battery_model
        self.dcdc_efficiency = dcdc_converter_efficiency
        self._d_soc = drive_battery_capacity * 36.0 * 2
        self.init_soc = initial_drive_battery_state_of_charge
        self.drive_battery_load = drive_battery_load
        self.reset()
        self._dcdc_p = self.service.nominal_voltage / 1e3

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.service.reset()
        self._prev_current = 0
        self._prev_soc = self.init_soc

    # noinspection PyAttributeOutsideInit
    def compile(self):
        self._ocv = self.ocv * self.ns / 1e3
        self.constant = self.r0 <= 0

        self.v_b = self.ocv * self.ns / 2
        self.v_b2 = self.v_b ** 2

        if self.constant:
            self._r0 = self.v_ac4 = 0
        else:
            n = self.np / self.r0
            self.c_b, self.c_ac4 = -self.ocv * n / 2, 1e3 / self.ns * n
            self.c_b2 = self.c_b ** 2

            self.v_ac4 = self.ns / n * 1e3

            self._r0 = self.ns / n / 1e3

    def fit(self, currents, voltages):
        from sklearn.linear_model import RANSACRegressor
        m = RANSACRegressor(random_state=0)
        try:
            m.fit(currents[:, None], voltages)
            r0, ocv = float(m.estimator_.coef_), float(m.estimator_.intercept_)
        except ValueError:  # RANSAC failure switch to constant model.
            r0 = ocv = dfl.EPS
        if r0 <= dfl.EPS:
            r0, ocv = 0, np.mean(voltages)
        self.r0 = r0 * self.np / self.ns
        self.ocv = ocv / self.ns
        self.compile()
        return self

    def currents(self, powers):
        if self.constant:
            return powers / self._ocv
        cur = np.sqrt(np.maximum(self.c_b2 + self.c_ac4 * powers, 0)) + self.c_b
        return cur

    def powers(self, currents):
        return (self._ocv + currents * self._r0) * currents

    def voltages(self, powers):
        vol = self.v_b + np.sqrt(np.maximum(self.v_b2 + self.v_ac4 * powers, 0))
        return vol

    def __call__(self, current, time, motive_power, acceleration, on_engine,
                 starter_current=0, prev_soc=None, update=True,
                 service_kw=None):
        dt = time - self.service._prev_time
        from ..motors.p4 import calculate_motor_p4_powers_v1 as f
        dcdc_p = f(self.service(
            time, motive_power, acceleration, on_engine, starter_current,
            update=update, **(service_kw or {})
        )[2] * self._dcdc_p, self.dcdc_efficiency) + self.drive_battery_load

        if prev_soc is None:
            prev_soc = self._prev_soc

        current = current + self.currents(dcdc_p)
        dsoc = (current + self._prev_current) * dt / self._d_soc
        soc = max(0.0, min(prev_soc + dsoc, 100.0))

        if update:
            self._prev_soc, self._prev_current = soc, current

        return soc


@sh.add_function(dsp, outputs=['drive_battery_model'])
def calibrate_drive_battery_model(
        service_battery_model, initial_drive_battery_state_of_charge,
        drive_battery_capacity, drive_battery_n_parallel_cells,
        drive_battery_n_series_cells, drive_battery_currents,
        dcdc_converter_efficiency, drive_battery_load, drive_battery_voltages):
    """
    Calibrate the drive battery current model.

    :param service_battery_model:
         Service battery model.
    :type service_battery_model: ServiceBatteryModel

    :param initial_drive_battery_state_of_charge:
        Initial state of charge of the drive battery [%].
    :type initial_drive_battery_state_of_charge: float

    :param drive_battery_capacity:
        Drive battery capacity [Ah].
    :type drive_battery_capacity: float

    :param drive_battery_n_parallel_cells:
        Number of battery cells in parallel [-].
    :type drive_battery_n_parallel_cells: int

    :param drive_battery_n_series_cells:
        Number of battery cells in series [-].
    :type drive_battery_n_series_cells: int

    :param drive_battery_currents:
        Drive battery current vector [A].
    :type drive_battery_currents: numpy.array

    :param dcdc_converter_efficiency:
        DC/DC converter efficiency [-].
    :type dcdc_converter_efficiency: float

    :param drive_battery_load:
        Drive electric load [kW].
    :type drive_battery_load: float

    :param drive_battery_voltages:
        Drive battery voltage [V].
    :type drive_battery_voltages: numpy.array

    :return:
        Drive battery current model.
    :rtype: DriveBatteryModel
    """
    return DriveBatteryModel(
        service_battery_model, drive_battery_load,
        initial_drive_battery_state_of_charge, drive_battery_capacity,
        dcdc_converter_efficiency,
        n_parallel_cells=drive_battery_n_parallel_cells,
        n_series_cells=drive_battery_n_series_cells
    ).fit(drive_battery_currents, drive_battery_voltages)


@sh.add_function(dsp, outputs=['drive_battery_r0', 'drive_battery_ocv'])
def get_drive_battery_r0_and_ocv(drive_battery_model):
    """
    Returns drive battery resistance and open circuit voltage [ohm, V].

    :param drive_battery_model:
        Drive battery current model.
    :type drive_battery_model: DriveBatteryModel

    :return:
        Drive battery resistance and open circuit voltage [ohm, V].
    :rtype: float, float
    """
    return drive_battery_model.r0, drive_battery_model.ocv


dsp.add_data('drive_battery_n_cells', 1)


@sh.add_function(dsp, outputs=['drive_battery_nominal_voltage'])
def identify_drive_battery_nominal_voltage(drive_battery_voltages):
    """
    Identify the drive battery nominal voltage [V].

    :param drive_battery_voltages:
        Drive battery voltage [V].
    :type drive_battery_voltages: numpy.array

    :return:
        Drive battery nominal voltage [V].
    :rtype: float
    """
    return np.median(drive_battery_voltages[drive_battery_voltages > dfl.EPS])


@sh.add_function(dsp, outputs=['drive_battery_voltages'], weight=sh.inf(15, 0))
def define_drive_battery_voltages(times, drive_battery_nominal_voltage):
    """
    Defines drive battery voltage vector [V].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param drive_battery_nominal_voltage:
        Drive battery nominal voltage [V].
    :type drive_battery_nominal_voltage: float

    :return:
        Drive battery voltage [V].
    :rtype: numpy.array
    """
    return np.tile(drive_battery_nominal_voltage, times.size)


@sh.add_function(dsp, outputs=['drive_battery_n_parallel_cells'])
def calculate_drive_battery_n_parallel_cells(
        drive_battery_n_cells, drive_battery_n_series_cells):
    """
    Calculate the number of battery cells in parallel [-].

    :param drive_battery_n_cells:
        Number of battery cells [-].
    :type drive_battery_n_cells: int

    :param drive_battery_n_series_cells:
        Number of battery cells in series [-].
    :type drive_battery_n_series_cells: int

    :return:
        Number of battery cells in parallel [-].
    :rtype: int
    """
    return drive_battery_n_cells / drive_battery_n_series_cells


dsp.add_data('drive_battery_technology', dfl.values.drive_battery_technology)


@sh.add_function(dsp, outputs=['drive_battery_n_parallel_cells'], weight=5)
def calculate_drive_battery_n_parallel_cells_v1(
        drive_battery_technology, drive_battery_nominal_voltage,
        drive_battery_n_cells):
    """
    Calculate the number of battery cells in parallel [-].

    :param drive_battery_technology:
        Drive battery technology type (e.g., NiMH, Li-NCA (Li-Ni-Co-Al), etc.).
    :type drive_battery_technology: str

    :param drive_battery_n_cells:
        Number of battery cells [-].
    :type drive_battery_n_cells: int

    :param drive_battery_nominal_voltage:
        Drive battery nominal voltage [V].
    :type drive_battery_nominal_voltage: float

    :return:
        Number of battery cells in parallel [-].
    :rtype: int
    """
    v = dfl.functions.calculate_drive_battery_n_parallel_cells_v1.reference_volt
    v = v.get(drive_battery_technology, v['unknown'])
    n = np.ceil(v / drive_battery_nominal_voltage * drive_battery_n_cells)
    n = int(min(n, drive_battery_n_cells))
    while n < drive_battery_n_cells and drive_battery_n_cells % n:
        n += 1
    return n


@sh.add_function(dsp, outputs=['drive_battery_n_series_cells'])
def calculate_drive_battery_n_series_cells(
        drive_battery_n_cells, drive_battery_n_parallel_cells):
    """
    Calculate the number of battery cells in parallel [-].

    :param drive_battery_n_cells:
        Number of battery cells [-].
    :type drive_battery_n_cells: int

    :param drive_battery_n_parallel_cells:
        Number of battery cells in parallel [-].
    :type drive_battery_n_parallel_cells: int

    :return:
        Number of battery cells in series [-].
    :rtype: int
    """
    return drive_battery_n_cells / drive_battery_n_parallel_cells


@sh.add_function(dsp, outputs=['drive_battery_n_cells'])
def calculate_drive_battery_n_cells(
        drive_battery_n_parallel_cells, drive_battery_n_series_cells):
    """
    Calculate the number of battery cells [-].

    :param drive_battery_n_parallel_cells:
        Number of battery cells in parallel [-].
    :type drive_battery_n_parallel_cells: int

    :param drive_battery_n_series_cells:
        Number of battery cells in series [-].
    :type drive_battery_n_series_cells: int

    :return:
        Number of battery cells [-].
    :rtype: int
    """
    return drive_battery_n_parallel_cells * drive_battery_n_series_cells


@sh.add_function(dsp, outputs=['drive_battery_model'])
def define_drive_battery_model(
        service_battery_model, drive_battery_load, dcdc_converter_efficiency,
        initial_drive_battery_state_of_charge, drive_battery_capacity,
        drive_battery_r0, drive_battery_ocv, drive_battery_n_parallel_cells,
        drive_battery_n_series_cells):
    """
    Define the drive battery current model.

    :param service_battery_model:
         Service battery model.
    :type service_battery_model: ServiceBatteryModel

    :param dcdc_converter_efficiency:
        DC/DC converter efficiency [-].
    :type dcdc_converter_efficiency: float

    :param drive_battery_load:
        Drive electric load [kW].
    :type drive_battery_load: float

    :param initial_drive_battery_state_of_charge:
        Initial state of charge of the drive battery [%].
    :type initial_drive_battery_state_of_charge: float

    :param drive_battery_capacity:
        Drive battery capacity [Ah].
    :type drive_battery_capacity: float

    :param drive_battery_r0:
        Drive battery resistance [ohm].
    :type drive_battery_r0: float

    :param drive_battery_ocv:
        Drive battery open circuit voltage [V].
    :type drive_battery_ocv: float

    :param drive_battery_n_parallel_cells:
        Number of battery cells in parallel [-].
    :type drive_battery_n_parallel_cells: int

    :param drive_battery_n_series_cells:
        Number of battery cells in series [-].
    :type drive_battery_n_series_cells: int

    :return:
        Drive battery current model.
    :rtype: DriveBatteryModel
    """
    return DriveBatteryModel(
        service_battery_model, drive_battery_load,
        initial_drive_battery_state_of_charge, drive_battery_capacity,
        dcdc_converter_efficiency, drive_battery_r0, drive_battery_ocv,
        drive_battery_n_parallel_cells, drive_battery_n_series_cells
    )


@sh.add_function(dsp, outputs=['drive_battery_voltages'])
def calculate_drive_battery_voltages_v1(
        drive_battery_electric_powers, drive_battery_model):
    """
    Calculate the drive battery current vector [A].

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param drive_battery_model:
        Drive battery current model.
    :type drive_battery_model: DriveBatteryModel

    :return:
        Drive battery current vector [A].
    :rtype: numpy.array
    """
    return drive_battery_model.voltages(drive_battery_electric_powers)


@sh.add_function(dsp, outputs=['motors_electric_powers'])
def calculate_motors_electric_powers(
        motor_p0_electric_powers, motor_p1_electric_powers,
        motor_p2_planetary_electric_powers, motor_p2_electric_powers,
        motor_p3_front_electric_powers, motor_p3_rear_electric_powers,
        motor_p4_front_electric_powers, motor_p4_rear_electric_powers):
    """
    Calculate motors electric power [kW].

    :param motor_p0_electric_powers:
        Electric power of motor P0 [kW].
    :type motor_p0_electric_powers: numpy.array | float

    :param motor_p1_electric_powers:
        Electric power of motor P1 [kW].
    :type motor_p1_electric_powers: numpy.array | float

    :param motor_p2_planetary_electric_powers:
        Electric power of planetary motor P2 [kW].
    :type motor_p2_planetary_electric_powers: numpy.array | float

    :param motor_p2_electric_powers:
        Electric power of motor P2 [kW].
    :type motor_p2_electric_powers: numpy.array | float

    :param motor_p3_front_electric_powers:
        Electric power of motor P3 front [kW].
    :type motor_p3_front_electric_powers: numpy.array | float

    :param motor_p3_rear_electric_powers:
        Electric power of motor P3 rear [kW].
    :type motor_p3_rear_electric_powers: numpy.array | float

    :param motor_p4_front_electric_powers:
        Electric power of motor P4 front [kW].
    :type motor_p4_front_electric_powers: numpy.array | float

    :param motor_p4_rear_electric_powers:
        Electric power of motor P4 rear [kW].
    :type motor_p4_rear_electric_powers: numpy.array | float

    :return:
        Cumulative motors electric power [kW].
    :rtype: numpy.array | float
    """
    p = motor_p0_electric_powers + motor_p1_electric_powers
    p += motor_p2_planetary_electric_powers
    p += motor_p2_electric_powers
    p += motor_p3_front_electric_powers
    p += motor_p3_rear_electric_powers
    p += motor_p4_front_electric_powers
    p += motor_p4_rear_electric_powers
    return p


@sh.add_function(dsp, outputs=['motors_electric_powers'])
def calculate_motors_electric_powers_v1(
        drive_battery_electric_powers, dcdc_converter_electric_powers_demand,
        drive_battery_loads):
    """
    Calculates drive battery load vector [kW].

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param drive_battery_loads:
        Drive battery load vector [kW].
    :type drive_battery_loads: numpy.array

    :param dcdc_converter_electric_powers_demand:
        DC/DC converter electric power demand [kW].
    :type dcdc_converter_electric_powers_demand: numpy.array

    :return:
        Cumulative motors electric power [kW].
    :rtype: numpy.array
    """
    p = drive_battery_loads - drive_battery_electric_powers
    p += dcdc_converter_electric_powers_demand
    return p
