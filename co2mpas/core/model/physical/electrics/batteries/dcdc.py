# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the DC/DC Converter.
"""
import numpy as np
import schedula as sh

dsp = sh.BlueDispatcher(
    name='DC/DC Converter',
    description='Models the DC/DC Converter.'
)


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True,
    outputs=['dcdc_converter_electric_powers_demand']
)
def calculate_dcdc_converter_electric_powers_demand(
        dcdc_converter_electric_powers, dcdc_converter_efficiency=.95):
    """
    Calculate DC/DC converter electric power demand [kW].

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array | float

    :param dcdc_converter_efficiency:
        DC/DC converter efficiency [-].
    :type dcdc_converter_efficiency: float

    :return:
        DC/DC converter electric power demand [kW].
    :rtype: numpy.array | float
    """
    from ..motors.p4 import calculate_motor_p4_powers_v1 as func
    return func(dcdc_converter_electric_powers, dcdc_converter_efficiency)


@sh.add_function(dsp, outputs=['dcdc_converter_electric_powers'])
def calculate_dcdc_converter_electric_powers_v1(
        dcdc_converter_electric_powers_demand, dcdc_converter_efficiency):
    """
    Calculate DC/DC converter electric power [kW].

    :param dcdc_converter_electric_powers_demand:
        DC/DC converter electric power demand [kW].
    :type dcdc_converter_electric_powers_demand: numpy.array | float

    :param dcdc_converter_efficiency:
        DC/DC converter efficiency [-].
    :type dcdc_converter_efficiency: float

    :return:
        DC/DC converter electric power [kW].
    :rtype: numpy.array | float
    """
    from ..motors.p4 import calculate_motor_p4_electric_powers as f
    return f(dcdc_converter_electric_powers_demand, dcdc_converter_efficiency)


@sh.add_function(dsp, outputs=['dcdc_converter_electric_powers'])
def calculate_dcdc_converter_electric_powers(
        dcdc_converter_currents, service_battery_nominal_voltage):
    """
    Calculates DC/DC converter electric powers [kW].

    :param dcdc_converter_currents:
        DC/DC converter currents [A].
    :type dcdc_converter_currents: numpy.array | float

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :return:
        DC/DC converter electric power [kW].
    :rtype: numpy.array | float
    """
    return dcdc_converter_currents * (service_battery_nominal_voltage / 1000)


@sh.add_function(dsp, outputs=['dcdc_converter_currents'])
def calculate_dcdc_converter_currents(
        dcdc_converter_electric_powers, service_battery_nominal_voltage):
    """
    Calculate DC/DC converter current [A].

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array | float

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :return:
        DC/DC converter currents [A].
    :rtype: numpy.array | float
    """
    c = 1000 / service_battery_nominal_voltage
    return dcdc_converter_electric_powers * c


@sh.add_function(dsp, outputs=['dcdc_current_model'])
def define_dcdc_current_model(dcdc_charging_currents):
    """
    Defines an DC/DC converter current model that predicts its current [A].

    :param dcdc_charging_currents:
        Mean charging currents of the DC/DC converter (for negative and positive
        power)[A].
    :type dcdc_charging_currents: (float, float)

    :return:
        DC/DC converter current model.
    :rtype: callable
    """
    from ..motors.alternator.current import define_alternator_current_model
    return define_alternator_current_model(dcdc_charging_currents)


@sh.add_function(dsp, outputs=['dcdc_current_model'])
def calibrate_dcdc_current_model(
        dcdc_converter_currents, times, service_battery_state_of_charges,
        service_battery_charging_statuses, service_battery_initialization_time):
    """
    Calibrates an DC/DC current model that predicts DC/DC current [A].

    :param dcdc_converter_currents:
        DC/DC converter currents [A].
    :type dcdc_converter_currents: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :param service_battery_charging_statuses:
        Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-].
    :type service_battery_charging_statuses: numpy.array

    :param service_battery_initialization_time:
        Service battery initialization time delta [s].
    :type service_battery_initialization_time: float

    :return:
        DC/DC converter current model.
    :rtype: callable
    """
    from ..motors.alternator.current import AlternatorCurrentModel
    statuses = service_battery_charging_statuses
    return AlternatorCurrentModel().fit(
        dcdc_converter_currents, np.in1d(statuses, (1, 3)), times,
        service_battery_state_of_charges, statuses,
        init_time=service_battery_initialization_time
    )


@sh.add_function(dsp, outputs=['dcdc_converter_currents'])
def predict_dcdc_converter_currents(
        dcdc_current_model, times, service_battery_state_of_charges,
        service_battery_charging_statuses, service_battery_initialization_time):
    """
    Predict DC/DC converter current [A].

    :param dcdc_current_model:
        DC/DC converter current model.
    :type dcdc_current_model: callable

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :param service_battery_charging_statuses:
        Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-].
    :type service_battery_charging_statuses: numpy.array

    :param service_battery_initialization_time:
        Service battery initialization time delta [s].
    :type service_battery_initialization_time: float

    :return:
        DC/DC converter currents [A].
    :rtype: numpy.array
    """
    return dcdc_current_model.predict(np.column_stack((
        times, service_battery_state_of_charges,
        service_battery_charging_statuses
    )), init_time=service_battery_initialization_time)


@sh.add_function(dsp, outputs=['dcdc_converter_currents'], weight=sh.inf(10, 3))
def default_dcdc_converter_currents(times, is_hybrid):
    """
    Return zero current if the vehicle is not hybrid [A].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        DC/DC converter currents [A].
    :rtype: numpy.array
    """
    from ..motors.p4 import default_motor_p4_powers as func
    return func(times, is_hybrid)
