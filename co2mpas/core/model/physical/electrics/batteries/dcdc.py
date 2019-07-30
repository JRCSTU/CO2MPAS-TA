# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the DC/DC Converter.
"""

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
    return dcdc_converter_electric_powers / dcdc_converter_efficiency


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
    return dcdc_converter_currents * service_battery_nominal_voltage / 1000


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
        dcdc_converter_currents, on_engine, times,
        service_battery_state_of_charges, service_battery_charging_statuses,
        clutch_tc_powers, accelerations, service_battery_initialization_time):
    """
    Calibrates an alternator current model that predicts alternator current [A].

    :param dcdc_converter_currents:
        DC/DC converter currents [A].
    :type dcdc_converter_currents: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

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

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param service_battery_initialization_time:
        Service battery initialization time delta [s].
    :type service_battery_initialization_time: float

    :return:
        DC/DC converter current model.
    :rtype: callable
    """
    from ..motors.alternator.current import calibrate_alternator_current_model
    return calibrate_alternator_current_model(
        dcdc_converter_currents, on_engine, times,
        service_battery_state_of_charges, service_battery_charging_statuses,
        clutch_tc_powers, accelerations, service_battery_initialization_time
    )
