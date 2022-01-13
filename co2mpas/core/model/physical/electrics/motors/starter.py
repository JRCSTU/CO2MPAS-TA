# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the starter.
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl

dsp = sh.BlueDispatcher(
    name='Starter',
    description='Models the starter (motor downstream the engine).'
)

dsp.add_data('starter_efficiency', dfl.values.starter_efficiency)
dsp.add_data('delta_time_engine_starter', dfl.values.delta_time_engine_starter)


@sh.add_function(dsp, outputs=['start_demand_function'])
def define_engine_start_demand_function(engine_moment_inertia):
    """
    Define the energy required to start the engine function [kJ].

    :param engine_moment_inertia:
        Engine moment of inertia [kg*m2].
    :type engine_moment_inertia: float

    :return:
        Energy required to start the engine function.
    :rtype: function
    """
    coef = (np.pi / 30) ** 2 / 2000 * engine_moment_inertia

    def _func(start_engine_speed):
        return coef * start_engine_speed ** 2

    return _func


@sh.add_function(dsp, outputs=['starter_powers'])
def calculate_starter_powers(
        start_demand_function, times, on_engine, delta_time_engine_starter,
        engine_speeds_out):
    """
    Calculates starter power [kW].

    :param start_demand_function:
        Energy required to start the engine function.
    :type start_demand_function: function

    :param times:
        Time vector.
    :type times: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param delta_time_engine_starter:
        Time elapsed to turn on the engine with electric starter [s].
    :type delta_time_engine_starter: float

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :return:
        Starter power [kW].
    :rtype: numpy.array
    """
    i = np.where(np.bitwise_xor(on_engine[:-1], on_engine[1:]))[0]
    start = on_engine[i + 1]
    j = np.searchsorted(times, times[i] + delta_time_engine_starter + dfl.EPS)
    e = start_demand_function(engine_speeds_out[i + start.astype(int)])
    e /= (times.take(j, mode='clip') - times[i])
    e[~start] *= -1
    p = np.zeros_like(times, float)
    for i, j, e in zip(i, j, e):
        p[i:j] += e
    return p


@sh.add_function(dsp, outputs=['starter_electric_powers'])
def calculate_starter_electric_powers(starter_powers, starter_efficiency):
    """
    Calculates starter electric power [kW].

    :param starter_powers:
        Starter power [kW].
    :type starter_powers: numpy.array | float

    :param starter_efficiency:
        Starter efficiency [-].
    :type starter_efficiency: float

    :return:
        Starter electric power [kW].
    :rtype: numpy.array | float
    """
    eff = starter_efficiency
    return starter_powers * np.where(starter_powers <= 0, eff, 1 / eff)


@sh.add_function(dsp, outputs=['starter_currents'])
def calculate_starter_currents(
        starter_electric_powers, starter_nominal_voltage):
    """
    Calculates starter currents [A].

    :param starter_electric_powers:
        Starter electric power [kW].
    :type starter_electric_powers: numpy.array | float

    :param starter_nominal_voltage:
        Starter nominal voltage [V].
    :type starter_nominal_voltage: float

    :return:
        Starter currents [A].
    :rtype: numpy.array | float
    """
    from .alternator import calculate_alternator_currents as func
    return func(starter_electric_powers, starter_nominal_voltage)


# noinspection PyMissingOrEmptyDocstring
class StarterModel:
    def __init__(self, start_demand_function, starter_nominal_voltage,
                 starter_efficiency, delta_time_engine_starter):
        self.power_demand = start_demand_function
        self.nominal_voltage = starter_nominal_voltage
        self.efficiency = starter_efficiency
        self.time = delta_time_engine_starter

    def __call__(self, start_engine_speed):
        return calculate_starter_electric_powers(
            self.power_demand(start_engine_speed), self.efficiency
        )


@sh.add_function(dsp, outputs=['starter_model'])
def define_starter_model(
        start_demand_function, starter_nominal_voltage, starter_efficiency,
        delta_time_engine_starter):
    """
    Defines the starter model.

    :param start_demand_function:
        Energy required to start the engine function.
    :type start_demand_function: function

    :param starter_nominal_voltage:
        Starter nominal voltage [V].
    :type starter_nominal_voltage: float

    :param starter_efficiency:
        Starter efficiency [-].
    :type starter_efficiency: float

    :param delta_time_engine_starter:
        Time elapsed to turn on the engine with electric starter [s].
    :type delta_time_engine_starter: float

    :return:
        Starter model.
    :rtype: StarterModel
    """
    return StarterModel(
        start_demand_function, starter_nominal_voltage, starter_efficiency,
        delta_time_engine_starter
    )
