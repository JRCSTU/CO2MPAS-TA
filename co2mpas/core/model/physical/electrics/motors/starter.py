# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the starter.
"""
import numpy as np
import schedula as sh
from ...defaults import dfl

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
        start_demand_function, times, engine_starts, delta_time_engine_starter,
        engine_speeds_out):
    """
    Calculates starter power [kW].

    :param start_demand_function:
        Energy required to start the engine function.
    :type start_demand_function: function

    :param times:
        Time vector.
    :type times: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

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
    ts, dt = times[engine_starts], delta_time_engine_starter + dfl.EPS
    k = np.searchsorted(times, np.column_stack((ts, ts + dt)))
    e = start_demand_function(engine_speeds_out[k[:, 1]])
    e /= np.diff(times[k]).ravel()
    p = np.zeros_like(times, float)
    for (i, j), v in zip(k, e):
        p[i:j] += v
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
    return starter_powers / starter_efficiency
