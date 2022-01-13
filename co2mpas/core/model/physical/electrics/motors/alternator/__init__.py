# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the alternator.
Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics.motors.alternator

.. autosummary::
    :nosignatures:
    :toctree: alternator/

    current
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
from .current import dsp as _current

dsp = sh.BlueDispatcher(name='Alternator', description='Models the alternator.')


@sh.add_function(dsp, outputs=['alternator_electric_powers'])
def calculate_alternator_electric_powers(
        alternator_currents, alternator_nominal_voltage):
    """
    Calculates alternator electric powers [kW].

    :param alternator_currents:
        Alternator currents [A].
    :type alternator_currents: numpy.array | float

    :param alternator_nominal_voltage:
        Alternator nominal voltage [V].
    :type alternator_nominal_voltage: float

    :return:
        Alternator electric powers [kW].
    :rtype: numpy.array | float
    """
    return alternator_currents * alternator_nominal_voltage / 1000


@sh.add_function(dsp, outputs=['alternator_currents'])
def calculate_alternator_currents(
        alternator_electric_powers, alternator_nominal_voltage):
    """
    Calculates alternator currents [A].

    :param alternator_electric_powers:
        Alternator power [kW].
    :type alternator_electric_powers: numpy.array | float

    :param alternator_nominal_voltage:
        Alternator nominal voltage [V].
    :type alternator_nominal_voltage: float

    :return:
        Alternator currents [A].
    :rtype: numpy.array | float
    """
    return alternator_electric_powers / alternator_nominal_voltage * 1000


dsp.add_data('alternator_efficiency', dfl.values.alternator_efficiency)


@sh.add_function(dsp, outputs=['alternator_powers'])
def calculate_alternator_powers(
        alternator_electric_powers, alternator_efficiency):
    """
    Calculates alternator power [kW].

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array | float

    :param alternator_efficiency:
        Alternator efficiency [-].
    :type alternator_efficiency: float

    :return:
        Alternator power [kW].
    :rtype: numpy.array | float
    """
    from ..p4 import calculate_motor_p4_powers_v1 as func
    return func(alternator_electric_powers, alternator_efficiency)


@sh.add_function(dsp, outputs=['alternator_electric_powers'])
def calculate_alternator_electric_powers_v1(
        alternator_powers, alternator_efficiency):
    """
    Calculates alternator electric power [kW].

    :param alternator_powers:
        Alternator power [kW].
    :type alternator_powers: numpy.array | float

    :param alternator_efficiency:
        Alternator efficiency [-].
    :type alternator_efficiency: float

    :return:
        Alternator electric power [kW].
    :rtype: numpy.array | float
    """
    from ..p4 import calculate_motor_p4_electric_powers as func
    return func(alternator_powers, alternator_efficiency)


dsp.add_dispatcher(
    dsp_id='current_model',
    dsp=_current,
    inputs=(
        'service_battery_state_of_charges', 'service_battery_charging_statuses',
        'service_battery_initialization_time', 'alternator_charging_currents',
        'accelerations', 'on_engine', 'alternator_currents', 'motive_powers',
        'alternator_current_model', 'times',
    ),
    outputs=('alternator_current_model', 'alternator_currents')
)


@sh.add_function(dsp, outputs=['alternator_currents'], weight=sh.inf(10, 3))
def default_alternator_currents(times, is_hybrid):
    """
    Return zero current if the vehicle is hybrid [A].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        Alternator currents [A].
    :rtype: numpy.array
    """
    if is_hybrid:
        return np.zeros_like(times, float)
    return sh.NONE
