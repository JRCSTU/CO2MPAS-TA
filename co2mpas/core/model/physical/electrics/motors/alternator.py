# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the alternator.
"""
import schedula as sh

dsp = sh.BlueDispatcher(
    name='Alternator',
    description='Models the alternator.'
)


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
    return alternator_electric_powers / alternator_efficiency


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
    return alternator_powers * alternator_efficiency
