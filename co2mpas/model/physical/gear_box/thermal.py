# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions to calculate torque losses and the gear box temperature.
"""

import schedula as sh
import math


def evaluate_gear_box_torque_in(
        gear_box_torque_out, gear_box_speed_in, gear_box_speed_out,
        gear_box_efficiency_parameters):
    """
    Calculates torque required according to the temperature profile [N*m].

    :param gear_box_torque_out:
        Torque gear_box [N*m].
    :type gear_box_torque_out: float

    :param gear_box_speed_in:
        Engine speed [RPM].
    :type gear_box_speed_in: float

    :param gear_box_speed_out:
        Wheel speed [RPM].
    :type gear_box_speed_out: float

    :param gear_box_efficiency_parameters:
        Parameters of gear box efficiency model (`gbp00`, `gbp10`, `gbp01`).
    :type gear_box_efficiency_parameters: dict

    :return:
        Torque required [N*m].
    :rtype: float
    """

    tgb, es, ws = gear_box_torque_out, gear_box_speed_in, gear_box_speed_out
    par = gear_box_efficiency_parameters

    if tgb < 0 < es and ws > 0:
        return (par['gbp01'] * tgb - par['gbp10'] * ws - par['gbp00']) * ws / es
    elif es > 0 and ws > 0:
        return (tgb - par['gbp10'] * es - par['gbp00']) / par['gbp01']
    return 0


def calculate_gear_box_torque_in(
        gear_box_torque_out, gear_box_speed_in, gear_box_speed_out,
        gear_box_temperature, gear_box_efficiency_parameters_cold_hot,
        gear_box_temperature_references):
    """
    Calculates torque required according to the temperature profile [N*m].

    :param gear_box_torque_out:
        Torque gear box [N*m].
    :type gear_box_torque_out: float

    :param gear_box_speed_in:
        Engine speed [RPM].
    :type gear_box_speed_in: float

    :param gear_box_speed_out:
        Wheel speed [RPM].
    :type gear_box_speed_out: float

    :param gear_box_temperature:
        Gear box temperature [°C].
    :type gear_box_temperature: float

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model for cold/hot phases:

            - 'hot': `gbp00`, `gbp10`, `gbp01`
            - 'cold': `gbp00`, `gbp10`, `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :param gear_box_temperature_references:
        Cold and hot reference temperatures [°C].
    :type gear_box_temperature_references: tuple

    :return:
        Torque required according to the temperature profile [N*m].
    :rtype: float
    """

    par = gear_box_efficiency_parameters_cold_hot
    T_cold, T_hot = gear_box_temperature_references
    t_out = gear_box_torque_out
    e_s, gb_s = gear_box_speed_in, gear_box_speed_out

    t = evaluate_gear_box_torque_in(t_out, e_s, gb_s, par['hot'])

    if not T_cold == T_hot and gear_box_temperature <= T_hot:

        t_cold = evaluate_gear_box_torque_in(t_out, e_s, gb_s, par['cold'])

        t += (T_hot - gear_box_temperature) / (T_hot - T_cold) * (t_cold - t)

    return t


def correct_gear_box_torque_in(
        gear_box_torque_out, gear_box_torque_in, gear, gear_box_ratios):
    """
    Corrects the torque when the gear box ratio is equal to 1.

    :param gear_box_torque_out:
        Torque gear_box [N*m].
    :type gear_box_torque_out: float

    :param gear_box_torque_in:
        Torque required [N*m].
    :type gear_box_torque_in: float

    :param gear:
        Gear [-].
    :type gear: int

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int | float]

    :return:
        Corrected torque required [N*m].
    :rtype: float
    """

    gbr = gear_box_ratios
    if gbr is None or gear is None:
        return gear_box_torque_in

    return gear_box_torque_out if gbr.get(gear, 0) == 1 else gear_box_torque_in


def calculate_gear_box_efficiency(
        gear_box_power_out, gear_box_speed_in, gear_box_torque_out,
        gear_box_torque_in):
    """
    Calculates the gear box efficiency [N*m].

    :param gear_box_power_out:
        Power at wheels [kW].
    :type gear_box_power_out: float

    :param gear_box_speed_in:
        Engine speed [RPM].
    :type gear_box_speed_in: float

    :param gear_box_torque_out:
        Torque gear_box [N*m].
    :type gear_box_torque_out: float

    :param gear_box_torque_in:
        Torque required [N*m].
    :type gear_box_torque_in: float

    :return:
        Gear box efficiency [-].
    :rtype: float
    """

    if gear_box_torque_in == gear_box_torque_out or gear_box_power_out == 0:
        eff = 1
    else:
        s_in = gear_box_speed_in
        eff = s_in * gear_box_torque_in / gear_box_power_out * (math.pi / 30000)
        if gear_box_power_out > 0:
            eff = 1 / eff if eff else 1

    return max(0, min(1, eff))


def calculate_gear_box_temperature(
        gear_box_heat, starting_temperature, equivalent_gear_box_heat_capacity,
        thermostat_temperature):
    """
    Calculates the gear box temperature not finalized [°C].

    :param gear_box_heat:
        Gear box heat [W].
    :type gear_box_heat: float

    :param starting_temperature:
        Starting temperature [°C].
    :type starting_temperature: float

    :param equivalent_gear_box_heat_capacity:
        Equivalent gear box capacity (from cold start model) [W/°C].
    :type equivalent_gear_box_heat_capacity: float

    :param thermostat_temperature:
        Engine thermostat temperature [°C].
    :type thermostat_temperature: float

    :return:
        Gear box temperature not finalized [°C].
    :rtype: float
    """

    temp = starting_temperature
    temp += gear_box_heat / equivalent_gear_box_heat_capacity

    return min(temp, thermostat_temperature - 5.0)


def calculate_gear_box_heat(
        gear_box_efficiency, gear_box_power_out, delta_time):
    """
    Calculates the gear box temperature heat [W].

    :param gear_box_efficiency:
        Gear box efficiency [-].
    :type gear_box_efficiency: float

    :param gear_box_power_out:
        Power at wheels [kW].
    :type gear_box_power_out: float

    :param delta_time:
        Time step [s].
    :type delta_time: float

    :return:
        Gear box heat [W].
    :rtype: float
    """

    if gear_box_efficiency and gear_box_power_out:
        eff = gear_box_efficiency
        return abs(gear_box_power_out) * (1.0 - eff) * 1000.0 * delta_time

    return 0.0


def thermal():
    """
    Defines the gear box thermal sub model.

    .. dispatcher:: d

        >>> d = thermal()

    :return:
        The gear box thermal sub model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Gear box thermal sub model',
        description='Calculates temperature, efficiency, '
                    'torque loss of gear box'
    )

    from ..defaults import dfl
    d.add_data(
        data_id='gear_box_temperature_references',
        default_value=dfl.values.gear_box_temperature_references
    )

    d.add_function(
        function=calculate_gear_box_torque_in,
        inputs=['gear_box_torque_out', 'gear_box_speed_in',
                'gear_box_speed_out', 'gear_box_temperature',
                'gear_box_efficiency_parameters_cold_hot',
                'gear_box_temperature_references'],
        outputs=['gear_box_torque_in<0>']
    )

    d.add_function(
        function=correct_gear_box_torque_in,
        inputs=['gear_box_torque_out', 'gear_box_torque_in<0>', 'gear',
                'gear_box_ratios'],
        outputs=['gear_box_torque_in']
    )

    d.add_function(
        function=calculate_gear_box_efficiency,
        inputs=['gear_box_power_out', 'gear_box_speed_in',
                'gear_box_torque_out', 'gear_box_torque_in'],
        outputs=['gear_box_efficiency'],
    )

    d.add_function(
        function=calculate_gear_box_heat,
        inputs=['gear_box_efficiency', 'gear_box_power_out', 'delta_time'],
        outputs=['gear_box_heat']
    )

    d.add_function(
        function=calculate_gear_box_temperature,
        inputs=['gear_box_heat', 'gear_box_temperature',
                'equivalent_gear_box_heat_capacity', 'thermostat_temperature'],
        outputs=['gear_box_temperature']
    )

    return d


def _thermal(
        gear_box_ratios, thermostat_temperature,
        equivalent_gear_box_heat_capacity,
        gear_box_efficiency_parameters_cold_hot,
        gear_box_temperature_references, gear_box_temperature, delta_time,
        gear_box_power_out, gear_box_speed_out, gear_box_speed_in,
        gear_box_torque_out, gear):

    gear_box_torque_in = calculate_gear_box_torque_in(
        gear_box_torque_out, gear_box_speed_in, gear_box_speed_out,
        gear_box_temperature, gear_box_efficiency_parameters_cold_hot,
        gear_box_temperature_references)

    gear_box_torque_in = correct_gear_box_torque_in(
        gear_box_torque_out, gear_box_torque_in, gear, gear_box_ratios)

    gear_box_efficiency = calculate_gear_box_efficiency(
        gear_box_power_out, gear_box_speed_in, gear_box_torque_out,
        gear_box_torque_in)

    gear_box_heat = calculate_gear_box_heat(
        gear_box_efficiency, gear_box_power_out, delta_time)

    gear_box_temperature = calculate_gear_box_temperature(
        gear_box_heat, gear_box_temperature, equivalent_gear_box_heat_capacity,
        thermostat_temperature)

    return list((gear_box_temperature, gear_box_torque_in, gear_box_efficiency))
