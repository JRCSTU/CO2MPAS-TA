# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model torque losses and temperature of gearbox.
"""
import math
import schedula as sh
from co2mpas.defaults import dfl

dsp = sh.BlueDispatcher(
    name='Gear box thermal sub model',
    description='Calculates temperature, efficiency, torque loss of gear box.'
)


@sh.add_function(dsp, outputs=['gear_box_torque'])
def calculate_gear_box_torque(
        gear_box_power_out, gear_box_speed_out, gear_box_speed_in,
        min_engine_on_speed):
    """
    Calculates torque entering the gear box [N*m].

    :param gear_box_power_out:
        Power at wheels [kW].
    :type gear_box_power_out: float

    :param gear_box_speed_in:
        Engine speed [RPM].
    :type gear_box_speed_in: float

    :param gear_box_speed_out:
        Wheel speed [RPM].
    :type gear_box_speed_out: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Torque gear box vector [N*m].
    :rtype: numpy.array | float

    .. note:: Torque entering the gearbox can be from engine side
       (power mode or from wheels in motoring mode)
    """
    if gear_box_power_out > 0:
        x = gear_box_speed_in
    else:
        x = gear_box_speed_out
    if x <= min_engine_on_speed:
        return 0
    return gear_box_power_out / x * 30000.0 / math.pi


dsp.add_data(
    data_id='gear_box_temperature_references',
    default_value=dfl.values.gear_box_temperature_references
)


def _evaluate_gear_box_torque_in(
        min_engine_on_speed, gear_box_torque, gear_box_speed_in,
        gear_box_speed_out, gear_box_efficiency_parameters):
    """
    Calculates torque required according to the temperature profile [N*m].
    """

    tgb, es, ws = gear_box_torque, gear_box_speed_in, gear_box_speed_out
    par = gear_box_efficiency_parameters

    if tgb < 0 < es and ws > 0:
        return (par['gbp01'] * tgb - par['gbp10'] * ws - par['gbp00']) * ws / es
    elif es > min_engine_on_speed and ws > min_engine_on_speed:
        return (tgb - par['gbp10'] * es - par['gbp00']) / par['gbp01']
    return 0


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['gear_box_torque_in<0>'])
def calculate_gear_box_torque_in(
        gear_box_torque, gear_box_speed_in, gear_box_speed_out,
        gear_box_temperature, gear_box_efficiency_parameters_cold_hot,
        gear_box_temperature_references, min_engine_on_speed):
    """
    Calculates torque required according to the temperature profile [N*m].

    :param gear_box_torque:
        Torque gear box [N*m].
    :type gear_box_torque: float

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

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Torque required according to the temperature profile [N*m].
    :rtype: float
    """

    par = gear_box_efficiency_parameters_cold_hot
    T_cold, T_hot = gear_box_temperature_references
    t_out = gear_box_torque
    min_es = min_engine_on_speed
    e_s, gb_s = gear_box_speed_in, gear_box_speed_out

    t = _evaluate_gear_box_torque_in(min_es, t_out, e_s, gb_s, par['hot'])

    if not T_cold == T_hot and gear_box_temperature <= T_hot:
        t_cold = _evaluate_gear_box_torque_in(
            min_es, t_out, e_s, gb_s, par['cold']
        )

        t += (T_hot - gear_box_temperature) / (T_hot - T_cold) * (t_cold - t)

    return t


@sh.add_function(
    dsp,
    inputs=['gear_box_torque', 'gear_box_torque_in<0>', 'gear',
            'gear_box_ratios'],
    outputs=['gear_box_torque_in']
)
def correct_gear_box_torque_in(
        gear_box_torque, gear_box_torque_in, gear, gear_box_ratios):
    """
    Corrects the torque when the gear box ratio is equal to 1.

    :param gear_box_torque:
        Torque gear_box [N*m].
    :type gear_box_torque: float

    :param gear_box_torque_in:
        Torque required [N*m].
    :type gear_box_torque_in: float

    :param gear:
        Gear [-].
    :type gear: int

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int]

    :return:
        Corrected torque required [N*m].
    :rtype: float
    """

    gbr = gear_box_ratios
    if gbr is None or gear is None:
        return gear_box_torque_in

    return gear_box_torque if gbr.get(gear, 0) == 1 else gear_box_torque_in


@sh.add_function(dsp, outputs=['gear_box_efficiency'])
def calculate_gear_box_efficiency(
        gear_box_power_out, gear_box_speed_in, gear_box_torque,
        gear_box_torque_in, min_engine_on_speed):
    """
    Calculates the gear box efficiency [N*m].

    :param gear_box_power_out:
        Power at wheels [kW].
    :type gear_box_power_out: float

    :param gear_box_speed_in:
        Engine speed [RPM].
    :type gear_box_speed_in: float

    :param gear_box_torque:
        Torque gear_box [N*m].
    :type gear_box_torque: float

    :param gear_box_torque_in:
        Torque required [N*m].
    :type gear_box_torque_in: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Gear box efficiency [-].
    :rtype: float
    """

    if gear_box_torque_in == gear_box_torque or gear_box_power_out == 0 or \
            gear_box_speed_in < min_engine_on_speed:
        eff = 1
    else:
        s_in = gear_box_speed_in
        eff = s_in * gear_box_torque_in / gear_box_power_out * (math.pi / 30000)
        if gear_box_power_out > 0:
            eff = 1 / eff if eff else 1

    return max(0, min(1, eff))


@sh.add_function(dsp, outputs=['gear_box_heat'])
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


@sh.add_function(dsp, outputs=['gear_box_temperature'])
def calculate_next_gear_box_temperature(
        gear_box_heat, gear_box_temperature, equivalent_gear_box_heat_capacity):
    """
    Calculates the gear box temperature not finalized [°C].

    :param gear_box_heat:
        Gear box heat [W].
    :type gear_box_heat: float

    :param gear_box_temperature:
        Starting gear box temperature not finalized [°C].
    :type gear_box_temperature: float

    :param equivalent_gear_box_heat_capacity:
        Equivalent gear box capacity (from cold start model) [W/°C].
    :type equivalent_gear_box_heat_capacity: float

    :return:
        Gear box temperature not finalized [°C].
    :rtype: float
    """
    dtemp = gear_box_heat / equivalent_gear_box_heat_capacity
    return gear_box_temperature + dtemp


def _thermal(
        gear_box_temperature, gear_box_torque, gear, delta_time,
        gear_box_power_out, gear_box_speed_out, gear_box_speed_in,
        equivalent_gear_box_heat_capacity,
        gear_box_efficiency_parameters_cold_hot,
        gear_box_temperature_references, gear_box_ratios=None,
        min_engine_on_speed=None):
    if gear_box_torque is None:
        gear_box_torque = calculate_gear_box_torque(
            gear_box_power_out, gear_box_speed_out, gear_box_speed_in,
            min_engine_on_speed
        )

    gear_box_torque_in = calculate_gear_box_torque_in(
        gear_box_torque, gear_box_speed_in, gear_box_speed_out,
        gear_box_temperature, gear_box_efficiency_parameters_cold_hot,
        gear_box_temperature_references, min_engine_on_speed
    )
    gear_box_torque_in = correct_gear_box_torque_in(
        gear_box_torque, gear_box_torque_in, gear, gear_box_ratios
    )
    gear_box_efficiency = calculate_gear_box_efficiency(
        gear_box_power_out, gear_box_speed_in, gear_box_torque,
        gear_box_torque_in, min_engine_on_speed
    )
    gear_box_heat = calculate_gear_box_heat(
        gear_box_efficiency, gear_box_power_out, delta_time
    )
    gear_box_temperature = calculate_next_gear_box_temperature(
        gear_box_heat, gear_box_temperature, equivalent_gear_box_heat_capacity
    )
    return gear_box_temperature, gear_box_torque_in, gear_box_efficiency
