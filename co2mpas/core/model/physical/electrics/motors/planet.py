# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the electric motor in planetary P2 position.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl
from co2mpas.defaults import dfl

dsp = sh.BlueDispatcher(
    name='Motor P2 planetary',
    description='Models the planetary motor P2 '
                '(motor acting on the planetary gear-set).'
)


@sh.add_function(dsp, outputs=['motor_p2_planetary_maximum_power'])
def identify_motor_p2_planetary_maximum_power(motor_p2_planetary_powers):
    """
    Identify the maximum power of planetary motor P2 [kW].

    :param motor_p2_planetary_powers:
        Power at planetary motor P2 [kW].
    :type motor_p2_planetary_powers: numpy.array

    :return:
        Maximum power of planetary motor P2 [kW].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_power as func
    return func(motor_p2_planetary_powers)


@sh.add_function(dsp, outputs=['motor_p2_planetary_maximum_torque'])
def identify_motor_p2_planetary_maximum_torque(motor_p2_planetary_torques):
    """
    Identify the maximum torque of planetary motor P2 [N*m].

    :param motor_p2_planetary_torques:
        Torque at planetary motor P2 [N*m].
    :type motor_p2_planetary_torques: numpy.array

    :return:
        Maximum torque of planetary motor P2 [N*m].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_torque as func
    return func(motor_p2_planetary_torques)


@sh.add_function(dsp, outputs=['motor_p2_planetary_maximum_power'])
def calculate_motor_p2_planetary_maximum_power(
        motor_p2_planetary_rated_speed, motor_p2_planetary_maximum_torque):
    """
    Calculate the maximum power of planetary motor P2 [kW].

    :param motor_p2_planetary_rated_speed:
        Rated speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_rated_speed: float

    :param motor_p2_planetary_maximum_torque:
        Maximum torque of planetary motor P2 [N*m].
    :type motor_p2_planetary_maximum_torque: float

    :return:
        Maximum power of planetary motor P2 [kW].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_power as f
    return f(motor_p2_planetary_rated_speed, motor_p2_planetary_maximum_torque)


@sh.add_function(dsp, outputs=['motor_p2_planetary_rated_speed'])
def calculate_motor_p2_planetary_rated_speed(
        motor_p2_planetary_maximum_power, motor_p2_planetary_maximum_torque):
    """
    Calculate the rated speed of planetary motor P2 [RPM].

    :param motor_p2_planetary_maximum_power:
        Maximum power of planetary motor P2 [kW].
    :type motor_p2_planetary_maximum_power: float

    :param motor_p2_planetary_maximum_torque:
        Maximum torque of planetary motor P2 [N*m].
    :type motor_p2_planetary_maximum_torque: float

    :return:
        Rated speed of planetary motor P2 [RPM].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_rated_speed as func
    return func(
        motor_p2_planetary_maximum_power, motor_p2_planetary_maximum_torque
    )


@sh.add_function(dsp, outputs=['motor_p2_planetary_maximum_torque'])
def calculate_motor_p2_planetary_maximum_torque(
        motor_p2_planetary_maximum_power, motor_p2_planetary_rated_speed):
    """
    Calculate the maximum torque of planetary motor P2 [N*m].

    :param motor_p2_planetary_maximum_power:
        Maximum power of planetary motor P2 [kW].
    :type motor_p2_planetary_maximum_power: float

    :param motor_p2_planetary_rated_speed:
        Rated speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_rated_speed: float

    :return:
        Maximum torque of planetary motor P2 [N*m].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_torque as f
    return f(motor_p2_planetary_maximum_power, motor_p2_planetary_rated_speed)


@sh.add_function(dsp, outputs=['motor_p2_planetary_maximum_power_function'])
def define_motor_p2_planetary_maximum_power_function(
        motor_p2_planetary_maximum_power, motor_p2_planetary_rated_speed):
    """
    Define the maximum power function of planetary motor P2.

    :param motor_p2_planetary_maximum_power:
        Maximum power of planetary motor P2 [kW].
    :type motor_p2_planetary_maximum_power: float

    :param motor_p2_planetary_rated_speed:
        Rated speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_rated_speed: float

    :return:
        Maximum power function of planetary motor P2.
    :rtype: function
    """
    from .p4 import define_motor_p4_maximum_power_function as f
    func = f(motor_p2_planetary_maximum_power, motor_p2_planetary_rated_speed)

    def _maximum_power_function(speeds):
        return func(np.abs(speeds))

    return _maximum_power_function


@sh.add_function(dsp, outputs=['motor_p2_planetary_maximum_powers'])
def calculate_motor_p2_planetary_maximum_powers(
        motor_p2_planetary_speeds, motor_p2_planetary_maximum_power_function):
    """
    Calculate the maximum power vector of planetary motor P2 [kW].

    :param motor_p2_planetary_speeds:
        Rotating speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_speeds: numpy.array | float

    :param motor_p2_planetary_maximum_power_function:
        Maximum power function of planetary motor P2.
    :type motor_p2_planetary_maximum_power_function: function

    :return:
        Maximum power vector of planetary motor P2 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_maximum_powers as func
    return func(
        motor_p2_planetary_speeds, motor_p2_planetary_maximum_power_function
    )


@sh.add_function(dsp, outputs=['motor_p2_planetary_speed_ratio'])
def identify_motor_p2_planetary_speed_ratio(
        planetary_speeds_in, motor_p2_planetary_speeds):
    """
    Identifies planetary motor P2 speed ratio.

    :param planetary_speeds_in:
        Planetary speed vector [RPM].
    :type planetary_speeds_in: numpy.array | float

    :param motor_p2_planetary_speeds:
        Rotating speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_speeds: numpy.array | float

    :return:
        Ratio between planetary motor P2 speed and planetary speed [-].
    :rtype: float
    """
    from .p4 import identify_motor_p4_speed_ratio as func
    return func(planetary_speeds_in, motor_p2_planetary_speeds)


dsp.add_data('motor_p2_planetary_speed_ratio', 1, sh.inf(10, 1))


@sh.add_function(dsp, outputs=['motor_p2_planetary_speeds'])
def calculate_motor_p2_planetary_speeds(
        planetary_speeds_in, motor_p2_planetary_speed_ratio):
    """
    Calculates rotating speed of planetary motor P2 [RPM].

    :param planetary_speeds_in:
        Planetary speed vector [RPM].
    :type planetary_speeds_in: numpy.array | float

    :param motor_p2_planetary_speed_ratio:
        Ratio between planetary motor P2 speed and planetary speed [-].
    :type motor_p2_planetary_speed_ratio: float

    :return:
        Rotating speed of planetary motor P2 [RPM].
    :rtype: numpy.array | float
    """
    return planetary_speeds_in * motor_p2_planetary_speed_ratio


@sh.add_function(dsp, inputs_kwargs=True, outputs=['planetary_speeds_in'])
def calculate_planetary_speeds_in(
        motor_p2_planetary_speeds, motor_p2_planetary_speed_ratio=1):
    """
    Calculates Gear box speed vector [RPM].

    :param motor_p2_planetary_speeds:
        Rotating speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_speeds: numpy.array | float

    :param motor_p2_planetary_speed_ratio:
        Ratio between planetary motor P2 speed and planetary speed [-].
    :type motor_p2_planetary_speed_ratio: float

    :return:
        Planetary speed vector [RPM].
    :rtype: numpy.array | float
    """
    return motor_p2_planetary_speeds / motor_p2_planetary_speed_ratio


@sh.add_function(dsp, outputs=['motor_p2_planetary_torques'])
def calculate_motor_p2_planetary_torques(
        motor_p2_planetary_powers, motor_p2_planetary_speeds):
    """
    Calculates torque at planetary motor P2 [N*m].

    :param motor_p2_planetary_powers:
        Power at planetary motor P2 [kW].
    :type motor_p2_planetary_powers: numpy.array | float

    :param motor_p2_planetary_speeds:
        Rotating speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_speeds: numpy.array | float

    :return:
        Torque at planetary motor P2 [N*m].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques as func
    return func(motor_p2_planetary_powers, motor_p2_planetary_speeds)


@sh.add_function(dsp, outputs=['motor_p2_planetary_powers'])
def calculate_motor_p2_planetary_powers(
        motor_p2_planetary_torques, motor_p2_planetary_speeds):
    """
    Calculates power at planetary motor P2 [kW].

    :param motor_p2_planetary_torques:
        Torque at planetary motor P2 [N*m].
    :type motor_p2_planetary_torques: numpy.array | float

    :param motor_p2_planetary_speeds:
        Rotating speed of planetary motor P2 [RPM].
    :type motor_p2_planetary_speeds: numpy.array | float

    :return:
        Power at planetary motor P2 [kW].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_powers as func
    return func(motor_p2_planetary_torques, motor_p2_planetary_speeds)


@sh.add_function(dsp, outputs=['motor_p2_planetary_speeds'])
def calculate_motor_p2_planetary_speeds_v1(
        motor_p2_planetary_powers, motor_p2_planetary_torques):
    """
    Calculates rotating speed of planetary motor P2 [RPM].

    :param motor_p2_planetary_powers:
        Power at planetary motor P2 [kW].
    :type motor_p2_planetary_powers: numpy.array | float

    :param motor_p2_planetary_torques:
        Torque at planetary motor P2 [N*m].
    :type motor_p2_planetary_torques: numpy.array | float

    :return:
        Rotating speed of planetary motor P2 [RPM].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques as func
    return func(motor_p2_planetary_powers, motor_p2_planetary_torques)


dsp.add_data('motor_p2_planetary_efficiency', 0.9)


@sh.add_function(dsp, outputs=['motor_p2_planetary_electric_powers'])
def calculate_motor_p2_planetary_electric_powers(
        motor_p2_planetary_powers, motor_p2_planetary_efficiency):
    """
    Calculates planetary motor P2 electric power [kW].

    :param motor_p2_planetary_powers:
        Power at planetary motor P2 [kW].
    :type motor_p2_planetary_powers: numpy.array | float

    :param motor_p2_planetary_efficiency:
        Motor P2 efficiency [-].
    :type motor_p2_planetary_efficiency: float

    :return:
        Electric power of planetary motor P2 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_electric_powers as func
    return func(motor_p2_planetary_powers, motor_p2_planetary_efficiency)


@sh.add_function(dsp, outputs=['motor_p2_planetary_powers'])
def calculate_motor_p2_planetary_powers_v1(
        motor_p2_planetary_electric_powers, motor_p2_planetary_efficiency):
    """
    Calculate planetary motor P2 power from electric power and losses [kW].

    :param motor_p2_planetary_electric_powers:
        Electric power of planetary motor P2 [kW].
    :type motor_p2_planetary_electric_powers: numpy.array | float

    :param motor_p2_planetary_efficiency:
        Motor P2 efficiency [-].
    :type motor_p2_planetary_efficiency: float

    :return:
        Power at planetary motor P2 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_powers_v1 as f
    return f(motor_p2_planetary_electric_powers, motor_p2_planetary_efficiency)


dsp.add_data('has_motor_p2_planetary', False, sh.inf(10, 3))


@sh.add_function(dsp, outputs=['has_motor_p2_planetary'])
def identify_has_motor_p2(motor_p2_planetary_maximum_power):
    """
    Identify if the vehicle has a planetary motor P2 [kW].

    :param motor_p2_planetary_maximum_power:
        Maximum power of planetary motor P2 [kW].
    :type motor_p2_planetary_maximum_power: float

    :return:
        Has the vehicle a motor in planetary P2?
    :rtype: bool
    """
    from .p4 import identify_has_motor_p4 as func
    return func(motor_p2_planetary_maximum_power)


@sh.add_function(dsp, outputs=['motor_p2_planetary_powers'])
def default_motor_p2_planetary_powers(times, has_motor_p2_planetary):
    """
    Return zero power if the vehicle has not a planetary motor P2 [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param has_motor_p2_planetary:
        Has the vehicle a motor in planetary P2?
    :type has_motor_p2_planetary: bool

    :return:
        Power at planetary motor P2 [kW].
    :rtype: numpy.array
    """
    from .p4 import default_motor_p4_powers as func
    return func(times, has_motor_p2_planetary)


@sh.add_function(dsp, outputs=['planetary_speeds_in'])
def calculate_planetary_speeds_in(
        engine_speeds_out, final_drive_speeds_in, planetary_ratio):
    """
    Calculates the planetary speed [RPM].

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param final_drive_speeds_in:
        Final drive speed in [RPM].
    :type final_drive_speeds_in: numpy.array

    :param planetary_ratio:
        Fundamental planetary speed ratio [-].
    :type planetary_ratio: float

    :return:
        Planetary speed vector [RPM].
    :rtype: numpy.array
    """
    r = planetary_ratio
    return engine_speeds_out * (1 + r) - final_drive_speeds_in * r


@sh.add_function(dsp, outputs=['engine_speeds_out'])
def calculate_engine_speeds_out(
        planetary_speeds_in, final_drive_speeds_in, planetary_ratio):
    """
    Calculates the engine speed [RPM].

    :param planetary_speeds_in:
        Planetary speed vector [RPM].
    :type planetary_speeds_in: numpy.array

    :param final_drive_speeds_in:
        Final drive speed in [RPM].
    :type final_drive_speeds_in: numpy.array

    :param planetary_ratio:
        Fundamental planetary speed ratio [-].
    :type planetary_ratio: float

    :return:
        Engine speed vector [RPM].
    :rtype: numpy.array
    """
    if planetary_ratio == -1:
        return sh.NONE
    r = planetary_ratio
    return (planetary_speeds_in + final_drive_speeds_in * r) / (1 + r)


@sh.add_function(dsp, outputs=['final_drive_speeds_in'])
def calculate_final_drive_speeds_in(
        planetary_speeds_in, engine_speeds_out, planetary_ratio):
    """
    Calculates final drive speed [RPM].

    :param planetary_speeds_in:
        Planetary speed vector [RPM].
    :type planetary_speeds_in: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param planetary_ratio:
        Fundamental planetary speed ratio [-].
    :type planetary_ratio: float

    :return:
        Final drive speed in [RPM].
    :rtype: numpy.array
    """
    if not planetary_ratio:
        return sh.NONE
    r = planetary_ratio
    return (engine_speeds_out * (1 + r) - planetary_speeds_in) / r


@sh.add_function(dsp, outputs=['planetary_ratio'])
def identify_planetary_ratio(
        planetary_speeds_in, engine_speeds_out, final_drive_speeds_in):
    """
    Calculates final drive speed [RPM].

    :param planetary_speeds_in:
        Planetary speed vector [RPM].
    :type planetary_speeds_in: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param final_drive_speeds_in:
        Final drive speed in [RPM].
    :type final_drive_speeds_in: float

    :return:
        Fundamental planetary speed ratio [-].
    :rtype: numpy.array
    """
    r = planetary_speeds_in - engine_speeds_out
    r /= engine_speeds_out - final_drive_speeds_in
    return co2_utl.reject_outliers(r)


@sh.add_function(dsp, outputs=['planetary_mean_efficiency'])
def default_planetary_mean_efficiency(has_motor_p2_planetary):
    """
    Returns the default planetary mean efficiency [-].

    :param has_motor_p2_planetary:
        Has the vehicle a motor in planetary P2?
    :type has_motor_p2_planetary: bool

    :return:
        Planetary mean efficiency [-].
    :rtype: float
    """
    if has_motor_p2_planetary:
        return dfl.functions.default_planetary_mean_efficiency.efficiency
    return 1


@sh.add_function(dsp, outputs=['planetary_ratio'], weight=sh.inf(10, 5))
def default_planetary_ratio(has_motor_p2_planetary):
    """
    Returns the default fundamental planetary speed ratio [-].

    :param has_motor_p2_planetary:
        Has the vehicle a motor in planetary P2?
    :type has_motor_p2_planetary: bool

    :return:
        Fundamental planetary speed ratio [-].
    :rtype: float
    """
    if has_motor_p2_planetary:
        return dfl.functions.default_planetary_ratio.ratio
    return 0
