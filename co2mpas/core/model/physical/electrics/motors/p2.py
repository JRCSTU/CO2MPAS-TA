# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the electric motor in P2 position.
"""
import schedula as sh

dsp = sh.BlueDispatcher(
    name='Motor P2',
    description='Models the motor P2 (motor upstream the gear box).'
)


@sh.add_function(dsp, outputs=['motor_p2_maximum_power'])
def identify_motor_p2_maximum_power(motor_p2_powers):
    """
    Identify the maximum power of motor P2 [kW].

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array

    :return:
        Maximum power of motor P2 [kW].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_power as func
    return func(motor_p2_powers)


@sh.add_function(dsp, outputs=['motor_p2_maximum_torque'])
def identify_motor_p2_maximum_torque(motor_p2_torques):
    """
    Identify the maximum torque of motor P2 [N*m].

    :param motor_p2_torques:
        Torque at motor P2 [N*m].
    :type motor_p2_torques: numpy.array

    :return:
        Maximum torque of motor P2 [N*m].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_torque as func
    return func(motor_p2_torques)


@sh.add_function(dsp, outputs=['motor_p2_maximum_power'])
def calculate_motor_p2_maximum_power(
        motor_p2_rated_speed, motor_p2_maximum_torque):
    """
    Calculate the maximum power of motor P2 [kW].

    :param motor_p2_rated_speed:
        Rated speed of motor P2 [RPM].
    :type motor_p2_rated_speed: float

    :param motor_p2_maximum_torque:
        Maximum torque of motor P2 [N*m].
    :type motor_p2_maximum_torque: float

    :return:
        Maximum power of motor P2 [kW].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_power as func
    return func(motor_p2_rated_speed, motor_p2_maximum_torque)


@sh.add_function(dsp, outputs=['motor_p2_rated_speed'])
def calculate_motor_p2_rated_speed(
        motor_p2_maximum_power, motor_p2_maximum_torque):
    """
    Calculate the rated speed of motor P2 [RPM].

    :param motor_p2_maximum_power:
        Maximum power of motor P2 [kW].
    :type motor_p2_maximum_power: float

    :param motor_p2_maximum_torque:
        Maximum torque of motor P2 [N*m].
    :type motor_p2_maximum_torque: float

    :return:
        Rated speed of motor P2 [RPM].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_rated_speed as func
    return func(motor_p2_maximum_power, motor_p2_maximum_torque)


@sh.add_function(dsp, outputs=['motor_p2_maximum_torque'])
def calculate_motor_p2_maximum_torque(
        motor_p2_maximum_power, motor_p2_rated_speed):
    """
    Calculate the maximum torque of motor P2 [N*m].

    :param motor_p2_maximum_power:
        Maximum power of motor P2 [kW].
    :type motor_p2_maximum_power: float

    :param motor_p2_rated_speed:
        Rated speed of motor P2 [RPM].
    :type motor_p2_rated_speed: float

    :return:
        Maximum torque of motor P2 [N*m].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_torque as func
    return func(motor_p2_maximum_power, motor_p2_rated_speed)


@sh.add_function(dsp, outputs=['motor_p2_maximum_power_function'])
def define_motor_p2_maximum_power_function(
        motor_p2_maximum_power, motor_p2_rated_speed):
    """
    Define the maximum power function of motor P2.

    :param motor_p2_maximum_power:
        Maximum power of motor P2 [kW].
    :type motor_p2_maximum_power: float

    :param motor_p2_rated_speed:
        Rated speed of motor P2 [RPM].
    :type motor_p2_rated_speed: float

    :return:
        Maximum power function of motor P2.
    :rtype: function
    """
    from .p4 import define_motor_p4_maximum_power_function as func
    return func(motor_p2_maximum_power, motor_p2_rated_speed)


@sh.add_function(dsp, outputs=['motor_p2_maximum_powers'])
def calculate_motor_p2_maximum_powers(
        motor_p2_speeds, motor_p2_maximum_power_function):
    """
    Calculate the maximum power vector of motor P2 [kW].

    :param motor_p2_speeds:
        Rotating speed of motor P2 [RPM].
    :type motor_p2_speeds: numpy.array | float

    :param motor_p2_maximum_power_function:
        Maximum power function of motor P2.
    :type motor_p2_maximum_power_function: function

    :return:
        Maximum power vector of motor P2 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_maximum_powers as func
    return func(motor_p2_speeds, motor_p2_maximum_power_function)


@sh.add_function(dsp, outputs=['motor_p2_speed_ratio'])
def identify_motor_p2_speed_ratio(gear_box_speeds_in, motor_p2_speeds):
    """
    Identifies motor P2 speed ratio.

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :param motor_p2_speeds:
        Rotating speed of motor P2 [RPM].
    :type motor_p2_speeds: numpy.array | float

    :return:
        Motor P2 speed ratio [-].
    :rtype: float
    """
    from .p4 import identify_motor_p4_speed_ratio as func
    return func(gear_box_speeds_in, motor_p2_speeds)


dsp.add_data('motor_p2_speed_ratio', 1, sh.inf(10, 1))


@sh.add_function(dsp, outputs=['motor_p2_speeds'])
def calculate_motor_p2_speeds(gear_box_speeds_in, motor_p2_speed_ratio):
    """
    Calculates rotating speed of motor P2 [RPM].

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :param motor_p2_speed_ratio:
        Ratio between motor P2 speed and gear box speed [-].
    :type motor_p2_speed_ratio: float

    :return:
        Rotating speed of motor P2 [RPM].
    :rtype: numpy.array | float
    """
    return gear_box_speeds_in * motor_p2_speed_ratio


@sh.add_function(dsp, inputs_kwargs=True, outputs=['gear_box_speeds_in'])
def calculate_gear_box_speeds_in(motor_p2_speeds, motor_p2_speed_ratio=1):
    """
    Calculates Gear box speed vector [RPM].

    :param motor_p2_speeds:
        Rotating speed of motor P2 [RPM].
    :type motor_p2_speeds: numpy.array | float

    :param motor_p2_speed_ratio:
        Ratio between motor P2 speed and gear box speed [-].
    :type motor_p2_speed_ratio: float

    :return:
        Gear box speed vector [RPM].
    :rtype: numpy.array | float
    """
    return motor_p2_speeds / motor_p2_speed_ratio


@sh.add_function(dsp, outputs=['motor_p2_torques'])
def calculate_motor_p2_torques(motor_p2_powers, motor_p2_speeds):
    """
    Calculates torque at motor P2 [N*m].

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array | float

    :param motor_p2_speeds:
        Rotating speed of motor P2 [RPM].
    :type motor_p2_speeds: numpy.array | float

    :return:
        Torque at motor P2 [N*m].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p2_powers, motor_p2_speeds)


@sh.add_function(dsp, outputs=['motor_p2_powers'])
def calculate_motor_p2_powers(motor_p2_torques, motor_p2_speeds):
    """
    Calculates power at motor P2 [kW].

    :param motor_p2_torques:
        Torque at motor P2 [N*m].
    :type motor_p2_torques: numpy.array | float

    :param motor_p2_speeds:
        Rotating speed of motor P2 [RPM].
    :type motor_p2_speeds: numpy.array | float

    :return:
        Power at motor P2 [kW].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_powers
    return calculate_wheel_powers(motor_p2_torques, motor_p2_speeds)


@sh.add_function(dsp, outputs=['motor_p2_speeds'])
def calculate_motor_p2_speeds_v1(motor_p2_powers, motor_p2_torques):
    """
    Calculates rotating speed of motor P2 [RPM].

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array | float

    :param motor_p2_torques:
        Torque at motor P2 [N*m].
    :type motor_p2_torques: numpy.array | float

    :return:
        Rotating speed of motor P2 [RPM].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p2_powers, motor_p2_torques)


dsp.add_data('motor_p2_efficiency', 0.9)


@sh.add_function(dsp, outputs=['motor_p2_electric_powers'])
def calculate_motor_p2_electric_powers(motor_p2_powers, motor_p2_efficiency):
    """
    Calculates motor P2 electric power [kW].

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array | float

    :param motor_p2_efficiency:
        Motor P2 efficiency [-].
    :type motor_p2_efficiency: float

    :return:
        Electric power of motor P2 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_electric_powers as func
    return func(motor_p2_powers, motor_p2_efficiency)


@sh.add_function(dsp, outputs=['motor_p2_powers'])
def calculate_motor_p2_powers_v1(motor_p2_electric_powers, motor_p2_efficiency):
    """
    Calculate motor P2 power from electric power and electric power losses [kW].

    :param motor_p2_electric_powers:
        Electric power of motor P2 [kW].
    :type motor_p2_electric_powers: numpy.array | float

    :param motor_p2_efficiency:
        Motor P2 efficiency [-].
    :type motor_p2_efficiency: float

    :return:
        Power at motor P2 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_powers_v1 as func
    return func(motor_p2_electric_powers, motor_p2_efficiency)


dsp.add_data('has_motor_p2', False, sh.inf(10, 3))


@sh.add_function(dsp, outputs=['has_motor_p2'])
def identify_has_motor_p2(motor_p2_maximum_power):
    """
    Identify if the vehicle has a motor P2 [kW].

    :param motor_p2_maximum_power:
        Maximum power of motor P2 [kW].
    :type motor_p2_maximum_power: float

    :return:
        Has the vehicle a motor in P2?
    :rtype: bool
    """
    from .p4 import identify_has_motor_p4 as func
    return func(motor_p2_maximum_power)


@sh.add_function(dsp, outputs=['motor_p2_powers'])
def default_motor_p2_powers(times, has_motor_p2):
    """
    Return zero power if the vehicle has not a motor P2 [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param has_motor_p2:
        Has the vehicle a motor in P2?
    :type has_motor_p2: bool

    :return:
        Power at motor P2 [kW].
    :rtype: numpy.array
    """
    from .p4 import default_motor_p4_powers as func
    return func(times, has_motor_p2)
