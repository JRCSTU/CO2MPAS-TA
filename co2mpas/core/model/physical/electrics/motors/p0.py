# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the electric motor in P0 position.
"""
import schedula as sh

dsp = sh.BlueDispatcher(
    name='Motor P0',
    description='Models the motor P0 (motor on the engine service belt).'
)


@sh.add_function(dsp, outputs=['motor_p0_maximum_power'])
def identify_motor_p0_maximum_power(motor_p0_powers):
    """
    Identify the maximum power of motor P0 [kW].

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array

    :return:
        Maximum power of motor P0 [kW].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_power as func
    return func(motor_p0_powers)


@sh.add_function(dsp, outputs=['motor_p0_maximum_torque'])
def identify_motor_p0_maximum_torque(motor_p0_torques):
    """
    Identify the maximum torque of motor P0 [N*m].

    :param motor_p0_torques:
        Torque at motor P0 [N*m].
    :type motor_p0_torques: numpy.array

    :return:
        Maximum torque of motor P0 [N*m].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_torque as func
    return func(motor_p0_torques)


@sh.add_function(dsp, outputs=['motor_p0_maximum_power'])
def calculate_motor_p0_maximum_power(
        motor_p0_rated_speed, motor_p0_maximum_torque):
    """
    Calculate the maximum power of motor P0 [kW].

    :param motor_p0_rated_speed:
        Rated speed of motor P0 [RPM].
    :type motor_p0_rated_speed: float

    :param motor_p0_maximum_torque:
        Maximum torque of motor P0 [N*m].
    :type motor_p0_maximum_torque: float

    :return:
        Maximum power of motor P0 [kW].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_power as func
    return func(motor_p0_rated_speed, motor_p0_maximum_torque)


@sh.add_function(dsp, outputs=['motor_p0_rated_speed'])
def calculate_motor_p0_rated_speed(
        motor_p0_maximum_power, motor_p0_maximum_torque):
    """
    Calculate the rated speed of motor P0 [RPM].

    :param motor_p0_maximum_power:
        Maximum power of motor P0 [kW].
    :type motor_p0_maximum_power: float

    :param motor_p0_maximum_torque:
        Maximum torque of motor P0 [N*m].
    :type motor_p0_maximum_torque: float

    :return:
        Rated speed of motor P0 [RPM].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_rated_speed as func
    return func(motor_p0_maximum_power, motor_p0_maximum_torque)


@sh.add_function(dsp, outputs=['motor_p0_maximum_torque'])
def calculate_motor_p0_maximum_torque(
        motor_p0_maximum_power, motor_p0_rated_speed):
    """
    Calculate the maximum torque of motor P0 [N*m].

    :param motor_p0_maximum_power:
        Maximum power of motor P0 [kW].
    :type motor_p0_maximum_power: float

    :param motor_p0_rated_speed:
        Rated speed of motor P0 [RPM].
    :type motor_p0_rated_speed: float

    :return:
        Maximum torque of motor P0 [N*m].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_torque as func
    return func(motor_p0_maximum_power, motor_p0_rated_speed)


@sh.add_function(dsp, outputs=['motor_p0_maximum_power_function'])
def define_motor_p0_maximum_power_function(
        motor_p0_maximum_power, motor_p0_rated_speed):
    """
    Define the maximum power function of motor P0.

    :param motor_p0_maximum_power:
        Maximum power of motor P0 [kW].
    :type motor_p0_maximum_power: float

    :param motor_p0_rated_speed:
        Rated speed of motor P0 [RPM].
    :type motor_p0_rated_speed: float

    :return:
        Maximum power function of motor P0.
    :rtype: function
    """
    from .p4 import define_motor_p4_maximum_power_function as func
    return func(motor_p0_maximum_power, motor_p0_rated_speed)


@sh.add_function(dsp, outputs=['motor_p0_maximum_powers'])
def calculate_motor_p0_maximum_powers(
        motor_p0_speeds, motor_p0_maximum_power_function):
    """
    Calculate the maximum power vector of motor P0 [kW].

    :param motor_p0_speeds:
        Rotating speed of motor P0 [RPM].
    :type motor_p0_speeds: numpy.array | float

    :param motor_p0_maximum_power_function:
        Maximum power function of motor P0.
    :type motor_p0_maximum_power_function: function

    :return:
        Maximum power vector of motor P0 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_maximum_powers as func
    return func(motor_p0_speeds, motor_p0_maximum_power_function)


@sh.add_function(dsp, outputs=['motor_p0_speed_ratio'])
def identify_motor_p0_speed_ratio(engine_speeds_out, motor_p0_speeds):
    """
    Identifies motor P0 speed ratio.

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param motor_p0_speeds:
        Rotating speed of motor P0 [RPM].
    :type motor_p0_speeds: numpy.array | float

    :return:
        Motor P0 speed ratio [-].
    :rtype: float
    """
    from .p4 import identify_motor_p4_speed_ratio as func
    return func(engine_speeds_out, motor_p0_speeds)


dsp.add_data('motor_p0_speed_ratio', 3, sh.inf(10, 1))


@sh.add_function(dsp, inputs_kwargs=True, outputs=['motor_p0_speeds'])
def calculate_motor_p0_speeds(engine_speeds_out, motor_p0_speed_ratio=3):
    """
    Calculates rotating speed of motor P0 [RPM].

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param motor_p0_speed_ratio:
        Ratio between motor P0 speed and engine speed [-].
    :type motor_p0_speed_ratio: float

    :return:
        Rotating speed of motor P0 [RPM].
    :rtype: numpy.array | float
    """
    return engine_speeds_out * motor_p0_speed_ratio


@sh.add_function(dsp, inputs_kwargs=True, outputs=['engine_speeds_out'])
def calculate_engine_speeds_out(motor_p0_speeds, motor_p0_speed_ratio=3):
    """
    Calculates the engine speed [RPM].

    :param motor_p0_speeds:
        Rotating speed of motor P0 [RPM].
    :type motor_p0_speeds: numpy.array | float

    :param motor_p0_speed_ratio:
        Ratio between motor P0 speed and engine speed [-].
    :type motor_p0_speed_ratio: float

    :return:
        Engine speed [RPM].
    :rtype: numpy.array | float
    """
    return motor_p0_speeds / motor_p0_speed_ratio


@sh.add_function(dsp, outputs=['motor_p0_torques'])
def calculate_motor_p0_torques(motor_p0_powers, motor_p0_speeds):
    """
    Calculates torque at motor P0 [N*m].

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array | float

    :param motor_p0_speeds:
        Rotating speed of motor P0 [RPM].
    :type motor_p0_speeds: numpy.array | float

    :return:
        Torque at motor P0 [N*m].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p0_powers, motor_p0_speeds)


@sh.add_function(dsp, outputs=['motor_p0_powers'])
def calculate_motor_p0_powers(motor_p0_torques, motor_p0_speeds):
    """
    Calculates power at motor P0 [kW].

    :param motor_p0_torques:
        Torque at motor P0 [N*m].
    :type motor_p0_torques: numpy.array | float

    :param motor_p0_speeds:
        Rotating speed of motor P0 [RPM].
    :type motor_p0_speeds: numpy.array | float

    :return:
        Power at motor P0 [kW].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_powers
    return calculate_wheel_powers(motor_p0_torques, motor_p0_speeds)


@sh.add_function(dsp, outputs=['motor_p0_speeds'])
def calculate_motor_p0_speeds_v1(motor_p0_powers, motor_p0_torques):
    """
    Calculates rotating speed of motor P0 [RPM].

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array | float

    :param motor_p0_torques:
        Torque at motor P0 [N*m].
    :type motor_p0_torques: numpy.array | float

    :return:
        Rotating speed of motor P0 [RPM].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p0_powers, motor_p0_torques)


dsp.add_data('motor_p0_efficiency', 0.9)


@sh.add_function(dsp, outputs=['motor_p0_electric_powers'])
def calculate_motor_p0_electric_powers(motor_p0_powers, motor_p0_efficiency):
    """
    Calculates motor P0 electric power [kW].

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array | float

    :param motor_p0_efficiency:
        Motor P0 efficiency [-].
    :type motor_p0_efficiency: float

    :return:
        Electric power of motor P0 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_electric_powers as func
    return func(motor_p0_powers, motor_p0_efficiency)


@sh.add_function(dsp, outputs=['motor_p0_powers'])
def calculate_motor_p0_powers_v1(motor_p0_electric_powers, motor_p0_efficiency):
    """
    Calculate motor P0 power from electric power and electric power losses [kW].

    :param motor_p0_electric_powers:
        Electric power of motor P0 [kW].
    :type motor_p0_electric_powers: numpy.array | float

    :param motor_p0_efficiency:
        Motor P0 efficiency [-].
    :type motor_p0_efficiency: float

    :return:
        Power at motor P0 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_powers_v1 as func
    return func(motor_p0_electric_powers, motor_p0_efficiency)


dsp.add_data('has_motor_p0', False, sh.inf(10, 3))


@sh.add_function(dsp, outputs=['has_motor_p0'])
def identify_has_motor_p0(motor_p0_maximum_power):
    """
    Identify if the vehicle has a motor P0 [kW].

    :param motor_p0_maximum_power:
        Maximum power of motor P0 [kW].
    :type motor_p0_maximum_power: float

    :return:
        Has the vehicle a motor in P0?
    :rtype: bool
    """
    from .p4 import identify_has_motor_p4 as func
    return func(motor_p0_maximum_power)


@sh.add_function(dsp, outputs=['motor_p0_powers'])
def default_motor_p0_powers(times, has_motor_p0):
    """
    Return zero power if the vehicle has not a motor P0 [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param has_motor_p0:
        Has the vehicle a motor in P0?
    :type has_motor_p0: bool

    :return:
        Power at motor P0 [kW].
    :rtype: numpy.array
    """
    from .p4 import default_motor_p4_powers as func
    return func(times, has_motor_p0)
