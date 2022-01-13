# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the electric motor in P3 position.
"""
import schedula as sh

dsp = sh.BlueDispatcher(
    name='Motor P3',
    description='Models the motor P3 (motor upstream the final drive).'
)


@sh.add_function(
    dsp, inputs=['motor_p3_front_powers'],
    outputs=['motor_p3_front_maximum_power']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_powers'],
    outputs=['motor_p3_rear_maximum_power']
)
def identify_motor_p3_maximum_power(motor_p3_powers):
    """
    Identify the maximum power of motor P3 [kW].

    :param motor_p3_powers:
        Power at motor P3 [kW].
    :type motor_p3_powers: numpy.array

    :return:
        Maximum power of motor P3 [kW].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_power as func
    return func(motor_p3_powers)


@sh.add_function(
    dsp, inputs=['motor_p3_front_torques'],
    outputs=['motor_p3_front_maximum_torque']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_torques'],
    outputs=['motor_p3_rear_maximum_torque']
)
def identify_motor_p3_maximum_torque(motor_p3_torques):
    """
    Identify the maximum torque of motor P3 [N*m].

    :param motor_p3_torques:
        Torque at motor P3 [N*m].
    :type motor_p3_torques: numpy.array

    :return:
        Maximum torque of motor P3 [N*m].
    :rtype: float
    """
    from .p4 import identify_motor_p4_maximum_torque as func
    return func(motor_p3_torques)


@sh.add_function(
    dsp, inputs=['motor_p3_front_rated_speed', 'motor_p3_front_maximum_torque'],
    outputs=['motor_p3_front_maximum_power']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_rated_speed', 'motor_p3_rear_maximum_torque'],
    outputs=['motor_p3_rear_maximum_power']
)
def calculate_motor_p3_maximum_power(
        motor_p3_rated_speed, motor_p3_maximum_torque):
    """
    Calculate the maximum power of motor P3 [kW].

    :param motor_p3_rated_speed:
        Rated speed of motor P3 [RPM].
    :type motor_p3_rated_speed: float

    :param motor_p3_maximum_torque:
        Maximum torque of motor P3 [N*m].
    :type motor_p3_maximum_torque: float

    :return:
        Maximum power of motor P3 [kW].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_power as func
    return func(motor_p3_rated_speed, motor_p3_maximum_torque)


@sh.add_function(
    dsp, outputs=['motor_p3_front_rated_speed'],
    inputs=['motor_p3_front_maximum_power', 'motor_p3_front_maximum_torque']
)
@sh.add_function(
    dsp, outputs=['motor_p3_rear_rated_speed'],
    inputs=['motor_p3_rear_maximum_power', 'motor_p3_rear_maximum_torque']
)
def calculate_motor_p3_rated_speed(
        motor_p3_maximum_power, motor_p3_maximum_torque):
    """
    Calculate the rated speed of motor P3 [RPM].

    :param motor_p3_maximum_power:
        Maximum power of motor P3 [kW].
    :type motor_p3_maximum_power: float

    :param motor_p3_maximum_torque:
        Maximum torque of motor P3 [N*m].
    :type motor_p3_maximum_torque: float

    :return:
        Rated speed of motor P3 [RPM].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_rated_speed as func
    return func(motor_p3_maximum_power, motor_p3_maximum_torque)


@sh.add_function(
    dsp, outputs=['motor_p3_front_maximum_torque'],
    inputs=['motor_p3_front_maximum_power', 'motor_p3_front_rated_speed']
)
@sh.add_function(
    dsp, outputs=['motor_p3_rear_maximum_torque'],
    inputs=['motor_p3_rear_maximum_power', 'motor_p3_rear_rated_speed']
)
def calculate_motor_p3_maximum_torque(
        motor_p3_maximum_power, motor_p3_rated_speed):
    """
    Calculate the maximum torque of motor P3 [N*m].

    :param motor_p3_maximum_power:
        Maximum power of motor P3 [kW].
    :type motor_p3_maximum_power: float

    :param motor_p3_rated_speed:
        Rated speed of motor P3 [RPM].
    :type motor_p3_rated_speed: float

    :return:
        Maximum torque of motor P3 [N*m].
    :rtype: float
    """
    from .p4 import calculate_motor_p4_maximum_torque as func
    return func(motor_p3_maximum_power, motor_p3_rated_speed)


@sh.add_function(
    dsp, inputs=['motor_p3_front_maximum_power', 'motor_p3_front_rated_speed'],
    outputs=['motor_p3_front_maximum_power_function']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_maximum_power', 'motor_p3_rear_rated_speed'],
    outputs=['motor_p3_rear_maximum_power_function']
)
def define_motor_p3_maximum_power_function(
        motor_p3_maximum_power, motor_p3_rated_speed):
    """
    Define the maximum power function of motor P3.

    :param motor_p3_maximum_power:
        Maximum power of motor P3 [kW].
    :type motor_p3_maximum_power: float

    :param motor_p3_rated_speed:
        Rated speed of motor P3 [RPM].
    :type motor_p3_rated_speed: float

    :return:
        Maximum power function of motor P3.
    :rtype: function
    """
    from .p4 import define_motor_p4_maximum_power_function as func
    return func(motor_p3_maximum_power, motor_p3_rated_speed)


@sh.add_function(
    dsp, outputs=['motor_p3_front_maximum_powers'],
    inputs=['motor_p3_front_speeds', 'motor_p3_front_maximum_power_function']
)
@sh.add_function(
    dsp, outputs=['motor_p3_rear_maximum_powers'],
    inputs=['motor_p3_rear_speeds', 'motor_p3_rear_maximum_power_function']
)
def calculate_motor_p3_maximum_powers(
        motor_p3_speeds, motor_p3_maximum_power_function):
    """
    Calculate the maximum power vector of motor P3 [kW].

    :param motor_p3_speeds:
        Rotating speed of motor P3 [RPM].
    :type motor_p3_speeds: numpy.array | float

    :param motor_p3_maximum_power_function:
        Maximum power function of motor P3.
    :type motor_p3_maximum_power_function: function

    :return:
        Maximum power vector of motor P3 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_maximum_powers as func
    return func(motor_p3_speeds, motor_p3_maximum_power_function)


@sh.add_function(
    dsp, inputs=['final_drive_speeds_in', 'motor_p3_front_speeds'],
    outputs=['motor_p3_front_speed_ratio']
)
@sh.add_function(
    dsp, inputs=['final_drive_speeds_in', 'motor_p3_rear_speeds'],
    outputs=['motor_p3_rear_speed_ratio']
)
def identify_motor_p3_speed_ratio(final_drive_speeds_in, motor_p3_speeds):
    """
    Identifies motor P3 speed ratio.

    :param final_drive_speeds_in:
        Final drive speed in [RPM].
    :type final_drive_speeds_in: numpy.array

    :param motor_p3_speeds:
        Rotating speed of motor P3 [RPM].
    :type motor_p3_speeds: numpy.array | float

    :return:
        Motor P3 speed ratio [-].
    :rtype: float
    """
    from .p4 import identify_motor_p4_speed_ratio as func
    return func(final_drive_speeds_in, motor_p3_speeds)


dsp.add_data('motor_p3_front_speed_ratio', 1, sh.inf(10, 1))
dsp.add_data('motor_p3_rear_speed_ratio', 1, sh.inf(10, 1))


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['motor_p3_front_speeds'],
    inputs=['final_drive_speeds_in', 'motor_p3_front_speed_ratio']
)
@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['motor_p3_rear_speeds'],
    inputs=['final_drive_speeds_in', 'motor_p3_rear_speed_ratio']
)
def calculate_motor_p3_speeds(final_drive_speeds_in, motor_p3_speed_ratio=1):
    """
    Calculates rotating speed of motor P3 [RPM].

    :param final_drive_speeds_in:
        Final drive speed in [RPM].
    :type final_drive_speeds_in: numpy.array

    :param motor_p3_speed_ratio:
        Ratio between motor P3 speed and final drive speed in [-].
    :type motor_p3_speed_ratio: float

    :return:
        Rotating speed of motor P3 [RPM].
    :rtype: numpy.array | float
    """
    return final_drive_speeds_in * motor_p3_speed_ratio


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['final_drive_speeds_in'],
    inputs=['motor_p3_front_speeds', 'motor_p3_front_speed_ratio']
)
@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['final_drive_speeds_in'],
    inputs=['motor_p3_rear_speeds', 'motor_p3_rear_speed_ratio']
)
def calculate_final_drive_speeds_in(motor_p3_speeds, motor_p3_speed_ratio=1):
    """
    Calculates final drive speed [RPM].

    :param motor_p3_speeds:
        Rotating speed of motor P3 [RPM].
    :type motor_p3_speeds: numpy.array | float

    :param motor_p3_speed_ratio:
        Ratio between motor P3 speed and final drive speed in [-].
    :type motor_p3_speed_ratio: float

    :return:
        Final drive speed in [RPM].
    :rtype: numpy.array | float
    """
    return motor_p3_speeds / motor_p3_speed_ratio


@sh.add_function(
    dsp, inputs=['motor_p3_front_powers', 'motor_p3_front_speeds'],
    outputs=['motor_p3_front_torques']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_powers', 'motor_p3_rear_speeds'],
    outputs=['motor_p3_rear_torques']
)
def calculate_motor_p3_torques(motor_p3_powers, motor_p3_speeds):
    """
    Calculates torque at motor P3 [N*m].

    :param motor_p3_powers:
        Power at motor P3 [kW].
    :type motor_p3_powers: numpy.array | float

    :param motor_p3_speeds:
        Rotating speed of motor P3 [RPM].
    :type motor_p3_speeds: numpy.array | float

    :return:
        Torque at motor P3 [N*m].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p3_powers, motor_p3_speeds)


@sh.add_function(
    dsp, inputs=['motor_p3_front_torques', 'motor_p3_front_speeds'],
    outputs=['motor_p3_front_powers']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_torques', 'motor_p3_rear_speeds'],
    outputs=['motor_p3_rear_powers']
)
def calculate_motor_p3_powers(motor_p3_torques, motor_p3_speeds):
    """
    Calculates power at motor P3 [kW].

    :param motor_p3_torques:
        Torque at motor P3 [N*m].
    :type motor_p3_torques: numpy.array | float

    :param motor_p3_speeds:
        Rotating speed of motor P3 [RPM].
    :type motor_p3_speeds: numpy.array | float

    :return:
        Power at motor P3 [kW].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_powers
    return calculate_wheel_powers(motor_p3_torques, motor_p3_speeds)


@sh.add_function(
    dsp, inputs=['motor_p3_front_powers', 'motor_p3_front_torques'],
    outputs=['motor_p3_front_speeds']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_powers', 'motor_p3_rear_torques'],
    outputs=['motor_p3_rear_speeds']
)
def calculate_motor_p3_speeds_v1(motor_p3_powers, motor_p3_torques):
    """
    Calculates rotating speed of motor P3 [RPM].

    :param motor_p3_powers:
        Power at motor P3 [kW].
    :type motor_p3_powers: numpy.array | float

    :param motor_p3_torques:
        Torque at motor P3 [N*m].
    :type motor_p3_torques: numpy.array | float

    :return:
        Rotating speed of motor P3 [RPM].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p3_powers, motor_p3_torques)


dsp.add_data('motor_p3_front_efficiency', 0.9)
dsp.add_data('motor_p3_rear_efficiency', 0.9)


@sh.add_function(
    dsp, inputs=['motor_p3_front_powers', 'motor_p3_front_efficiency'],
    outputs=['motor_p3_front_electric_powers']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_powers', 'motor_p3_rear_efficiency'],
    outputs=['motor_p3_rear_electric_powers']
)
def calculate_motor_p3_electric_powers(motor_p3_powers, motor_p3_efficiency):
    """
    Calculates motor P3 electric power [kW].

    :param motor_p3_powers:
        Power at motor P3 [kW].
    :type motor_p3_powers: numpy.array | float

    :param motor_p3_efficiency:
        Motor P3 efficiency [-].
    :type motor_p3_efficiency: float

    :return:
        Electric power of motor P3 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_electric_powers as func
    return func(motor_p3_powers, motor_p3_efficiency)


@sh.add_function(
    dsp, inputs=['motor_p3_front_electric_powers', 'motor_p3_front_efficiency'],
    outputs=['motor_p3_front_powers']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_electric_powers', 'motor_p3_rear_efficiency'],
    outputs=['motor_p3_rear_powers']
)
def calculate_motor_p3_powers_v1(motor_p3_electric_powers, motor_p3_efficiency):
    """
    Calculate motor P3 power from electric power and electric power losses [kW].

    :param motor_p3_electric_powers:
        Electric power of motor P3 [kW].
    :type motor_p3_electric_powers: numpy.array | float

    :param motor_p3_efficiency:
        Motor P3 efficiency [-].
    :type motor_p3_efficiency: float

    :return:
        Power at motor P3 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_powers_v1 as func
    return func(motor_p3_electric_powers, motor_p3_efficiency)


dsp.add_data('has_motor_p3_front', False, sh.inf(10, 3))
dsp.add_data('has_motor_p3_rear', False, sh.inf(10, 3))


@sh.add_function(
    dsp, inputs=['motor_p3_front_maximum_power'], outputs=['has_motor_p3_front']
)
@sh.add_function(
    dsp, inputs=['motor_p3_rear_maximum_power'], outputs=['has_motor_p3_rear']
)
def identify_has_motor_p3(motor_p3_maximum_power):
    """
    Identify if the vehicle has a motor P3 [kW].

    :param motor_p3_maximum_power:
        Maximum power of motor P3 [kW].
    :type motor_p3_maximum_power: float

    :return:
        Has the vehicle a motor in P3?
    :rtype: bool
    """
    from .p4 import identify_has_motor_p4 as func
    return func(motor_p3_maximum_power)


@sh.add_function(
    dsp, inputs=['times', 'has_motor_p3_front'],
    outputs=['motor_p3_front_powers']
)
@sh.add_function(
    dsp, inputs=['times', 'has_motor_p3_rear'], outputs=['motor_p3_rear_powers']
)
def default_motor_p3_powers(times, has_motor_p3):
    """
    Return zero power if the vehicle has not a motor P3 [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param has_motor_p3:
        Has the vehicle a motor in P3?
    :type has_motor_p3: bool

    :return:
        Power at motor P3 [kW].
    :rtype: numpy.array
    """
    from .p4 import default_motor_p4_powers as func
    return func(times, has_motor_p3)
