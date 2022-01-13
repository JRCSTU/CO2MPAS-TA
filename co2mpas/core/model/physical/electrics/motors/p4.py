# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the electric motor in P4 position.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Motor P4',
    description='Models the motor P4 (motor downstream the final drive).'
)


@sh.add_function(
    dsp, inputs=['motor_p4_front_powers'],
    outputs=['motor_p4_front_maximum_power']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_powers'],
    outputs=['motor_p4_rear_maximum_power']
)
def identify_motor_p4_maximum_power(motor_p4_powers):
    """
    Identify the maximum power of motor P4 [kW].

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array

    :return:
        Maximum power of motor P4 [kW].
    :rtype: float
    """
    return float(np.max(np.abs(motor_p4_powers)))


@sh.add_function(
    dsp, inputs=['motor_p4_front_torques'],
    outputs=['motor_p4_front_maximum_torque']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_torques'],
    outputs=['motor_p4_rear_maximum_torque']
)
def identify_motor_p4_maximum_torque(motor_p4_torques):
    """
    Identify the maximum torque of motor P4 [N*m].

    :param motor_p4_torques:
        Torque at motor P4 [N*m].
    :type motor_p4_torques: numpy.array

    :return:
        Maximum torque of motor P4 [N*m].
    :rtype: float
    """
    return float(np.max(np.abs(motor_p4_torques)))


@sh.add_function(
    dsp, inputs=['motor_p4_front_rated_speed', 'motor_p4_front_maximum_torque'],
    outputs=['motor_p4_front_maximum_power']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_rated_speed', 'motor_p4_rear_maximum_torque'],
    outputs=['motor_p4_rear_maximum_power']
)
def calculate_motor_p4_maximum_power(
        motor_p4_rated_speed, motor_p4_maximum_torque):
    """
    Calculate the maximum power of motor P4 [kW].

    :param motor_p4_rated_speed:
        Rated speed of motor P4 [RPM].
    :type motor_p4_rated_speed: float

    :param motor_p4_maximum_torque:
        Maximum torque of motor P4 [N*m].
    :type motor_p4_maximum_torque: float

    :return:
        Maximum power of motor P4 [kW].
    :rtype: float
    """
    from ...wheels import calculate_wheel_powers as func
    return func(motor_p4_maximum_torque, motor_p4_rated_speed)


@sh.add_function(
    dsp, outputs=['motor_p4_front_rated_speed'],
    inputs=['motor_p4_front_maximum_power', 'motor_p4_front_maximum_torque']
)
@sh.add_function(
    dsp, outputs=['motor_p4_rear_rated_speed'],
    inputs=['motor_p4_rear_maximum_power', 'motor_p4_rear_maximum_torque']
)
def calculate_motor_p4_rated_speed(
        motor_p4_maximum_power, motor_p4_maximum_torque):
    """
    Calculate the rated speed of motor P4 [RPM].

    :param motor_p4_maximum_power:
        Maximum power of motor P4 [kW].
    :type motor_p4_maximum_power: float

    :param motor_p4_maximum_torque:
        Maximum torque of motor P4 [N*m].
    :type motor_p4_maximum_torque: float

    :return:
        Rated speed of motor P4 [RPM].
    :rtype: float
    """
    from ...wheels import calculate_wheel_torques as func
    return func(motor_p4_maximum_power, motor_p4_maximum_torque)


@sh.add_function(
    dsp, outputs=['motor_p4_front_maximum_torque'],
    inputs=['motor_p4_front_maximum_power', 'motor_p4_front_rated_speed']
)
@sh.add_function(
    dsp, outputs=['motor_p4_rear_maximum_torque'],
    inputs=['motor_p4_rear_maximum_power', 'motor_p4_rear_rated_speed']
)
def calculate_motor_p4_maximum_torque(
        motor_p4_maximum_power, motor_p4_rated_speed):
    """
    Calculate the maximum torque of motor P4 [N*m].

    :param motor_p4_maximum_power:
        Maximum power of motor P4 [kW].
    :type motor_p4_maximum_power: float

    :param motor_p4_rated_speed:
        Rated speed of motor P4 [RPM].
    :type motor_p4_rated_speed: float

    :return:
        Maximum torque of motor P4 [N*m].
    :rtype: float
    """
    from ...wheels import calculate_wheel_torques as func
    return func(motor_p4_maximum_power, motor_p4_rated_speed)


@sh.add_function(
    dsp, inputs=['motor_p4_front_maximum_power', 'motor_p4_front_rated_speed'],
    outputs=['motor_p4_front_maximum_power_function']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_maximum_power', 'motor_p4_rear_rated_speed'],
    outputs=['motor_p4_rear_maximum_power_function']
)
def define_motor_p4_maximum_power_function(
        motor_p4_maximum_power, motor_p4_rated_speed):
    """
    Define the maximum power function of motor P4.

    :param motor_p4_maximum_power:
        Maximum power of motor P4 [kW].
    :type motor_p4_maximum_power: float

    :param motor_p4_rated_speed:
        Rated speed of motor P4 [RPM].
    :type motor_p4_rated_speed: float

    :return:
        Maximum power function of motor P4.
    :rtype: function
    """
    m = 0
    if motor_p4_rated_speed:
        m = motor_p4_maximum_power / motor_p4_rated_speed

    def _maximum_power_function(speeds):
        return np.minimum(motor_p4_maximum_power, speeds * m)

    return _maximum_power_function


@sh.add_function(
    dsp, outputs=['motor_p4_front_maximum_powers'],
    inputs=['motor_p4_front_speeds', 'motor_p4_front_maximum_power_function']
)
@sh.add_function(
    dsp, outputs=['motor_p4_rear_maximum_powers'],
    inputs=['motor_p4_rear_speeds', 'motor_p4_rear_maximum_power_function']
)
def calculate_motor_p4_maximum_powers(
        motor_p4_speeds, motor_p4_maximum_power_function):
    """
    Calculate the maximum power vector of motor P4 [kW].

    :param motor_p4_speeds:
        Rotating speed of motor P4 [RPM].
    :type motor_p4_speeds: numpy.array | float

    :param motor_p4_maximum_power_function:
        Maximum power function of motor P4.
    :type motor_p4_maximum_power_function: function

    :return:
        Maximum power vector of motor P4 [kW].
    :rtype: numpy.array | float
    """
    return motor_p4_maximum_power_function(motor_p4_speeds)


@sh.add_function(
    dsp, inputs=['wheel_speeds', 'motor_p4_front_speeds'],
    outputs=['motor_p4_front_speed_ratio']
)
@sh.add_function(
    dsp, inputs=['wheel_speeds', 'motor_p4_rear_speeds'],
    outputs=['motor_p4_rear_speed_ratio']
)
def identify_motor_p4_speed_ratio(wheel_speeds, motor_p4_speeds):
    """
    Identifies motor P4 speed ratio.

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array | float

    :param motor_p4_speeds:
        Rotating speed of motor P4 [RPM].
    :type motor_p4_speeds: numpy.array | float

    :return:
        Motor P4 speed ratio [-].
    :rtype: float
    """
    b = wheel_speeds > 0
    return co2_utl.reject_outliers(motor_p4_speeds[b] / wheel_speeds[b])[0]


dsp.add_data('motor_p4_front_speed_ratio', 1, sh.inf(10, 1))
dsp.add_data('motor_p4_rear_speed_ratio', 1, sh.inf(10, 1))


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['motor_p4_front_speeds'],
    inputs=['wheel_speeds', 'motor_p4_front_speed_ratio']
)
@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['motor_p4_rear_speeds'],
    inputs=['wheel_speeds', 'motor_p4_rear_speed_ratio']
)
def calculate_motor_p4_speeds(wheel_speeds, motor_p4_speed_ratio=1):
    """
    Calculates rotating speed of motor P4 [RPM].

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array | float

    :param motor_p4_speed_ratio:
        Ratio between motor P4 speed and wheel speed [-].
    :type motor_p4_speed_ratio: float

    :return:
        Rotating speed of motor P4 [RPM].
    :rtype: numpy.array | float
    """
    return wheel_speeds * motor_p4_speed_ratio


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['wheel_speeds'],
    inputs=['motor_p4_front_speeds', 'motor_p4_front_speed_ratio']
)
@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['wheel_speeds'],
    inputs=['motor_p4_rear_speeds', 'motor_p4_rear_speed_ratio']
)
def calculate_wheel_speeds(motor_p4_speeds, motor_p4_speed_ratio=1):
    """
    Calculates rotating speed of the wheels [RPM].

    :param motor_p4_speeds:
        Rotating speed of motor P4 [RPM].
    :type motor_p4_speeds: numpy.array | float

    :param motor_p4_speed_ratio:
        Ratio between motor P4 speed and wheel speed [-].
    :type motor_p4_speed_ratio: float

    :return:
        Rotating speed of the wheel [RPM].
    :rtype: numpy.array | float
    """
    return motor_p4_speeds / motor_p4_speed_ratio


@sh.add_function(
    dsp, inputs=['motor_p4_front_powers', 'motor_p4_front_speeds'],
    outputs=['motor_p4_front_torques']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_powers', 'motor_p4_rear_speeds'],
    outputs=['motor_p4_rear_torques']
)
def calculate_motor_p4_torques(motor_p4_powers, motor_p4_speeds):
    """
    Calculates torque at motor P4 [N*m].

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array | float

    :param motor_p4_speeds:
        Rotating speed of motor P4 [RPM].
    :type motor_p4_speeds: numpy.array | float

    :return:
        Torque at motor P4 [N*m].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p4_powers, motor_p4_speeds)


@sh.add_function(
    dsp, inputs=['motor_p4_front_torques', 'motor_p4_front_speeds'],
    outputs=['motor_p4_front_powers']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_torques', 'motor_p4_rear_speeds'],
    outputs=['motor_p4_rear_powers']
)
def calculate_motor_p4_powers(motor_p4_torques, motor_p4_speeds):
    """
    Calculates power at motor P4 [kW].

    :param motor_p4_torques:
        Torque at motor P4 [N*m].
    :type motor_p4_torques: numpy.array | float

    :param motor_p4_speeds:
        Rotating speed of motor P4 [RPM].
    :type motor_p4_speeds: numpy.array | float

    :return:
        Power at motor P4 [kW].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_powers
    return calculate_wheel_powers(motor_p4_torques, motor_p4_speeds)


@sh.add_function(
    dsp, inputs=['motor_p4_front_powers', 'motor_p4_front_torques'],
    outputs=['motor_p4_front_speeds']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_powers', 'motor_p4_rear_torques'],
    outputs=['motor_p4_rear_speeds']
)
def calculate_motor_p4_speeds_v1(motor_p4_powers, motor_p4_torques):
    """
    Calculates rotating speed of motor P4 [RPM].

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array | float

    :param motor_p4_torques:
        Torque at motor P4 [N*m].
    :type motor_p4_torques: numpy.array | float

    :return:
        Rotating speed of motor P4 [RPM].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p4_powers, motor_p4_torques)


dsp.add_data('motor_p4_front_efficiency', 0.9)
dsp.add_data('motor_p4_rear_efficiency', 0.9)


@sh.add_function(
    dsp, inputs=['motor_p4_front_powers', 'motor_p4_front_efficiency'],
    outputs=['motor_p4_front_electric_powers']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_powers', 'motor_p4_rear_efficiency'],
    outputs=['motor_p4_rear_electric_powers']
)
def calculate_motor_p4_electric_powers(motor_p4_powers, motor_p4_efficiency):
    """
    Calculates motor P4 electric power [kW].

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array | float

    :param motor_p4_efficiency:
        Motor P4 efficiency [-].
    :type motor_p4_efficiency: float

    :return:
        Electric power of motor P4 [kW].
    :rtype: numpy.array | float
    """
    p = motor_p4_powers
    return p * np.where(p >= 0, 1 / motor_p4_efficiency, motor_p4_efficiency)


@sh.add_function(
    dsp, inputs=['motor_p4_front_electric_powers', 'motor_p4_front_efficiency'],
    outputs=['motor_p4_front_powers'], weight=5
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_electric_powers', 'motor_p4_rear_efficiency'],
    outputs=['motor_p4_rear_powers'], weight=5
)
def calculate_motor_p4_powers_v1(motor_p4_electric_powers, motor_p4_efficiency):
    """
    Calculate motor P4 power from electric power and electric power losses [kW].

    :param motor_p4_electric_powers:
        Electric power of motor P4 [kW].
    :type motor_p4_electric_powers: numpy.array | float

    :param motor_p4_efficiency:
        Motor P4 efficiency [-].
    :type motor_p4_efficiency: float

    :return:
        Power at motor P4 [kW].
    :rtype: numpy.array | float
    """
    p = motor_p4_electric_powers
    return p * np.where(p < 0, 1 / motor_p4_efficiency, motor_p4_efficiency)


dsp.add_data('has_motor_p4_front', False, sh.inf(10, 3))
dsp.add_data('has_motor_p4_rear', False, sh.inf(10, 3))


@sh.add_function(
    dsp, inputs=['motor_p4_front_maximum_power'], outputs=['has_motor_p4_front']
)
@sh.add_function(
    dsp, inputs=['motor_p4_rear_maximum_power'], outputs=['has_motor_p4_rear']
)
def identify_has_motor_p4(motor_p4_maximum_power):
    """
    Identify if the vehicle has a motor P4 [kW].

    :param motor_p4_maximum_power:
        Maximum power of motor P4 [kW].
    :type motor_p4_maximum_power: float

    :return:
        Has the vehicle a motor in P4?
    :rtype: bool
    """
    return not np.isclose(motor_p4_maximum_power, 0)


@sh.add_function(
    dsp, inputs=['times', 'has_motor_p4_front'],
    outputs=['motor_p4_front_powers']
)
@sh.add_function(
    dsp, inputs=['times', 'has_motor_p4_rear'], outputs=['motor_p4_rear_powers']
)
def default_motor_p4_powers(times, has_motor_p4):
    """
    Return zero power if the vehicle has not a motor P4 [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param has_motor_p4:
        Has the vehicle a motor in P4?
    :type has_motor_p4: bool

    :return:
        Power at motor P4 [kW].
    :rtype: numpy.array
    """
    if not has_motor_p4:
        return np.zeros_like(times, float)
    return sh.NONE
