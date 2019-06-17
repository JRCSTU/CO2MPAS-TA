# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the electric motor in P4 position.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Motor P4',
    description='Models the motor P4 (motor downstream the final drive).'
)


@sh.add_function(dsp, outputs=['motor_p4_speed_ratio'])
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


dsp.add_data('motor_p4_speed_ratio', 1, sh.inf(10, 1))


@sh.add_function(dsp, inputs_kwargs=True, outputs=['motor_p4_speeds'])
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


@sh.add_function(dsp, outputs=['motor_p4_torques'])
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


@sh.add_function(dsp, outputs=['motor_p4_powers'])
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


@sh.add_function(dsp, outputs=['motor_p4_speeds'])
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


dsp.add_data('motor_p4_efficiency', 0.9)


@sh.add_function(
    dsp, outputs=['motor_p4_electric_power_loss_function'], weight=1
)
def define_motor_p4_electric_power_loss_function(motor_p4_efficiency):
    """
    Define motor P4 electric power loss function from constant efficiency input.

    :param motor_p4_efficiency:
        Motor P4 efficiency [-].
    :type motor_p4_efficiency: float

    :return:
        Motor P4 electric power loss function.
    :rtype: function
    """
    eff = 1.0 / motor_p4_efficiency - 1.0, motor_p4_efficiency - 1.0

    # noinspection PyUnusedLocal
    def motor_p4_electric_power_loss_function(motor_p4_powers, *args):
        return motor_p4_powers * np.where(motor_p4_powers >= 0, *eff)

    return motor_p4_electric_power_loss_function


@sh.add_function(dsp, outputs=['motor_p4_electric_power_loss_function'])
def define_motor_p4_electric_power_loss_function_v1(
        motor_p4_loss_param_a, motor_p4_loss_param_b):
    """
        Define motor P4 electric power loss function from power loss model.

        :param motor_p4_loss_param_a:
            Motor P4 electric power loss parameter a [-].
        :type motor_p4_loss_param_a: float

        :param motor_p4_loss_param_b:
            Motor P4 electric power loss parameter b [-].
        :type motor_p4_loss_param_b: float

        :return:
            Motor P4 electric power loss function.
        :rtype: function
        """
    a, b = motor_p4_loss_param_a, motor_p4_loss_param_b

    # noinspection PyUnusedLocal
    def motor_p4_electric_power_loss_function(
            motor_p4_powers, motor_p4_torques, motor_p4_speeds):
        return a * motor_p4_speeds ** 2 + b * motor_p4_torques ** 2

    return motor_p4_electric_power_loss_function


@sh.add_function(dsp, outputs=['motor_p4_electric_power_losses'])
def calculate_motor_p4_electric_power_losses(
        motor_p4_electric_power_loss_function, motor_p4_powers,
        motor_p4_torques, motor_p4_speeds):
    """
    Calculates motor P4 electric power losses [kW].

    :param motor_p4_electric_power_loss_function:
        Function that calculates motor P4 electric power losses [kW].
    :type motor_p4_electric_power_loss_function: function

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array | float

    :param motor_p4_torques:
        Torque at motor P4 [N*m].
    :type motor_p4_torques: numpy.array | float

    :param motor_p4_speeds:
        Rotating speed of motor P4 [RPM].
    :type motor_p4_speeds: numpy.array | float

    :return:
        Electric power losses of motor P4 [kW].
    :rtype: numpy.array | float
    """
    func = motor_p4_electric_power_loss_function
    return func(motor_p4_powers, motor_p4_torques, motor_p4_speeds)


@sh.add_function(dsp, outputs=['motor_p4_electric_powers'])
def calculate_motor_p4_electric_powers(
        motor_p4_powers, motor_p4_electric_power_losses):
    """
    Calculates motor P4 electric power [kW].

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array | float

    :param motor_p4_electric_power_losses:
        Electric power losses of motor P4 [kW].
    :type motor_p4_electric_power_losses: numpy.array | float

    :return:
        Electric power of motor P4 [kW].
    :rtype: numpy.array | float
    """
    return motor_p4_powers + motor_p4_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p4_powers'])
def calculate_motor_p4_powers_v1(
        motor_p4_electric_powers, motor_p4_electric_power_losses):
    """
    Calculate motor P4 power from electric power and electric power losses [kW].

    :param motor_p4_electric_powers:
        Electric power of motor P4 [kW].
    :type motor_p4_electric_powers: numpy.array | float

    :param motor_p4_electric_power_losses:
        Electric power losses of motor P4 [kW].
    :type motor_p4_electric_power_losses: numpy.array | float

    :return:
        Power at motor P4 [kW].
    :rtype: numpy.array | float
    """
    return motor_p4_electric_powers - motor_p4_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p4_efficiency_ratios'])
def calculate_motor_p4_efficiency_ratios(
        motor_p4_powers, motor_p4_electric_powers):
    """
    Calculates motor P4 efficiency ratio [-]

    :param motor_p4_powers:
        Power at motor P4 [kW].
    :type motor_p4_powers: numpy.array | float

    :param motor_p4_electric_powers:
        Electric power of motor P4 [kW].
    :type motor_p4_electric_powers: numpy.array | float

    :return:
        Motor P4 efficiency ratio [-].
    :rtype: numpy.array | float
    """
    return motor_p4_powers / motor_p4_electric_powers
