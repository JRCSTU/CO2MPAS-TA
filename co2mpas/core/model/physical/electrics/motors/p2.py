# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the electric motor in P2 position.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Motor P2',
    description='Models the motor P2 (motor upstream the gear box).'
)


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


@sh.add_function(
    dsp, outputs=['motor_p2_electric_power_loss_function'], weight=1
)
def define_motor_p2_electric_power_loss_function(motor_p2_efficiency):
    """
    Define motor P2 electric power loss function from constant efficiency input.

    :param motor_p2_efficiency:
        Motor P2 efficiency [-].
    :type motor_p2_efficiency: float

    :return:
        Motor P2 electric power loss function.
    :rtype: function
    """
    from .p4 import define_motor_p4_electric_power_loss_function as func
    return func(motor_p2_efficiency)


@sh.add_function(dsp, outputs=['motor_p2_electric_power_loss_function'])
def define_motor_p2_electric_power_loss_function_v1(
        motor_p2_loss_param_a, motor_p2_loss_param_b):
    """
        Define motor P2 electric power loss function from power loss model.

        :param motor_p2_loss_param_a:
            Motor P2 electric power loss parameter a [-].
        :type motor_p2_loss_param_a: float

        :param motor_p2_loss_param_b:
            Motor P2 electric power loss parameter b [-].
        :type motor_p2_loss_param_b: float

        :return:
            Motor P2 electric power loss function.
        :rtype: function
        """
    from .p4 import define_motor_p4_electric_power_loss_function_v1 as func
    return func(motor_p2_loss_param_a, motor_p2_loss_param_b)


@sh.add_function(dsp, outputs=['motor_p2_electric_power_losses'])
def calculate_motor_p2_electric_power_losses(
        motor_p2_electric_power_loss_function, motor_p2_powers,
        motor_p2_torques, motor_p2_speeds):
    """
    Calculates motor P2 electric power losses [kW].

    :param motor_p2_electric_power_loss_function:
        Function that calculates motor P2 electric power losses [kW].
    :type motor_p2_electric_power_loss_function: function

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array | float

    :param motor_p2_torques:
        Torque at motor P2 [N*m].
    :type motor_p2_torques: numpy.array | float

    :param motor_p2_speeds:
        Rotating speed of motor P2 [RPM].
    :type motor_p2_speeds: numpy.array | float

    :return:
        Electric power losses of motor P2 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_electric_power_losses as func
    return func(
        motor_p2_electric_power_loss_function, motor_p2_powers,
        motor_p2_torques, motor_p2_speeds
    )


@sh.add_function(dsp, outputs=['motor_p2_electric_powers'])
def calculate_motor_p2_electric_powers(
        motor_p2_powers, motor_p2_electric_power_losses):
    """
    Calculates motor P2 electric power [kW].

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array | float

    :param motor_p2_electric_power_losses:
        Electric power losses of motor P2 [kW].
    :type motor_p2_electric_power_losses: numpy.array | float

    :return:
        Electric power of motor P2 [kW].
    :rtype: numpy.array | float
    """
    return motor_p2_powers + motor_p2_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p2_powers'])
def calculate_motor_p2_powers_v1(
        motor_p2_electric_powers, motor_p2_electric_power_losses):
    """
    Calculate motor P2 power from electric power and electric power losses [kW].

    :param motor_p2_electric_powers:
        Electric power of motor P2 [kW].
    :type motor_p2_electric_powers: numpy.array | float

    :param motor_p2_electric_power_losses:
        Electric power losses of motor P2 [kW].
    :type motor_p2_electric_power_losses: numpy.array | float

    :return:
        Power at motor P2 [kW].
    :rtype: numpy.array | float
    """
    return motor_p2_electric_powers - motor_p2_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p2_efficiency_ratios'])
def calculate_motor_p2_efficiency_ratios(
        motor_p2_powers, motor_p2_electric_powers):
    """
    Calculates motor P2 efficiency ratio [-].

    :param motor_p2_powers:
        Power at motor P2 [kW].
    :type motor_p2_powers: numpy.array | float

    :param motor_p2_electric_powers:
        Electric power of motor P2 [kW].
    :type motor_p2_electric_powers: numpy.array | float

    :return:
        Motor P2 efficiency ratio [-].
    :rtype: numpy.array | float
    """
    return motor_p2_powers / motor_p2_electric_powers
