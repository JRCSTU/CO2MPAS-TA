# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the electric motor in P0 position.
"""
import schedula as sh

dsp = sh.BlueDispatcher(
    name='Motor P0',
    description='Models the motor P0 (motor on the engine service belt).'
)


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
def calculate_motor_p0_speeds(engine_speeds_out, motor_p0_speed_ratio=1):
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


@sh.add_function(
    dsp, outputs=['motor_p0_electric_power_loss_function'], weight=1
)
def define_motor_p0_electric_power_loss_function(motor_p0_efficiency):
    """
    Define motor P0 electric power loss function from constant efficiency input.

    :param motor_p0_efficiency:
        Motor P0 efficiency [-].
    :type motor_p0_efficiency: float

    :return:
        Motor P0 electric power loss function.
    :rtype: function
    """
    from .p4 import define_motor_p4_electric_power_loss_function as func
    return func(motor_p0_efficiency)


@sh.add_function(dsp, outputs=['motor_p0_electric_power_loss_function'])
def define_motor_p0_electric_power_loss_function_v1(
        motor_p0_loss_param_a, motor_p0_loss_param_b):
    """
        Define motor P0 electric power loss function from power loss model.

        :param motor_p0_loss_param_a:
            Motor P0 electric power loss parameter a [-].
        :type motor_p0_loss_param_a: float

        :param motor_p0_loss_param_b:
            Motor P0 electric power loss parameter b [-].
        :type motor_p0_loss_param_b: float

        :return:
            Motor P0 electric power loss function.
        :rtype: function
        """
    from .p4 import define_motor_p4_electric_power_loss_function_v1 as func
    return func(motor_p0_loss_param_a, motor_p0_loss_param_b)


@sh.add_function(dsp, outputs=['motor_p0_electric_power_losses'])
def calculate_motor_p0_electric_power_losses(
        motor_p0_electric_power_loss_function, motor_p0_powers,
        motor_p0_torques, motor_p0_speeds):
    """
    Calculates motor P0 electric power losses [kW].

    :param motor_p0_electric_power_loss_function:
        Function that calculates motor P0 electric power losses [kW].
    :type motor_p0_electric_power_loss_function: function

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array | float

    :param motor_p0_torques:
        Torque at motor P0 [N*m].
    :type motor_p0_torques: numpy.array | float

    :param motor_p0_speeds:
        Rotating speed of motor P0 [RPM].
    :type motor_p0_speeds: numpy.array | float

    :return:
        Electric power losses of motor P0 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_electric_power_losses as func
    return func(
        motor_p0_electric_power_loss_function, motor_p0_powers,
        motor_p0_torques, motor_p0_speeds
    )


@sh.add_function(dsp, outputs=['motor_p0_electric_powers'])
def calculate_motor_p0_electric_powers(
        motor_p0_powers, motor_p0_electric_power_losses):
    """
    Calculates motor P0 electric power [kW].

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array | float

    :param motor_p0_electric_power_losses:
        Electric power losses of motor P0 [kW].
    :type motor_p0_electric_power_losses: numpy.array | float

    :return:
        Electric power of motor P0 [kW].
    :rtype: numpy.array | float
    """
    return motor_p0_powers + motor_p0_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p0_powers'])
def calculate_motor_p0_powers_v1(
        motor_p0_electric_powers, motor_p0_electric_power_losses):
    """
    Calculate motor P0 power from electric power and electric power losses [kW].

    :param motor_p0_electric_powers:
        Electric power of motor P0 [kW].
    :type motor_p0_electric_powers: numpy.array | float

    :param motor_p0_electric_power_losses:
        Electric power losses of motor P0 [kW].
    :type motor_p0_electric_power_losses: numpy.array | float

    :return:
        Power at motor P0 [kW].
    :rtype: numpy.array | float
    """
    return motor_p0_electric_powers - motor_p0_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p0_efficiency_ratios'])
def calculate_motor_p0_efficiency_ratios(
        motor_p0_powers, motor_p0_electric_powers):
    """
    Calculates motor P0 efficiency ratio [-].

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array | float

    :param motor_p0_electric_powers:
        Electric power of motor P0 [kW].
    :type motor_p0_electric_powers: numpy.array | float

    :return:
        Motor P0 efficiency ratio [-].
    :rtype: numpy.array | float
    """
    return motor_p0_powers / motor_p0_electric_powers
