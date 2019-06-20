# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the electric motor in P1 position.
"""
import schedula as sh

dsp = sh.BlueDispatcher(
    name='Motor P1',
    description='Models the motor P1 (motor downstream the engine).'
)


@sh.add_function(dsp, outputs=['motor_p1_speed_ratio'])
def identify_motor_p1_speed_ratio(engine_speeds_out, motor_p1_speeds):
    """
    Identifies motor P1 speed ratio.

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array | float

    :param motor_p1_speeds:
        Rotating speed of motor P1 [RPM].
    :type motor_p1_speeds: numpy.array | float

    :return:
        Motor P1 speed ratio [-].
    :rtype: float
    """
    from .p4 import identify_motor_p4_speed_ratio
    return identify_motor_p4_speed_ratio(engine_speeds_out, motor_p1_speeds)


dsp.add_data('motor_p1_speed_ratio', 1, sh.inf(10, 1))


@sh.add_function(dsp, outputs=['motor_p1_speeds'])
def calculate_motor_p1_speeds(engine_speeds_out, motor_p1_speed_ratio):
    """
    Calculates rotating speed of motor P1 [RPM].

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array | float

    :param motor_p1_speed_ratio:
        Ratio between motor P1 speed and engine speed [-].
    :type motor_p1_speed_ratio: float

    :return:
        Rotating speed of motor P1 [RPM].
    :rtype: numpy.array | float
    """
    return engine_speeds_out * motor_p1_speed_ratio


@sh.add_function(dsp, inputs_kwargs=True, outputs=['engine_speeds_out'])
def calculate_engine_speeds_out(motor_p1_speeds, motor_p1_speed_ratio=1):
    """
    Calculates the engine speed [RPM].

    :param motor_p1_speeds:
        Rotating speed of motor P1 [RPM].
    :type motor_p1_speeds: numpy.array | float

    :param motor_p1_speed_ratio:
        Ratio between motor P1 speed and wheel speed [-].
    :type motor_p1_speed_ratio: float

    :return:
        Engine speed [RPM].
    :rtype: numpy.array | float
    """
    return motor_p1_speeds / motor_p1_speed_ratio


@sh.add_function(dsp, outputs=['motor_p1_torques'])
def calculate_motor_p1_torques(motor_p1_powers, motor_p1_speeds):
    """
    Calculates torque at motor P1 [N*m].

    :param motor_p1_powers:
        Power at motor P1 [kW].
    :type motor_p1_powers: numpy.array | float

    :param motor_p1_speeds:
        Rotating speed of motor P1 [RPM].
    :type motor_p1_speeds: numpy.array | float

    :return:
        Torque at motor P1 [N*m].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p1_powers, motor_p1_speeds)


@sh.add_function(dsp, outputs=['motor_p1_powers'])
def calculate_motor_p1_powers(motor_p1_torques, motor_p1_speeds):
    """
    Calculates power at motor P1 [kW].

    :param motor_p1_torques:
        Torque at motor P1 [N*m].
    :type motor_p1_torques: numpy.array | float

    :param motor_p1_speeds:
        Rotating speed of motor P1 [RPM].
    :type motor_p1_speeds: numpy.array | float

    :return:
        Power at motor P1 [kW].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_powers
    return calculate_wheel_powers(motor_p1_torques, motor_p1_speeds)


@sh.add_function(dsp, outputs=['motor_p1_speeds'])
def calculate_motor_p1_speeds_v1(motor_p1_powers, motor_p1_torques):
    """
    Calculates rotating speed of motor P1 [RPM].

    :param motor_p1_powers:
        Power at motor P1 [kW].
    :type motor_p1_powers: numpy.array | float

    :param motor_p1_torques:
        Torque at motor P1 [N*m].
    :type motor_p1_torques: numpy.array | float

    :return:
        Rotating speed of motor P1 [RPM].
    :rtype: numpy.array | float
    """
    from ...wheels import calculate_wheel_torques
    return calculate_wheel_torques(motor_p1_powers, motor_p1_torques)


dsp.add_data('motor_p1_efficiency', 0.9)


@sh.add_function(
    dsp, outputs=['motor_p1_electric_power_loss_function'], weight=1
)
def define_motor_p1_electric_power_loss_function(motor_p1_efficiency):
    """
    Define motor P1 electric power loss function from constant efficiency input.

    :param motor_p1_efficiency:
        Motor P1 efficiency [-].
    :type motor_p1_efficiency: float

    :return:
        Motor P1 electric power loss function.
    :rtype: function
    """
    from .p4 import define_motor_p4_electric_power_loss_function as func
    return func(motor_p1_efficiency)


@sh.add_function(dsp, outputs=['motor_p1_electric_power_loss_function'])
def define_motor_p1_electric_power_loss_function_v1(
        motor_p1_loss_param_a, motor_p1_loss_param_b):
    """
        Define motor P1 electric power loss function from power loss model.

        :param motor_p1_loss_param_a:
            Motor P1 electric power loss parameter a [-].
        :type motor_p1_loss_param_a: float

        :param motor_p1_loss_param_b:
            Motor P1 electric power loss parameter b [-].
        :type motor_p1_loss_param_b: float

        :return:
            Motor P1 electric power loss function.
        :rtype: function
        """
    from .p4 import define_motor_p4_electric_power_loss_function_v1 as func
    return func(motor_p1_loss_param_a, motor_p1_loss_param_b)


@sh.add_function(dsp, outputs=['motor_p1_electric_power_losses'])
def calculate_motor_p1_electric_power_losses(
        motor_p1_electric_power_loss_function, motor_p1_powers,
        motor_p1_torques, motor_p1_speeds):
    """
    Calculates motor P1 electric power losses [kW].

    :param motor_p1_electric_power_loss_function:
        Function that calculates motor P1 electric power losses [kW].
    :type motor_p1_electric_power_loss_function: function

    :param motor_p1_powers:
        Power at motor P1 [kW].
    :type motor_p1_powers: numpy.array | float

    :param motor_p1_torques:
        Torque at motor P1 [N*m].
    :type motor_p1_torques: numpy.array | float

    :param motor_p1_speeds:
        Rotating speed of motor P1 [RPM].
    :type motor_p1_speeds: numpy.array | float

    :return:
        Electric power losses of motor P1 [kW].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_electric_power_losses as func
    return func(
        motor_p1_electric_power_loss_function, motor_p1_powers,
        motor_p1_torques, motor_p1_speeds
    )


@sh.add_function(dsp, outputs=['motor_p1_electric_powers'])
def calculate_motor_p1_electric_powers(
        motor_p1_powers, motor_p1_electric_power_losses):
    """
    Calculates motor P1 electric power [kW].

    :param motor_p1_powers:
        Power at motor P1 [kW].
    :type motor_p1_powers: numpy.array | float

    :param motor_p1_electric_power_losses:
        Electric power losses of motor P1 [kW].
    :type motor_p1_electric_power_losses: numpy.array | float

    :return:
        Electric power of motor P1 [kW].
    :rtype: numpy.array | float
    """
    return motor_p1_powers + motor_p1_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p1_powers'])
def calculate_motor_p1_powers_v1(
        motor_p1_electric_powers, motor_p1_electric_power_losses):
    """
    Calculate motor P1 power from electric power and electric power losses [kW].

    :param motor_p1_electric_powers:
        Electric power of motor P1 [kW].
    :type motor_p1_electric_powers: numpy.array | float

    :param motor_p1_electric_power_losses:
        Electric power losses of motor P1 [kW].
    :type motor_p1_electric_power_losses: numpy.array | float

    :return:
        Power at motor P1 [kW].
    :rtype: numpy.array | float
    """
    return motor_p1_electric_powers - motor_p1_electric_power_losses


@sh.add_function(dsp, outputs=['motor_p1_efficiency_ratios'])
def calculate_motor_p1_efficiency_ratios(
        motor_p1_powers, motor_p1_electric_powers):
    """
    Calculates motor P1 efficiency ratio [-].

    :param motor_p1_powers:
        Power at motor P1 [kW].
    :type motor_p1_powers: numpy.array | float

    :param motor_p1_electric_powers:
        Electric power of motor P1 [kW].
    :type motor_p1_electric_powers: numpy.array | float

    :return:
        Motor P1 efficiency ratio [-].
    :rtype: numpy.array | float
    """
    from .p4 import calculate_motor_p4_efficiency_ratios as func
    return func(motor_p1_powers, motor_p1_electric_powers)
