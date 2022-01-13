# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the CMV Approach with Cold/Hot.
"""
import schedula as sh
from co2mpas.defaults import dfl
from .core import prediction_gears_gsm, GSMColdHot

dsp = sh.BlueDispatcher(name='Corrected Matrix Velocity Approach with Cold/Hot')


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True, outputs=['CMV_Cold_Hot']
)
def calibrate_gear_shifting_cmv_cold_hot(
        correct_gear, times, gears, engine_speeds_out, velocities,
        accelerations, motive_powers, velocity_speed_ratios,
        time_cold_hot_transition=dfl.values.time_cold_hot_transition,
        stop_velocity=dfl.values.stop_velocity):
    """
    Calibrates a corrected matrix velocity for cold and hot phases to predict
    gears.

    :param correct_gear:
        A function to correct the predicted gear.
    :type correct_gear: callable

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param time_cold_hot_transition:
        Time at cold hot transition phase [s].
    :type time_cold_hot_transition: float

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :returns:
        Two corrected matrix velocities for cold and hot phases.
    :rtype: dict
    """
    from .cmv import CMV
    model = GSMColdHot(time_cold_hot_transition=time_cold_hot_transition).fit(
        CMV, times, correct_gear, gears, engine_speeds_out, times, velocities,
        accelerations, motive_powers, velocity_speed_ratios, stop_velocity
    )

    return model or sh.NONE


# predict gears with corrected matrix velocity
dsp.add_function(
    function=prediction_gears_gsm,
    inputs=[
        'correct_gear', 'gear_filter', 'CMV_Cold_Hot', 'times', 'velocities',
        'accelerations', 'motive_powers', 'cycle_type', 'velocity_speed_ratios'
    ],
    outputs=['gears']
)
