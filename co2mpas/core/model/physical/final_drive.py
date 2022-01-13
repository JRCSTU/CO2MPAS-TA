# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the final drive.
"""

import logging
import collections
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl

log = logging.getLogger(__name__)

dsp = sh.BlueDispatcher(
    name='Final drive', description='Models the final drive.'
)
dsp.add_data('final_drive_ratio', dfl.values.final_drive_ratio)


@sh.add_function(dsp, inputs_kwargs=True, outputs=['final_drive_ratios'])
def calculate_final_drive_ratios(final_drive_ratio, n_gears=1):
    """
    Defines final drive ratios for each gear [-].

    :param final_drive_ratio:
        Final drive ratio [-].
    :type final_drive_ratio: float

    :param n_gears:
        Number of gears [-].
    :type n_gears: int, optional

    :return:
        Final drive ratios [-].
    :rtype: dict
    """
    d = collections.defaultdict(lambda: final_drive_ratio)
    d.update(dict.fromkeys(range(0, int(n_gears + 1)), final_drive_ratio))
    return d


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def is_not_manual_or_automatic(gear_box_type, *args):
    return gear_box_type not in ('manual', 'automatic')


dsp.add_function(
    function=sh.add_args(calculate_final_drive_ratios),
    inputs=['gear_box_type', 'final_drive_ratio'],
    outputs=['final_drive_ratios'],
    input_domain=is_not_manual_or_automatic
)


@sh.add_function(dsp, outputs=['final_drive_ratio_vector'])
def calculate_final_drive_ratio_vector(final_drive_ratios, gears):
    """
    Calculates the final drive ratio vector [-].

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        Final drive ratio vector [-].
    :rtype: numpy.array
    """
    fdr0 = final_drive_ratios[min(final_drive_ratios)]
    d = collections.defaultdict(lambda: fdr0)
    d.update(final_drive_ratios)
    return np.vectorize(lambda k: d[k])(gears)


@sh.add_function(dsp, outputs=['final_drive_speeds_in'])
def calculate_final_drive_speeds_in(
        final_drive_speeds_out, final_drive_ratio_vector):
    """
    Calculates final drive speed [RPM].

    :param final_drive_speeds_out:
        Rotating speed of the wheel [RPM].
    :type final_drive_speeds_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :return:
        Final drive speed in [RPM].
    :rtype: numpy.array | float
    """
    return final_drive_speeds_out * final_drive_ratio_vector


dsp.add_data('wheel_drive', dfl.values.wheel_drive)


@sh.add_function(dsp, outputs=['n_wheel_drive'])
def define_n_wheel_drive(wheel_drive):
    """
    Defines the default number of wheel drive [-].

    :param wheel_drive:
        Wheel drive (i.e., front, rear, front+rear).
    :type wheel_drive: str

    :return:
        Number of wheel drive [-].
    :rtype: int
    """
    if 'front+rear' == wheel_drive:
        return 4
    return 2


@sh.add_function(dsp, outputs=['final_drive_efficiency'])
def default_final_drive_efficiency(n_wheel_drive):
    """
    Returns the default final drive efficiency [-].

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :return:
        Final drive efficiency [-].
    :rtype: float
    """
    from asteval import Interpreter as Interp
    formula = dfl.functions.default_final_drive_efficiency.formula
    return Interp(dict(n_wheel_drive=n_wheel_drive)).eval(formula)


@sh.add_function(dsp, outputs=['final_drive_powers_in'])
def calculate_final_drive_powers_in(
        final_drive_powers_out, final_drive_efficiency):
    """
    Calculate final drive power in [kW].

    :param final_drive_powers_out:
        Final drive power out [kW].
    :type final_drive_powers_out: numpy.array | float

    :param final_drive_efficiency:
        Final drive efficiency [-].
    :type final_drive_efficiency: float

    :return:
        Final drive power in [kW].
    :rtype: numpy.array | float
    """
    eff, p = final_drive_efficiency, final_drive_powers_out
    return np.where(p < 0, eff, 1 / eff) * p


@sh.add_function(dsp, outputs=['final_drive_powers_out'])
def calculate_final_drive_powers_out(
        final_drive_powers_in, final_drive_efficiency):
    """
    Calculate final drive power out [kW].

    :param final_drive_powers_in:
        Final drive power in [kW].
    :type final_drive_powers_in: numpy.array | float

    :param final_drive_efficiency:
        Final drive efficiency [-].
    :type final_drive_efficiency: float

    :return:
        Final drive power out [kW].
    :rtype: numpy.array | float
    """
    return calculate_final_drive_powers_in(
        final_drive_powers_in, 1 / final_drive_efficiency
    )


@sh.add_function(dsp, outputs=['final_drive_torques_in'])
def calculate_final_drive_torques_in(
        final_drive_powers_in, final_drive_speeds_in):
    """
    Calculate final drive power in [kW].

    :param final_drive_powers_in:
        Final drive power in [kW].
    :type final_drive_powers_in: numpy.array | float

    :param final_drive_speeds_in:
        Final drive speed in [RPM].
    :type final_drive_speeds_in: numpy.array | float

    :return:
        Final drive torque in [N*m].
    :rtype: numpy.array | float
    """
    from .wheels import calculate_wheel_torques as func
    return func(final_drive_powers_in, final_drive_speeds_in)


dsp.add_function(
    function=sh.bypass,
    inputs=['final_drive_efficiency'],
    outputs=['final_drive_mean_efficiency']
)


# noinspection PyUnusedLocal
def domain_final_drive_torque_losses_v1(n_dyno_axes, n_wheel_drive, *args):
    """
    Check the validity of number of wheel drive respect to the dyno axes
    assuming 2 wheels per axes.

    :param n_dyno_axes:
        Number of dyno axes [-].
    :type n_dyno_axes: int

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :return:
        True and log a waring if `n_wheel_drive` does not respect the domain.
    :rtype: bool
    """

    if n_dyno_axes < n_wheel_drive / 2:
        msg = 'WARNING: n_dyno_axes(%d) < n_wheel_drive(%d) / 2!'
        log.warning(msg, n_dyno_axes, n_wheel_drive)
    return True
