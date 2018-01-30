#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides the model to calculate theoretical times, velocities, and gears.

Sub-Modules:

.. currentmodule:: co2mpas.model.physical.cycle

.. autosummary::
    :nosignatures:
    :toctree: cycle/

    NEDC
    WLTP

"""

import schedula as sh
import numpy as np


def is_nedc(kwargs):
    return kwargs['cycle_type'] == 'NEDC'


def is_wltp(kwargs):
    return kwargs['cycle_type'] == 'WLTP'


def cycle_times(frequency, time_length):
    """
    Returns the time vector with constant time step [s].

    :param frequency:
        Time frequency [1/s].
    :type frequency: float

    :param time_length:
        Length of the time vector [-].
    :type time_length: float

    :return:
        Time vector [s].
    :rtype: numpy.array
    """

    dt = 1 / frequency

    return np.arange(0.0, time_length,  dtype=float) * dt


def calculate_time_length(frequency, max_time):
    """
    Returns the length of the time vector [-].

    :param frequency:
        Time frequency [1/s].
    :type frequency: float

    :param max_time:
        Maximum time [s].
    :type max_time: float

    :return:
        length of the time vector [-].
    :rtype: int
    """
    return np.floor(max_time * frequency) + 1


def select_phases_integration_times(cycle_type):
    """
    Selects the cycle phases integration times [s].

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :return:
        Cycle phases integration times [s].
    :rtype: tuple
    """

    from ..defaults import dfl
    v = dfl.functions.select_phases_integration_times.INTEGRATION_TIMES
    return tuple(sh.pairwise(v[cycle_type.upper()]))


def _extract_indices(bag_phases):
    pit, bag_phases = [], np.asarray(bag_phases)
    n = len(bag_phases) - 1
    for bf in np.unique(bag_phases):
        i = np.where(bf == bag_phases)[0]
        pit.append((i.min(), min(i.max() + 1, n)))
    return sorted(pit)


def extract_phases_integration_times(times, bag_phases):
    """
    Extracts the cycle phases integration times [s] from bag phases vector.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param bag_phases:
        Bag phases [-].
    :type bag_phases: numpy.array

    :return:
        Cycle phases integration times [s].
    :rtype: tuple
    """

    return tuple((times[i], times[j]) for i, j in _extract_indices(bag_phases))


def cycle():
    """
    Defines the cycle model.

    .. dispatcher:: d

        >>> d = cycle()

    :return:
        The cycle model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Cycle model',
        description='Returns the theoretical times, velocities, and gears.'
    )

    from ..defaults import dfl
    d.add_data(
        data_id='time_sample_frequency',
        default_value=dfl.values.time_sample_frequency
    )

    from .NEDC import nedc_cycle
    d.add_dispatcher(
        include_defaults=True,
        dsp=nedc_cycle(),
        inputs=(
            'gear_box_type', 'gears', 'k1', 'k2', 'k5', 'max_gear', 'times',
            {'cycle_type': sh.SINK}),
        outputs=('gears', 'initial_temperature', 'max_time', 'velocities'),
        input_domain=is_nedc
    )

    from .WLTP import wltp_cycle
    d.add_dispatcher(
        include_defaults=True,
        dsp=wltp_cycle(),
        inputs=(
            'accelerations', 'climbing_force', 'downscale_factor',
            'downscale_factor_threshold', 'downscale_phases',
            'engine_max_power', 'engine_speed_at_max_power',
            'full_load_curve', 'gear_box_type', 'gears', 'idle_engine_speed',
            'inertial_factor', 'max_speed_velocity_ratio', 'max_velocity',
            'motive_powers', 'road_loads', 'speed_velocity_ratios', 'times',
            'unladen_mass', 'vehicle_mass', 'velocities', 'wltp_base_model',
            'engine_max_speed',
            'wltp_class', {'cycle_type': sh.SINK}),
        outputs=('gears', 'initial_temperature', 'max_time', 'velocities'),
        input_domain=is_wltp
    )

    d.add_function(
        function=calculate_time_length,
        inputs=['time_sample_frequency', 'max_time'],
        outputs=['time_length']
    )

    d.add_function(
        function=cycle_times,
        inputs=['time_sample_frequency', 'time_length'],
        outputs=['times']
    )

    d.add_function(
        function=len,
        inputs=['velocities'],
        outputs=['time_length']
    )

    d.add_function(
        function=len,
        inputs=['gears'],
        outputs=['time_length'],
        weight=1
    )

    d.add_function(
        function=extract_phases_integration_times,
        inputs=['times', 'bag_phases'],
        outputs=['phases_integration_times']
    )

    d.add_function(
        function=select_phases_integration_times,
        inputs=['cycle_type'],
        outputs=['phases_integration_times'],
        weight=10
    )

    return d
