# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the vehicle control strategy.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.control

.. autosummary::
    :nosignatures:
    :toctree: control/

    ecms
    start_stop
"""
import numpy as np
import schedula as sh
from ..defaults import dfl
from .start_stop import dsp as _start_stop

dsp = sh.BlueDispatcher(
    name='Control', description='Models the vehicle control strategy.'
)

dsp.add_data(
    'min_time_engine_on_after_start', dfl.values.min_time_engine_on_after_start
)


@sh.add_function(dsp, outputs=['on_engine'])
def identify_on_engine(
        times, engine_speeds_out, idle_engine_speed,
        min_time_engine_on_after_start):
    """
    Identifies if the engine is on [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param min_time_engine_on_after_start:
        Minimum time of engine on after a start [s].
    :type min_time_engine_on_after_start: float

    :return:
        If the engine is on [-].
    :rtype: numpy.array
    """

    on_engine = engine_speeds_out > idle_engine_speed[0] - idle_engine_speed[1]
    mask = np.where(identify_engine_starts(on_engine))[0] + 1
    ts = np.asarray(times[mask], dtype=float)
    ts += min_time_engine_on_after_start + dfl.EPS
    for i, j in np.column_stack((mask, np.searchsorted(times, ts))):
        on_engine[i:j] = True

    return on_engine


@sh.add_function(dsp, outputs=['engine_starts'])
def identify_engine_starts(on_engine):
    """
    Identifies when the engine starts [-].

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :return:
        When the engine starts [-].
    :rtype: numpy.array
    """

    engine_starts = np.zeros_like(on_engine, dtype=bool)
    engine_starts[:-1] = on_engine[1:] & (on_engine[:-1] != on_engine[1:])
    return engine_starts


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_start_stop,
    dsp_id='start_stop',
    inputs=(
        'accelerations', 'correct_start_stop_with_gears', 'start_stop_model',
        'engine_coolant_temperatures', 'engine_starts', 'use_basic_start_stop',
        'on_engine', 'gears', 'gear_box_type', 'start_stop_activation_time',
        'has_start_stop', 'is_hybrid', 'times', 'velocities'
    ),
    outputs=(
        'use_basic_start_stop', 'on_engine', 'start_stop_prediction_model',
        'start_stop_model', 'correct_start_stop_with_gears',
        'start_stop_activation_time'
    )
)
