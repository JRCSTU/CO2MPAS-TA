#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to define theoretical times, velocities, and gears.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.cycle

.. autosummary::
    :nosignatures:
    :toctree: cycle/

    NEDC
    WLTP

"""
import numpy as np
import schedula as sh
from ..defaults import dfl
from co2mpas.utils import BaseModel
from .NEDC import dsp as _nedc_cycle, is_manual
from .WLTP import dsp as _wltp_cycle

dsp = sh.BlueDispatcher(
    name='Cycle model',
    description='Returns the theoretical times, velocities, and gears.'
)
dsp.add_data('time_sample_frequency', dfl.values.time_sample_frequency)


# noinspection PyMissingOrEmptyDocstring
def is_nedc(kwargs):
    return kwargs.get('cycle_type') == 'NEDC'


# noinspection PyMissingOrEmptyDocstring
def is_wltp(kwargs):
    return kwargs.get('cycle_type') == 'WLTP'


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_nedc_cycle,
    inputs=(
        'gear_box_type', 'gears', 'k1', 'k2', 'k5', 'max_gear', 'times',
        {'cycle_type': sh.SINK}
    ),
    outputs=('gears', 'initial_temperature', 'max_time', 'velocities'),
    input_domain=is_nedc
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_wltp_cycle,
    inputs=(
        'accelerations', 'climbing_force', 'downscale_factor', 'motive_powers',
        'downscale_factor_threshold', 'downscale_phases', 'engine_max_power',
        'engine_speed_at_max_power', 'full_load_curve', 'gear_box_type',
        'gears', 'idle_engine_speed', 'speed_velocity_ratios', 'max_velocity',
        'max_speed_velocity_ratio', 'inertial_factor', 'road_loads', 'times',
        'unladen_mass', 'vehicle_mass', 'velocities', 'wltp_base_model',
        'engine_max_speed', 'wltp_class', {
            'cycle_type': sh.SINK
        }
    ),
    outputs=('gears', 'initial_temperature', 'max_time', 'velocities'),
    input_domain=is_wltp
)


@sh.add_function(dsp, outputs=['time_length'])
def calculate_time_length(time_sample_frequency, max_time):
    """
    Returns the length of the time vector [-].

    :param time_sample_frequency:
        Time frequency [1/s].
    :type time_sample_frequency: float

    :param max_time:
        Maximum time [s].
    :type max_time: float

    :return:
        length of the time vector [-].
    :rtype: int
    """
    return np.floor(max_time * time_sample_frequency) + 1


@sh.add_function(dsp, outputs=['times'])
def cycle_times(time_sample_frequency, time_length):
    """
    Returns the time vector with constant time step [s].

    :param time_sample_frequency:
        Time frequency [1/s].
    :type time_sample_frequency: float

    :param time_length:
        Length of the time vector [-].
    :type time_length: float

    :return:
        Time vector [s].
    :rtype: numpy.array
    """

    dt = 1 / time_sample_frequency

    return np.arange(0.0, time_length, dtype=float) * dt


dsp.add_function(function=len, inputs=['velocities'], outputs=['time_length'])
dsp.add_function(
    function=len, inputs=['gears'], outputs=['time_length'], weight=1
)


def _extract_indices(bag_phases):
    pit, bag_phases = [], np.asarray(bag_phases)
    n = len(bag_phases) - 1
    for bf in np.unique(bag_phases):
        i = np.where(bf == bag_phases)[0]
        pit.append((i.min(), min(i.max() + 1, n)))
    return sorted(pit)


@sh.add_function(dsp, outputs=['phases_integration_times'])
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


@sh.add_function(dsp, outputs=['phases_integration_times'], weight=10)
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


# noinspection PyMissingOrEmptyDocstring
class CycleModel(BaseModel):
    key_outputs = 'times', 'accelerations'
    contract_outputs = 'times',
    types = {float: set(key_outputs)}

    def __init__(self, driver_model=None, outputs=None):
        self.driver_model = driver_model
        super(CycleModel, self).__init__(outputs)

    def init_driver(self, *models):
        keys = 'times', 'accelerations'

        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            times, acc = sh.selector(keys, self._outputs, output_type='list')
            n = len(times) - 1

            def _next(i):
                if i > n:
                    raise StopIteration
                return times[min(i + 1, n)], acc[i]

            return _next
        return self.driver_model.init_results(*models)

    def init_results(self, *models):
        times, acc = self.outputs['times'], self.outputs['accelerations']

        d_gen = self.init_driver(*models)

        def _next(i):
            t, a = d_gen(i)
            acc[i] = a
            try:
                times[i + 1] = t
            except IndexError:
                pass
            return times[i], a

        return _next


@sh.add_function(dsp, outputs=['cycle_prediction_model'])
def define_fake_cycle_prediction_model(times, accelerations):
    """
    Defines a fake vehicle prediction model.

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array

    :param wheel_powers:
        Power at the wheels [kW].
    :type wheel_powers: numpy.array

    :param wheel_torques:
        Torque at the wheel [N*m].
    :type wheel_torques: numpy.array

    :return:
        Wheels prediction model.
    :rtype: WheelsModel
    """
    return CycleModel(outputs=dict(times=times, accelerations=accelerations))


@sh.add_function(dsp, outputs=['cycle_prediction_model'], weight=4000)
def define_vehicle_prediction_model(driver_model):
    """
    Defines the vehicle prediction model.

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Wheels prediction model.
    :rtype: WheelsModel
    """
    return CycleModel(driver_model)
