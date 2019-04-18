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
dsp.add_data('use_driver', dfl.values.use_driver)


# noinspection PyMissingOrEmptyDocstring
def is_nedc(kw):
    return kw.get('cycle_type') == 'NEDC' and not kw.get('use_driver', True)


# noinspection PyMissingOrEmptyDocstring
def is_wltp(kw):
    return kw.get('cycle_type') == 'WLTP' and not kw.get('use_driver', True)


# noinspection PyMissingOrEmptyDocstring
def is_driver(kw):
    return kw.get('use_driver')


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_nedc_cycle,
    inputs=(
        'gear_box_type', 'gears', 'k1', 'k2', 'k5', 'max_gear', 'times',
        {'cycle_type': sh.SINK, 'use_driver': sh.SINK}
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
            'cycle_type': sh.SINK, 'use_driver': sh.SINK
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


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True,
    outputs=['driver_style_ratio']
)
def default_driver_style_ratio(driver_style='normal'):
    """
    Return the default driver style ratio [-].

    :param driver_style:
        Driver style (aggressive, normal, gentle).
    :type driver_style: str

    :return:
        Driver style ratio [-].
    :rtype: float
    """
    from ...physical.defaults import dfl
    return dfl.functions.default_driver_style_ratio.ratios[driver_style]


class SimulationModel:
    def __init__(self, models, outputs, index=0):
        self.index = index
        self.models = models
        self.outputs = outputs

    def __call__(self, acceleration, next_time):
        i = self.index
        self.outputs['accelerations'][i] = acceleration
        try:
            self.outputs['times'][i + 1] = next_time
        except IndexError:
            pass
        for m in self.models:
            m(i)
        return self

    def select(self, *items, di=0):
        i = max(self.index + di, 0)
        res = sh.selector(items, self.outputs, output_type='list')
        res = [v[i] for v in res]
        if len(res) == 1:
            return res[0]
        return res


# noinspection PyMissingOrEmptyDocstring
class CycleModel(BaseModel):
    key_outputs = 'times', 'accelerations'
    contract_outputs = 'times',
    types = {float: set(key_outputs)}

    def __init__(self, path_velocities=None, path_distances=None,
                 full_load_curve=None, time_sample_frequency=None,
                 road_loads=None, vehicle_mass=None, inertial_factor=None,
                 driver_style_ratio=None, static_friction=None,
                 wheel_drive_load_fraction=None, outputs=None):
        from .logic import dsp as _logic, define_max_acceleration_model as f
        if path_distances is not None:
            self.stop_distance = path_distances[-1]
            d = _logic.register(memo={})
            d.set_default_value('path_distances', path_distances)
            d.set_default_value('path_velocities', path_velocities)
            d.set_default_value('full_load_curve', full_load_curve)
            d.set_default_value(
                'max_acceleration_model',
                f(road_loads, vehicle_mass, inertial_factor,
                  static_friction, wheel_drive_load_fraction)
            )
            d.set_default_value('time_sample_frequency', time_sample_frequency)
            d.set_default_value('driver_style_ratio', driver_style_ratio)
            self.model = sh.DispatchPipe(
                d, inputs=['simulation_model'],
                outputs=('next_time', 'acceleration')
            )
        super(CycleModel, self).__init__(outputs)

    def init_driver(self, *models):
        keys = 'times', 'accelerations'
        simulation = SimulationModel(models, self.outputs)
        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            times, acc = sh.selector(keys, self._outputs, output_type='list')
            n = len(times) - 1

            def _next(i):
                if i > n:
                    raise StopIteration
                simulation.index, a, t = i, acc[i], times[min(i + 1, n)]
                simulation(a, t)
                return t, a

            return _next

        times, acc = sh.selector(keys, self.outputs, output_type='list')
        times[0] = 0

        def _next(i):
            simulation.index = i
            t, a = self.model(simulation)

            if simulation(a, t).select('distances') >= self.stop_distance:
                raise StopIteration
            return t, a

        return _next

    def init_results(self, velocities, distances, *models):
        times, acc = self.outputs['times'], self.outputs['accelerations']

        d_gen = self.init_driver(velocities, distances, *models)

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
def define_cycle_prediction_model(
        path_velocities, path_distances, full_load_curve, time_sample_frequency,
        road_loads, vehicle_mass, inertial_factor, driver_style_ratio,
        static_friction, wheel_drive_load_fraction):
    """
    Defines the vehicle prediction model.

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Wheels prediction model.
    :rtype: WheelsModel
    """
    return CycleModel(
        path_velocities, path_distances, full_load_curve, time_sample_frequency,
        road_loads, vehicle_mass, inertial_factor, driver_style_ratio,
        static_friction, wheel_drive_load_fraction
    )
