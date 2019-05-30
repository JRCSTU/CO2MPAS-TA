#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to define driver strategies.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.driver

.. autosummary::
    :nosignatures:
    :toctree: driver/

    NEDC
    WLTP
    logic

"""
import numpy as np
import schedula as sh
from ..defaults import dfl
from co2mpas.utils import BaseModel
from .NEDC import dsp as _nedc_cycle, is_manual
from .WLTP import dsp as _wltp_cycle

dsp = sh.BlueDispatcher(
    name='Driver model',
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
        'accelerations', 'downscale_factor', 'motive_powers',
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


# noinspection PyMissingOrEmptyDocstring
class DriverModel(BaseModel):
    key_outputs = 'times', 'accelerations'
    contract_outputs = 'times',
    types = {float: set(key_outputs)}

    def __init__(self, path_velocities=None, path_distances=None,
                 full_load_curve=None, time_sample_frequency=None,
                 road_loads=None, vehicle_mass=None, inertial_factor=None,
                 driver_style_ratio=None, static_friction=None,
                 wheel_drive_load_fraction=None, gear_box_type=None,
                 auxiliaries_power_loss=None, auxiliaries_torque_loss=None,
                 maximum_velocity=None, engine_moment_inertia=None,
                 idle_engine_speed=None, outputs=None):
        from .logic import dsp as _logic, define_max_acceleration_model as f
        if path_distances is not None:
            self.stop_distance = path_distances[-1]
            d = _logic.register(memo={})
            d.set_default_value('path_distances', path_distances)
            d.set_default_value('path_velocities', path_velocities)
            d.set_default_value('full_load_curve', full_load_curve)
            d.set_default_value('engine_moment_inertia', engine_moment_inertia)
            d.set_default_value('idle_engine_speed', idle_engine_speed)
            d.set_default_value(
                'max_acceleration_model',
                f(road_loads, vehicle_mass, inertial_factor,
                  static_friction, wheel_drive_load_fraction, gear_box_type,
                  maximum_velocity)
            )
            d.set_default_value('delta_time', 1 / time_sample_frequency)
            d.set_default_value('driver_style_ratio', driver_style_ratio)
            d.set_default_value(
                'auxiliaries_power_loss', auxiliaries_power_loss
            )
            d.set_default_value(
                'auxiliaries_torque_loss', auxiliaries_torque_loss
            )
            self.model = sh.DispatchPipe(
                d, inputs=['simulation_model'],
                outputs=('next_time', 'acceleration')
            )
        super(DriverModel, self).__init__(outputs)

    def init_driver(self, *models):
        keys = 'times', 'accelerations'
        from .logic import SimulationModel
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


@sh.add_function(dsp, outputs=['driver_prediction_model'])
def define_fake_driver_prediction_model(times, accelerations):
    """
    Defines a fake driver prediction model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :return:
        Driver prediction model.
    :rtype: DriverModel
    """
    return DriverModel(outputs=dict(times=times, accelerations=accelerations))


dsp.add_data('path_velocities', wildcard=True)
dsp.add_data('path_distances', wildcard=True)


@sh.add_function(dsp, outputs=['desired_velocities'])
def calculate_desired_velocities(path_distances, path_velocities, distances):
    """
    Calculates the desired velocity vector [km/h].

    :param path_distances:
        Cumulative distance vector [m].
    :type path_distances: numpy.array

    :param path_velocities:
        Desired velocity vector [km/h].
    :type path_velocities: numpy.array

    :param distances:
        Cumulative distance vector [m].
    :type distances: numpy.array

    :return:
        Desired velocity vector [km/h].
    :rtype: numpy.array
    """
    i = np.searchsorted(path_distances, distances, side='right')
    return path_velocities.take(i, mode='clip')


@sh.add_function(dsp, outputs=['driver_prediction_model'], weight=4000)
def define_driver_prediction_model(
        path_velocities, path_distances, full_load_curve, time_sample_frequency,
        road_loads, vehicle_mass, inertial_factor, driver_style_ratio,
        static_friction, wheel_drive_load_fraction, gear_box_type,
        auxiliaries_power_loss, auxiliaries_torque_loss, maximum_velocity,
        engine_moment_inertia, idle_engine_speed):
    """
    Defines the driver prediction model.

    :param path_velocities:
        Desired velocity vector [km/h].
    :type path_velocities: numpy.array

    :param path_distances:
        Cumulative distance vector [m].
    :type path_distances: numpy.array

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param time_sample_frequency:
        Time frequency [1/s].
    :type time_sample_frequency: float

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param inertial_factor:
        Factor that considers the rotational inertia [%].
    :type inertial_factor: float

    :param driver_style_ratio:
        Driver style ratio [-].
    :type driver_style_ratio: float

    :param static_friction:
        Static friction coefficient [-].
    :type static_friction: float

    :param wheel_drive_load_fraction:
        Repartition of the load on wheel drive axles [-].
    :type wheel_drive_load_fraction: float

    :param gear_box_type:
        Gear box type (manual or automatic or cvt).
    :type gear_box_type: str

    :param auxiliaries_power_loss:
        Constant power loss due to engine auxiliaries [kW].
    :type auxiliaries_power_loss: float

    :param auxiliaries_torque_loss:
        Constant torque loss due to engine auxiliaries [N*m].
    :type auxiliaries_torque_loss: float

    :param maximum_velocity:
        Maximum velocity [km/h].
    :type maximum_velocity: float

    :param engine_moment_inertia:
        Engine moment of inertia [kg*m2].
    :type engine_moment_inertia: float

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Diver prediction model.
    :rtype: DriverModel
    """
    return DriverModel(
        path_velocities, path_distances, full_load_curve, time_sample_frequency,
        road_loads, vehicle_mass, inertial_factor, driver_style_ratio,
        static_friction, wheel_drive_load_fraction, gear_box_type,
        auxiliaries_power_loss, auxiliaries_torque_loss, maximum_velocity,
        engine_moment_inertia, idle_engine_speed
    )
