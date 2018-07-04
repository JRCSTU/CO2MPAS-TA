# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions that model the engine start stop strategy.
"""

import sklearn.tree as sk_tree
import sklearn.pipeline as sk_pip
import sklearn.feature_selection as sk_fsel
import numpy as np
import co2mpas.model.physical.defaults as defaults
import schedula as sh


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
    ts += min_time_engine_on_after_start + defaults.dfl.EPS
    for i, j in np.column_stack((mask, np.searchsorted(times, ts))):
        on_engine[i:j] = True

    return on_engine


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


def default_use_basic_start_stop_model(is_hybrid):
    """
    Returns a flag that defines if basic or complex start stop model is applied.

    ..note:: The basic start stop model is function of velocity and
      acceleration. While, the complex model is function of velocity,
      acceleration, temperature, and battery state of charge.

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        If True the basic start stop model is applied, otherwise complex one.
    :rtype: bool
    """

    return not is_hybrid


def calibrate_start_stop_model(
        on_engine, velocities, accelerations, engine_coolant_temperatures,
        state_of_charges):
    """
    Calibrates an start/stop model to predict if the engine is on.

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param state_of_charges:
        State of charge of the battery [%].

        .. note::

            `state_of_charges` = 99 is equivalent to 99%.
    :type state_of_charges: numpy.array

    :return:
        Start/stop model.
    :rtype: callable
    """

    soc = np.zeros_like(state_of_charges)
    soc[0], soc[1:] = state_of_charges[0], state_of_charges[:-1]
    model = StartStopModel()
    model.fit(
        on_engine, velocities, accelerations, engine_coolant_temperatures,
        state_of_charges
    )

    return model


# noinspection PyShadowingBuiltins,PyUnusedLocal,PyMissingOrEmptyDocstring
class StartStopModel(object):
    VEL = defaults.dfl.functions.StartStopModel.stop_velocity
    ACC = defaults.dfl.functions.StartStopModel.plateau_acceleration

    def __init__(self):
        self.simple = self.complex = self.base

    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def base(self, velocity, acceleration, *args):
        return (velocity > self.VEL) | (acceleration > self.ACC)

    def fit_simple(self, on_engine, velocities, accelerations, *args):
        model = sk_tree.DecisionTreeClassifier(
            random_state=0, max_depth=3
        )
        model.fit(np.column_stack((velocities, accelerations)), on_engine)
        predict = model.predict

        def simple(velocity, acceleration, *a):
            return predict([[velocity, acceleration]])[0]

        self.simple = simple

    def fit_complex(self, on_engine, velocities, accelerations,
                    engine_coolant_temperatures, state_of_charges):
        model = sk_tree.DecisionTreeClassifier(random_state=0, max_depth=4)
        model = sk_pip.Pipeline([
            ('feature_selection',
             sk_fsel.SelectFromModel(model)),
            ('classification', model)
        ])
        model.fit(np.column_stack((
            velocities, accelerations, engine_coolant_temperatures,
            state_of_charges
        )), on_engine)
        predict = model.predict

        def complex(velocity, acceleration, temperature, prev_soc):
            return predict([[velocity, acceleration, temperature, prev_soc]])[0]

        self.complex = complex

    def fit(self, on_engine, velocities, accelerations,
            engine_coolant_temperatures, state_of_charges):
        if on_engine.all():
            self.simple = self.complex = self.base
        else:
            self.fit_simple(on_engine, velocities, accelerations)
            self.fit_complex(on_engine, velocities, accelerations,
                             engine_coolant_temperatures, state_of_charges)
        return self


# noinspection PyMissingOrEmptyDocstring
class EngineStartStopModel:
    key_outputs = ['on_engine', 'engine_starts']
    types = {bool: {'on_engine', 'engine_starts'}}

    def __init__(self, start_stop_model=None, start_stop_activation_time=None,
                 correct_start_stop_with_gears=False,
                 min_time_engine_on_after_start=0.0, has_start_stop=True,
                 use_basic_start_stop=True, outputs=None):
        self.start_stop_activation_time = start_stop_activation_time
        self.correct_start_stop_with_gears = correct_start_stop_with_gears
        self.min_time_engine_on_after_start = min_time_engine_on_after_start
        self.has_start_stop = has_start_stop
        self.use_basic_start_stop = use_basic_start_stop
        self.start_stop_model = start_stop_model
        self._outputs = outputs or {}
        self.outputs = None

    def __call__(self, times, *args, **kwargs):
        self.set_outputs(times.shape[0])
        for _ in self.yield_results(times, *args, **kwargs):
            pass
        return sh.selector(self.key_outputs, self.outputs, output_type='list')

    def set_outputs(self, n, outputs=None):
        if outputs is None:
            outputs = {}
        outputs.update(self._outputs or {})

        for t, names in self.types.items():
            names = names - set(outputs)
            if names:
                outputs.update(zip(names, np.empty((len(names), n), dtype=t)))
        self.outputs = outputs

    def yield_results(self, times, velocities, accelerations,
                      engine_coolant_temperatures, state_of_charges,
                      gears=None):
        keys = ['on_engine', 'engine_starts']
        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            yield from zip(
                *sh.selector(keys, self._outputs, output_type='list'))
        elif not self.has_start_stop:
            outputs = self.outputs
            outputs['on_engine'][0], outputs['engine_starts'][0] = True, False
            yield True, False
            for i, prev in enumerate(outputs['on_engine'][:-1], 1):
                outputs['engine_starts'][i] = start = not prev
                outputs['on_engine'][i] = True
                yield True, start
        else:
            outputs, t_switch_on, can_off = self.outputs, times[0], False
            base = self.start_stop_model.base

            if self.use_basic_start_stop:
                predict = self.start_stop_model.simple
            else:
                predict = self.start_stop_model.complex

            it = enumerate(zip(times, zip(
                velocities, accelerations, engine_coolant_temperatures,
                state_of_charges
            )))
            prev = True
            for i, (t, v) in it:
                if i > 0:
                    if outputs['engine_starts'][i - 1]:
                        t_switch_on = t + self.min_time_engine_on_after_start
                        can_off = False

                    if not can_off:
                        can_off = base(*v)

                    prev = outputs['on_engine'].take(i - 1, mode='clip')

                # noinspection PyPep8
                on = t <= self.start_stop_activation_time \
                     or \
                     self.correct_start_stop_with_gears and gears[i] > 0 \
                     or \
                     not (can_off and t >= t_switch_on) \
                     or \
                     ((prev or base(*v)) and predict(*v))
                outputs['on_engine'][i] = on
                outputs['engine_starts'][i] = start = on and prev != on
                yield on, start


def define_engine_start_stop_prediction_model(
        start_stop_model, start_stop_activation_time,
        correct_start_stop_with_gears, min_time_engine_on_after_start,
        has_start_stop, use_basic_start_stop):
    """
    Defines the engine start/stop prediction model.

    :param start_stop_model:
        Start/stop model.
    :type start_stop_model: StartStopModel

    :param start_stop_activation_time:
        Start-stop activation time threshold [s].
    :type start_stop_activation_time: float

    :param correct_start_stop_with_gears:
        A flag to impose engine on when there is a gear > 0.
    :type correct_start_stop_with_gears: bool

    :param min_time_engine_on_after_start:
        Minimum time of engine on after a start [s].
    :type min_time_engine_on_after_start: float

    :param has_start_stop:
        Does the vehicle have start/stop system?
    :type has_start_stop: bool

    :param use_basic_start_stop:
        If True the basic start stop model is applied, otherwise complex one.

        ..note:: The basic start stop model is function of velocity and
          acceleration. While, the complex model is function of velocity,
          acceleration, temperature, and battery state of charge.
    :type use_basic_start_stop: bool

    :return:
        Engine start/stop prediction model.
    :rtype: EngineStartStopModel
    """
    model = EngineStartStopModel(
        start_stop_model, start_stop_activation_time,
        correct_start_stop_with_gears, min_time_engine_on_after_start,
        has_start_stop, use_basic_start_stop
    )
    return model


def define_fake_engine_start_stop_prediction_model(on_engine, engine_starts):
    """
    Defines a fake engine start/stop prediction model.

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

    :return:
        Engine start stop prediction model.
    :rtype: EngineStartStopModel
    """
    model = EngineStartStopModel(outputs={
        'on_engine': on_engine, 'engine_starts': engine_starts
    })

    return model


def predict_engine_start_stop(
        start_stop_prediction_model, times, velocities, accelerations,
        engine_coolant_temperatures, state_of_charges, gears):
    """
    Predicts if the engine is on and when the engine starts.

    :param start_stop_prediction_model:
        Engine start/stop prediction model.
    :type start_stop_prediction_model: EngineStartStopModel

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param state_of_charges:
        State of charge of the battery [%].

        .. note::

            `state_of_charges` = 99 is equivalent to 99%.
    :type state_of_charges: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        If the engine is on and when the engine starts [-, -].
    :rtype: numpy.array, numpy.array
    """

    return start_stop_prediction_model(
        times, velocities, accelerations, engine_coolant_temperatures,
        state_of_charges, gears)


def default_correct_start_stop_with_gears(gear_box_type):
    """
    Defines a flag that imposes the engine on when there is a gear > 0.

    :param gear_box_type:
        Gear box type (manual or automatic or cvt).
    :type gear_box_type: str

    :return:
        A flag to impose engine on when there is a gear > 0.
    :rtype: bool
    """

    return gear_box_type == 'manual'


def default_start_stop_activation_time(has_start_stop):
    """
    Returns the default start stop activation time threshold [s].
    
    :return:
        Start-stop activation time threshold [s].
    :rtype: float
    """
    d = defaults.dfl.functions
    if not has_start_stop or d.ENABLE_ALL_FUNCTIONS or \
            d.default_start_stop_activation_time.ENABLE:
        return d.default_start_stop_activation_time.threshold
    return sh.NONE


def start_stop():
    """
    Defines the engine start/stop model.

    .. dispatcher:: d

        >>> d = start_stop()

    :return:
        The engine start/stop model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='start_stop',
        description='Models the engine start/stop strategy.'
    )

    d.add_function(
        function=identify_on_engine,
        inputs=['times', 'engine_speeds_out', 'idle_engine_speed',
                'min_time_engine_on_after_start'],
        outputs=['on_engine']
    )

    d.add_function(
        function=identify_engine_starts,
        inputs=['on_engine'],
        outputs=['engine_starts']
    )

    d.add_function(
        function=default_start_stop_activation_time,
        inputs=['has_start_stop'],
        outputs=['start_stop_activation_time']
    )

    d.add_function(
        function=calibrate_start_stop_model,
        inputs=['on_engine', 'velocities', 'accelerations',
                'engine_coolant_temperatures', 'state_of_charges'],
        outputs=['start_stop_model']
    )

    d.add_function(
        function=default_correct_start_stop_with_gears,
        inputs=['gear_box_type'],
        outputs=['correct_start_stop_with_gears']
    )

    d.add_data(
        data_id='min_time_engine_on_after_start',
        default_value=defaults.dfl.values.min_time_engine_on_after_start
    )

    d.add_data(
        data_id='has_start_stop',
        default_value=defaults.dfl.values.has_start_stop
    )

    d.add_data(
        data_id='is_hybrid',
        default_value=defaults.dfl.values.is_hybrid
    )

    d.add_function(
        function=default_use_basic_start_stop_model,
        inputs=['is_hybrid'],
        outputs=['use_basic_start_stop']
    )

    d.add_function(
        function=define_engine_start_stop_prediction_model,
        inputs=['start_stop_model', 'start_stop_activation_time',
                'correct_start_stop_with_gears',
                'min_time_engine_on_after_start',
                'has_start_stop', 'use_basic_start_stop'],
        outputs=['start_stop_prediction_model'],
        weight=4000
    )

    d.add_function(
        function=define_fake_engine_start_stop_prediction_model,
        inputs=['on_engine', 'engine_starts'],
        outputs=['start_stop_prediction_model']
    )

    d.add_function(
        function=predict_engine_start_stop,
        inputs=['start_stop_prediction_model', 'times', 'velocities',
                'accelerations', 'engine_coolant_temperatures',
                'state_of_charges', 'gears'],
        outputs=['on_engine', 'engine_starts']
    )

    return d
