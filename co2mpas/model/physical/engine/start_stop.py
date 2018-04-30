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
import sklearn.preprocessing as sk_prep
import sklearn.feature_selection as sk_fsel
import numpy as np
import co2mpas.model.physical.defaults as defaults
import schedula as sh
import co2mpas.utils as co2_utl


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
        on_engine, velocities, accelerations, engine_coolant_temperatures, soc
    )

    return model


class DefaultStartStopModel(object):
    @staticmethod
    def predict(X):
        X = np.asarray(X)
        VEL = defaults.dfl.functions.DefaultStartStopModel.stop_velocity
        ACC = defaults.dfl.functions.DefaultStartStopModel.plateau_acceleration
        return (X[:, 0] > VEL) | (X[:, 1] > ACC)


class StartStopModel(object):
    def __init__(self):
        self.base = self.model = DefaultStartStopModel()

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def fit(self, on_engine, velocities, accelerations, *args):
        if on_engine.all():
            self.base = self.model = DefaultStartStopModel()
        else:
            X = np.column_stack((velocities, accelerations) + args)
            model = sk_tree.DecisionTreeClassifier(random_state=0, max_depth=4)
            self.model = sk_pip.Pipeline([
                ('feature_selection',
                 sk_fsel.SelectFromModel(model)),
                ('classification', model)
            ])
            self.model.fit(X, on_engine)
            model = sk_tree.DecisionTreeClassifier(random_state=0, max_depth=3)
            self.base = sk_pip.Pipeline([
                ('feature_selection',
                 sk_prep.FunctionTransformer(lambda X: X[:, :2])),
                ('classification', model)
            ])
            self.base.fit(X, on_engine)
        return self

    def predict(self, times, velocities, accelerations, *args,
                start_stop_activation_time=None, gears=None,
                correct_start_stop_with_gears=False,
                min_time_engine_on_after_start=0.0, has_start_stop=True,
                use_basic_start_stop=True):

        gen = map(tuple, self.yield_on_start(
            times, velocities, accelerations, *args, gears=gears,
            correct_start_stop_with_gears=correct_start_stop_with_gears,
            start_stop_activation_time=start_stop_activation_time,
            min_time_engine_on_after_start=min_time_engine_on_after_start,
            has_start_stop=has_start_stop,
            use_basic_start_stop=use_basic_start_stop
        ))
        dtype = [('on', bool), ('starts', bool)]
        return co2_utl.fromiter(gen, dtype=dtype, count=len(times))

    def yield_on_start(self, times, velocities, accelerations, *args,
                       start_stop_activation_time=None, gears=None,
                       correct_start_stop_with_gears=False,
                       min_time_engine_on_after_start=0.0,
                       has_start_stop=True, use_basic_start_stop=True):
        if has_start_stop:
            to_predict = self.when_predict_on_engine(
                times, start_stop_activation_time, gears,
                correct_start_stop_with_gears
            )
            gen = self._yield_on_start(
                times, to_predict, velocities, accelerations, *args,
                min_time_engine_on_after_start=min_time_engine_on_after_start,
                use_basic_start_stop=use_basic_start_stop
            )
        else:
            gen = self._yield_no_start_stop(times)
        return gen

    def _yield_on_start(self, times, to_predict, velocities, accelerations,
                        *args, min_time_engine_on_after_start=0.0,
                        use_basic_start_stop=True):
        base = DefaultStartStopModel().predict
        on, prev, t_switch_on, can_off = True, True, times[0], False
        model = self.base if use_basic_start_stop else self.model
        predict = model.predict
        args = (times, to_predict, velocities, accelerations) + args
        for v in np.column_stack(args):
            t, p, X = v[0], v[1], v[2:]
            if p and can_off and t >= t_switch_on:
                on = (prev or base([X])[0]) and predict([X])[0]
            else:
                on = True

            start = prev != on and on
            on_start = [on, start]
            yield on_start
            on = on_start[0]
            if on and prev != on:
                t_switch_on = t + min_time_engine_on_after_start
                can_off = False

            if not can_off:
                can_off = base([X])[0]

            prev = on

    @staticmethod
    def _yield_no_start_stop(times):
        on, prev = True, True
        for _ in times:
            on = True
            start = prev != on and on
            on_start = [on, start]
            yield on_start
            prev = on_start[0]

    @staticmethod
    def when_predict_on_engine(
            times, start_stop_activation_time=None, gears=None,
            correct_start_stop_with_gears=False):
        to_predict = np.ones_like(times, dtype=bool)

        if start_stop_activation_time is not None:
            to_predict[times <= start_stop_activation_time] = False

        if correct_start_stop_with_gears:
            to_predict[gears > 0] = False

        return to_predict


def predict_engine_start_stop(
        start_stop_model, times, velocities, accelerations,
        engine_coolant_temperatures, state_of_charges, gears,
        correct_start_stop_with_gears, start_stop_activation_time,
        min_time_engine_on_after_start, has_start_stop, use_basic_start_stop):
    """
    Predicts if the engine is on and when the engine starts.

    :param start_stop_model:
        Start/stop model.
    :type start_stop_model: StartStopModel

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

    :param correct_start_stop_with_gears:
        A flag to impose engine on when there is a gear > 0.
    :type correct_start_stop_with_gears: bool

    :param start_stop_activation_time:
        Start-stop activation time threshold [s].
    :type start_stop_activation_time: float

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
        If the engine is on and when the engine starts [-, -].
    :rtype: numpy.array, numpy.array
    """

    on_engine, engine_starts = start_stop_model(
        times, velocities, accelerations, engine_coolant_temperatures,
        state_of_charges, gears=gears,
        correct_start_stop_with_gears=correct_start_stop_with_gears,
        start_stop_activation_time=start_stop_activation_time,
        min_time_engine_on_after_start=min_time_engine_on_after_start,
        has_start_stop=has_start_stop, use_basic_start_stop=use_basic_start_stop
    )

    return on_engine, engine_starts


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
        function=predict_engine_start_stop,
        inputs=['start_stop_model', 'times', 'velocities', 'accelerations',
                'engine_coolant_temperatures', 'state_of_charges',
                'gears', 'correct_start_stop_with_gears',
                'start_stop_activation_time', 'min_time_engine_on_after_start',
                'has_start_stop', 'use_basic_start_stop'],
        outputs=['on_engine', 'engine_starts']
    )

    return d
