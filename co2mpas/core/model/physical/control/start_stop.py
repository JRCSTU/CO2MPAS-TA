# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the engine start stop strategy.
"""
import numpy as np
import schedula as sh
from ..defaults import dfl
from co2mpas.utils import BaseModel

dsp = sh.BlueDispatcher(
    name='start_stop', description='Models the engine start/stop strategy.'
)


@sh.add_function(dsp, outputs=['start_stop_activation_time'])
def default_start_stop_activation_time(has_start_stop):
    """
    Returns the default start stop activation time threshold [s].

    :return:
        Start-stop activation time threshold [s].
    :rtype: float
    """
    if not has_start_stop or dfl.functions.ENABLE_ALL_FUNCTIONS or \
            dfl.functions.default_start_stop_activation_time.ENABLE:
        return dfl.functions.default_start_stop_activation_time.threshold
    return sh.NONE


# noinspection PyShadowingBuiltins,PyUnusedLocal,PyMissingOrEmptyDocstring
class StartStopModel:
    VEL = dfl.functions.StartStopModel.stop_velocity
    ACC = dfl.functions.StartStopModel.plateau_acceleration

    def __init__(self):
        self.simple = self.complex = self.base

    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def base(self, velocity, acceleration, *args):
        return (velocity > self.VEL) | (acceleration > self.ACC)

    def fit_simple(self, on_engine, velocities, accelerations, *args):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=0, max_depth=3)
        model.fit(np.column_stack((velocities, accelerations)), on_engine)
        predict = model.predict

        def simple(velocity, acceleration, *a):
            return predict([[velocity, acceleration]])[0]

        self.simple = simple

    def fit_complex(self, on_engine, velocities, accelerations,
                    engine_coolant_temperatures, state_of_charges):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectFromModel

        model = DecisionTreeClassifier(random_state=0, max_depth=4)
        model = Pipeline([
            ('feature_selection', SelectFromModel(model)),
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


@sh.add_function(dsp, outputs=['start_stop_model'])
def calibrate_start_stop_model(
        on_engine, velocities, accelerations, engine_coolant_temperatures,
        state_of_charges, times, start_stop_activation_time):
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

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param start_stop_activation_time:
        Start-stop activation time threshold [s].
    :type start_stop_activation_time: float

    :return:
        Start/stop model.
    :rtype: callable
    """
    i = np.searchsorted(times, start_stop_activation_time)
    soc = np.zeros_like(state_of_charges)
    soc[0], soc[1:] = state_of_charges[0], state_of_charges[:-1]
    model = StartStopModel()
    model.fit(
        on_engine[i:], velocities[i:], accelerations[i:],
        engine_coolant_temperatures[i:], soc[i:]
    )

    return model


@sh.add_function(dsp, outputs=['correct_start_stop_with_gears'])
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


dsp.add_data('has_start_stop', dfl.values.has_start_stop)
dsp.add_data('is_hybrid', dfl.values.is_hybrid)


@sh.add_function(dsp, outputs=['use_basic_start_stop'])
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


# noinspection PyMissingOrEmptyDocstring
class EngineStartStopModel(BaseModel):
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
        super(EngineStartStopModel, self).__init__(outputs)

    def init_results(self, times, velocities, accelerations,
                     engine_coolant_temperatures, state_of_charges,
                     gears=None):
        keys = ['on_engine', 'engine_starts']
        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            one, st = self._outputs['on_engine'], self._outputs['engine_starts']
            _next = lambda i: (one[i], st[i])
        elif not self.has_start_stop:
            st, one = self.outputs['engine_starts'], self.outputs['on_engine']

            def _next(i):
                st[i] = start = not (i == 0 or one[i - 1])
                one[i] = True
                return True, start
        else:
            st, one = self.outputs['engine_starts'], self.outputs['on_engine']
            base = self.start_stop_model.base
            if self.use_basic_start_stop:
                predict = self.start_stop_model.simple
            else:
                predict = self.start_stop_model.complex

            def _next(i):
                t, v = times[i], (
                    velocities[i], accelerations[i],
                    engine_coolant_temperatures[i], state_of_charges[i]
                )

                on = t <= self.start_stop_activation_time
                on = on or self.correct_start_stop_with_gears and gears[i] > 0
                prev = i == 0 or one[i - 1]
                on = on or ((prev or base(*v)) and predict(*v))
                if not on:
                    t0 = t - self.min_time_engine_on_after_start
                    for ti, s in zip(times[i::-1], st[i::-1]):
                        if ti < t0:
                            break
                        elif s:
                            on = True
                            break
                one[i] = on
                st[i] = start = on and prev != on
                return on, start
        return _next


@sh.add_function(dsp, outputs=['start_stop_prediction_model'], weight=4000)
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


@sh.add_function(dsp, outputs=['start_stop_prediction_model'])
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


@sh.add_function(dsp, outputs=['on_engine', 'engine_starts'])
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
