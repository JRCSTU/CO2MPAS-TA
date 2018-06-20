# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions that model the engine cold start.
"""

import functools
import sklearn.metrics as sk_met
import sklearn.tree as sk_tree
import co2mpas.utils as co2_utl
import numpy as np
import schedula as sh
import lmfit


def identify_cold_start_speeds_phases(
        engine_coolant_temperatures, engine_thermostat_temperature, on_idle):
    """
    Identifies phases when engine speed is affected by the cold start [-].

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param engine_thermostat_temperature:
        Engine thermostat temperature [°C].
    :type engine_thermostat_temperature: float

    :param on_idle:
        If the engine is on idle [-].
    :type on_idle: numpy.array

    :return:
        Phases when engine speed is affected by the cold start [-].
    :rtype: numpy.array
    """
    temp = engine_coolant_temperatures
    i = co2_utl.argmax(temp > engine_thermostat_temperature)
    p = on_idle.copy()
    p[i:] = False
    return p


def identify_cold_start_speeds_delta(
        cold_start_speeds_phases, engine_speeds_out, engine_speeds_out_hot):
    """
    Identifies the engine speed delta due to the engine cold start [RPM].

    :param cold_start_speeds_phases:
        Phases when engine speed is affected by the cold start [-].
    :type cold_start_speeds_phases: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :return:
        Engine speed delta due to the engine cold start [RPM].
    :rtype: numpy.array
    """
    speeds = np.zeros_like(engine_speeds_out, dtype=float)
    b = cold_start_speeds_phases
    speeds[b] = np.maximum(0, engine_speeds_out[b] - engine_speeds_out_hot[b])
    return speeds


class ColdStartModel:
    def __init__(self, ds=0, m=np.inf, temp_limit=30):
        self.ds = ds
        self.m = m
        self.temp_limit = temp_limit

    def __repr__(self):
        s = '{}(ds={}, m={}, temp_limit={})'.format(
            self.__class__.__name__, self.ds, self.m, self.temp_limit
        )
        return s.replace('inf', "float('inf')")

    def set(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @staticmethod
    def initial_guess_temp_limit(
            cold_start_speeds_delta, engine_coolant_temperatures):
        reg = sk_tree.DecisionTreeRegressor(random_state=0, max_leaf_nodes=10)
        reg.fit(engine_coolant_temperatures[:, None], cold_start_speeds_delta)
        t = np.unique(engine_coolant_temperatures)
        i = np.searchsorted(t, np.unique(reg.tree_.threshold))
        n = len(t) - 1
        if i[-1] != n:
            i = np.append(i, (n,))
        return t[i]

    def correct_ds(self, min_t):
        if not np.isinf(self.m):
            self.ds = max(min(self.ds, (self.temp_limit - min_t) * self.m), 0)
        return self

    def fit(self, cold_start_speeds_delta, engine_coolant_temperatures,
        engine_speeds_out_hot, on_engine, idle_engine_speed,
        cold_start_speeds_phases):
        if not cold_start_speeds_phases.any():
            return self
        from .co2_emission import calibrate_model_params, _set_attr

        temp = engine_coolant_temperatures
        w = temp.max() + 1 - temp

        delta = cold_start_speeds_delta[cold_start_speeds_phases]
        temp = engine_coolant_temperatures[cold_start_speeds_phases]
        ds = delta / idle_engine_speed[0]

        def _err(x=None):
            if x is not None:
                self.set(**x.valuesdict())

            s = self(
                idle_engine_speed, on_engine, engine_coolant_temperatures,
                engine_speeds_out_hot
            )
            return float(np.mean(np.abs(s - cold_start_speeds_delta) * w))

        t_min, t_max = temp.min(), temp.max()
        d = dict(min=t_min, max=t_max) if t_min < t_max else dict(vary=False)
        p = lmfit.Parameters()
        p.add('temp_limit', 0, **d)
        p.add('ds', 0, min=0)
        p.add('m', 0, min=0)

        res = [(
            round(_err(), 1), 0,
            dict(ds=self.ds, m=self.m, temp_limit=self.temp_limit)
        )]

        for i, t in enumerate(self.initial_guess_temp_limit(delta, temp), 1):
            _set_attr(p, {'temp_limit': t}, attr='value')
            ds_max = ds[temp <= t].max()
            if not np.isclose(ds_max, 0.0):
                _set_attr(p, {'ds': ds_max}, attr='max')
                x = dict(calibrate_model_params(_err, p)[0].valuesdict())
                x['ds'] = self.set(**x).correct_ds(t_min).ds
                res.append((round(_err(), 1), i, x))
        self.set(**min(res)[-1])
        return self

    def __call__(self, idle_engine_speed, on_engine,
                 engine_coolant_temperatures, engine_speeds_out_hot):
        add_speeds = np.zeros_like(on_engine, dtype=float)
        if self.ds > 0:
            b = on_engine & (engine_coolant_temperatures <= self.temp_limit)
            if b.any():
                ds, m = self.ds, self.m
                if not np.isinf(m):
                    ds = np.minimum(
                        ds,
                        (self.temp_limit - engine_coolant_temperatures[b]) * m
                    )

                add_speeds[b] = np.maximum(
                    (ds + 1) * idle_engine_speed[0] - engine_speeds_out_hot[b],
                    0
                )
        return add_speeds


def calibrate_cold_start_speed_model(
        cold_start_speeds_phases, cold_start_speeds_delta, idle_engine_speed,
        on_engine, engine_coolant_temperatures, engine_speeds_out_hot):
    """
    Calibrates the engine cold start speed model.

    :param cold_start_speeds_phases:
        Phases when engine speed is affected by the cold start [-].
    :type cold_start_speeds_phases: numpy.array

    :param cold_start_speeds_delta:
        Engine speed delta due to the cold start [RPM].
    :type cold_start_speeds_delta: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :return:
        Cold start speed model.
    :rtype: ColdStartModel
    """

    model = ColdStartModel().fit(
        cold_start_speeds_delta, engine_coolant_temperatures,
        engine_speeds_out_hot, on_engine, idle_engine_speed,
        cold_start_speeds_phases
    )

    return model


def calculate_cold_start_speeds_delta(
        cold_start_speed_model, on_engine, engine_coolant_temperatures,
        engine_speeds_out_hot, idle_engine_speed):
    """
    Calculates the engine speed delta and phases due to the cold start [RPM, -].

    :param cold_start_speed_model:
        Cold start speed model.
    :type cold_start_speed_model: ColdStartModel

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Engine speed delta due to the cold start and its phases [RPM, -].
    :rtype: numpy.array, numpy.array
    """

    delta = cold_start_speed_model(
        idle_engine_speed, on_engine, engine_coolant_temperatures,
        engine_speeds_out_hot
    )

    return delta


def cold_start():
    """
    Defines the engine cold start model.

    .. dispatcher:: d

        >>> d = cold_start()

    :return:
        The engine start/stop model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='cold_start',
        description='Models the engine cold start strategy.'
    )

    d.add_function(
        function=identify_cold_start_speeds_phases,
        inputs=['engine_coolant_temperatures', 'engine_thermostat_temperature',
                'on_idle'],
        outputs=['cold_start_speeds_phases']
    )

    d.add_function(
        function=identify_cold_start_speeds_delta,
        inputs=['cold_start_speeds_phases', 'engine_speeds_out',
                'engine_speeds_out_hot'],
        outputs=['cold_start_speeds_delta']
    )

    d.add_function(
        function=calibrate_cold_start_speed_model,
        inputs=['cold_start_speeds_phases', 'cold_start_speeds_delta',
                'idle_engine_speed', 'on_engine', 'engine_coolant_temperatures',
                'engine_speeds_out_hot'],
        outputs=['cold_start_speed_model']
    )

    d.add_function(
        function=calculate_cold_start_speeds_delta,
        inputs=['cold_start_speed_model', 'on_engine',
                'engine_coolant_temperatures', 'engine_speeds_out_hot',
                'idle_engine_speed'],
        outputs=['cold_start_speeds_delta']
    )

    return d
