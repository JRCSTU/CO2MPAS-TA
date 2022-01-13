# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to model the engine coolant temperature.
"""
import itertools
import numpy as np
import xgboost as xgb
from co2mpas.defaults import dfl
import co2mpas.utils as co2_utl
# noinspection SpellCheckingInspection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RANSACRegressor


def _build_samples(temperature_derivatives, engine_coolant_temperatures, *args):
    col = itertools.chain(
        (engine_coolant_temperatures[:-1],),
        (a[1:] for a in args),
    )
    return np.column_stack(tuple(col)), temperature_derivatives[1:]


# noinspection PyPep8Naming
def _filter_temperature_samples(X, Y, on_engine, thermostat):
    adt = np.abs(Y)
    b = ~((adt <= 0.001) & on_engine[1:])
    b[:co2_utl.argmax(on_engine)] = False
    i = co2_utl.argmax(thermostat < X[:, 0])
    b[i:] = True
    # noinspection PyProtectedMember
    b[:i] &= adt[:i] < dfl.functions._filter_temperature_samples.max_abs_dt_cold
    return X[b], Y[b]


# noinspection PyMissingOrEmptyDocstring
class _SelectFromModel(SelectFromModel):
    def __init__(self, *args, in_mask=(), out_mask=(), **kwargs):
        super(_SelectFromModel, self).__init__(*args, **kwargs)
        self._in_mask = in_mask
        self._out_mask = out_mask
        self._cache_support_mask = None

    def fit(self, *args, **kwargs):
        self._cache_support_mask = None
        return super(_SelectFromModel, self).fit(*args, **kwargs)

    def _get_support_mask(self):
        if self._cache_support_mask is not None:
            return self._cache_support_mask
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError(
                'Either fit the model before transform or set "prefit=True"'
                ' while passing the fitted estimator to the constructor.')
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                importances = getattr(estimator, "feature_importances_", None)
            if importances is not None and np.isnan(importances).all():
                mask = np.ones(importances.shape, bool)
            else:
                mask = super(_SelectFromModel, self)._get_support_mask()
        except ValueError:
            sfm = SelectFromModel(
                estimator.estimator_, self.threshold, True
            )
            mask = sfm._get_support_mask()

        for i in self._out_mask:
            mask[i] = False

        for i in self._in_mask:
            mask[i] = True
        self._cache_support_mask = mask
        return mask

    def transform(self, X):
        return super(_SelectFromModel, self).transform(X).copy()


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class NoDelta:
    # noinspection PyUnusedLocal
    @staticmethod
    def predict(X, *args):
        return np.zeros(X.shape[0])


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class _SafeRANSACRegressor(RANSACRegressor):
    def fit(self, X, y, **kwargs):
        try:
            return super(_SafeRANSACRegressor, self).fit(X, y, **kwargs)
        except ValueError as ex:
            if self.residual_threshold is None:
                aym = np.abs(y - np.median(y))
                rt = np.median(aym)
                if np.isclose(rt, 0.0):
                    b = ~np.isclose(aym, 0.0)
                    # noinspection PyUnresolvedReferences
                    if b.any():
                        rt = np.median(aym[b])
                self.residual_threshold = rt + np.finfo(np.float32).eps * 10
                res = super(_SafeRANSACRegressor, self).fit(X, y, **kwargs)
                self.residual_threshold = None
                return res
            else:
                raise ex


# noinspection PyMissingOrEmptyDocstring
class _XGBRegressor(xgb.XGBRegressor):
    def __setattr__(self, key, value):
        if key != 'random_state':
            super(_XGBRegressor, self).__setattr__(key, value)
        else:
            super(_XGBRegressor, self).__setattr__(key, 0)

    def cache_params(self):
        self._cache_params = {}

    def get_params(self, deep=True):
        params = getattr(self, '_cache_params', {})
        if deep not in params:
            params[deep] = super(_XGBRegressor, self).get_params(deep)
        return params[deep]


# noinspection PyMethodMayBeStatic,PyMethodMayBeStatic,PyMissingOrEmptyDocstring
class ThermalModel:
    def __init__(self, engine_thermostat_temperature=100.0):
        self.on = self.off = lambda *args: 0
        self.ntemp = 5
        self.thermostat = engine_thermostat_temperature

    # noinspection PyProtectedMember,PyPep8Naming
    def fit(self, engine_coolant_temperatures, engine_temperature_derivatives,
            on_engine, velocities, engine_speeds_out, accelerations,
            after_treatment_warm_up_phases):
        from sklearn.pipeline import Pipeline
        # noinspection PyArgumentEqualDefault
        opt = dict(
            base_estimator=_XGBRegressor(
                random_state=0, objective='reg:squarederror'
            ),
            random_state=0, min_samples=0.85, max_trials=10
        )
        t = np.append(
            [engine_coolant_temperatures[0]], engine_coolant_temperatures[:-1]
        )
        x = np.column_stack((
            velocities, t, np.zeros_like(t), engine_speeds_out, accelerations,
            after_treatment_warm_up_phases
        ))
        n = self.ntemp
        x[np.searchsorted(t, (self.thermostat,))[0]:, 2] = 1
        x[:, 1] = np.round((self.thermostat + n - x[:, 1]) / n) * n
        b = on_engine & (np.abs(engine_temperature_derivatives) > dfl.EPS)
        # noinspection PyArgumentEqualDefault
        model = Pipeline([
            ('selection', _SelectFromModel(
                opt['base_estimator'], threshold='0.8*median', in_mask=(0, 2)
            )),
            ('regression', _SafeRANSACRegressor(**opt))
        ]).fit(x[b, 1:], engine_temperature_derivatives[b])
        model.steps[0][1].estimator_.cache_params()
        model.steps[0][1].estimator.cache_params()
        model.steps[1][1].base_estimator.cache_params()
        model.steps[1][1].estimator_.cache_params()
        self.on = model.predict
        b = ~on_engine
        if b.any():
            model = _SafeRANSACRegressor(**opt).fit(
                x[b, :2].copy(), engine_temperature_derivatives[b]
            )
            model.estimator_.cache_params()
            model.base_estimator.cache_params()
            self.off = model.predict
        return self

    def __call__(self, times, on_engine, velocities, engine_speeds_out,
                 accelerations, after_treatment_warm_up_phases,
                 initial_temperature=23, max_temp=100.0):
        t, temp = initial_temperature, np.zeros_like(times, dtype=float)
        it = enumerate(zip(
            np.ediff1d(times, to_begin=0), on_engine, velocities, accelerations,
            after_treatment_warm_up_phases, engine_speeds_out,
        ))
        x, t0, hot = np.array([[.0] * 6]), self.thermostat + self.ntemp, False
        for i, (dt, b, v, a, atp, s) in it:
            hot |= t > self.thermostat
            x[:] = v, t0 - t, hot, s, a, atp
            t += (self.on(x[:, 1:]) if b else self.off(x[:, :2])) * dt
            temp[i] = t = min(t, max_temp)
        return temp
