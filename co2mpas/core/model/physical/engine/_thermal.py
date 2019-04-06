# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to model the engine coolant temperature.
"""
import itertools
import numpy as np
import schedula as sh
import xgboost as xgb
from ..defaults import dfl
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


# noinspection PyMethodMayBeStatic,PyMethodMayBeStatic,PyMissingOrEmptyDocstring
class ThermalModel:
    def __init__(self, thermostat=100.0):
        default_model = NoDelta()
        self.model = default_model
        self.mask = None
        self.cold = default_model
        self.mask_cold = None
        self.base_model = _XGBRegressor
        self.thermostat = thermostat
        self.min_temp = -float('inf')

    # noinspection PyProtectedMember,PyPep8Naming
    def fit(self, idle_engine_speed, on_engine, temperature_derivatives,
            temperatures, *args):
        """
        Calibrates an engine temperature regression model to predict engine
        temperatures.

        This model returns the delta temperature function of temperature
        (previous), acceleration, and power at the wheel.

        :param idle_engine_speed:
            Engine speed idle median and std [RPM].
        :type idle_engine_speed: (float, float)

        :param on_engine:
            If the engine is on [-].
        :type on_engine: numpy.array

        :param temperature_derivatives:
            Derivative temperature vector [°C].
        :type temperature_derivatives: numpy.array

        :param temperatures:
            Temperature vector [°C].
        :type temperatures: numpy.array

        :return:
            The calibrated engine temperature regression model.
        :rtype: ThermalModel
        """
        import sklearn.pipeline as sk_pip
        X, Y = _build_samples(temperature_derivatives, temperatures, *args)
        self.thermostat = self._identify_thermostat(X, Y, idle_engine_speed)

        X, Y = _filter_temperature_samples(X, Y, on_engine, self.thermostat)
        opt = {
            'random_state': 0,
            'max_depth': 2,
            'n_estimators': int(min(300.0, 0.25 * (len(X) - 1)))
        }

        model = _SafeRANSACRegressor(
            base_estimator=self.base_model(**opt),
            random_state=0,
            min_samples=0.85,
            max_trials=10
        )

        model = sk_pip.Pipeline([
            ('feature_selection', _SelectFromModel(model, '0.8*median',
                                                   in_mask=(0, 2))),
            ('classification', model)
        ])
        model.fit(X, Y)

        self.model = model.steps[-1][-1]
        self.mask = np.where(model.steps[0][-1]._get_support_mask())[0]

        self.min_temp = X[:, 0].min()
        i = co2_utl.argmax(self.thermostat <= X[:, 0])
        X, Y = X[:i], Y[:i]

        if not X.any():
            self.min_temp = -float('inf')
            return self
        i = co2_utl.argmax(np.percentile(X[:, 0], 60) <= X[:, 0])
        X, Y = X[:i], Y[:i]
        opt = {
            'random_state': 0,
            'max_depth': 2,
            'n_estimators': int(min(300.0, 0.25 * (len(X) - 1)))
        }
        model = self.base_model(**opt)
        model = sk_pip.Pipeline([
            ('feature_selection', _SelectFromModel(model, '0.8*median',
                                                   in_mask=(1,))),
            ('classification', model)
        ])
        model.fit(X[:, 1:], Y)
        self.cold = model.steps[-1][-1]
        self.mask_cold = np.where(model.steps[0][-1]._get_support_mask())[0] + 1

        return self

    # noinspection PyPep8Naming
    def _identify_thermostat(self, X, Y, idle_engine_speed):
        X, Y = np.column_stack((Y, X[:, 1:])), X[:, 0]
        t_max, t_min = Y.max(), Y.min()
        b = (t_max - (t_max - t_min) / 3) <= Y

        model = xgb.XGBRegressor()
        model.fit(X[b], Y[b])
        ratio = np.arange(1, 1.5, 0.1) * idle_engine_speed[0]
        spl = np.zeros((len(ratio), 4))
        spl[:, 2] = ratio
        # noinspection PyTypeChecker
        return float(np.median(model.predict(spl)))

    def __call__(self, deltas_t, *args, initial_temperature=23, max_temp=100.0):
        func, temp = self.temperature, np.zeros(len(deltas_t) + 1, dtype=float)
        t = temp[0] = initial_temperature

        for i, a in enumerate(zip(*((deltas_t,) + args)), start=1):
            temp[i] = t = func(*a, prev_temp=t, max_temp=max_temp)

        return temp

    def temperature(self, dt, *args, prev_temp=23, max_temp=100.0):
        if prev_temp < self.min_temp:
            model, mask = self.cold, self.mask_cold
        else:
            model, mask = self.model, self.mask

        delta_temp = self._derivative(model, mask, prev_temp, *args) * dt

        return min(prev_temp + delta_temp, max_temp)

    @staticmethod
    def _derivative(model, mask, *args):
        return model.predict(np.array([args])[:, mask])[0]


# noinspection PyMissingOrEmptyDocstring
class EngineTemperatureModel(co2_utl.BaseModel):
    key_outputs = ('engine_coolant_temperatures',)
    types = {float: {'engine_coolant_temperatures'}}

    def __init__(self, initial_engine_temperature=None,
                 engine_temperature_regression_model=None,
                 max_engine_coolant_temperature=None, outputs=None):
        self.initial_engine_temperature = initial_engine_temperature
        self.engine_temperature_regression_model = \
            engine_temperature_regression_model
        self.max_engine_coolant_temperature = max_engine_coolant_temperature
        super(EngineTemperatureModel, self).__init__(outputs)

    def init_results(self, times, accelerations, final_drive_powers_in,
                     engine_speeds_out_hot):
        key = 'engine_coolant_temperatures'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]
        else:
            temp, max_t = self.outputs[key], self.max_engine_coolant_temperature
            temp[0] = self.initial_engine_temperature
            temperature = self.engine_temperature_regression_model.temperature
            acc, powers, speeds = (
                accelerations, final_drive_powers_in, engine_speeds_out_hot
            )

            def _next(i):
                j = i + 1
                dt = len(times) > j and times[j] - times[i] or 0
                temp[j] = t = temperature(
                    dt, powers[i], speeds[i], acc[i], prev_temp=temp[i],
                    max_temp=max_t
                )
                return t

            return _next
