# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the alternator.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics.alternator

.. autosummary::
    :nosignatures:
    :toctree: alternator/

    electrics_prediction
"""

import numpy as np
import schedula as sh
from ...defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(name='Alternator', description='Models the alternator.')


@sh.add_function(dsp, outputs=['max_alternator_current'])
def calculate_max_alternator_current(
        alternator_nominal_voltage, alternator_nominal_power,
        alternator_efficiency):
    """
    Calculates the max feasible alternator current [A].

    :param alternator_nominal_voltage:
        Alternator nominal voltage [V].
    :type alternator_nominal_voltage: float

    :param alternator_nominal_power:
        Alternator nominal power [kW].
    :type alternator_nominal_power: float

    :param alternator_efficiency:
        Alternator efficiency [-].
    :type alternator_efficiency: float

    :return:
        Max feasible alternator current [A].
    :rtype: float
    """

    c = alternator_nominal_power * 1000.0 * alternator_efficiency

    return c / alternator_nominal_voltage


dsp.add_data('stop_velocity', dfl.values.stop_velocity)
dsp.add_data('alternator_off_threshold', dfl.values.alternator_off_threshold)


# noinspection PyPep8
@sh.add_function(dsp, outputs=['alternator_current_threshold'])
def identify_alternator_current_threshold(
        alternator_currents, velocities, on_engine, stop_velocity,
        alternator_off_threshold):
    """
    Identifies the alternator current threshold [A] that identifies when the
    alternator is off.

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param alternator_off_threshold:
        Maximum negative current for being considered the alternator off [A].
    :type alternator_off_threshold: float

    :return:
        Alternator current threshold [A].
    :rtype: float
    """
    from sklearn.cluster import DBSCAN
    sample_weight = np.ones_like(alternator_currents, dtype=float)
    sample_weight[alternator_currents >= alternator_off_threshold] = 2.0
    sample_weight[velocities < stop_velocity] = 3.0
    sample_weight[~on_engine] = 4.0
    model = DBSCAN(eps=-alternator_off_threshold)
    model.fit(alternator_currents[:, None], sample_weight=sample_weight)
    c, lb = model.components_, model.labels_[model.core_sample_indices_]
    sample_weight = sample_weight[model.core_sample_indices_]
    threshold, w = [], []
    for i in range(lb.max() + 1):
        b = lb == i
        x = c[b].min()
        if x > alternator_off_threshold:
            threshold.append(x)
            w.append(np.sum(sample_weight[b]))

    if threshold:
        return min(0.0, np.average(threshold, weights=w))
    return 0.0


dsp.add_data('alternator_efficiency', dfl.values.alternator_efficiency)


def _set_alt_init_status(times, initialization_time, statuses):
    if initialization_time > 0:
        statuses[:co2_utl.argmax(times > (times[0] + initialization_time))] = 3
    return statuses


def _mask_boolean_phases(b):
    s = np.zeros(len(b) + 2, dtype=bool)
    s[1:-1] = b
    mask = np.column_stack((s[1:], s[:-1])) & (s[:-1] != s[1:])[:, None]
    return np.where(mask)[0].reshape((-1, 2))


@sh.add_function(dsp, outputs=['alternator_statuses'])
def identify_charging_statuses(
        times, alternator_currents, gear_box_powers_in, on_engine,
        alternator_current_threshold, starts_windows,
        alternator_initialization_time):
    """
    Identifies when the alternator is on due to 1:state of charge or 2:BERS or
    3: initialization [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :param starts_windows:
        Alternator starts windows [-].
    :type starts_windows: numpy.array

    :param alternator_initialization_time:
        Alternator initialization time delta [s].
    :type alternator_initialization_time: float

    :return:
        The alternator status (0: off, 1: on, due to state of charge, 2: on due
        to BERS, 3: on and initialize battery) [-].
    :rtype: numpy.array
    """

    b = (alternator_currents < alternator_current_threshold) & on_engine

    status = b.astype(int, copy=True)
    status[b & (gear_box_powers_in < 0)] = 2

    off = ~on_engine | starts_windows
    mask = _mask_boolean_phases(status != 1)
    for i, j in mask:
        # noinspection PyUnresolvedReferences
        if ((status[i:j] == 2) | off[i:j]).all():
            status[i:j] = 1

    _set_alt_init_status(times, alternator_initialization_time, status)

    return status


# noinspection PyPep8
@sh.add_function(dsp, outputs=['alternator_initialization_time'])
def identify_alternator_initialization_time(
        alternator_currents, gear_box_powers_in, on_engine, accelerations,
        state_of_charges, alternator_statuses, times,
        alternator_current_threshold):
    """
    Identifies the alternator initialization time delta [s].

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param state_of_charges:
        State of charge of the battery [%].

        .. note::

            `state_of_charges` = 99 is equivalent to 99%.
    :type state_of_charges: numpy.array

    :param alternator_statuses:
        The alternator status (0: off, 1: on, due to state of charge, 2: on due
        to BERS, 3: on and initialize battery) [-].
    :type alternator_statuses: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :return:
        Alternator initialization time delta [s].
    :rtype: float
    """
    alts, gb_p = alternator_statuses, gear_box_powers_in
    i = co2_utl.argmax(alts != 0)
    if alts[0] == 1 or (i and ((alts[:i] == 0) & (gb_p[:i] == 0)).all()):
        s = alternator_currents < alternator_current_threshold
        n, i = len(on_engine), int(co2_utl.argmax((s[:-1] != s[1:]) & s[:-1]))
        i = min(n - 1, i)
        opt = {
            'random_state': 0, 'max_depth': 2
        }

        import xgboost as xgb
        # noinspection PyProtectedMember
        from ...engine._thermal import _build_samples

        x, y = _build_samples(
            alternator_currents, state_of_charges, alts, gb_p, accelerations
        )

        j = min(i, int(n / 2))
        opt['n_estimators'] = int(min(100.0, 0.25 * (n - j))) or 1
        model = xgb.XGBRegressor(**opt)
        model.fit(x[j:], y[j:])
        err = np.abs(y - model.predict(x))
        sets = np.array(co2_utl.get_inliers(err)[0], dtype=int)[:i]
        if sum(sets) / i < 0.5 or i > j:
            from sklearn.tree import DecisionTreeClassifier
            reg = DecisionTreeClassifier(max_depth=1, random_state=0)
            reg.fit(times[1:i + 1, None], sets)
            s, r = reg.tree_.children_left[0], reg.tree_.children_right[0]
            s, r = np.argmax(reg.tree_.value[s]), np.argmax(reg.tree_.value[r])
            if s == 0 and r == 1:
                return reg.tree_.threshold[0] - times[0]
            elif r == 0 and not i > j:
                return times[i] - times[0]

    elif alts[0] == 3:
        i = co2_utl.argmax(alts != 3)
        return times[i] - times[0]
    return 0.0


@sh.add_function(
    dsp, outputs=['alternator_statuses', 'alternator_initialization_time'],
    weight=1
)
def identify_alternator_statuses_and_alternator_initialization_time(
        times, alternator_currents, gear_box_powers_in, on_engine,
        alternator_current_threshold, starts_windows, state_of_charges,
        accelerations):
    """
    Identifies when the alternator statuses [-] and alternator initialization
    time delta [s].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :param starts_windows:
        Alternator starts windows [-].
    :type starts_windows: numpy.array

    :param state_of_charges:
        State of charge of the battery [%].

        .. note::

            `state_of_charges` = 99 is equivalent to 99%.
    :type state_of_charges: numpy.array

    :param accelerations:
        Acceleration [m/s2].
    :type accelerations: numpy.array

    :return:
        The alternator status (0: off, 1: on, due to state of charge, 2: on due
        to BERS, 3: on and initialize battery) [-] and alternator initialization
        time delta [s].
    :rtype: numpy.array, float
    """
    statuses = identify_charging_statuses(
        times, alternator_currents, gear_box_powers_in, on_engine,
        alternator_current_threshold, starts_windows, 0)
    alternator_initialization_time = identify_alternator_initialization_time(
        alternator_currents, gear_box_powers_in, on_engine, accelerations,
        state_of_charges, statuses, times, alternator_current_threshold
    )
    _set_alt_init_status(times, alternator_initialization_time, statuses)
    return statuses, alternator_initialization_time


def _identify_balance_soc(times, state_of_charges):
    import lmfit
    parameters = lmfit.Parameters()
    parameters.add('B', value=np.median(state_of_charges), min=0.0, max=100.0)
    parameters.add('A', value=0)
    parameters.add('X0', value=1.0, min=0.0)
    x = (times - times[0]) / (times[-1] - times[0])
    n = len(x)

    # noinspection PyMissingOrEmptyDocstring
    def func(params):
        p = params.valuesdict()
        y = np.tile(p['B'], n)
        b = x < p['X0']
        y[b] += p['A'] * (x[b] - p['X0']) ** 2
        return y

    # noinspection PyMissingOrEmptyDocstring
    def error(params):
        return co2_utl.mae(state_of_charges, func(params))

    # noinspection PyProtectedMember
    from ...engine.co2_emission import _calibrate_model_params
    return _calibrate_model_params(error, parameters)[0].valuesdict()['B']


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class AlternatorStatusModel:
    def __init__(self, bers_pred=None, charge_pred=None, min_soc=0.0,
                 max_soc=100.0, current_threshold=0.0):
        self.bers = bers_pred
        self.charge = charge_pred
        self.max = max_soc
        self.min = min_soc
        self.current_threshold = current_threshold

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _fit_bers(self, alternator_statuses, gear_box_powers_in):
        b = alternator_statuses == 2
        threshold = 0.0
        if b.any():
            from sklearn.tree import DecisionTreeClassifier
            q = dfl.functions.AlternatorStatusModel.min_percentile_bers
            m = DecisionTreeClassifier(random_state=0, max_depth=2)
            c = alternator_statuses != 1
            # noinspection PyUnresolvedReferences
            m.fit(gear_box_powers_in[c, None], b[c])

            X = gear_box_powers_in[b, None]
            if (np.sum(m.predict(X)) / X.shape[0] * 100) >= q:
                self.bers = m.predict  # shortcut name
                return self.bers

            # noinspection PyUnresolvedReferences
            if not b.all():
                gb_p_s = gear_box_powers_in[_mask_boolean_phases(b)[:, 0]]

                threshold = min(threshold, np.percentile(gb_p_s, q))

        self.bers = lambda x: np.asarray(x) < threshold
        return self.bers

    # noinspection PyShadowingNames
    def _fit_charge(self, alternator_statuses, state_of_charges):
        b = alternator_statuses[1:] == 1
        if b.any():
            from sklearn.tree import DecisionTreeClassifier
            charge = DecisionTreeClassifier(random_state=0, max_depth=3)
            X = np.column_stack(
                (alternator_statuses[:-1], state_of_charges[1:])
            )
            charge.fit(X, b)
            self.charge = charge.predict
        else:
            self.charge = lambda X: np.zeros(len(X), dtype=bool)

    def _fit_boundaries(self, alternator_statuses, state_of_charges, times):
        n, b = len(alternator_statuses), alternator_statuses == 1
        mask = _mask_boolean_phases(b)
        self.max, self.min = 100.0, 0.0
        _max, _min, balance = [], [], ()
        from scipy.stats import linregress
        min_dt = dfl.functions.AlternatorStatusModel.min_delta_time_boundaries
        for i, j in mask:
            t = times[i:j]
            if t[-1] - t[0] <= min_dt:
                continue
            soc = state_of_charges[i:j]
            m, q = linregress(t, soc)[:2]
            if m >= 0:
                if i > 0:
                    _min.append(soc.min())
                if j < n:
                    _max.append(soc.max())

        min_delta_soc = dfl.functions.AlternatorStatusModel.min_delta_soc
        if _min:
            self.min = _min = max(self.min, min(100.0, min(_min)))

            _max = [m for m in _max if m >= _min]
            if _max:
                self.max = min(100.0, min(max(_max), _min + min_delta_soc))
            else:
                self.max = min(100.0, _min + min_delta_soc)
        elif _max:
            self.max = _max = min(self.max, max(0.0, max(_max)))
            self.min = _max - min_delta_soc
        elif b.any():
            balance = _identify_balance_soc(times, state_of_charges)
            # noinspection PyTypeChecker
            std = np.sqrt(np.mean((balance - state_of_charges) ** 2)) * 2
            std = min(min_delta_soc, std)
            self.max = min(balance + std, 100.0)
            self.min = max(balance - std, 0.0)

    def fit(self, times, alternator_statuses, state_of_charges,
            gear_box_powers_in):

        i = co2_utl.argmax(alternator_statuses != 3)

        status, soc = alternator_statuses[i:], state_of_charges[i:]

        self._fit_bers(status, gear_box_powers_in[i:])
        self._fit_charge(status, soc)
        self._fit_boundaries(status, soc, times[i:])

        return self

    def predict(self, has_energy_rec, init_time, time, prev, soc, power):
        status = 0

        if soc < 100:
            x = [(prev, soc)]
            if time < init_time:
                status = 3

            elif soc < self.min or (soc <= self.max and self.charge(x)[0]):
                status = 1

            elif has_energy_rec and self.bers([(power,)])[0]:
                status = 2

        return status


@sh.add_function(dsp, outputs=['alternator_status_model'], weight=10)
def calibrate_alternator_status_model(
        times, alternator_statuses, general_state_of_charges, gear_box_powers_in,
        alternator_current_threshold):
    """
    Calibrates the alternator status model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_statuses:
        The alternator status (0: off, 1: on, due to state of charge, 2: on due
        to BERS, 3: on and initialize battery) [-].
    :type alternator_statuses: numpy.array

    :param general_state_of_charges:
        State of charge of the low voltage battery [%].

        .. note::

            `general_state_of_charges` = 99 is equivalent to 99%.
    :type general_state_of_charges: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :return:
        A function that predicts the alternator status.
    :rtype: callable
    """

    model = AlternatorStatusModel(
        current_threshold=alternator_current_threshold
    )
    model.fit(
        times, alternator_statuses, general_state_of_charges, gear_box_powers_in
    )

    return model


@sh.add_function(dsp, outputs=['alternator_current_threshold'])
def get_alternator_current_threshold(alternator_status_model):
    """
    Gets the alternator current threshold [A] that identifies when the
    alternator is off from the alternator status model.

    :param alternator_status_model:
        A function that predicts the alternator status.
    :type alternator_status_model: AlternatorStatusModel

    :return:
        Alternator current threshold [A].
    :rtype: float
    """
    return alternator_status_model.current_threshold


@sh.add_function(dsp, outputs=['alternator_current_model'])
def define_alternator_current_model(alternator_charging_currents):
    """
    Defines an alternator current model that predicts alternator current [A].

    :param alternator_charging_currents:
        Mean charging currents of the alternator (for negative and positive
        power)[A].
    :type alternator_charging_currents: (float, float)

    :return:
        Alternator current model.
    :rtype: callable
    """

    return AlternatorCurrentModel(alternator_charging_currents)


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class AlternatorCurrentModel:
    def __init__(self, alternator_charging_currents=(0, 0)):
        def default_model(X):
            time, prev_soc, alt_status, gb_power, acc = X.T
            b = gb_power > 0 or (gb_power == 0 and acc >= 0)

            return np.where(b, *alternator_charging_currents)

        import xgboost as xgb
        self.model = default_model
        self.mask = None
        self.init_model = default_model
        self.init_mask = None
        self.base_model = xgb.XGBRegressor

    def predict(self, X, init_time=0.0):
        X = np.asarray(X)
        times = X[:, 0]
        b = times < (times[0] + init_time)
        curr = np.zeros_like(times, dtype=float)
        curr[b] = self.init_model(X[b][:, self.init_mask])
        b = ~b
        curr[b] = self.model(X[b][:, self.model])
        return curr

    # noinspection PyShadowingNames,PyProtectedMember
    def fit(self, currents, on_engine, times, soc, statuses, *args,
            init_time=0.0):
        b = (statuses[1:] > 0) & on_engine[1:]
        i = co2_utl.argmax(times > times[0] + init_time)
        from ...engine._thermal import _build_samples
        X, Y = _build_samples(currents, soc, statuses, *args)
        if b[i:].any():
            self.model, self.mask = self._fit_model(X[i:][b[i:]], Y[i:][b[i:]])
        elif b[:i].any():
            self.model, self.mask = self._fit_model(X[b], Y[b])
        else:
            self.model = lambda *args, **kwargs: [0.0]
            self.mask = np.array((0,))
        self.mask += 1

        if b[:i].any():
            init_spl = (times[1:i + 1] - times[0])[:, None], X[:i]
            init_spl = np.concatenate(init_spl, axis=1)[b[:i]]
            a = self._fit_model(init_spl, Y[:i][b[:i]], (0,), (2,))
            self.init_model, self.init_mask = a
        else:
            self.init_model, self.init_mask = self.model, self.mask

        return self

    # noinspection PyProtectedMember
    def _fit_model(self, X, Y, in_mask=(), out_mask=()):
        opt = {
            'random_state': 0,
            'max_depth': 2,
            'n_estimators': int(min(300.0, 0.25 * (len(X) - 1))) or 1
        }
        from sklearn.pipeline import Pipeline
        from ...engine._thermal import _SelectFromModel
        model = self.base_model(**opt)
        model = Pipeline([
            ('feature_selection', _SelectFromModel(model, '0.8*median',
                                                   in_mask=in_mask,
                                                   out_mask=out_mask)),
            ('classification', model)
        ])
        model.fit(X, Y)
        mask = np.where(model.steps[0][-1]._get_support_mask())[0]
        return model.steps[-1][-1].predict, mask

    def __call__(self, time, soc, status, *args):
        arr = np.array([(time, soc, status) + args])
        if status == 3:
            return min(0.0, self.init_model(arr[:, self.init_mask])[0])
        return min(0.0, self.model(arr[:, self.mask])[0])


@sh.add_function(dsp, outputs=['alternator_current_model'])
def calibrate_alternator_current_model(
        alternator_currents, on_engine, times, general_state_of_charges,
        alternator_statuses, gear_box_powers_in, accelerations,
        alternator_initialization_time):
    """
    Calibrates an alternator current model that predicts alternator current [A].

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param general_state_of_charges:
        State of charge of the low voltage battery [%].

        .. note::

            `general_state_of_charges` = 99 is equivalent to 99%.
    :type general_state_of_charges: numpy.array

    :param alternator_statuses:
        The alternator status (0: off, 1: on, due to state of charge, 2: on due
        to BERS, 3: on and initialize battery) [-].
    :type alternator_statuses: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param alternator_initialization_time:
        Alternator initialization time delta [s].
    :type alternator_initialization_time: float

    :return:
        Alternator current model.
    :rtype: callable
    """
    model = AlternatorCurrentModel()
    model.fit(
        alternator_currents, on_engine, times, general_state_of_charges,
        alternator_statuses, gear_box_powers_in, accelerations,
        init_time=alternator_initialization_time
    )

    return model
