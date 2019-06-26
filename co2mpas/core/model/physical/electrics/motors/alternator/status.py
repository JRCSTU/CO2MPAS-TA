# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the alternator status.
"""
import numpy as np
import schedula as sh
from ....defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Alternator status', description='Models the alternator status.'
)

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


dsp.add_data(
    'alternator_start_window_width', dfl.values.alternator_start_window_width
)


@sh.add_function(dsp, outputs=['alternator_starts_windows'])
def identify_alternator_starts_windows(
        times, engine_starts, alternator_currents,
        alternator_start_window_width, alternator_current_threshold):
    """
    Identifies the alternator starts windows [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param alternator_start_window_width:
        Alternator start window width [s].
    :type alternator_start_window_width: float

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :return:
        Alternator starts windows [-].
    :rtype: numpy.array
    """
    ts, dt = times[engine_starts], alternator_start_window_width / 2
    starts_windows = np.zeros_like(times, dtype=bool)
    ind = np.searchsorted(times, np.column_stack((ts - dt, ts + dt + dfl.EPS)))

    for i, j in ind:
        b = (alternator_currents[i:j] >= alternator_current_threshold).any()
        starts_windows[i:j] = b
    return starts_windows


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
        times, alternator_currents, clutch_tc_powers, on_engine,
        alternator_current_threshold, alternator_starts_windows,
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

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :param alternator_starts_windows:
        Alternator starts windows [-].
    :type alternator_starts_windows: numpy.array

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
    status[b & (clutch_tc_powers < 0)] = 2

    off = ~on_engine | alternator_starts_windows
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
        alternator_currents, clutch_tc_powers, on_engine, accelerations,
        service_battery_state_of_charges, alternator_statuses, times,
        alternator_current_threshold):
    """
    Identifies the alternator initialization time delta [s].

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

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
    alts, gb_p = alternator_statuses, clutch_tc_powers
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
        from ....engine._thermal import _build_samples

        x, y = _build_samples(
            alternator_currents, service_battery_state_of_charges, alts, gb_p,
            accelerations
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
        times, alternator_currents, clutch_tc_powers, on_engine,
        alternator_current_threshold, starts_windows,
        service_battery_state_of_charges, accelerations):
    """
    Identifies when the alternator statuses [-] and alternator initialization
    time delta [s].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :param starts_windows:
        Alternator starts windows [-].
    :type starts_windows: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

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
        times, alternator_currents, clutch_tc_powers, on_engine,
        alternator_current_threshold, starts_windows, 0)
    alternator_initialization_time = identify_alternator_initialization_time(
        alternator_currents, clutch_tc_powers, on_engine, accelerations,
        service_battery_state_of_charges, statuses, times,
        alternator_current_threshold
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
    from ....engine.co2_emission import _calibrate_model_params
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

    def _fit_bers(self, alternator_statuses, clutch_tc_powers):
        b = alternator_statuses == 2
        threshold = 0.0
        if b.any():
            from sklearn.tree import DecisionTreeClassifier
            q = dfl.functions.AlternatorStatusModel.min_percentile_bers
            m = DecisionTreeClassifier(random_state=0, max_depth=2)
            c = alternator_statuses != 1
            # noinspection PyUnresolvedReferences
            m.fit(clutch_tc_powers[c, None], b[c])

            X = clutch_tc_powers[b, None]
            if (np.sum(m.predict(X)) / X.shape[0] * 100) >= q:
                self.bers = m.predict  # shortcut name
                return self.bers

            # noinspection PyUnresolvedReferences
            if not b.all():
                gb_p_s = clutch_tc_powers[_mask_boolean_phases(b)[:, 0]]

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
            clutch_tc_powers):

        i = co2_utl.argmax(alternator_statuses != 3)

        status, soc = alternator_statuses[i:], state_of_charges[i:]

        self._fit_bers(status, clutch_tc_powers[i:])
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
        times, alternator_statuses, service_battery_state_of_charges,
        clutch_tc_powers, alternator_current_threshold):
    """
    Calibrates the alternator status model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_statuses:
        The alternator status (0: off, 1: on, due to state of charge, 2: on due
        to BERS, 3: on and initialize battery) [-].
    :type alternator_statuses: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

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
        times, alternator_statuses, service_battery_state_of_charges,
        clutch_tc_powers
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


@sh.add_function(dsp, outputs=['alternator_status_model'])
def define_alternator_status_model(
        service_battery_state_of_charge_balance,
        service_battery_state_of_charge_balance_window):
    """
    Defines the alternator status model.

    :param service_battery_state_of_charge_balance:
        Service battery state of charge balance [%].
    :type service_battery_state_of_charge_balance: float

    :param service_battery_state_of_charge_balance_window:
        Service battery state of charge balance [%].
    :type service_battery_state_of_charge_balance_window: float

    :return:
        A function that predicts the alternator status.
    :rtype: callable
    """

    # noinspection PyMissingOrEmptyDocstring,PyPep8Naming
    def bers_pred(X):
        return [X[0][0] < 0]

    m = service_battery_state_of_charge_balance
    w = service_battery_state_of_charge_balance_window / 2
    model = AlternatorStatusModel(
        charge_pred=lambda X: [X[0][0] == 1],
        bers_pred=bers_pred,
        min_soc=m - w,
        max_soc=m + w
    )

    return model


# noinspection PyPep8Naming
@sh.add_function(
    dsp,
    outputs=['service_battery_state_of_charge_balance',
             'service_battery_state_of_charge_balance_window']
)
def identify_service_battery_state_of_charge_balance_and_window(
        alternator_status_model):
    """
    Identify the service battery state of charge balance and window [%].

    :param alternator_status_model:
        A function that predicts the alternator status.
    :type alternator_status_model: AlternatorStatusModel

    :return:
        Service battery state of charge balance and window [%].
    :rtype: float, float
    """
    model = alternator_status_model
    min_soc, max_soc = model.min, model.max
    X = np.column_stack((np.ones(100), np.linspace(min_soc, max_soc, 100)))
    s = np.where(model.charge(X))[0]
    if s.shape[0]:
        min_soc, max_soc = max(min_soc, X[s[0], 1]), min(max_soc, X[s[-1], 1])

    state_of_charge_balance_window = max_soc - min_soc
    state_of_charge_balance = min_soc + state_of_charge_balance_window / 2
    return state_of_charge_balance, state_of_charge_balance_window
