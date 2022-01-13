# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the alternator status.
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Service Battery Status model',
    description='Models the Service Battery Charging Status.'
)


# noinspection PyPep8
@sh.add_function(
    dsp, outputs=['service_battery_electric_powers_supply_threshold']
)
def define_service_battery_electric_powers_supply_threshold(
        service_battery_capacity, service_battery_nominal_voltage):
    """
    Identifies the service battery electric powers supply threshold [kW] that
    define when the the service battery is not charging.

    :param service_battery_capacity:
        Service battery capacity [Ah].
    :type service_battery_capacity: float

    :param service_battery_nominal_voltage:
        Service battery nominal voltage [V].
    :type service_battery_nominal_voltage: float

    :return:
        Service battery not charging power threshold [kW].
    :rtype: float
    """
    d = dfl.functions.define_service_battery_electric_powers_supply_threshold
    c = max(-service_battery_capacity * 36 * d.min_soc, d.min_current)
    return c * service_battery_nominal_voltage / 1e3


dsp.add_data(
    'service_battery_start_window_width',
    dfl.values.service_battery_start_window_width
)


@sh.add_function(dsp, outputs=['service_battery_starts_windows'])
def identify_service_battery_starts_windows(
        times, engine_starts, alternator_electric_powers,
        service_battery_start_window_width,
        service_battery_electric_powers_supply_threshold):
    """
    Identifies the alternator starts windows [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array

    :param service_battery_start_window_width:
        Service battery start window width [s].
    :type service_battery_start_window_width: float

    :param service_battery_electric_powers_supply_threshold:
        Service battery not charging power threshold [kW].
    :type service_battery_electric_powers_supply_threshold: float

    :return:
        Service battery starts windows [-].
    :rtype: numpy.array
    """
    ts, dt = times[engine_starts], service_battery_start_window_width / 2
    starts_windows = np.zeros_like(times, dtype=bool)
    ind = np.searchsorted(times, np.column_stack((ts - dt, ts + dt + dfl.EPS)))
    b = service_battery_electric_powers_supply_threshold
    b = alternator_electric_powers >= b
    for i, j in ind:
        starts_windows[i:j] = b[i:j].any()
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


@sh.add_function(dsp, outputs=['service_battery_charging_statuses'])
def identify_service_battery_charging_statuses(
        times, alternator_electric_powers, dcdc_converter_electric_powers,
        motive_powers, on_engine,
        service_battery_electric_powers_supply_threshold,
        service_battery_starts_windows, service_battery_initialization_time):
    """
    Identifies service battery charging statuses: Discharge (0), Charging (1),
    BERS (2), and Initialization(3) [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param service_battery_electric_powers_supply_threshold:
        Service battery not charging power threshold [kW].
    :type service_battery_electric_powers_supply_threshold: float

    :param service_battery_starts_windows:
        Service battery starts windows [-].
    :type service_battery_starts_windows: numpy.array

    :param service_battery_initialization_time:
        Service battery initialization time delta [s].
    :type service_battery_initialization_time: float

    :return:
        Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-].
    :rtype: numpy.array
    """
    threshold = service_battery_electric_powers_supply_threshold
    b = (alternator_electric_powers < threshold) & on_engine
    status = b.astype(int, copy=True)
    status[b & (motive_powers < 0)] = 2

    off = ~on_engine | service_battery_starts_windows
    mask = _mask_boolean_phases(status != 1)
    for i, j in mask:
        # noinspection PyUnresolvedReferences
        if ((status[i:j] == 2) | off[i:j]).all():
            status[i:j] = 1

    status[dcdc_converter_electric_powers < threshold] = 1
    _set_alt_init_status(times, service_battery_initialization_time, status)

    return status


# noinspection PyPep8
@sh.add_function(dsp, outputs=['service_battery_initialization_time'])
def identify_service_battery_initialization_time(
        alternator_electric_powers, motive_powers,
        accelerations, service_battery_state_of_charges,
        service_battery_charging_statuses, times,
        service_battery_electric_powers_supply_threshold):
    """
    Identifies the alternator initialization time delta [s].

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :param service_battery_charging_statuses:
        Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-].
    :type service_battery_charging_statuses: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_electric_powers_supply_threshold:
        Service battery not charging power threshold [kW].
    :type service_battery_electric_powers_supply_threshold: float

    :return:
        Service battery initialization time delta [s].
    :rtype: float
    """
    bats, p = service_battery_charging_statuses, motive_powers
    i = co2_utl.argmax(bats != 0)
    if bats[0] == 1 or (i and ((bats[:i] == 0) & (p[:i] == 0)).all()):
        s = service_battery_electric_powers_supply_threshold
        s = alternator_electric_powers < s
        n, i = len(times), int(co2_utl.argmax((s[:-1] != s[1:]) & s[:-1]))
        i = min(n - 1, i)

        # noinspection PyProtectedMember
        from ....engine._thermal import _build_samples, _XGBRegressor

        x, y = _build_samples(
            alternator_electric_powers, service_battery_state_of_charges, bats,
            p, accelerations
        )

        j = min(i, int(n / 2))
        # noinspection PyArgumentEqualDefault
        model = _XGBRegressor(
            random_state=0,
            max_depth=2,
            n_estimators=int(min(100.0, 0.25 * (n - j))) or 1,
            objective='reg:squarederror'
        ).fit(x[j:], y[j:])
        err = np.abs(y - model.predict(x))
        sets = np.array(co2_utl.get_inliers(err)[0], dtype=int)[:i]
        if (i and sum(sets) / i < 0.5) or i > j:
            from sklearn.tree import DecisionTreeClassifier
            reg = DecisionTreeClassifier(max_depth=1, random_state=0)
            reg.fit(times[1:i + 1, None], sets)
            s, r = reg.tree_.children_left[0], reg.tree_.children_right[0]
            s, r = np.argmax(reg.tree_.value[s]), np.argmax(reg.tree_.value[r])
            if s == 0 and r == 1:
                return reg.tree_.threshold[0] - times[0]
            elif r == 0 and not i > j:
                return times[i] - times[0]

    elif bats[0] == 3:
        i = co2_utl.argmax(bats != 3)
        return times[i] - times[0]
    return 0.0


@sh.add_function(dsp, outputs=[
    'service_battery_charging_statuses', 'service_battery_initialization_time'
], weight=1)
def identify_service_battery_charging_statuses_and_initialization_time(
        times, accelerations, on_engine, alternator_electric_powers,
        dcdc_converter_electric_powers, motive_powers,
        service_battery_electric_powers_supply_threshold,
        service_battery_starts_windows, service_battery_state_of_charges):
    """
    Identifies the service battery charging statuses [-] and its initialization
    time delta [s].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param alternator_electric_powers:
        Alternator electric power [kW].
    :type alternator_electric_powers: numpy.array

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param service_battery_electric_powers_supply_threshold:
        Service battery not charging power threshold [kW].
    :type service_battery_electric_powers_supply_threshold: float

    :param service_battery_starts_windows:
        Service battery starts windows [-].
    :type service_battery_starts_windows: numpy.array

    :param accelerations:
        Acceleration [m/s2].
    :type accelerations: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :return:
       Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-] and its initialization time delta [s].
    :rtype: numpy.array, float
    """
    statuses = identify_service_battery_charging_statuses(
        times, alternator_electric_powers, dcdc_converter_electric_powers,
        motive_powers, on_engine,
        service_battery_electric_powers_supply_threshold,
        service_battery_starts_windows, 0
    )
    initialization_time = identify_service_battery_initialization_time(
        alternator_electric_powers, motive_powers, accelerations,
        service_battery_state_of_charges, statuses, times,
        service_battery_electric_powers_supply_threshold
    )
    _set_alt_init_status(times, initialization_time, statuses)
    return statuses, initialization_time


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
    from ....engine.fc import _calibrate_model_params
    return _calibrate_model_params(error, parameters)[0].valuesdict()['B']


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class BatteryStatusModel:
    def __init__(self, bers_pred=None, charge_pred=None, min_soc=0.0,
                 max_soc=100.0):
        self.charge = charge_pred or (lambda X: np.zeros(len(X), dtype=bool))
        self.bers = bers_pred
        self.max = max_soc
        self.min = min_soc

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _fit_bers(self, charging_statuses, motive_powers):
        b = charging_statuses == 2
        threshold = 0.0
        if b.any():
            from sklearn.tree import DecisionTreeClassifier
            q = dfl.functions.BatteryStatusModel.min_percentile_bers
            m = DecisionTreeClassifier(random_state=0, max_depth=2)
            c = charging_statuses != 1
            # noinspection PyUnresolvedReferences
            m.fit(motive_powers[c, None], b[c])

            X = motive_powers[b, None]
            if (np.sum(m.predict(X)) / X.shape[0] * 100) >= q:
                self.bers = m.predict  # shortcut name
                return self.bers

            # noinspection PyUnresolvedReferences
            if not b.all():
                gb_p_s = motive_powers[_mask_boolean_phases(b)[:, 0]]

                threshold = min(threshold, np.percentile(gb_p_s, q))

        self.bers = lambda x: np.asarray(x) < threshold
        return self.bers

    # noinspection PyShadowingNames
    def _fit_charge(self, charging_statuses, state_of_charges, times,
                    is_hybrid):
        b = charging_statuses[1:] == 1
        self.max, self.min = 100.0, 0.0
        if b.all():
            self.min = max(state_of_charges)
        elif b.any():
            from sklearn.tree import DecisionTreeClassifier
            charge = DecisionTreeClassifier(random_state=0, max_depth=3)
            X = np.column_stack(
                (charging_statuses[:-1], state_of_charges[1:])
            )
            charge.fit(X, b)
            soc = state_of_charges[charging_statuses == 1]
            X = np.column_stack((
                np.ones(100), np.linspace(soc.min(), soc.max(), 100)
            ))
            s = np.where(charge.predict(X))[0]
            if is_hybrid:
                self.charge = lambda x: [x[0][0] == 1]
            else:
                self.charge = charge.predict
            if s.shape[0]:
                self.min, self.max = X[s[0], 1], X[s[-1], 1]
        self._fit_boundaries(charging_statuses, state_of_charges, times)

    def _fit_boundaries(self, charging_statuses, state_of_charges, times):
        n, b = len(charging_statuses), charging_statuses == 1
        mask = _mask_boolean_phases(b)
        _max, _min, balance = [], [], ()
        from scipy.stats import linregress
        min_dt = dfl.functions.BatteryStatusModel.min_delta_time_boundaries
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

        min_delta_soc = dfl.functions.BatteryStatusModel.min_delta_soc
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
        elif b.any() and not b.all():
            balance = _identify_balance_soc(times, state_of_charges)
            # noinspection PyTypeChecker
            std = np.sqrt(np.mean((balance - state_of_charges) ** 2)) * 2
            std = min(min_delta_soc, std)
            self.max = min(balance + std, 100.0)
            self.min = max(balance - std, 0.0)

    def fit(self, is_hybrid, times, charging_statuses, state_of_charges,
            motive_powers):

        i = co2_utl.argmax(charging_statuses != 3)

        status, soc = charging_statuses[i:], state_of_charges[i:]

        self._fit_bers(status, motive_powers[i:])
        self._fit_charge(status, soc, times[i:], is_hybrid)

        return self

    def predict(self, has_energy_rec, init_time, time, prev, soc, power):
        status = 0

        if soc < 100:
            func = self.charge
            if time < init_time:
                status = 3

            elif soc < self.min or (soc <= self.max and func([[prev, soc]])[0]):
                status = 1

            elif has_energy_rec and self.bers([(power,)])[0]:
                status = 2

        return status


@sh.add_function(dsp, outputs=['service_battery_status_model'], weight=10)
def calibrate_service_battery_status_model(
        is_hybrid, times, service_battery_charging_statuses,
        service_battery_state_of_charges, motive_powers):
    """
    Calibrates the service battery charging status model.

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_charging_statuses:
        Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-].
    :type service_battery_charging_statuses: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :return:
        A function that predicts the service battery charging status.
    :rtype: callable
    """
    return BatteryStatusModel().fit(
        is_hybrid, times, service_battery_charging_statuses,
        service_battery_state_of_charges, motive_powers
    )


@sh.add_function(dsp, outputs=['service_battery_status_model'])
def define_service_battery_status_model(
        service_battery_state_of_charge_balance,
        service_battery_state_of_charge_balance_window):
    """
    Defines the service battery charging status model.

    :param service_battery_state_of_charge_balance:
        Service battery state of charge balance [%].
    :type service_battery_state_of_charge_balance: float

    :param service_battery_state_of_charge_balance_window:
        Service battery state of charge balance [%].
    :type service_battery_state_of_charge_balance_window: float

    :return:
        A function that predicts the service battery charging status.
    :rtype: callable
    """
    m = service_battery_state_of_charge_balance
    w = service_battery_state_of_charge_balance_window / 2
    return BatteryStatusModel(
        charge_pred=lambda x: [x[0][0] == 1],
        bers_pred=lambda x: [x[0][0] < 0],
        min_soc=m - w, max_soc=m + w
    )


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=[
    'service_battery_state_of_charge_balance',
    'service_battery_state_of_charge_balance_window'
])
def identify_service_battery_state_of_charge_balance_and_window(
        service_battery_status_model):
    """
    Identify the service battery state of charge balance and window [%].

    :param service_battery_status_model:
        A function that predicts the service battery charging status.
    :type service_battery_status_model: BatteryStatusModel

    :return:
        Service battery state of charge balance and window [%].
    :rtype: float, float
    """
    model = service_battery_status_model
    state_of_charge_balance_window = model.max - model.min
    state_of_charge_balance = model.min + state_of_charge_balance_window / 2
    return state_of_charge_balance, state_of_charge_balance_window
