# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the alternator current.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Alternator current', description='Models the alternator current.'
)


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

        # noinspection PyProtectedMember
        from ....engine._thermal import _XGBRegressor
        self.model = default_model
        self.mask = None
        self.init_model = default_model
        self.init_mask = None
        self.base_model = _XGBRegressor

    def _fit_charging_currents(self, X, y):
        gb_power, acc = X[:, -2:].T
        b = (gb_power > 0) | ((gb_power == 0) & (acc >= 0))
        if b.all() or ~b.all():
            return (np.median(y),) * 2
        return np.median(y[b]), np.median(y[~b])

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
        i = co2_utl.argmax(times >= times[0] + init_time)
        from ....engine._thermal import _build_samples
        X, Y = _build_samples(currents, soc, statuses, *args)
        if b[i:].any():
            self.model, self.mask = self._fit_model(X[i:][b[i:]], Y[i:][b[i:]])
            self.alternator_charging_currents = self._fit_charging_currents(
                X[i:][b[i:]], Y[i:][b[i:]]
            )
        elif b[:i].any():
            self.model, self.mask = self._fit_model(X[b], Y[b])
            self.alternator_charging_currents = self._fit_charging_currents(
                X[b], Y[b]
            )
        else:
            self.alternator_charging_currents = 0, 0
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
        from sklearn.pipeline import Pipeline
        from ....engine._thermal import _SelectFromModel
        # noinspection PyArgumentEqualDefault
        model = self.base_model(
            random_state=0,
            max_depth=2,
            objective='reg:squarederror',
            n_estimators=int(min(300.0, 0.25 * (len(X) - 1))) or 1
        )
        model = Pipeline([
            ('feature_selection', _SelectFromModel(
                model, threshold='0.8*median', in_mask=in_mask,
                out_mask=out_mask
            )),
            ('classification', model)
        ])
        model.fit(X, Y)
        model.steps[0][1].estimator_.cache_params()
        model.steps[0][1].estimator.cache_params()
        model.steps[1][1].cache_params()
        mask = np.where(model.steps[0][-1]._get_support_mask())[0]
        return model.steps[-1][-1].predict, mask

    def __call__(self, time, soc, status, *args):
        arr = np.array([(time, soc, status) + args], float)
        if status == 3:
            return min(0.0, self.init_model(arr[:, self.init_mask])[0])
        return min(0.0, self.model(arr[:, self.mask])[0])


@sh.add_function(dsp, outputs=['alternator_current_model'])
def calibrate_alternator_current_model(
        alternator_currents, on_engine, times, service_battery_state_of_charges,
        service_battery_charging_statuses, motive_powers, accelerations,
        service_battery_initialization_time):
    """
    Calibrates an alternator current model that predicts alternator current [A].

    :param alternator_currents:
        Alternator currents [A].
    :type alternator_currents: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :param service_battery_charging_statuses:
        Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-].
    :type service_battery_charging_statuses: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param service_battery_initialization_time:
        Service battery initialization time delta [s].
    :type service_battery_initialization_time: float

    :return:
        Alternator current model.
    :rtype: callable
    """
    return AlternatorCurrentModel().fit(
        alternator_currents, on_engine, times, service_battery_state_of_charges,
        service_battery_charging_statuses, motive_powers, accelerations,
        init_time=service_battery_initialization_time
    )


@sh.add_function(dsp, outputs=['alternator_currents'])
def predict_alternator_currents(
        alternator_current_model, times, service_battery_state_of_charges,
        service_battery_charging_statuses, motive_powers, accelerations,
        service_battery_initialization_time):
    """
    Predict alternator currents [A].

    :param alternator_current_model:
        Alternator current model.
    :type alternator_current_model: callable

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param service_battery_state_of_charges:
        State of charge of the service battery [%].
    :type service_battery_state_of_charges: numpy.array

    :param service_battery_charging_statuses:
        Service battery charging statuses (0: Discharge, 1: Charging, 2: BERS,
        3: Initialization) [-].
    :type service_battery_charging_statuses: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param service_battery_initialization_time:
        Service battery initialization time delta [s].
    :type service_battery_initialization_time: float

    :return:
        DC/DC converter current model.
    :rtype: callable
    """
    return alternator_current_model.predict(np.column_stack((
        times, service_battery_state_of_charges,
        service_battery_charging_statuses, motive_powers, accelerations
    )), init_time=service_battery_initialization_time)
