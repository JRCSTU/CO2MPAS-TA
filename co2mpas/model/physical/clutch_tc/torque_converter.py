# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions that model the basic mechanics of the torque converter.
"""
import xgboost as xgb
import sklearn.metrics as sk_met
import sklearn.pipeline as sk_pip
import schedula as sh
import numpy as np


def identify_torque_converter_speeds_delta(
        engine_speeds_out, engine_speeds_out_hot, cold_start_speeds_delta):
    """
    Calculates the engine speed delta due to the clutch [RPM].

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param cold_start_speeds_delta:
        Engine speed delta due to the cold start [RPM].
    :type cold_start_speeds_delta: numpy.array

    :return:
        Engine speed delta due to the clutch or torque converter [RPM].
    :rtype: numpy.array
    """

    return engine_speeds_out - engine_speeds_out_hot - cold_start_speeds_delta


class TorqueConverter(object):
    def __init__(self):
        self.predict = self.no_model
        self.regressor = None

    def _fit_regressor(self, X, y):
        from ..engine.thermal import _SelectFromModel
        model = xgb.XGBRegressor(
            seed=0,
            max_depth=2,
            n_estimators=int(min(300, 0.25 * (len(y) - 1)))
        )
        model = sk_pip.Pipeline([
            ('feature_selection', _SelectFromModel(model, '0.8*median')),
            ('classification', model)
        ])
        model.fit(X, y)
        return model

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _fit_sub_set(self, params, speeds_delta, calibration_speed_threshold,
                     stop_velocity, accelerations, velocities,
                     gear_box_speeds_in, gears, *args):
        b = np.isclose(accelerations, (0,)) & (velocities < stop_velocity)
        return ~(b & (abs(speeds_delta) > calibration_speed_threshold))

    def fit(self, times, params, calibration_speed_threshold,
            stop_velocity, speeds_delta, accelerations,
            velocities, gear_box_speeds_in, gears, *args):

        X = np.column_stack(
            (accelerations, velocities, gear_box_speeds_in, gears) + args
        )
        y = speeds_delta

        b = self._fit_sub_set(
            params, speeds_delta, calibration_speed_threshold, stop_velocity,
            accelerations, velocities, gear_box_speeds_in, gears
        )

        if b.any():
            self.regressor = self._fit_regressor(X[b, :], y[b])
            models = enumerate((self.model, self.no_model))
            a = times, params, self._prediction_inputs(X)
            error = sk_met.mean_absolute_error
            m = min([(error(y, m(*a)), i, m) for i, m in models])[-1]
        else:
            m = self.no_model

        self.predict = m

        return self

    @staticmethod
    def _prediction_inputs(X):
        return X

    @staticmethod
    def no_model(times, params, X):
        return np.zeros(X.shape[0])

    def model(self, times, params, X):
        lm_vel, lm_acc = params
        d = np.zeros(X.shape[0])
        a, v = X[:, 0], X[:, 1]
        # From issue #179 add lock up mode in torque converter.
        b = (v < lm_vel) & (a > lm_acc)
        if b.any():
            d[b] = self.regressor.predict(X[b])
        return d


def calibrate_torque_converter_model(
        times, lock_up_tc_limits, calibration_tc_speed_threshold, stop_velocity,
        torque_converter_speeds_delta, accelerations, velocities,
        gear_box_speeds_in, gears):
    """
    Calibrate torque converter model.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param lock_up_tc_limits:
        Limits (vel, acc) when torque converter is active [km/h, m/s].
    :type lock_up_tc_limits: (float, float)

    :param calibration_tc_speed_threshold:
        Calibration torque converter speeds delta threshold [RPM].
    :type calibration_tc_speed_threshold: float

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param torque_converter_speeds_delta:
        Engine speed delta due to the torque converter [RPM].
    :type torque_converter_speeds_delta: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        Torque converter model.
    :rtype: TorqueConverter
    """

    model = TorqueConverter()
    model.fit(times, lock_up_tc_limits, calibration_tc_speed_threshold,
              stop_velocity, torque_converter_speeds_delta,
              accelerations, velocities, gear_box_speeds_in, gears)

    return model


def predict_torque_converter_speeds_delta(
        times, lock_up_tc_limits, torque_converter_model, accelerations,
        velocities, gear_box_speeds_in, gears):
    """
    Predicts engine speed delta due to the torque converter [RPM].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param lock_up_tc_limits:
        Limits (vel, acc) when torque converter is active [km/h, m/s].
    :type lock_up_tc_limits: (float, float)

    :param torque_converter_model:
        Torque converter model.
    :type torque_converter_model: TorqueConverter

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        Engine speed delta due to the torque converter [RPM].
    :rtype: numpy.array
    """

    X = np.column_stack(
        (accelerations, velocities, gear_box_speeds_in, gears)
    )

    return torque_converter_model(times, lock_up_tc_limits, X)


def default_tc_k_factor_curve():
    """
    Returns a default k factor curve for a generic torque converter.

    :return:
        k factor curve.
    :rtype: callable
    """
    from ..defaults import dfl
    par = dfl.functions.default_tc_k_factor_curve
    a = par.STAND_STILL_TORQUE_RATIO, par.LOCKUP_SPEED_RATIO

    from . import define_k_factor_curve
    return define_k_factor_curve(*a)


def torque_converter():
    """
    Defines the torque converter model.

    .. dispatcher:: d

        >>> d = torque_converter()

    :return:
        The torque converter model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Torque_converter',
        description='Models the torque converter.'
    )

    from ..defaults import dfl
    d.add_data(
        data_id='calibration_tc_speed_threshold',
        default_value=dfl.values.calibration_tc_speed_threshold
    )

    d.add_data(
        data_id='stop_velocity',
        default_value=dfl.values.stop_velocity
    )

    d.add_data(
        data_id='lock_up_tc_limits',
        default_value=dfl.values.lock_up_tc_limits
    )

    d.add_function(
        function=identify_torque_converter_speeds_delta,
        inputs=['engine_speeds_out', 'engine_speeds_out_hot',
                'cold_start_speeds_delta'],
        outputs=['torque_converter_speeds_delta']
    )

    d.add_function(
        function=calibrate_torque_converter_model,
        inputs=['times', 'lock_up_tc_limits', 'calibration_tc_speed_threshold',
                'stop_velocity', 'torque_converter_speeds_delta',
                'accelerations', 'velocities', 'gear_box_speeds_in', 'gears'],
        outputs=['torque_converter_model']
    )

    d.add_function(
        function=predict_torque_converter_speeds_delta,
        inputs=['times', 'lock_up_tc_limits', 'torque_converter_model',
                'accelerations', 'velocities', 'gear_box_speeds_in', 'gears'],
        outputs=['torque_converter_speeds_delta']
    )

    from . import define_k_factor_curve
    d.add_function(
        function=define_k_factor_curve,
        inputs=['stand_still_torque_ratio', 'lockup_speed_ratio'],
        outputs=['k_factor_curve']
    )

    d.add_function(
        function=default_tc_k_factor_curve,
        outputs=['k_factor_curve'],
        weight=2
    )

    return d
