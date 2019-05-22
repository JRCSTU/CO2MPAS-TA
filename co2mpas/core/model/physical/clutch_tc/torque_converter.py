# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the mechanic of the torque converter.
"""
import numpy as np
import schedula as sh
from ..defaults import dfl

dsp = sh.BlueDispatcher(
    name='Torque_converter', description='Models the torque converter.'
)
dsp.add_data(
    'calibration_tc_speed_threshold', dfl.values.calibration_tc_speed_threshold
)
dsp.add_data('stop_velocity', dfl.values.stop_velocity)
dsp.add_data('lock_up_tc_limits', dfl.values.lock_up_tc_limits)


@sh.add_function(dsp, outputs=['torque_converter_speeds_delta'])
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


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class TorqueConverter:
    def __init__(self):
        self.predict = self.no_model
        self.regressor = None

    @staticmethod
    def _fit_regressor(X, y):
        # noinspection PyProtectedMember
        from ..engine._thermal import _SelectFromModel
        from sklearn.pipeline import Pipeline
        import xgboost as xgb

        model = xgb.XGBRegressor(
            max_depth=2,
            n_estimators=int(min(300., 0.25 * (len(y) - 1)))
        )
        model = Pipeline([
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

        # noinspection PyUnresolvedReferences
        if b.any():
            from co2mpas.utils import mae
            self.regressor = self._fit_regressor(X[b, :], y[b])
            models = enumerate((self.model, self.no_model))
            a = times, params, self._prediction_inputs(X)
            m = min([(mae(y, m(*a)), i, m) for i, m in models])[-1]
        else:
            m = self.no_model

        self.predict = m

        return self

    @staticmethod
    def _prediction_inputs(X):
        return X

    # noinspection PyUnusedLocal
    @staticmethod
    def no_model(times, params, X):
        return np.zeros(X.shape[0])

    def model(self, times, params, X):
        lm_vel, lm_acc = params
        d = np.zeros(X.shape[0])
        a, v = X[:, 0], X[:, 1]
        # From issue #179 add lock up mode in torque converter.
        b = (v < lm_vel) & (a > lm_acc)
        # noinspection PyUnresolvedReferences
        if b.any():
            d[b] = self.regressor.predict(X[b])
        return d

    def next(self, params, accelerations, velocities, gear_box_speeds_in, gears,
             *args):
        (lm_vel, lm_acc), predict = params, self.regressor.predict

        def _next(i):
            a, v = accelerations[i], velocities[i]
            # From issue #179 add lock up mode in torque converter.
            if v < lm_vel and a > lm_acc:
                return predict([[a, v, gear_box_speeds_in[i], gears[i]]])[0]
            return 0

        return _next


@sh.add_function(dsp, outputs=['torque_converter_model'])
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
        Limits (vel, acc) when torque converter is active [km/h, m/s2].
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


@sh.add_function(dsp, outputs=['torque_converter_speeds_delta'])
def predict_torque_converter_speeds_delta(
        times, lock_up_tc_limits, torque_converter_model, accelerations,
        velocities, gear_box_speeds_in, gears):
    """
    Predicts engine speed delta due to the torque converter [RPM].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param lock_up_tc_limits:
        Limits (vel, acc) when torque converter is active [km/h, m/s2].
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
    x = np.column_stack((accelerations, velocities, gear_box_speeds_in, gears))
    return torque_converter_model(times, lock_up_tc_limits, x)


@sh.add_function(dsp, outputs=['init_clutch_tc_speed_prediction_model'])
def define_init_clutch_tc_speed_prediction_model(
        torque_converter_model, lock_up_tc_limits):
    """
    Defines initialization function of the clutch tc speed prediction model.

    :param torque_converter_model:
        Torque converter model.
    :type torque_converter_model: TorqueConverter

    :param lock_up_tc_limits:
        Limits (vel, acc) when torque converter is active [km/h, m/s2].
    :type lock_up_tc_limits: (float, float)

    :return:
        Initialization function of the clutch tc speed prediction model.
    :rtype: function
    """
    import functools
    return functools.partial(torque_converter_model.next, lock_up_tc_limits)


@sh.add_function(dsp, inputs_kwargs=True, outputs=['k_factor_curve'])
def define_k_factor_curve(stand_still_torque_ratio=1.0, lockup_speed_ratio=0.0):
    """
    Defines k factor curve.

    :param stand_still_torque_ratio:
        Torque ratio when speed ratio==0.

        .. note:: The ratios are defined as follows:

           - Torque ratio = `gear box torque` / `engine torque`.
           - Speed ratio = `gear box speed` / `engine speed`.
    :type stand_still_torque_ratio: float

    :param lockup_speed_ratio:
        Minimum speed ratio where torque ratio==1.

        ..note::
            torque ratio==1 for speed ratio > lockup_speed_ratio.
    :type lockup_speed_ratio: float

    :return:
        k factor curve.
    :rtype: callable
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    if lockup_speed_ratio == 0:
        x = [0, 1]
        y = [1, 1]
    elif lockup_speed_ratio == 1:
        x = [0, 1]
        y = [stand_still_torque_ratio, 1]
    else:
        x = [0, lockup_speed_ratio, 1]
        y = [stand_still_torque_ratio, 1, 1]

    return InterpolatedUnivariateSpline(x, y, k=1)


@sh.add_function(dsp, outputs=['k_factor_curve'], weight=2)
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
    return define_k_factor_curve(*a)


@sh.add_function(
    dsp, outputs=['m1000_curve_ratios', 'm1000_curve_norm_torques'], weight=2
)
def default_tc_normalized_m1000_curve():
    """
    Returns default `m1000_curve_ratios` and `m1000_curve_norm_torques`.

    :return:
        Speed ratios and normalized torques of m1000 curve.
    :rtype: tuple[numpy.array]
    """

    from ..defaults import dfl
    curve = dfl.functions.default_tc_normailzed_m1000_curve.curve
    return np.array(curve['x']), np.array(curve['y'])


@sh.add_function(dsp, outputs=['normalized_m1000_curve'], weight=2)
def define_normalized_m1000_curve(m1000_curve_ratios, m1000_curve_norm_torques):
    """
    Defines normalized m1000 curve function.

    :param m1000_curve_ratios:
        Speed ratios of m1000 curve [-].
    :type m1000_curve_ratios: numpy.array

    :param m1000_curve_norm_torques:
        Normalized torques of m1000 curve [-].
    :type m1000_curve_norm_torques: numpy.array

    :return:
        Normalized m1000 curve function.
    :rtype: callable
    """
    from scipy.interpolate import interp1d
    return interp1d(m1000_curve_ratios, m1000_curve_norm_torques)


@sh.add_function(dsp, outputs=['m1000_curve_factor'])
def default_m1000_curve_factor(full_load_curve):
    """
    Returns the default value of the rescaling factor of m1000 curve [N*m].

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :return:
        Rescaling factor of m1000 curve [N*m].
    :rtype: float
    """
    from ..wheels import calculate_wheel_torques
    return calculate_wheel_torques(full_load_curve(1000), 1000) / 1e6
