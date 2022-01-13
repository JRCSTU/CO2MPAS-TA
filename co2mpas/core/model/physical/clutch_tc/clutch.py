# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and model `dsp` to model the mechanic of the clutch.
"""
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
from .torque_converter import define_k_factor_curve

dsp = sh.BlueDispatcher(name='Clutch', description='Models the clutch.')


@sh.add_function(dsp, outputs=['clutch_window'])
def default_clutch_window():
    """
    Returns a default clutching time window [s] for a generic clutch.

    :return:
        Clutching time window [s].
    :rtype: tuple
    """
    return dfl.functions.default_clutch_window.clutch_window


# noinspection PyUnusedLocal
def _no_model(times, **kwargs):
    return np.zeros_like(times)


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['clutch_speed_model'])
def calibrate_clutch_speed_model(
        clutch_phases, accelerations, clutch_tc_speeds_delta, velocities,
        gear_box_speeds_in, gears):
    """
    Calibrate clutch speed model.

    :param clutch_phases:
        When the clutch is active [-].
    :type clutch_phases: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param clutch_tc_speeds_delta:
        Engine speed delta due to the clutch [RPM].
    :type clutch_tc_speeds_delta: numpy.array

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
        Clutch speed model.
    :rtype: callable
    """
    model = _no_model
    if clutch_phases.sum() > 10:
        from co2mpas.utils import mae
        from sklearn.pipeline import Pipeline
        # noinspection PyProtectedMember
        from ..engine._thermal import _SelectFromModel, _XGBRegressor
        X = np.column_stack(
            (accelerations, velocities, gear_box_speeds_in, gears)
        )[clutch_phases]
        y = clutch_tc_speeds_delta[clutch_phases]

        # noinspection PyArgumentEqualDefault
        mdl = _XGBRegressor(
            max_depth=2,
            n_estimators=int(min(300., 0.25 * (len(y) - 1))),
            random_state=0,
            objective='reg:squarederror'
        )
        mdl = Pipeline([
            ('feature_selection',
             _SelectFromModel(mdl, threshold='0.8*median')),
            ('classification', mdl)
        ])
        mdl.fit(X, y)
        mdl.steps[0][1].estimator_.cache_params()
        mdl.steps[0][1].estimator.cache_params()
        mdl.steps[1][1].cache_params()
        if mae(mdl.predict(X), y) < mae(0, y):
            keys = 'accelerations', 'velocities', 'gear_box_speeds_in', 'gears'

            # noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
            def model(times, **kwargs):
                return mdl.predict(np.column_stack(sh.selector(keys, kwargs)))
    return model


dsp.add_function(
    function=define_k_factor_curve,
    inputs=['stand_still_torque_ratio', 'lockup_speed_ratio'],
    outputs=['k_factor_curve']
)


@sh.add_function(dsp, outputs=['k_factor_curve'], weight=2)
def default_clutch_k_factor_curve():
    """
    Returns a default k factor curve for a generic clutch.

    :return:
        k factor curve.
    :rtype: callable
    """
    from co2mpas.defaults import dfl
    par = dfl.functions.default_clutch_k_factor_curve
    a = par.STAND_STILL_TORQUE_RATIO, par.LOCKUP_SPEED_RATIO
    from .torque_converter import define_k_factor_curve
    return define_k_factor_curve(*a)
