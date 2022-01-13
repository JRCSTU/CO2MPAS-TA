# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the DT Approach.
"""
import numpy as np
import schedula as sh
from .cmv import CMV
from .core import prediction_gears_gsm as _prediction_gears_gsm

dsp = sh.BlueDispatcher(name='Decision Tree Approach')


# noinspection PyMissingOrEmptyDocstring,PyCallByClass,PyUnusedLocal
# noinspection PyTypeChecker,PyPep8Naming
class DTGS:
    def __init__(self, velocity_speed_ratios):
        from xgboost import XGBClassifier
        self.tree = XGBClassifier()
        self.model = self.gears = None
        self.velocity_speed_ratios = velocity_speed_ratios

    def fit(self, gears, velocities, accelerations, motive_powers,
            engine_temperatures):
        i = np.arange(-1, gears.shape[0] - 1)
        i[0] = 0
        # noinspection PyProtectedMember
        from ...engine._thermal import _SelectFromModel
        from sklearn.pipeline import Pipeline
        model = self.tree
        self.model = Pipeline([
            ('feature_selection', _SelectFromModel(
                model, threshold='0.8*median', in_mask=(0, 1)
            )),
            ('classification', model)
        ])
        X = np.column_stack((
            gears[i], velocities, accelerations, motive_powers,
            engine_temperatures
        ))
        self.model.fit(X, gears)

        self.gears = np.unique(gears)
        return self

    def _init_gear(self, times, velocities, accelerations, motive_powers,
                   engine_temperatures):
        from co2mpas.utils import List
        predict = self.model.predict
        pars = (
            velocities, accelerations, motive_powers,
            engine_temperatures
        )

        if any(isinstance(v, List) for v in pars) or np.isnan(pars[-1]).any():
            x = np.empty((1, 5), float)

            def _next(gear, i):
                x[:, :] = (
                    gear, velocities[i], accelerations[i], motive_powers[i],
                    engine_temperatures[i]
                )
                return predict(x)[0]
        else:
            matrix = {}
            x = np.column_stack((np.empty_like(velocities),) + pars)
            for g in self.velocity_speed_ratios.keys():
                x[:, 0] = g
                matrix[g] = predict(x)
            del x

            def _next(gear, index):
                return matrix[gear][index]
        return _next

    def init_gear(self, *args, **kwargs):
        return CMV.init_gear(self, *args, **kwargs)

    def init_speed(self, *args, **kwargs):
        return CMV.init_speed(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return CMV.predict(self, *args, **kwargs)

    def convert(self, velocity_speed_ratios):
        self.velocity_speed_ratios = velocity_speed_ratios
        return self


@sh.add_function(dsp, outputs=['DTGS'])
def calibrate_gear_shifting_decision_tree(
        velocity_speed_ratios, gears, velocities, accelerations, motive_powers,
        engine_temperatures):
    """
    Calibrates a decision tree to predict gears.

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :returns:
        A decision tree to predict gears.
    :rtype: DTGS
    """

    model = DTGS(velocity_speed_ratios).fit(
        gears, velocities, accelerations, motive_powers,
        engine_temperatures
    )
    return model


def prediction_gears_gsm(*a):
    """
    Predicts gears with a gear shifting model (cmv or gspv or dtgs or mgs) [-].

    :param a:
        Arguments of `_prediction_gears_gsm`.
    :type a: tuple

    :return:
        Predicted gears.
    :rtype: numpy.array
    """
    return _prediction_gears_gsm(*a[:-1], engine_temperatures=a[-1])


prediction_gears_gsm.__doc__ = _prediction_gears_gsm.__doc__
dsp.add_function(
    function=prediction_gears_gsm,
    inputs=[
        'correct_gear', 'gear_filter', 'DTGS', 'times', 'velocities',
        'accelerations', 'motive_powers', 'engine_temperatures'
    ],
    outputs=['gears']
)
