# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the DT Approach.
"""
import numpy as np
import schedula as sh
import sklearn.tree as sk_tree
import sklearn.pipeline as sk_pip
from .cmv import CMV
from .core import prediction_gears_gsm

dsp = sh.BlueDispatcher(name='Decision Tree Approach')


# noinspection PyMissingOrEmptyDocstring,PyCallByClass,PyUnusedLocal
# noinspection PyTypeChecker,PyPep8Naming
class DTGS:
    def __init__(self, velocity_speed_ratios):
        self.tree = sk_tree.DecisionTreeClassifier(random_state=0)
        self.model = self.gears = None
        self.velocity_speed_ratios = velocity_speed_ratios

    def fit(self, gears, velocities, accelerations, motive_powers,
            engine_coolant_temperatures):
        i = np.arange(-1, gears.shape[0] - 1)
        i[0] = 0
        # noinspection PyProtectedMember
        from ...engine.thermal import _SelectFromModel
        model = self.tree
        self.model = sk_pip.Pipeline([
            ('feature_selection', _SelectFromModel(
                model, '0.8*median', in_mask=(0, 1)
            )),
            ('classification', model)
        ])
        X = np.column_stack((
            gears[i], velocities, accelerations, motive_powers,
            engine_coolant_temperatures
        ))
        self.model.fit(X, gears)

        self.gears = np.unique(gears)
        return self

    def _prepare(self, times, velocities, accelerations, motive_powers,
                 engine_coolant_temperatures):
        keys = sorted(self.velocity_speed_ratios.keys())
        matrix, r, c = {}, velocities.shape[0], len(keys) - 1
        func = self.model.predict
        for i, g in enumerate(keys):
            matrix[g] = func(np.column_stack((
                np.tile(g, r), velocities, accelerations, motive_powers,
                engine_coolant_temperatures
            )))
        return matrix

    @staticmethod
    def get_gear(gear, index, gears, times, velocities, accelerations,
                 motive_powers, engine_coolant_temperatures, matrix):
        return matrix[gear][index]

    def yield_gear(self, *args, **kwargs):
        return CMV.yield_gear(self, *args, **kwargs)

    def yield_speed(self, *args, **kwargs):
        return CMV.yield_speed(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return CMV.predict(self, *args, **kwargs)

    def convert(self, velocity_speed_ratios):
        self.velocity_speed_ratios = velocity_speed_ratios


@sh.add_function(dsp, outputs=['DTGS'])
def calibrate_gear_shifting_decision_tree(
        velocity_speed_ratios, gears, velocities, accelerations, motive_powers,
        engine_coolant_temperatures):
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

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :returns:
        A decision tree to predict gears.
    :rtype: DTGS
    """

    model = DTGS(velocity_speed_ratios).fit(
        gears, velocities, accelerations, motive_powers,
        engine_coolant_temperatures
    )
    return model


dsp.add_function(
    function=prediction_gears_gsm,
    inputs=[
        'correct_gear', 'gear_filter', 'DTGS', 'times', 'velocities',
        'accelerations', 'motive_powers', 'engine_coolant_temperatures'
    ],
    outputs=['gears']
)
