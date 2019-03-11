# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains models to compare/select the calibrated co2_params.
"""

import schedula as sh
from co2mpas.utils import mae
from ._core import define_sub_model
from ...physical.engine.co2_emission import dsp as _co2_emission

#: Model name.
name = 'co2_params'

#: Parameters that constitute the model.
models = [
    'co2_params_calibrated', 'calibration_status', 'initial_friction_params',
    'engine_idle_fuel_consumption'
]

#: Inputs required to run the model.
inputs = ['co2_emissions_model']

#: Relevant outputs of the model.
outputs = ['co2_emissions', 'calibration_status']

#: Targets to compare the outputs of the model.
targets = ['identified_co2_emissions', 'calibration_status']

#: Weights coefficients to compute the model score.
weights = sh.map_list(targets, 1, None)


# noinspection PyUnusedLocal
def metric_calibration_status(y_true, y_pred):
    """
    Metric for the `calibration_status`.

    :param y_true:
        Reference calibration_status.
    :type y_true: list

    :param y_pred:
        Predicted calibration_status.
    :type y_pred: list

    :return:
        Error.
    :rtype: list
    """
    return [v[0] for v in y_pred]


#: Metrics to compare outputs with targets.
metrics = sh.map_list(targets, mae, metric_calibration_status)

#: Upper score limits to raise the warnings.
up_limit = {'identified_co2_emissions': 0.5}

#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_co2_emission, inputs, outputs, models)._set_cls(
    define_sub_model
)
