# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It provides CO2MPAS architecture model `dsp`.

Sub-Modules:

.. currentmodule:: co2mpas.core.model

.. autosummary::
    :nosignatures:
    :toctree: model/

    physical
    selector
"""
import numpy as np
import schedula as sh
from .physical import dsp as _physical
from .selector import dsp as _selector, calibration_cycles, prediction_cycles

dsp = sh.BlueDispatcher(
    name='CO2MPAS model',
    description='Calibrates the models with WLTP data and predicts NEDC cycle.'
)

_physical = sh.SubDispatch(_physical)

dsp.add_data(
    data_id='input.calibration.wltp_h',
    description='User input data of WLTP-H calibration stage.',
    default_value=None,
    filters=[lambda x: {} if x is None else x]
)
dsp.add_data(
    data_id='input.calibration.wltp_l',
    description='User input data of WLTP-L calibration stage.',
    default_value=None,
    filters=[lambda x: {} if x is None else x]
)
dsp.add_data(
    data_id='input.calibration.wltp_m',
    description='User input data of WLTP-M calibration stage.',
    default_value=None,
    filters=[lambda x: {} if x is None else x]
)
dsp.add_data(
    data_id='output.calibration.wltp_h',
    description='Output data of WLTP-H calibration stage.'
)
dsp.add_data(
    data_id='output.calibration.wltp_l',
    description='Output data of WLTP-L calibration stage.'
)
dsp.add_data(
    data_id='output.calibration.wltp_m',
    description='Output data of WLTP-M calibration stage.'
)
dsp.add_function(
    function_id='calibrate_with_wltp_h',
    function=_physical,
    inputs=['input.calibration.wltp_h'],
    outputs=['output.calibration.wltp_h'],
    description='Wraps all functions needed to calibrate the models to '
                'predict CO2 emissions.'
)

dsp.add_function(
    function_id='calibrate_with_wltp_l',
    function=_physical,
    inputs=['input.calibration.wltp_l'],
    outputs=['output.calibration.wltp_l'],
    description='Wraps all functions needed to calibrate the models to '
                'predict CO2 emissions.'
)
dsp.add_function(
    function_id='calibrate_with_wltp_m',
    function=_physical,
    inputs=['input.calibration.wltp_m'],
    outputs=['output.calibration.wltp_m'],
    description='Wraps all functions needed to calibrate the models to '
                'predict CO2 emissions.'
)

dsp.add_data(
    data_id='input.prediction.wltp_h', default_value=None,
    description='User input data of WLTP-H prediction stage.',
    filters=[lambda x: {} if x is None else x]
)
dsp.add_data(
    data_id='input.prediction.wltp_l', default_value=None,
    description='User input data of WLTP-L prediction stage.',
    filters=[lambda x: {} if x is None else x]
)
dsp.add_data(
    data_id='input.prediction.wltp_m', default_value=None,
    description='User input data of WLTP-M prediction stage.',
    filters=[lambda x: {} if x is None else x]
)
dsp.add_data(
    data_id='data.prediction.models_wltp_h',
    description='Calibrated models for WLTP-H prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_wltp_l',
    description='Calibrated models for WLTP-L prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_wltp_m',
    description='Calibrated models for WLTP-M prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.wltp_h',
    description='Input data of WLTP-H prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.wltp_l',
    description='Input data of WLTP-L prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.wltp_m',
    description='Input data of WLTP-M prediction stage.'
)


@sh.add_function(
    dsp, inputs=['output.calibration.wltp_h', 'data.prediction.models_wltp_h',
                 'input.prediction.wltp_h'], outputs=['data.prediction.wltp_h']
)
@sh.add_function(
    dsp, inputs=['output.calibration.wltp_l', 'data.prediction.models_wltp_l',
                 'input.prediction.wltp_l'], outputs=['data.prediction.wltp_l']
)
@sh.add_function(
    dsp, inputs=['output.calibration.wltp_m', 'data.prediction.models_wltp_m',
                 'input.prediction.wltp_m'], outputs=['data.prediction.wltp_m']
)
def select_prediction_data(calibration_data, models_data, user_data):
    """
    Selects the data required to predict the CO2 emissions with CO2MPAS model.

    :param calibration_data:
        Output data of calibration stage
    :type calibration_data: dict

    :param models_data:
        Calibrated models for prediction stage.
    :type models_data: dict

    :param user_data:
        User input data of prediction stage.
    :type user_data: dict

    :return:
        Data required to predict the CO2 emissions with CO2MPAS model.
    :rtype: dict
    """
    from co2mpas.defaults import dfl
    kw = dict(calibration=calibration_data, models=models_data, user=user_data)
    data, xp = [], None
    for k, v in dfl.functions.select_prediction_data.prediction_data[::-1]:
        d = kw[k]
        if v != 'all':
            d = sh.selector(v, d, allow_miss=True)
        if 'times' in d:
            if xp is None:
                xp = d['times']
            else:
                x, inter = d['times'], np.interp
                d = {k: inter(x, xp, fp) for k, fp in d.items() if k != 'times'}
        data.append(d)
    return sh.combine_dicts(*data[::-1])


dsp.add_data(
    data_id='input.prediction.models', default_value={},
    description='User input models for prediction stages.'
)
dsp.add_data(
    data_id='enable_selector', default_value=False,
    description='Enable the selection of the best model to predict '
                'H/L/M cycles.'
)
dsp.add_data(
    data_id='data.calibration.model_scores',
    description='Scores of calibrated models.'
)

_prediction_models = tuple(map('models_{}'.format, prediction_cycles))
dsp.add_function(
    function=sh.SubDispatchPipe(
        _selector,
        function_id='extract_calibrated_models',
        inputs=('enable_selector', 'default_models',) + calibration_cycles,
        outputs=('selections',) + _prediction_models
    ),
    inputs=[
        'enable_selector', 'input.prediction.models',
        'output.calibration.wltp_h', 'output.calibration.wltp_l',
        'output.calibration.wltp_m'
    ],
    outputs=['data.calibration.model_scores'] + [
        'data.prediction.%s' % k for k in _prediction_models
    ]
)

dsp.add_data(
    data_id='output.prediction.wltp_h',
    description='Output data of WLTP-H prediction stage.'
)
dsp.add_data(
    data_id='output.prediction.wltp_l',
    description='Output data of WLTP-L prediction stage.'
)
dsp.add_data(
    data_id='output.prediction.wltp_m',
    description='Output data of WLTP-M prediction stage.'
)
dsp.add_function(
    function_id='predict_wltp_h',
    function=_physical,
    inputs=['data.prediction.wltp_h'],
    outputs=['output.prediction.wltp_h'],
    description='Wraps all functions needed to predict CO2 emissions.'
)

dsp.add_function(
    function_id='predict_wltp_l',
    function=_physical,
    inputs=['data.prediction.wltp_l'],
    outputs=['output.prediction.wltp_l'],
    description='Wraps all functions needed to predict CO2 emissions.'
)

dsp.add_function(
    function_id='predict_wltp_m',
    function=_physical,
    inputs=['data.prediction.wltp_m'],
    outputs=['output.prediction.wltp_m'],
    description='Wraps all functions needed to predict CO2 emissions.'
)

dsp.add_data(
    data_id='input.prediction.nedc_h',
    description='User input data of NEDC-H prediction stage.'
)
dsp.add_data(
    data_id='input.prediction.nedc_l',
    description='User input data of NEDC-L prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_nedc_h',
    description='Calibrated models for NEDC-H prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_nedc_l',
    description='Calibrated models for NEDC-L prediction stage.'
)
dsp.add_data(
    data_id='output.prediction.nedc_h',
    description='Output data of NEDC-H prediction stage.'
)
dsp.add_data(
    data_id='output.prediction.nedc_l',
    description='Output data of NEDC-L prediction stage.'
)
dsp.add_function(
    function_id='predict_nedc_h',
    function=_physical,
    inputs=['data.prediction.models_nedc_h', 'input.prediction.nedc_h'],
    outputs=['output.prediction.nedc_h'],
    description='Wraps all functions needed to predict CO2 emissions.'
)

dsp.add_function(
    function_id='predict_nedc_l',
    function=_physical,
    inputs=['data.prediction.models_nedc_l', 'input.prediction.nedc_l'],
    outputs=['output.prediction.nedc_l'],
    description='Wraps all functions needed to predict CO2 emissions.'
)
