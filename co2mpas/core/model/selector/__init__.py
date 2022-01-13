# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to compare/select the calibrated models.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.selector

.. autosummary::
    :nosignatures:
    :toctree: selector/

    models
"""
import pkgutil
import functools
import schedula as sh
import os.path as osp
from .models import mdl_selector, calibration_cycles, prediction_cycles

dsp = sh.BlueDispatcher(
    name='Models selector', description='Select the calibrated models.'
)

dsp.add_function(
    function=functools.partial(sh.map_list, calibration_cycles),
    inputs=calibration_cycles,
    outputs=['CO2MPAS_results']
)

MODELS = [
    v.name
    for v in pkgutil.iter_modules([osp.join(osp.dirname(__file__), 'models')])
    if not v.name.startswith('_')
]

dsp.add_data(
    'enable_selector', False, filters=[lambda x: {'enable_selector': x}]
)

for name in MODELS:
    dsp.add_function(
        function=sh.SubDispatch(
            mdl_selector('.models.%s' % name, __name__),
            outputs=['model', 'score'], output_type='list'
        ),
        function_id='%s selector' % name,
        inputs=['CO2MPAS_results', 'enable_selector'],
        outputs=['models', 'scores']
    )


def merge_model_selection(adict):
    """
    Merge models or scores selection.

    :param adict:
        Models or scores selection.
    :type adict: dict

    :return:
        Merged models or scores selection.
    :rtype: dict
    """
    return {' '.join(k.split(' ')[:-1]): v for k, v in adict.items() if v}


dsp.add_data(data_id='models', function=merge_model_selection, wait_inputs=True)
dsp.add_data(data_id='scores', function=merge_model_selection, wait_inputs=True)


# noinspection PyShadowingNames
@sh.add_function(dsp, outputs=['selections'] + list(
    map('models_{}'.format, prediction_cycles)
))
def split_prediction_models(scores, models, default_models):
    """
    Split prediction models.

    :param scores:
        Models score.
    :type scores: dict

    :param models:
        Calibrated models.
    :type models: dict

    :param default_models:
        Default calibrated models.
    :type default_models: dict

    :return:
        Scores and prediction models.
    :rtype: tuple
    """
    sbm, model_sel, par = {}, {}, {}
    for (k, c), v in sh.stack_nested_keys(scores, depth=2):
        r = sh.selector(['models'], v, allow_miss=True)

        for m in r.get('models', ()):
            sh.get_nested_dicts(par, m, 'calibration')[c] = c

        r.update(v.get('score', {}))
        sh.get_nested_dicts(sbm, k)[c] = r
        r = sh.selector(['success'], r, allow_miss=True)
        r = sh.map_dict({'success': 'status'}, r, {'from': c})
        sh.get_nested_dicts(model_sel, k, 'calibration')[c] = r

    p = {i: dict.fromkeys(default_models, 'input') for i in prediction_cycles}

    mdls = {i: default_models.copy() for i in prediction_cycles}

    for k, n in sorted(models.items()):
        d = n.get(sh.NONE, (None, True, {}))

        for i in prediction_cycles:
            c, s, m = n.get(i, d)
            if m:
                s = {'from': c, 'status': s}
                sh.get_nested_dicts(model_sel, k, 'prediction')[i] = s
                mdls[i].update(m)
                p[i].update(dict.fromkeys(m, c))

    for k, v in sh.stack_nested_keys(p, ('prediction',), depth=2):
        sh.get_nested_dicts(par, k[-1], *k[:-2])[k[-2]] = v

    s = {
        'param_selections': par,
        'model_selections': model_sel,
        'score_by_model': sbm,
        'scores': scores
    }
    return (s,) + tuple(mdls.get(k, {}) for k in prediction_cycles)
