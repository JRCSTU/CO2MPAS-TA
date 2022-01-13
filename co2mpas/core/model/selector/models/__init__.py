# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to compare/select the calibrated models.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.selector.models

.. autosummary::
    :nosignatures:
    :toctree: models/

    _core
    at_model
    after_treatment_model
    clutch_torque_converter_model
    co2_params
    control_model
    electrics_model
    engine_coolant_temperature_model
    engine_speed_model
"""
import logging
import functools
import numpy as np
import schedula as sh
from collections.abc import Iterable

log = logging.getLogger(__name__)
calibration_cycles = 'wltp_h', 'wltp_l', 'wltp_m'
prediction_cycles = 'nedc_h', 'nedc_l', 'wltp_h', 'wltp_l', 'wltp_m'

_select_data = functools.partial(sh.selector, allow_miss=True)
_map_list = functools.partial(sh.map_list, ['calibrated_models', 'data'])


def select_predictions(outputs, targets, results):
    """
    Select the prediction results.

    :param outputs:
        Output keys.
    :type outputs: list | tuple

    :param targets:
        Targets keys.
    :type targets: list | tuple

    :param results:
        Calibration results.
    :type results: dict

    :return:
        Prediction results.
    :rtype: dict
    """
    results = sh.selector(outputs, results, allow_miss=True)
    return sh.map_dict(dict(zip(outputs, targets)), results)


def calculate_errors(references, predictions, metrics, metrics_kwargs):
    """
    Calculate error coefficients.

    :param references:
        Reference results.
    :type references: dict

    :param predictions:
        Prediction results.
    :type predictions: dict

    :param metrics:
        Metric functions.
    :type metrics: dict[str, function]

    :param metrics_kwargs:
        Kwargs to pass to metric functions.
    :type metrics_kwargs: dict

    :return:
        Error coefficients.
    :rtype: dict
    """
    it = set(predictions).intersection(references).intersection(metrics)
    try:
        return {
            k: metrics[k](references[k], predictions[k], **metrics_kwargs)
            for k in it
        }
    except TypeError:
        return {}


def _check_limit(limit, errors, check=lambda e, l: e <= l):
    it = set(limit).intersection(errors)
    return {k: check(errors[k], limit[k]) for k in it}


def calculate_calibration_status(errors, up_limit=None, dn_limit=None):
    """
    Calculate calibration statuses.

    :param errors:
        Error coefficients.
    :type errors: dict

    :param up_limit:
        Upper limits.
    :type up_limit: dict

    :param dn_limit:
        Lower limits.
    :type dn_limit: dict

    :return:
        Calibration statuses.
    :rtype: dict
    """
    status = {}

    limit = _check_limit(up_limit or {}, errors, check=lambda e, l: e <= l)
    if limit:
        status['up_limit'] = limit

    limit = _check_limit(dn_limit or {}, errors, check=lambda e, l: e >= l)
    if limit:
        status['up_limit'] = limit

    return status


def _mdl_error(mdl):
    dsp = sh.BlueDispatcher(
        name=mdl.name,
        description='Calculates the error of calibrated model of a reference.',
    )
    dsp.add_data('inputs_map', getattr(mdl, 'inputs_map', {}))

    dsp.add_function(
        function_id='select_inputs',
        function=sh.map_dict,
        inputs=['inputs_map', 'data'],
        outputs=['inputs<0>']
    )

    dsp.add_data('inputs', getattr(mdl, 'inputs', []))
    dsp.add_function(
        function_id='select_inputs',
        function=_select_data,
        inputs=['inputs', 'inputs<0>'],
        outputs=['inputs<1>']
    )

    dsp.add_function(
        function=sh.combine_dicts,
        inputs=['calibrated_models', 'inputs<1>'],
        outputs=['prediction_inputs']
    )

    dsp.add_data('targets', getattr(mdl, 'targets', []))
    dsp.add_function(
        function_id='select_targets', function=_select_data,
        inputs=['targets', 'data'], outputs=['references']
    )

    dsp.add_function(
        function=sh.SubDispatch(mdl.dsp),
        inputs=['prediction_inputs', 'calibrated_models'],
        outputs=['results']
    )

    dsp.add_data('outputs', getattr(mdl, 'outputs', []))
    dsp.add_func(select_predictions, outputs=['predictions'])

    dsp.add_data('metrics_inputs', getattr(mdl, 'metrics_inputs', {}))
    dsp.add_function(
        function_id='select_metrics_inputs',
        function=_select_data,
        inputs=['metrics_inputs', 'data'],
        outputs=['metrics_kwargs']
    )

    dsp.add_data('metrics', getattr(mdl, 'metrics', {}))
    dsp.add_func(calculate_errors, outputs=['errors'])

    dsp.add_data('up_limit', getattr(mdl, 'up_limit', None))
    dsp.add_data('dn_limit', getattr(mdl, 'dn_limit', None))
    dsp.add_func(
        calculate_calibration_status, inputs_kwargs=True, outputs=['status']
    )

    return dsp


def _mdl_errors(mdl, data_id, err_func):
    name = '%s-%s errors' % (mdl.name, data_id)
    dsp = sh.BlueDispatcher(
        name=name, description='Calculates the error of calibrated model.'
    )
    dsp.add_data('models', mdl.models)

    dsp.add_function(
        function_id='select_models',
        function=getattr(mdl, 'select_models', _select_data),
        inputs=['models', data_id],
        outputs=['calibrated_models']
    )
    dsp.add_data('data_in', data_id)

    for o in calibration_cycles:
        dsp.add_function(
            function=_map_list,
            inputs=['calibrated_models', o],
            outputs=['input/%s' % o]
        )

        dsp.add_function(
            function=err_func,
            inputs=['input/%s' % o],
            outputs=['error/%s' % o]
        )
    return dsp, name


def _mean(values, weights=None):
    if isinstance(weights, Iterable):
        values = [v * w for v, w in zip(values, weights) if w]

    v = np.asarray(values)
    return np.average(v)


def _key_score(x):
    s = 1 if np.isnan(x['score']) else -int(x['success'])
    return s, -x['n'], x['score']


def _key_scores(x):
    return tuple(y[:2] for y in x)


def _sorting_func(x):
    return _key_score(x[0]) + _key_scores(x[1]) + (x[3],)


def sort_models(*data, weights=None):
    """
    Sort models by scores.

    :param weights:
        Weights coefficients.
    :type weights: dict

    :return:
        Models rank.
    :rtype: list
    """
    weights = weights or {}
    rank = []

    for d in data:
        errors = {k[6:]: v for k, v in d.items() if k.startswith('error/')}
        scores = []

        def _sort(x):
            return x[0], x[1], tuple(x[2].values()), x[3]

        for k, v in errors.items():
            if v[0]:
                l = [list(m.values()) for l, m in sorted(v[1].items()) if m]
                l = _mean(l) if l else 1
                keys, m = zip(*v[0].items())
                e = l, _mean(m, weights=[weights.get(i, 1) for i in keys])
                scores.append((e, l, v[0], k, v[1]))

        scores = sorted(scores, key=_sort)
        if scores:
            score = tuple(np.mean([e[0] for e in scores], axis=0))

            models = d['calibrated_models']

            if models:
                score = {
                    'success': score[0] == 1,
                    'n': len(models),
                    'score': score[1]
                }

            rank.append([score, scores, errors, d['data_in'], models])

    return sorted(rank, key=_sorting_func)


def _check_success(score):
    try:
        return score['success']
    except IndexError:
        return True


def format_score(rank):
    """
    Format score output.

    :param rank:
        Models rank.
    :type rank: list

    :return:
        Score output.
    :rtype: dict
    """
    res = {}
    for score, scores, errors, name, models in rank:
        res[name] = d = {'models': sorted(models.keys())}
        if scores:
            d.update({
                'score': score,
                'errors': {k: v[0] for k, v in errors.items()},
                'limits': {k: v[1] for k, v in errors.items()}
            })
    return res


def select_best_model(rank, enable_selector, selector_id=''):
    """
    Select the best model.

    :param rank:
        Models rank.
    :type rank: list

    :param enable_selector:
        Enable the selection of the best model to predict both H/L cycles.
    :type enable_selector: bool

    :param selector_id:
        Selector id.
    :type selector_id: str

    :return:
        Best model.
    :rtype: dict
    """

    models, _map = {}, {}
    if not enable_selector:
        from co2mpas.defaults import dfl
        _map = dfl.functions.select_best_model.MAP
        _map = _map.get(selector_id, _map[None])

    for i, (score, scores, err, name, mdl) in enumerate(rank):
        k = ((sh.NONE,) if enable_selector and i == 0 else ())
        k += _map.get(name, ())
        if not k:
            continue
        success = _check_success(score)
        models.update(dict.fromkeys(k, (name, success, mdl)))
        if not success:
            log.info(
                '\n  %s: Models (%s) to predict %s failed the calibration.',
                '%s warning' % selector_id.replace('_', ' ').capitalize(),
                ','.join(sorted(set(mdl))), ','.join(k)
            )
    return models


def mdl_selector(name, package=None):
    """
    Defines a model selector for a specific model.

    :param name:
        Model name.
    :type name: str

    :param package:
        Package name.
    :type package: str

    :return:
        Model selector.
    :rtype: schedula.utils.blue.BlueDispatcher
    """
    import importlib
    mdl = importlib.import_module(name, package)
    dsp = sh.BlueDispatcher(
        name='%s selector' % mdl.name,
        description='Select the calibrated %s.' % mdl.name
    )

    err_func = sh.SubDispatch(
        _mdl_error(mdl), outputs=['errors', 'status'], output_type='list'
    )

    for data_id in calibration_cycles:
        mdl_err, mdl_err_name = _mdl_errors(mdl, data_id, err_func)
        dsp.add_func(
            sh.SubDispatchFunction(
                mdl_err, function_id=mdl_err_name,
                inputs=[data_id] + [
                    k for k in calibration_cycles if k != data_id
                ]
            ),
            outputs=['error/%s' % data_id]
        )

    # noinspection PyTypeChecker
    dsp.add_function(
        function=functools.partial(
            sort_models, weights=getattr(mdl, 'weights', None)
        ),
        inputs=list(map('error/{}'.format, calibration_cycles)),
        outputs=['rank']
    )

    dsp.add_func(format_score, outputs=['score'])

    dsp.add_func(functools.partial(
        select_best_model, selector_id='%s selector' % mdl.name
    ), outputs=['model'])

    return dsp
