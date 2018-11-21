# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions to compare/select the CO2MPAS calibrated models.

Docstrings should provide sufficient understanding for any individual function.

Modules:

.. currentmodule:: co2mpas.model.selector

.. autosummary::
    :nosignatures:
    :toctree: selector/

    co2_params
"""

import schedula as sh
import sklearn.metrics as sk_met
import logging
import collections
import pprint
import functools
import numpy as np
import co2mpas.utils as co2_utl

log = logging.getLogger(__name__)


def _mean(values, weights=None):
    if isinstance(weights, collections.Iterable):
        values = [v * w for v, w in zip(values, weights) if w]

    v = np.asarray(values)
    return np.average(v)


def sort_models(*data, weights=None):
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

        scores = list(sorted(scores, key=_sort))
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

    return list(sorted(rank, key=_sorting_func))


def _sorting_func(x):
    return _key_score(x[0]) + _key_scores(x[1]) + (x[3],)


def _key_score(x):
    s = 1 if np.isnan(x['score']) else -int(x['success'])
    return s, -x['n'], x['score']


def _key_scores(x):
    return tuple(y[:2] for y in x)


def _check(best):
    try:
        return best[0]['success']
    except IndexError:
        return True


def _sort_rank_for_selecting_best(rank, select=(), **kwargs):
    select = tuple(k.lower().replace('-', '_') for k in sh.stlp(select))
    mw = len(select) + 1
    w = {k: v for v, k in enumerate(select)}
    rank = sorted(((w.get(m[3], mw), i), m) for i, m in enumerate(rank))
    return [v[-1] for v in rank]


def get_best_model(
        rank, settings=None, models_wo_err=None, selector_id=''):
    settings = settings or {}
    scores = collections.OrderedDict()
    rank = _sort_rank_for_selecting_best(rank, **settings)
    for m in rank:
        if m[1]:
            scores[m[3]] = {
                'score': m[0],
                'errors': {k: v[0] for k, v in m[2].items()},
                'limits': {k: v[1] for k, v in m[2].items()},
                'models': list(sorted(m[-1].keys()))
            }
        else:
            scores[m[3]] = {'models': list(sorted(m[-1].keys()))}
    if not rank:
        m = {}
    else:
        m = _select_models(rank, scores, models_wo_err, selector_id, **settings)

    return m, scores


def _select_models(rank, scores, models_wo_err, selector_id, select=None, **kw):
    select = select or {}
    indices = {0: [sh.NONE]}
    func = functools.partial(sh.get_nested_dicts, indices, default=list)
    for k, v in select.items():
        func(_select_index(rank, v)).append(k)

    models = {}
    func = functools.partial(_check_model, rank, scores, models_wo_err,
                             selector_id)

    log.debug('Scores: %s.', pprint.pformat(scores))
    for i, v in indices.items():
        models.update(dict.fromkeys(v, func(i, v)))

    return models


def _select_index(rank, select=()):
    for k in sh.stlp(select):
        if k is None:
            i = -1
            break
        cycle = k.lower().replace('-', '_')
        gen = (i for i, m in enumerate(rank) if m[3] == cycle)
        i = next(gen, sh.NONE)
        if i is not sh.NONE:
            break
    else:
        i = 0

    return i


def _check_model(rank, scores, models_wo_err, selector_id, index, cycles):
    if index < 0:
        return None, False, {}
    m = rank[index]
    s = scores[m[3]]
    models_wo_err = models_wo_err or []

    if 'score' not in s and not set(s['models']).issubset(models_wo_err):
        msg = '\n  Selection error (%s):\n' \
              '  Models %s need a score. \n' \
              '  Please report this bug to CO2MPAS team, \n' \
              '  providing the data to replicate it.'
        m = set(s['models']).difference(models_wo_err)
        raise ValueError(msg % (selector_id[:-9], str(m)))

    msg = '\n  Models %s to predict %s are selected ' \
          'from %s respect to targets %s.\n'

    log.debug(msg, s['models'], cycles, m[3], tuple(m[4].keys()))
    c = _check(m)
    if not c:
        msg = '\n  %s warning: Models %s to predict %s failed the calibration.'
        selector_name = selector_id.replace('_', ' ').capitalize()
        log.info(msg, selector_name, str(set(s['models'])), cycles)
    return m[3], c, m[-1]


def select_outputs(names, outputs, targets, results):
    results = sh.selector(outputs, results, allow_miss=True)
    results = sh.map_dict(dict(zip(outputs, targets)), results)
    it = ((k, results[v]) for k, v in zip(names, targets) if v in results)
    return collections.OrderedDict(it)


def make_metrics(metrics, ref, pred, kwargs):
    metric = collections.OrderedDict()

    for k, p in pred.items():
        if k in ref:
            m, r = metrics[k], ref[k]

            if m is not None:
                metric[k] = m(r, p, **kwargs)

    return metric


def _check_limit(limit, errors, check=lambda e, l: e <= l):
    if limit:
        l = collections.OrderedDict()
        for k, e in errors.items():
            if limit[k] is not None:
                l[k] = check(e, limit[k])
        return l


def check_limits(errors, up_limit=None, dn_limit=None):
    status = {}

    limit = _check_limit(up_limit, errors, check=lambda e, l: e <= l)
    if limit:
        status['up_limit'] = limit

    limit = _check_limit(dn_limit, errors, check=lambda e, l: e >= l)
    if limit:
        status['up_limit'] = limit

    return status


# noinspection PyUnusedLocal
def define_sub_model(d, inputs, outputs, models, **kwargs):
    missing = set(outputs).difference(d.nodes)
    if missing:
        outputs = set(outputs).difference(missing)
    if inputs is not None:
        inputs = set(inputs).union(models)
    return sh.SubDispatch(d.shrink_dsp(inputs, outputs))


# noinspection PyUnusedLocal
def metric_calibration_status(y_true, y_pred):
    return [v[0] for v in y_pred]


def metric_engine_speed_model(
        y_true, y_pred, times, velocities, gear_shifts, on_engine,
        stop_velocity):
    from ..physical.clutch_tc.clutch import calculate_clutch_phases
    b = ~calculate_clutch_phases(times, 1, 1, gear_shifts, 0, (-4.0, 4.0))
    b &= (velocities > stop_velocity) & (times > 100) & on_engine
    return sk_met.mean_absolute_error(y_true[b], y_pred[b])


def metric_engine_cold_start_speed_model(
        y_true, y_pred, cold_start_speeds_phases, engine_coolant_temperatures):
    b = cold_start_speeds_phases
    if b.any():
        t = engine_coolant_temperatures
        w = (t.max() + 1) - t[b]
        return sk_met.mean_absolute_error(y_true[b], y_pred[b], w)
    else:
        return 0


def metric_clutch_torque_converter_model(y_true, y_pred, on_engine):
    return sk_met.mean_absolute_error(y_true[on_engine], y_pred[on_engine])


def split_engine_coolant_temperatures(
        engine_coolant_temperatures, engine_thermostat_temperature):
    i = np.searchsorted(
        engine_coolant_temperatures, (engine_thermostat_temperature,)
    )[0]
    return engine_coolant_temperatures[:i], engine_coolant_temperatures[i:]


def metric_engine_coolant_temperature_model_cold(y_true, y_pred):
    if len(y_pred) > 2:
        return sk_met.mean_absolute_error(y_true[:len(y_pred)], y_pred)
    return 0


def metric_engine_coolant_temperature_model_hot(y_true, y_pred):
    if len(y_pred) > 2:
        return sk_met.mean_absolute_error(y_true[-len(y_pred):], y_pred)
    return 0


def combine_outputs(outputs):
    return {k[:-9]: v for k, v in outputs.items() if v}


def sub_models():
    models = {}

    from ..physical.engine.thermal import thermal
    models['engine_coolant_temperature_model'] = {
        'd': thermal(),
        'models': ['engine_temperature_regression_model',
                   'max_engine_coolant_temperature'],
        'inputs': ['times', 'accelerations', 'final_drive_powers_in',
                   'engine_speeds_out_hot', 'initial_engine_temperature'],
        'outputs': ['engine_coolant_temperatures'],
        'targets': ['engine_coolant_temperatures'],
        'metrics': [sk_met.mean_absolute_error],
        'up_limit': [3],
    }

    from ..physical.engine.start_stop import start_stop
    models['start_stop_model'] = {
        'd': start_stop(),
        'models': ['start_stop_model', 'use_basic_start_stop'],
        'inputs': ['times', 'velocities', 'accelerations',
                   'engine_coolant_temperatures', 'state_of_charges',
                   'gears', 'correct_start_stop_with_gears',
                   'start_stop_activation_time',
                   'min_time_engine_on_after_start', 'has_start_stop'],
        'outputs': ['on_engine', 'engine_starts'],
        'targets': ['on_engine', 'engine_starts'],
        'metrics': [sk_met.accuracy_score] * 2,
        'weights': [-1, -1],
        'dn_limit': [0.7] * 2,
    }

    from ..physical import physical

    models['engine_speed_model'] = {
        'd': physical(),
        'select_models': tyre_models_selector,
        'models': ['final_drive_ratios', 'gear_box_ratios',
                   'idle_engine_speed_median', 'idle_engine_speed_std',
                   'CVT', 'max_speed_velocity_ratio',
                   'tyre_dynamic_rolling_coefficient'],
        'inputs': ['velocities', 'gears', 'times', 'on_engine', 'gear_box_type',
                   'accelerations', 'final_drive_powers_in',
                   'engine_thermostat_temperature', 'tyre_code'],
        'outputs': ['engine_speeds_out_hot'],
        'targets': ['engine_speeds_out'],
        'metrics_inputs': ['times', 'velocities', 'gear_shifts', 'on_engine',
                           'stop_velocity'],
        'metrics': [metric_engine_speed_model],
        'up_limit': [40],
    }

    from ..physical.engine import calculate_engine_speeds_out
    from ..physical.engine.cold_start import cold_start
    d = cold_start()

    d.add_function(
        function=calculate_engine_speeds_out,
        inputs=['on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
                'cold_start_speeds_delta'],
        outputs=['engine_speeds_out']
    )

    models['engine_cold_start_speed_model'] = {
        'd': d,
        'models': ['cold_start_speed_model'],
        'inputs': ['engine_speeds_out_hot', 'engine_coolant_temperatures',
                   'on_engine', 'idle_engine_speed'],
        'outputs': ['engine_speeds_out'],
        'targets': ['engine_speeds_out'],
        'metrics_inputs': ['cold_start_speeds_phases',
                           'engine_coolant_temperatures'],
        'metrics': [metric_engine_cold_start_speed_model],
        'up_limit': [160],
    }

    from ..physical.clutch_tc import clutch_torque_converter

    d = clutch_torque_converter()

    d.add_function(
        function=calculate_engine_speeds_out,
        inputs=['on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
                'clutch_tc_speeds_delta'],
        outputs=['engine_speeds_out']
    )

    models['clutch_torque_converter_model'] = {
        'd': d,
        'models': ['clutch_window', 'clutch_model', 'torque_converter_model'],
        'inputs': ['gear_box_speeds_in', 'on_engine', 'idle_engine_speed',
                   'gear_box_type', 'gears', 'accelerations', 'times',
                   'gear_shifts', 'engine_speeds_out_hot', 'velocities',
                   'lock_up_tc_limits', 'has_torque_converter'],
        'define_sub_model': lambda d, **kwargs: sh.SubDispatch(d),
        'outputs': ['engine_speeds_out'],
        'targets': ['engine_speeds_out'],
        'metrics_inputs': ['on_engine'],
        'metrics': [metric_clutch_torque_converter_model],
        'up_limit': [100],
    }

    from ..physical.engine.co2_emission import co2_emission
    from .co2_params import co2_params_selector
    models['co2_params'] = {
        'd': co2_emission(),
        'model_selector': co2_params_selector,
        'models': ['co2_params_calibrated', 'calibration_status',
                   'initial_friction_params', 'engine_idle_fuel_consumption'],
        'inputs': ['co2_emissions_model'],
        'outputs': ['co2_emissions', 'calibration_status'],
        'targets': ['identified_co2_emissions', 'calibration_status'],
        'metrics': [sk_met.mean_absolute_error, metric_calibration_status],
        'up_limit': [0.5, None],
        'weights': [1, None]
    }

    from ..physical.electrics import electrics

    models['alternator_model'] = {
        'd': electrics(),
        'models': ['alternator_status_model', 'alternator_nominal_voltage',
                   'alternator_current_model', 'max_battery_charging_current',
                   'start_demand', 'electric_load', 'alternator_nominal_power',
                   'alternator_efficiency', 'alternator_initialization_time'],
        'inputs': [
            'battery_capacity', 'alternator_nominal_voltage',
            'initial_state_of_charge', 'times', 'gear_box_powers_in',
            'on_engine', 'engine_starts', 'accelerations'],
        'outputs': ['alternator_currents', 'battery_currents',
                    'state_of_charges', 'alternator_statuses'],
        'targets': ['alternator_currents', 'battery_currents',
                    'state_of_charges', 'alternator_statuses'],
        'metrics': [sk_met.mean_absolute_error] * 3 + [sk_met.accuracy_score],
        'up_limit': [60, 60, None, None],
        'weights': [1, 1, 0, 0]
    }

    from ..physical.gear_box.at_gear import at_gear
    at_pred_inputs = [
        'idle_engine_speed', 'full_load_curve', 'road_loads', 'vehicle_mass',
        'accelerations', 'motive_powers', 'engine_speeds_out',
        'engine_coolant_temperatures', 'time_cold_hot_transition', 'times',
        'use_dt_gear_shifting', 'specific_gear_shifting',
        'velocity_speed_ratios', 'velocities', 'MVL', 'fuel_saving_at_strategy',
        'change_gear_window_width', 'stop_velocity', 'plateau_acceleration',
        'max_velocity_full_load_correction', 'cycle_type'
    ]

    models['at_model'] = {
        'd': at_gear(),
        'select_models': functools.partial(
            at_models_selector, at_gear(), at_pred_inputs
        ),
        'models': ['MVL', 'CMV', 'CMV_Cold_Hot', 'DTGS', 'GSPV',
                   'GSPV_Cold_Hot',
                   'specific_gear_shifting', 'change_gear_window_width',
                   'max_velocity_full_load_correction', 'plateau_acceleration'],
        'inputs': at_pred_inputs,
        'define_sub_model': lambda d, **kwargs: sh.SubDispatch(d),
        'outputs': ['gears', 'max_gear'],
        'targets': ['gears', 'max_gear'],
        'metrics': [sk_met.accuracy_score, None],
        'weights': [-1, 0]
    }

    return models


def tyre_models_selector(models_ids, data):
    models = sh.selector(models_ids, data, allow_miss=True)
    if 'tyre_dynamic_rolling_coefficient' in models:
        models.pop('r_dynamic', None)
    return models


def at_models_selector(d, at_pred_inputs, models_ids, data):
    sgs = 'specific_gear_shifting'
    # Namespace shortcuts.
    try:
        vel, vsr = data['velocities'], data['velocity_speed_ratios']
        t_eng, t_gears = data['engine_speeds_out'], data['gears']
        sv, at_m = data['stop_velocity'], data[sgs]
    except KeyError:
        return {}

    t_e = ('mean_absolute_error', 'accuracy_score', 'correlation_coefficient')

    # at_models to be assessed.
    at_m = {'CMV', 'CMV_Cold_Hot', 'DTGS', 'GSPV',
            'GSPV_Cold_Hot'} if at_m == 'ALL' else {at_m}

    # Other models to be taken from calibration output.
    models = sh.selector(set(models_ids) - at_m, data, allow_miss=True)

    # Inputs to predict the gears.
    inputs = sh.selector(at_pred_inputs, data, allow_miss=True)

    from ..physical.gear_box.at_gear import calculate_error_coefficients
    from ..physical.gear_box.mechanical import calculate_gear_box_speeds_in

    def _err(model_id, model):
        gears = d.dispatch(
            inputs=sh.combine_dicts(inputs, {sgs: model_id, model_id: model}),
            outputs=['gears']
        )['gears']

        eng = calculate_gear_box_speeds_in(gears, vel, vsr, sv)
        err = calculate_error_coefficients(
            t_gears, gears, t_eng, eng, vel, sv
        )
        return err

    def _sort(v):
        e = sh.selector(t_e, v[0], output_type='list')
        return (e[0], -e[1], -e[2]), v[1]

    # Sort by error.
    rank = sorted((
        (_err(k, m), k, m)
        for k, m in sorted(sh.selector(at_m, data, allow_miss=True).items())
    ), key=_sort)

    if rank:
        data['at_scores'] = collections.OrderedDict((k, e) for e, k, m in rank)
        e, k, m = rank[0]
        models[sgs], models[k] = k, m
        log.debug('at_gear_shifting_model: %s with mean_absolute_error %.3f '
                  '[RPM], accuracy_score %.3f, and correlation_coefficient '
                  '%.3f.', k, *sh.selector(t_e, e, output_type='list'))

    return models


def split_prediction_models(
        scores, calibrated_models, input_models, cycle_ids=()):
    sbm, model_sel, par = {}, {}, {}
    for (k, c), v in sh.stack_nested_keys(scores, depth=2):
        r = sh.selector(['models'], v, allow_miss=True)

        for m in r.get('models', ()):
            sh.get_nested_dicts(par, m, 'calibration')[c] = c

        r.update(v.get('score', {}))
        sh.get_nested_dicts(sbm, k, c, default=co2_utl.ret_v(r))
        r = sh.selector(['success'], r, allow_miss=True)
        r = sh.map_dict({'success': 'status'}, r, {'from': c})
        sh.get_nested_dicts(model_sel, k, 'calibration')[c] = r

    p = {i: dict.fromkeys(input_models, 'input') for i in cycle_ids}

    models = {i: input_models.copy() for i in cycle_ids}

    for k, n in sorted(calibrated_models.items()):
        d = n.get(sh.NONE, (None, True, {}))

        for i in cycle_ids:
            c, s, m = n.get(i, d)
            if m:
                s = {'from': c, 'status': s}
                sh.get_nested_dicts(model_sel, k, 'prediction')[i] = s
                models[i].update(m)
                p[i].update(dict.fromkeys(m, c))

    for k, v in sh.stack_nested_keys(p, ('prediction',), depth=2):
        sh.get_nested_dicts(par, k[-1], *k[:-1], default=co2_utl.ret_v(v))

    s = {
        'param_selections': par,
        'model_selections': model_sel,
        'score_by_model': sbm,
        'scores': scores
    }
    return (s,) + tuple(models.get(k, {}) for k in cycle_ids)


def selector(*data, pred_cyl_ids=('nedc_h', 'nedc_l', 'wltp_h', 'wltp_l')):
    """
    Defines the models' selector model.

    .. dispatcher:: d

        >>> d = selector()

    :return:
        The models' selector model.
    :rtype: SubDispatchFunction
    """

    data = data or ('wltp_h', 'wltp_l')

    d = sh.Dispatcher(
        name='Models selector',
        description='Select the calibrated models.'
    )

    d.add_function(
        function=functools.partial(sh.map_list, data),
        inputs=data,
        outputs=['CO2MPAS_results']
    )

    d.add_data(
        data_id='models',
        function=combine_outputs,
        wait_inputs=True
    )

    d.add_data(
        data_id='scores',
        function=combine_outputs,
        wait_inputs=True
    )

    setting = sub_models()

    d.add_data(
        data_id='selector_settings',
        default_value={}
    )

    m = list(setting)
    d.add_function(
        function=functools.partial(split_selector_settings, m),
        inputs=['selector_settings'],
        outputs=['selector_settings/%s' % k for k in m]
    )

    for k, v in setting.items():
        v['names'] = v.get('names', v['targets'])
        v['dsp'] = v.pop('define_sub_model', define_sub_model)(v.pop('d'), **v)
        v['metrics'] = sh.map_list(v['names'], *v['metrics'])
        d.add_function(
            function=v.pop('model_selector', _selector)(k, data, data, v),
            function_id='%s selector' % k,
            inputs=['CO2MPAS_results', 'selector_settings/%s' % k],
            outputs=['models', 'scores']
        )

    pred_mdl_ids = ['models_%s' % k for k in pred_cyl_ids]
    d.add_function(
        function=functools.partial(split_prediction_models,
                                   cycle_ids=pred_cyl_ids),
        inputs=['scores', 'models', 'default_models'],
        outputs=['selections'] + pred_mdl_ids
    )

    func = sh.SubDispatchFunction(
        dsp=d,
        function_id='models_selector',
        inputs=('selector_settings', 'default_models') + data,
        outputs=['selections'] + pred_mdl_ids
    )

    return func


def split_selector_settings(models_ids, selector_settings):
    config = selector_settings.get('config', {})
    return tuple(config.get(k, {}) for k in models_ids)


def define_selector_settings(selector_settings, node_ids=()):
    return tuple(selector_settings.get(k, {}) for k in node_ids)


def select_targets(names, targets, data):
    return {k: data[i] for k, i in zip(names, targets) if i in data}


def _selector(name, data_in, data_out, setting):
    d = sh.Dispatcher(
        name='%s selector' % name,
        description='Select the calibrated %s.' % name
    )

    errors, setting = [], setting or {}
    _sort_models = setting.pop('sort_models', sort_models)

    if 'weights' in setting:
        _weights = sh.map_list(
            setting.get('names', setting['targets']), *setting.pop('weights')
        )
    else:
        _weights = None

    _get_best_model = functools.partial(
        setting.pop('get_best_model', get_best_model),
        models_wo_err=setting.pop('models_wo_err', None),
        selector_id=d.name
    )

    d.add_data(
        data_id='selector_settings',
        default_value={})

    node_ids = ['error_settings', 'best_model_settings']

    d.add_function(
        function=functools.partial(define_selector_settings, node_ids=node_ids),
        inputs=['selector_settings'],
        outputs=node_ids
    )

    for i in data_in:
        e = 'error/%s' % i

        errors.append(e)

        d.add_function(
            function=_errors(name, i, data_out, setting),
            inputs=['error_settings', i] + [k for k in data_out if k != i],
            outputs=[e]
        )

    d.add_function(
        function_id='sort_models',
        function=functools.partial(_sort_models, weights=_weights),
        inputs=errors,
        outputs=['rank']
    )

    d.add_function(
        function_id='get_best_model',
        function=_get_best_model,
        inputs=['rank', 'best_model_settings'],
        outputs=['model', 'errors']
    )

    return sh.SubDispatch(d, outputs=['model', 'errors'],
                          output_type='list')


def _errors(name, data_id, data_out, setting):
    d = sh.Dispatcher(
        name='%s-%s errors' % (name, data_id),
        description='Calculates the error of calibrated model.',
    )

    setting = setting.copy()

    d.add_data(
        data_id='models',
        default_value=setting.pop('models', [])
    )

    select_data = functools.partial(sh.selector, allow_miss=True)

    d.add_function(
        function_id='select_models',
        function=setting.pop('select_models', select_data),
        inputs=['models', data_id],
        outputs=['calibrated_models']
    )

    d.add_data(
        data_id='data_in',
        default_value=data_id
    )

    d.add_data(
        data_id='error_settings',
        default_value={}
    )
    err = _error(name, setting)
    for o in data_out:
        d.add_function(
            function=functools.partial(
                sh.map_list, ['calibrated_models', 'data']
            ),
            inputs=['calibrated_models', o],
            outputs=['input/%s' % o]
        )

        d.add_function(
            function=err,
            inputs=['input/%s' % o, 'error_settings'],
            outputs=['error/%s' % o]
        )

    i = ['error_settings', data_id] + [k for k in data_out if k != data_id]
    func = sh.SubDispatchFunction(
        dsp=d,
        function_id=d.name,
        inputs=i
    )

    return func


def _error(name, setting):
    d = sh.Dispatcher(
        name=name,
        description='Calculates the error of calibrated model of a reference.',
    )

    default_settings = {
        'inputs_map': {},
        'targets': [],
        'metrics_inputs': {},
        'up_limit': None,
        'dn_limit': None
    }

    default_settings.update(setting)
    default_settings['names'] = default_settings.get(
        'names', default_settings['targets']
    )

    it = sh.selector(['up_limit', 'dn_limit'], default_settings).items()

    for k, v in it:
        if v is not None:
            default_settings[k] = sh.map_list(setting['names'], *v)

    d.add_function(
        function_id='select_inputs',
        function=sh.map_dict,
        inputs=['inputs_map', 'data'],
        outputs=['inputs<0>']
    )

    d.add_function(
        function_id='select_inputs',
        function=functools.partial(sh.selector, allow_miss=True),
        inputs=['inputs', 'inputs<0>'],
        outputs=['inputs<1>']
    )

    d.add_function(
        function=sh.combine_dicts,
        inputs=['calibrated_models', 'inputs<1>'],
        outputs=['prediction_inputs']
    )

    d.add_function(
        function=select_targets,
        inputs=['names', 'targets', 'data'],
        outputs=['references']
    )

    d.add_function(
        function=functools.partial(
            default_settings.pop('dsp', lambda x: x), {}
        ),
        inputs=['prediction_inputs', 'calibrated_models'],
        outputs=['results']
    )

    d.add_function(
        function=select_outputs,
        inputs=['names', 'outputs', 'targets', 'results'],
        outputs=['predictions']
    )

    d.add_function(
        function_id='select_metrics_inputs',
        function=functools.partial(sh.selector, allow_miss=True),
        inputs=['metrics_inputs', 'data'],
        outputs=['metrics_args']
    )

    d.add_function(
        function=make_metrics,
        inputs=['metrics', 'references', 'predictions', 'metrics_args'],
        outputs=['errors']
    )

    d.add_function(
        function=check_limits,
        inputs=['errors', 'up_limit', 'dn_limit'],
        outputs=['status']
    )

    for k, v in default_settings.items():
        d.add_data(k, v)

    func = sh.SubDispatch(
        dsp=d,
        outputs=['errors', 'status'],
        output_type='list'
    )

    return func
