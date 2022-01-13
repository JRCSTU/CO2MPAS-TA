# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to produce output report and summary.
"""
import functools
import collections
import numpy as np
import schedula as sh

dsp = sh.BlueDispatcher(
    name='make_report',
    description='Produces a vehicle report from CO2MPAS outputs.'
)


def _split_by_data_format(data):
    d = {}
    p = ('full_load_speeds', 'full_load_torques', 'full_load_powers')
    try:
        s = max(v.size for k, v in data.items()
                if k not in p and isinstance(v, np.ndarray))
    except ValueError:
        s = None

    get_d = functools.partial(
        sh.get_nested_dicts, d, default=collections.OrderedDict
    )

    for k, v in data.items():
        if isinstance(v, np.ndarray) and s == v.size:  # series
            get_d('ts')[k] = v
        else:  # params
            get_d('pa')[k] = v

    return d


def _is_equal(v, iv):
    try:
        if v == iv:
            return True
    except ValueError:
        # noinspection PyUnresolvedReferences
        if (v == iv).all():
            return True
    return False


def _re_sample_targets(data):
    res = {}
    for k, v in sh.stack_nested_keys(data.get('target', {}), depth=2):
        if sh.are_in_nested_dicts(data, 'output', *k):
            o = sh.get_nested_dicts(data, 'output', *k)
            o = _split_by_data_format(o)
            t = sh.selector(o, _split_by_data_format(v), allow_miss=True)

            if 'times' not in t.get('ts', {}) or 'times' not in o['ts']:
                t.pop('ts', None)
            else:
                time_series = t['ts']
                x, xp = o['ts']['times'], time_series.pop('times')
                if not _is_equal(x, xp):
                    for i, fp in time_series.items():
                        time_series[i] = np.interp(x, xp, fp)
            v = sh.combine_dicts(*t.values())
            sh.get_nested_dicts(res, *k[:-1])[k[-1]] = v

    return res


def _correlation_coefficient(t, o):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.corrcoef(t, o)[0, 1] if t.size > 1 else np.nan


def _prediction_target_ratio(t, o):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.mean(o / t)


@functools.lru_cache(None)
def _get_metrics():
    from co2mpas.utils import mae
    import sklearn.metrics as sk_met
    metrics = {
        'mean_absolute_error': mae,
        'correlation_coefficient': _correlation_coefficient,
        'accuracy_score': sk_met.accuracy_score,
        'prediction_target_ratio': _prediction_target_ratio
    }
    return metrics


def _compare(t, o, metrics):
    res = {}

    def _asarray(*x):
        x = np.asarray(x)
        if x.dtype is np.dtype(np.bool):
            x = np.asarray(x, dtype=int)
        return x

    try:
        t, o = _asarray(t), _asarray(o)
        for k, v in metrics.items():
            # noinspection PyBroadException
            try:
                m = v(t, o)
                if not np.isnan(m):
                    res[k] = m
            except Exception:
                pass
    except (ValueError, TypeError):
        pass
    return res


def _compare_outputs_vs_targets(data):
    res = {}
    metrics = _get_metrics()
    with np.errstate(divide='ignore', invalid='ignore'):
        for k, t in sh.stack_nested_keys(data.get('target', {}), depth=3):
            if not sh.are_in_nested_dicts(data, 'output', *k):
                continue

            o = sh.get_nested_dicts(data, 'output', *k)
            v = _compare(t, o, metrics=metrics)
            if v:
                sh.get_nested_dicts(res, *k[:-1])[k[-1]] = v

    return res


def _get_values(data, keys, tag=(), update=lambda k, v: v, base=None):
    k = ('input', 'target', 'output')
    data = sh.selector(k, data, allow_miss=True)

    base = {} if base is None else base
    for k, v in sh.stack_nested_keys(data, depth=3):
        k = k[::-1]
        v = sh.selector(keys, v, allow_miss=True)
        v = update(k, v)

        if v:
            sh.get_nested_dicts(base, *tag, *k[:-1])[k[-1]] = v

    return base


def _get_phases_values(data, what='co2_emission', base=None):
    p_wltp, p_nedc = ('low', 'medium', 'high', 'extra_high'), ('UDC', 'EUDC')
    keys = tuple('_'.join((what, v)) for v in (p_wltp + p_nedc + ('value',)))
    keys += ('phases_%ss' % what,)

    def _update(k, v):
        if keys[-1] in v:
            o = v.pop(keys[-1])
            _map = p_nedc if 'nedc' in k[0] else p_wltp
            if len(_map) != len(o):
                v.update(_format_dict(enumerate(o), '{} phase %d'.format(what)))
            else:
                v.update(_format_dict(zip(_map, o), '{}_%s'.format(what)))
        return v

    return _get_values(data, keys, tag=(what,), update=_update, base=base)


def _get_summary_results(data):
    res = {}
    for k in ('declared_co2_emission', 'corrected_co2_emission', 'co2_emission',
              'fuel_consumption', 'declared_sustaining_co2_emission',
              'corrected_sustaining_co2_emission'):
        _get_phases_values(data, what=k, base=res)
    keys = ('f0', 'f1', 'f2', 'vehicle_mass', 'gear_box_type', 'has_start_stop',
            'r_dynamic', 'ki_multiplicative', 'ki_addittive', 'fuel_type',
            'engine_capacity', 'engine_is_turbo', 'engine_max_power',
            'engine_speed_at_max_power', 'drive_battery_delta_state_of_charge',
            'service_battery_delta_state_of_charge', 'is_hybrid')
    _get_values(data, keys, tag=('vehicle',), base=res)

    return res


def _format_selection(score_by_model, depth=-1, index='model_id'):
    res = {}
    for k, v in sorted(sh.stack_nested_keys(score_by_model, depth=depth)):
        v = v.copy()
        v[index] = k[0]
        sh.get_nested_dicts(res, *k[1:], default=list).append(v)
    return res


def _get_selection(data):
    n = ('data', 'calibration', 'model_scores', 'model_selections')
    if sh.are_in_nested_dicts(data, *n):
        return _format_selection(sh.get_nested_dicts(data, *n), 3)
    return {}


def _format_report_summary(data):
    summary = {}
    comparison = _compare_outputs_vs_targets(data)
    if comparison:
        summary['comparison'] = comparison

    selection = _get_selection(data)
    if selection:
        summary['selection'] = selection

    results = _get_summary_results(data)
    if results:
        summary['results'] = results

    return summary


def _add_special_data2report(data, report, to_keys, *from_keys):
    if from_keys[-1] != 'times' and \
            sh.are_in_nested_dicts(data, *from_keys):
        v = sh.get_nested_dicts(data, *from_keys)
        n = to_keys + ('{}.{}'.format(from_keys[0], from_keys[-1]),)
        sh.get_nested_dicts(report, *n[:-1],
                            default=collections.OrderedDict)[n[-1]] = v
        return True, v
    return False, None


def _format_report_output(data):
    res = {}
    func = functools.partial(sh.get_nested_dicts,
                             default=collections.OrderedDict)
    for k, v in sh.stack_nested_keys(data.get('output', {}), depth=3):
        _add_special_data2report(data, res, k[:-1], 'target', *k)

        s, iv = _add_special_data2report(data, res, k[:-1], 'input', *k)
        if not s or (s and not _is_equal(iv, v)):
            func(res, *k[:-1])[k[-1]] = v

    output = {}
    for k, v in sh.stack_nested_keys(res, depth=2):
        v = _split_by_data_format(v)
        sh.get_nested_dicts(output, *k[:-1])[k[-1]] = v

    return output


def _format_scores(scores):
    res = {}
    for k, j in sh.stack_nested_keys(scores, depth=3):
        if k[-1] in ('limits', 'errors'):
            model_id = k[0]
            extra_field = ('score',) if k[-1] == 'errors' else ()
            for i, v in sh.stack_nested_keys(j):
                i = (model_id, i[-1], k[1],) + i[:-1] + extra_field
                sh.get_nested_dicts(res, *i[:-1])[i[-1]] = v
    sco = {}
    for k, v in sorted(sh.stack_nested_keys(res, depth=4)):
        v.update(sh.map_list(['model_id', 'param_id'], *k[:2]))
        sh.get_nested_dicts(sco, *k[2:], default=list).append(v)
    return sco


def _format_report_scores(data):
    res = {}
    scores = 'data', 'calibration', 'model_scores'
    if sh.are_in_nested_dicts(data, *scores):
        n = scores + ('param_selections',)
        v = _format_selection(sh.get_nested_dicts(data, *n), 2, 'param_id')
        if v:
            sh.get_nested_dicts(res, *n[:-1])[n[-1]] = v

        n = scores + ('model_selections',)
        v = _format_selection(sh.get_nested_dicts(data, *n), 3)
        if v:
            sh.get_nested_dicts(res, *n[:-1])[n[-1]] = v

        n = scores + ('score_by_model',)
        v = _format_selection(sh.get_nested_dicts(data, *n), 2)
        if v:
            sh.get_nested_dicts(res, *n[:-1])[n[-1]] = v

        n = scores + ('scores',)
        v = _format_scores(sh.get_nested_dicts(data, *n))
        if v:
            sh.get_nested_dicts(res, *n[:-1])[n[-1]] = v

    return res


def _map_cycle_report_graphs():
    _map = collections.OrderedDict()

    _map['fuel_consumptions'] = {
        'label': 'fuel consumption',
        'set': {
            'title': {'name': 'Fuel consumption'},
            'y_axis': {'name': 'Fuel consumption [g/s]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['engine_speeds_out'] = {
        'label': 'engine speed',
        'set': {
            'title': {'name': 'Engine speed [RPM]'},
            'y_axis': {'name': 'Engine speed [RPM]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['engine_powers_out'] = {
        'label': 'engine power',
        'set': {
            'title': {'name': 'Engine power [kW]'},
            'y_axis': {'name': 'Engine power [kW]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['velocities'] = {
        'label': 'velocity',
        'set': {
            'title': {'name': 'Velocity [km/h]'},
            'y_axis': {'name': 'Velocity [km/h]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['engine_coolant_temperatures'] = {
        'label': 'engine coolant temperature',
        'set': {
            'title': {'name': 'Engine temperature [°C]'},
            'y_axis': {'name': 'Engine temperature [°C]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['service_battery_state_of_charges'] = {
        'label': 'SOC',
        'set': {
            'title': {'name': 'Service battery state of charge [%]'},
            'y_axis': {'name': 'Service battery state of charge [%]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['service_battery_electric_powers'] = {
        'label': 'service battery electric power',
        'set': {
            'title': {'name': 'Service battery electric power [kW]'},
            'y_axis': {'name': 'Service battery electric power [kW]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['drive_battery_state_of_charges'] = {
        'label': 'SOC',
        'set': {
            'title': {'name': 'Drive battery state of charge [%]'},
            'y_axis': {'name': 'Drive battery state of charge [%]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['drive_battery_electric_powers'] = {
        'label': 'drive electric power',
        'set': {
            'title': {'name': 'Drive battery power [kW]'},
            'y_axis': {'name': 'Drive battery power [kW]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['alternator_currents'] = {
        'label': 'alternator current',
        'set': {
            'title': {'name': 'Alternator current [A]'},
            'y_axis': {'name': 'Alternator current [A]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    _map['gear_box_temperatures'] = {
        'label': 'gear box temperature',
        'set': {
            'title': {'name': 'Gear box temperature [°C]'},
            'y_axis': {'name': 'Gear box temperature [°C]'},
            'x_axis': {'name': 'Time [s]'},
            'legend': {'position': 'bottom'}
        }
    }

    return _map


# noinspection PyProtectedMember
def _get_chart_reference(report):
    from .write.excel import _sheet_name
    from .load.excel import _re_params_name
    r, _map = {}, _map_cycle_report_graphs()
    out = report.get('output', {})
    it = sh.stack_nested_keys(out, key=('output',), depth=3)
    for k, v in sorted(it):
        if k[-1] == 'ts' and 'times' in v:
            label = '{}/%s'.format(_sheet_name(k))
            for i, j in sorted(v.items()):
                param_id = _re_params_name.match(i)['param']
                m = _map.get(param_id, None)
                if m:
                    d = {
                        'x': k + ('times',),
                        'y': k + (i,),
                        'label': label % i
                    }
                    n = k[2], param_id, 'series'
                    sh.get_nested_dicts(r, *n, default=list).append(d)

    for k, v in sh.stack_nested_keys(r, depth=2):
        m = _map[k[1]]
        m.pop('label', None)
        v.update(m)

    return r


@sh.add_function(dsp, outputs=['report'])
def format_report_output_data(output_data):
    """
    Produces a vehicle output report from CO2MPAS outputs.

    :param output_data:
        CO2MPAS outputs.
    :type output_data: dict

    :return:
        Vehicle output report.
    :rtype: dict
    """
    output_data = output_data.copy()

    report = {}

    if 'pipe' in output_data:
        report['pipe'] = output_data['pipe']

    target = _re_sample_targets(output_data)
    if target:
        output_data['target'] = target

    summary = _format_report_summary(output_data)
    if summary:
        report['summary'] = summary

    output = _format_report_output(output_data)
    if output:
        report['output'] = output

    scores = _format_report_scores(output_data)
    if scores:
        sh.combine_nested_dicts(scores, base=report)

    graphs = _get_chart_reference(report)
    if graphs:
        report['graphs'] = graphs

    return report


def _extract_summary_from_summary(report, extracted, augmented_summary=False):
    n = ('summary', 'results')
    keys = (
        'corrected_sustaining_co2_emission', 'declared_sustaining_co2_emission',
        'declared_co2_emission', 'co2_emission', 'corrected_co2_emission'
    )
    if augmented_summary:
        keys += 'fuel_consumption', 'vehicle'
    if sh.are_in_nested_dicts(report, *n):
        for j, w in sh.get_nested_dicts(report, *n).items():
            if j in keys:
                for k, v in sh.stack_nested_keys(w, depth=3):
                    if v:
                        sh.get_nested_dicts(extracted, *k).update(v)
    n = ('summary', 'comparison')
    if augmented_summary and sh.are_in_nested_dicts(report, *n):
        comp = sh.get_nested_dicts(report, *n)
        for k, v in sh.stack_nested_keys(comp, key=('comparison',), depth=2):
            sh.get_nested_dicts(extracted, *k[::-1]).update({
                '/'.join(i): j for i, j in sh.stack_nested_keys(v)
            })


def _param_names_values(data):
    # noinspection PyProtectedMember
    from .load.excel import _re_params_name
    for k, v in data.items():
        m = _re_params_name.match(k)
        yield m['usage'] or 'output', m['param'], v


def _format_dict(gen, str_format='%s', func=lambda x: x):
    return {str_format % k: func(v) for k, v in gen}


def _extract_summary_from_output(report, extracted, augmented_summary=False):
    for k, v in sh.stack_nested_keys(report.get('output', {}), depth=2):
        k = k[::-1]
        for u, i, j in _param_names_values(v.get('pa', {})):
            o = {}
            if i in ('has_sufficient_power',):
                o = {i: j}
            elif augmented_summary:
                if i == 'co2_params_calibrated':
                    o = _format_dict(j.valuesdict().items(), 'co2_params %s')
                elif i == 'calibration_status':
                    o = _format_dict(
                        enumerate(j), 'status co2_params step %d',
                        lambda x: x[0]
                    )
                elif i == 'willans_factors':
                    o = j
                elif i == 'phases_willans_factors':
                    for n, m in enumerate(j):
                        o.update(_format_dict(
                            m.items(), '%s phase {}'.format(n)
                        ))
                elif i == 'co2_rescaling_scores':
                    o = sh.map_list(
                        ['rescaling_mean', 'rescaling_std', 'rescaling_n'], *j
                    )

            if o:
                sh.get_nested_dicts(extracted, *(k + (u,))).update(o)


def _extract_summary_from_model_scores(
        report, extracted, augmented_summary=False):
    n = ('data', 'calibration', 'model_scores', 'model_selections')
    if not augmented_summary or not sh.are_in_nested_dicts(report, *n):
        return False

    sel = sh.get_nested_dicts(report, *n)
    s = ('data', 'calibration', 'model_scores', 'score_by_model')
    score = sh.get_nested_dicts(report, *s)
    s = ('data', 'calibration', 'model_scores', 'scores')
    scores = sh.get_nested_dicts(report, *s)

    for k, v in sh.stack_nested_keys(extracted, depth=3):
        n = k[1::-1]
        if k[-1] == 'output' and sh.are_in_nested_dicts(sel, *n):
            gen = sh.get_nested_dicts(sel, *n)
            gen = ((d['model_id'], d['status']) for d in gen if 'status' in d)
            o = _format_dict(gen, 'status %s')
            v.update(o)
            if k[1] == 'calibration' and k[0] in score:
                gen = score[k[0]]
                gen = ((d['model_id'], d['score']) for d in gen if 'score' in d)
                o = _format_dict(gen, 'score %s')
                v.update(o)
                for i, j in scores[k[0]].items():
                    gen = (
                        ('/'.join((d['model_id'], d['param_id'])), d['score'])
                        for d in j if 'score' in d
                    )
                    o = _format_dict(gen, 'score {}/%s'.format(i))
                    v.update(o)

    return True


@sh.add_function(dsp, outputs=['summary'], inputs_kwargs=True)
def extract_summary(report, augmented_summary=False):
    """
    Extract a summary report.

    :param report:
        Vehicle output report.
    :type report: dict

    :param augmented_summary:
        Add more outputs to the summary.
    :type augmented_summary: bool

    :return:
        Summary report.
    :rtype: dict
    """
    extracted = {}

    _extract_summary_from_summary(report, extracted, augmented_summary)

    _extract_summary_from_output(report, extracted, augmented_summary)

    _extract_summary_from_model_scores(report, extracted, augmented_summary)

    return extracted
