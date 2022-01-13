# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to convert CO2MPAS output report to DataFrames.
"""
import regex
import functools
import itertools
import logging
import collections
import schedula as sh

log = logging.getLogger(__name__)
_re_units = regex.compile(r'(\[.*\])')


def _parse_name(name, _standard_names=None):
    """
    Parses a column/row name.

    :param name:
        Name to be parsed.
    :type name: str

    :return:
        The parsed name.
    :rtype: str
    """

    if _standard_names and name in _standard_names:
        return _standard_names[name]

    return name.replace('_', ' ').capitalize()


@functools.lru_cache(None)
def _get_doc_description():
    from ..model.physical import dsp

    doc_descriptions = {}

    d = dsp.register(memo={})
    for k, v in d.data_nodes.items():
        if k in doc_descriptions or v['type'] != 'data':
            continue
        des = d.get_node(k, node_attr='description')[0]
        if not des or len(des.split(' ')) > 4:

            unit = _re_units.search(des)
            if unit:
                unit = ' %s' % unit.group()
            else:
                unit = ''
            doc_descriptions[k] = '%s%s.' % (_parse_name(k), unit)
        else:
            doc_descriptions[k] = des
    return doc_descriptions


def _param_parts(param_id):
    # noinspection PyProtectedMember
    from ..load.excel import _re_params_name
    match = _re_params_name.match(param_id).groupdict().items()
    return {i: regex.sub(r"[\W]", "_", (j or '').lower()) for i, j in match}


def _time_series2df(data, data_descriptions):
    df = collections.OrderedDict()
    for k, v in data.items():
        df[(_parse_name(_param_parts(k)['param'], data_descriptions), k)] = v
    import pandas as pd
    return pd.DataFrame(df)


def _parameters2df(data, data_descriptions, write_validator):
    import schema
    df = []
    for k, v in data.items():
        try:
            param_id, vl = write_validator(_param_parts(k)['param'], v)
            if vl is not sh.NONE:
                df.append({
                    'Parameter': _parse_name(param_id, data_descriptions),
                    'Model Name': k,
                    'Value': vl
                })
        except schema.SchemaError as ex:
            raise ValueError(k, v, ex)

    if df:
        import pandas as pd
        df = pd.DataFrame(df)
        df.set_index(['Parameter', 'Model Name'], inplace=True)
        return df
    else:
        return None


def _cycle2df(data):
    res, out = {}, data.get('output', {})
    from .excel import _sheet_name
    from ..load.schema import define_data_validation
    write_validator = define_data_validation(read=False)
    data_descriptions = _get_doc_description()
    for k, v in sh.stack_nested_keys(out, key=('output',), depth=3):
        n, k = _sheet_name(k), k[-1]
        if 'ts' == k:
            df = _time_series2df(v, data_descriptions)
        elif 'pa' == k:
            df = _parameters2df(v, data_descriptions, write_validator)
        else:
            continue

        if df is not None:
            res[n] = df
    return res


def _dd2df(dd, index=None, depth=0, col_key=None, row_key=None):
    frames = []
    import inspect
    import pandas as pd
    for k, v in sh.stack_nested_keys(dd, depth=depth):
        df = pd.DataFrame(v)
        df.drop_duplicates(subset=index, inplace=True)
        if index is not None:
            df.set_index(index, inplace=True)

        df.columns = pd.MultiIndex.from_tuples([k + (i,) for i in df.columns])
        frames.append(df)

    df = pd.concat(frames, **sh.selector(
        inspect.getfullargspec(pd.concat).args,
        dict(copy=False, axis=1, verify_integrity=True, sort=False),
        allow_miss=True
    ))

    if col_key is not None:
        ax = sorted(df.columns, key=col_key)
        if isinstance(df.columns, pd.MultiIndex):
            ax = pd.MultiIndex.from_tuples(ax)

        # noinspection PyUnresolvedReferences
        df = df.reindex(ax, axis='columns', copy=False)

    if row_key is not None:
        ax = sorted(df.index, key=row_key)
        if isinstance(df.index, pd.MultiIndex):
            ax = pd.MultiIndex.from_tuples(ax)
        df = df.reindex(ax, axis='index', copy=False)

    if index is not None:
        if len(index) == 1 and isinstance(df.index, pd.MultiIndex):
            df.index = pd.Index(df.index, name=index[0])
        else:
            df.index.set_names(index, inplace=True)

    return df


@functools.lru_cache(None)
def _param_orders():
    x = ('declared_co2_emission', 'co2_emission', 'fuel_consumption')
    y = ('low', 'medium', 'high', 'extra_high', 'UDC', 'EUDC', 'value')
    param = x + tuple(map('_'.join, itertools.product(x, y))) + ('status',)

    param += (
        'av_velocities', 'distance', 'init_temp', 'av_temp', 'end_temp',
        'av_vel_pos_mov_pow', 'av_pos_motive_powers',
        'av_missing_powers_pos_pow', 'sec_pos_mov_pow', 'av_neg_motive_powers',
        'sec_neg_mov_pow', 'av_pos_accelerations',
        'av_engine_speeds_out_pos_pow', 'av_pos_engine_powers_out',
        'engine_bmep_pos_pow', 'mean_piston_speed_pos_pow', 'fuel_mep_pos_pow',
        'fuel_consumption_pos_pow', 'willans_a', 'willans_b',
        'specific_fuel_consumption', 'indicated_efficiency',
        'willans_efficiency', 'times'
    )

    _map = {
        'scope': ('plan', 'flag', 'base'),
        'usage': ('target', 'output', 'input', 'data'),
        'stage': ('precondition', 'prediction', 'calibration'),
        'cycle': (
            'all', 'nedc_h', 'nedc_l', 'wltp_h', 'wltp_l','wltp_m', 'wltp_p'
        ),
        'type': ('pa', 'ts', 'pl', 'mt'),
        'param': param
    }
    _map = {k: {j: str(i).zfill(3) for i, j in enumerate(v)}
            for k, v in _map.items()}

    return _map


# noinspection PyShadowingBuiltins
def _match_part(map, *parts, default=None):
    part = parts[-1]
    try:
        return map[part],
    except KeyError:
        for k, v in sorted(map.items(), key=lambda x: x[1]):
            if k in part:
                return v, 0, part
        part = part if default is None else default
        if len(parts) <= 1:
            return max(map.values()) if map else None, 1, part
        return _match_part(map, *parts[:-1], default=part)


def _sort_key(
        parts, score_map=None,
        p_keys=('scope', 'param', 'cycle', 'usage', 'stage', 'type')):
    score_map = score_map or _param_orders()
    it = itertools.zip_longest(parts, p_keys, fillvalue=None)
    return tuple(_match_part(score_map.get(k, {}), p) for p, k in it)


def _scores2df(data):
    n = ('data', 'calibration', 'model_scores')
    if not sh.are_in_nested_dicts(data, *n):
        return {}

    scores = sh.get_nested_dicts(data, *n)

    it = (('model_selections', ['model_id'], 2, ('stage', 'cycle'), ()),
          ('score_by_model', ['model_id'], 1, ('cycle',), ()),
          ('scores', ['model_id', 'param_id'], 2, ('cycle', 'cycle'), ()),
          ('param_selections', ['param_id'], 2, ('stage', 'cycle'), ()))
    dfs = []
    for k, idx, depth, col_keys, row_keys in it:
        if k not in scores:
            continue
        df = _dd2df(
            scores[k], idx, depth=depth,
            col_key=functools.partial(_sort_key, p_keys=col_keys),
            row_key=functools.partial(_sort_key, p_keys=row_keys)
        )
        setattr(df, 'name', k)
        dfs.append(df)
    if dfs:
        return {'.'.join(n): dfs}
    else:
        return {}


@functools.lru_cache(None)
def _summary_map(short=True):
    keys = (
        'declared_sustaining_co2_emission_value', 'declared_co2_emission_value',
        'corrected_sustaining_co2_emission_value', 'co2_emission_value',
        'corrected_co2_emission_value',
    )
    if short:
        _map = {k: 'value' for k in keys}
    else:
        _map = {k: k.replace('co2_emission_', '') for k in keys}

    _map.update({
        'co2_params a': 'a',
        'co2_params a2': 'a2',
        'co2_params b': 'b',
        'co2_params b2': 'b2',
        'co2_params c': 'c',
        'co2_params l': 'l',
        'co2_params l2': 'l2',
        'co2_params t0': 't0',
        'co2_params dt': 'dt',
        'co2_params t1': 't1',
        'co2_params trg': 'trg',
        'co2_emission_low': 'low',
        'co2_emission_medium': 'medium',
        'co2_emission_high': 'high',
        'co2_emission_extra_high': 'extra_high',
        'co2_emission_UDC': 'UDC',
        'co2_emission_EUDC': 'EUDC',
        'vehicle_mass': 'mass',
    })

    return _map


def _search_unit(units, default, *keys):
    try:
        return units[keys[-1]]
    except KeyError:
        try:
            return _search_unit(units, sh.EMPTY, *keys[:-1])
        except IndexError:
            if default is sh.EMPTY:
                raise IndexError
            for i, u in units.items():
                if any(i in k for k in keys):
                    return u
            return default


@functools.lru_cache(None)
def _param_units():
    units = (
        (k, _re_units.search(v)) for k, v in _get_doc_description().items()
    )
    units = {k: v.group() for k, v in units if v}
    units.update({
        'co2_params a': '[-]',
        'co2_params b': '[s/m]',
        'co2_params c': '[(s/m)^2]',
        'co2_params a2': '[1/bar]',
        'co2_params b2': '[s/(bar*m)]',
        'co2_params l': '[bar]',
        'co2_params l2': '[bar*(s/m)^2]',
        'co2_params t': '[-]',
        'co2_params trg': '[°C]',
        'fuel_consumption': '[l/100km]',
        'co2_emission': '[CO2g/km]',
        'declared_co2_emission_value': '[CO2g/km]',
        'av_velocities': '[kw/h]',
        'av_vel_pos_mov_pow': '[kw/h]',
        'av_pos_motive_powers': '[kW]',
        'av_neg_motive_powers': '[kW]',
        'distance': '[km]',
        'init_temp': '[°C]',
        'av_temp': '[°C]',
        'end_temp': '[°C]',
        'sec_pos_mov_pow': '[s]',
        'sec_neg_mov_pow': '[s]',
        'av_pos_accelerations': '[m/s2]',
        'av_engine_speeds_out_pos_pow': '[RPM]',
        'av_pos_engine_powers_out': '[kW]',
        'engine_bmep_pos_pow': '[bar]',
        'mean_piston_speed_pos_pow': '[m/s]',
        'fuel_mep_pos_pow': '[bar]',
        'fuel_consumption_pos_pow': '[g/sec]',
        'willans_a': '[g/kW]',
        'willans_b': '[g]',
        'specific_fuel_consumption': '[g/kWh]',
        'indicated_efficiency': '[-]',
        'willans_efficiency': '[-]',
    })

    return units


def _add_units(gen, default=' ', short=True):
    p_map = _summary_map(short=short).get
    units = functools.partial(_search_unit, _param_units(), default)
    return [k[:-1] + (p_map(k[-1], k[-1]), units(*k)) for k in gen]


def _summary2df(data):
    res = []
    summary = data.get('summary', {})

    if 'results' in summary:
        r = {}
        index = ['cycle', 'stage', 'usage']

        for k, v in sh.stack_nested_keys(summary['results'], depth=4):
            l = sh.get_nested_dicts(r, k[0], default=list)
            l.append(sh.combine_dicts(sh.map_list(index, *k[1:]), v))

        if r:
            df = _dd2df(
                r, index=index, depth=2,
                col_key=functools.partial(_sort_key, p_keys=('param',) * 2),
                row_key=functools.partial(_sort_key, p_keys=index)
            )
            import pandas as pd
            df.columns = pd.MultiIndex.from_tuples(_add_units(df.columns))
            setattr(df, 'name', 'results')
            res.append(df)

    if 'selection' in summary:
        df = _dd2df(
            summary['selection'], ['model_id'], depth=2,
            col_key=functools.partial(_sort_key, p_keys=('stage', 'cycle')),
            row_key=functools.partial(_sort_key, p_keys=())
        )
        setattr(df, 'name', 'selection')
        res.append(df)

    if 'comparison' in summary:
        r = {}
        for k, v in sh.stack_nested_keys(summary['comparison'], depth=3):
            v = sh.combine_dicts(v, base={'param_id': k[-1]})
            sh.get_nested_dicts(r, *k[:-1], default=list).append(v)
        if r:
            df = _dd2df(
                r, ['param_id'], depth=2,
                col_key=functools.partial(_sort_key, p_keys=('stage', 'cycle')),
                row_key=functools.partial(_sort_key, p_keys=())
            )
            setattr(df, 'name', 'comparison')
            res.append(df)

    if res:
        return {'summary': res}
    return {}


def _co2mpas_info2df(start_time, main_flags=None):
    import socket
    import datetime
    from co2mpas import __version__
    from ..load.schema import define_flags_validation
    time_elapsed = (datetime.datetime.today() - start_time).total_seconds()
    hostname = socket.gethostname()
    info = [
        ('CO2MPAS version', __version__),
        ('Simulation started', start_time.strftime('%Y/%m/%d-%H:%M:%S')),
        ('Time elapsed', '%.3f sec' % time_elapsed),
        ('Hostname', hostname),
    ]

    if main_flags:
        validate = define_flags_validation(read=False)
        main_flags = dict(validate(k, v) for k, v in main_flags.items())
        info.extend(sorted(main_flags.items()))
    import pandas as pd
    df = pd.DataFrame(info, columns=['Parameter', 'Value'])
    df.set_index(['Parameter'], inplace=True)
    setattr(df, 'name', 'info')
    return df


_re_list = regex.compile(br'^[^\[]*(?P<list>\[.*\])[^\]]*$', regex.DOTALL)


@functools.lru_cache()
def _get_installed_packages():
    import json
    from subprocess import check_output
    try:
        out = check_output("conda list --json".split())
        m = _re_list.match(out)
        if m:
            return json.loads(m.group('list'))
        msg = 'Invalid JSON!'
    except Exception as ex:
        msg = ex
    log.info("Failed collecting installation info.\n%s", msg)


def _pipe2list(pipe, i=0, source=()):
    res, max_l = [], i
    idx = {'nodes L%d' % i: str(v) for i, v in enumerate(source)}
    node_id = 'nodes L%d' % i
    for k, v in pipe.items():
        k = sh.stlp(k)
        d = {node_id: str(k)}

        if 'error' in v:
            d['error'] = v['error']

        j, s = v['task'][2]
        n = s.workflow.nodes.get(j, {})
        if 'duration' in n:
            d['duration'] = n['duration']

        d.update(idx)
        res.append(d)

        if 'sub_pipe' in v:
            l, ml = _pipe2list(v['sub_pipe'], i=i + 1, source=source + (k,))
            max_l = max(max_l, ml)
            res.extend(l)

    return res, max_l


def _proc_info2df(data, start_time, main_flags):
    res = _co2mpas_info2df(start_time, main_flags),

    df = _get_installed_packages()
    if df:
        import pandas as pd
        df = pd.DataFrame(df).set_index('name')
        setattr(df, 'name', 'packages')
        res += df,

    df, max_l = _pipe2list(data.get('pipe', {}))

    if df:
        import pandas as pd
        df = pd.DataFrame(df)
        setattr(df, 'name', 'pipe')
        res += df,

    return {'proc_info': res}


def _dice2df(dice):
    if dice:
        import pandas as pd
        df = pd.DataFrame(sorted(dice.items()), columns=['Parameter', 'Value'])
        df.set_index(['Parameter'], inplace=True)
        setattr(df, 'name', 'dice')
        return {'dice': [df]}
    return {}


def convert2df(report, start_time, flag, dice):
    """
    Convert vehicle output report to DataFrames.

    :param report:
        Vehicle output report.
    :type report: dict

    :param start_time:
        Run start time.
    :type start_time: datetime.datetime

    :param flag:
        Command line flags.
    :type flag: dict

    :param dice:
        DICE data.
    :type dice: dict

    :return:
        DataFrames of vehicle output report.
    :rtype: dict[str, pandas.DataFrame]
    """
    res = {'graphs.%s' % k: v for k, v in report.get('graphs', {}).items()}

    res.update(_cycle2df(report))

    res.update(_scores2df(report))

    res.update(_summary2df(report))

    res.update(_dice2df(dice))

    res.update(_proc_info2df(report, start_time, flag or {}))

    res['summary'] = [res['proc_info'][0]] + res.get('summary', [])

    return res
