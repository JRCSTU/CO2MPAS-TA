# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to read inputs from excel.
"""
import io
import math
import regex
import logging
import collections
import numpy as np
import os.path as osp
import schedula as sh
from collections.abc import Iterable
from xlref.parser import Ref, _re_xl_ref_parser, FILTERS

log = logging.getLogger(__name__)

_base_params = r"""
    ^((?P<scope>base)[.\s]+)?
    ((?P<usage>(target|input|output|data))s?[.\s]+)?
    ((?P<stage>(calibration|prediction))s?[.\s]+)?
    ((?P<cycle>(WLTP|NEDC|ALL)([-_][HLM])?)[.\s]+)?
    (?P<param>[^\s.]*)\s*$
    |
    ^((?P<scope>base)[.\s]+)?
    ((?P<usage>(target|input|output|data))s?[.\s]+)?
    ((?P<stage>(calibration|prediction))s?[.\s]+)?
    ((?P<param>[^\s.]*))?
    ((.|\s+)(?P<cycle>(WLTP|NEDC|ALL)([-_][HLM])?))?\s*$
"""

_flag_params = r"""
^(?P<scope>flag)[.\s]+(?P<flag>(input_version|vehicle_family_id))([.\s]+ALL)?\s*$
"""

_dice_params = r"""
    ^(?P<scope>dice)[.\s]+(?P<dice>.+)[.\s]+ALL\s*$
    |
    ^(?P<scope>dice)[.\s]+(?P<dice>.+)\s*$    
"""

_meta_params = r"""
    ^(?P<scope>meta)[.\s]+((?P<meta>.+)[.\s]+)?((?P<param>[^\s.]+))
    ((.|\s+)(?P<cycle>(WLTP|NEDC|ALL)([-_][HLM])?))?\s*$
"""

_plan_params = r"""
    ^(?P<scope>plan)[.\s]+(
     (?P<index>(id|base|run_base))\s*$
     |
""" + _flag_params.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     |
""" + _dice_params.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     |
""" + _meta_params.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     |
""" + _base_params.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     )
"""

_re_params_name = regex.compile(
    r"""
        ^(?P<param>((plan|base|flag|dice)|
                    (target|input|output|data|meta)|
                    ((calibration|prediction)s?)|
                    ((WLTP|NEDC|ALL)([-_][HLM])?)))\s*$
        |
    """ + _flag_params + r"""
        |
    """ + _dice_params + r"""
        |
    """ + _meta_params + r"""
        |
    """ + _plan_params + r"""
        |
    """ + _base_params, regex.IGNORECASE | regex.X | regex.DOTALL)

_base_sheet = r"""
    ^((?P<scope>base)([.\s]+)?)?
    ((?P<usage>(target|input|output|data))s?([.\s]+)?)?
    ((?P<stage>(calibration|prediction))s?([.\s]+)?)?
    ((?P<cycle>(WLTP|NEDC|ALL)([-_][HLM])?)([.\s]+)?)?
    (?P<type>(pa|ts|pl|mt))?\s*$
"""

_flag_sheet = r"""^(?P<scope>flag)([.\s]+(?P<type>(pa|ts|pl|mt)))?\s*$"""

_dice_sheet = r"""^(?P<scope>dice)([.\s]+(?P<type>(pa|ts|pl|mt)))?\s*$"""

_meta_sheet = r"""
    ^(?P<scope>meta)([.\s]+(?P<meta>.+))?[.\s]+(?P<type>(pa|ts|pl|mt))\s*$
"""

_plan_sheet = r"""
    ^(?P<scope>plan)([.\s]+(
""" + _flag_sheet.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     |
""" + _dice_sheet.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     |
""" + _meta_sheet.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     |
""" + _base_sheet.replace('<scope>', '<v_scope>').replace('^(', '(') + r"""
     ))?\s*$
"""

_re_input_sheet_name = regex.compile(r'|'.join(
    (_flag_sheet, _dice_sheet, _meta_sheet, _plan_sheet, _base_sheet)
), regex.IGNORECASE | regex.X | regex.DOTALL)

_re_space_dot = regex.compile(r'(\s*\.\s*|\s+)')

_xl_ref = {
    'pa': '#%s!B2:C_[{"fun": "dict", "key": "ref", "value": "ref"}]',
    'ts': '#%s!A2(R):._:RD["T", "dict"]',
    'pl': '#%s!A1(R):._:R[{"fun": "recursive", "dtype": "object"}]',
    'mt': '#%s!B1:._:R[{"fun": "recursive", "dtype": "object"}, "matrix"]'
}


def _matrix(parent, x):
    from pandas import isnull
    b, f = np.append([False], ~isnull(x[0][1:])), '{}.{}'.format
    keys = [
        None if isnull(k) else f(k.strip(' '), c.strip(' '))
        for k in x[1:, 0] for c in x[0][b]]

    return dict(zip(keys, x[1:, b].ravel()))


def _vector(parent, x):
    from pandas import notnull
    x = x.ravel()
    return x if notnull(x).any() else []


def _idict(parent, x):
    return dict(enumerate(x, 1))


def _empty(parent, x):
    return x if len(x) else None


FILTERS['empty'] = _empty
FILTERS['idict'] = _idict
FILTERS['vector'] = _vector
FILTERS['matrix'] = _matrix


class Rererence(Ref):
    _re = regex.compile(
        _re_xl_ref_parser.pattern.replace('?P<filters>', r'?P<filters>:?\s*'),
        _re_xl_ref_parser.flags
    )

    def _match(self, ref):
        d = super(Rererence, self)._match(ref)
        if (d.get('filters') or '').startswith(':'):
            d['filters'] = None
            if d['nd_col'] == d['nd_row'] == '.' and d['nd_mov']:
                d['range_exp'] = d['nd_mov'] + (d['range_exp'] or '')
                d['nd_mov'] = None
        return d


# noinspection PyShadowingBuiltins,PyUnusedLocal
def _get_sheet_type(
        type=None, usage=None, cycle=None, scope='base', **kw):
    if type:
        pass
    elif scope == 'plan':
        type = 'pl'
    elif scope in ('flag', 'dice') or not cycle:
        type = 'pa'
    else:
        type = 'ts'
    return type


def _check_none(v):
    if v is None:
        return True
    elif isinstance(v, Iterable) and not isinstance(v, str) \
            and len(v) <= 1:
        # noinspection PyTypeChecker
        return _check_none(next(iter(v))) if len(v) == 1 else True
    return False


def _isempty(val):
    return isinstance(val, float) and math.isnan(val) or _check_none(val)


# noinspection PyUnusedLocal
def _get_cycle(cycle=None, usage=None, **kw):
    if cycle is None or cycle == 'all':
        cycle = 'nedc_h', 'nedc_l', 'wltp_h', 'wltp_l', 'wltp_m'
    elif cycle == 'wltp':
        cycle = 'wltp_h', 'wltp_l', 'wltp_m'
    elif cycle == 'nedc':
        cycle = 'nedc_h', 'nedc_l'
    elif cycle in ('all-h', 'all_h'):
        cycle = 'nedc_h', 'wltp_h'
    elif cycle in ('all-l', 'all_l'):
        cycle = 'nedc_l', 'wltp_l'
    elif cycle in ('all-m', 'all_m'):
        cycle = 'wltp_m',
    elif isinstance(cycle, str):
        cycle = cycle.replace('-', '_')

    return cycle


# noinspection PyUnusedLocal
def _get_default_stage(stage=None, cycle=None, usage=None, **kw):
    if stage is None:
        if 'nedc' in cycle or usage == 'target':
            stage = 'prediction'
        else:
            stage = 'calibration'

    return stage.replace(' ', '')


def _parse_key(scope='base', usage='input', **match):
    if scope == 'flag':
        if match['flag'] == 'vehicle_family_id':
            scope = 'dice'
        yield scope, match['flag']
    elif scope == 'dice':
        yield scope, match['dice']
        if match['dice'] == 'input_type':
            yield from _parse_key(cycle='all', param='input_type')
    elif scope == 'meta':
        meta = _re_space_dot.sub(match.get('meta', ''), '.').replace('-', '_')
        param = match['param']
        if match.get('cycle', 'ALL') != 'ALL':
            param = match['cycle'] + param
        yield scope, meta, param
    elif scope == 'plan':
        if 'param' in match:
            m = _re_params_name.match('.'.join((scope, match['param'])))
            if m:
                m = {i: j for i, j in m.groupdict().items() if j}
                if 'index' in m:
                    match = m

        if 'index' in match:
            yield scope, match['index']
        else:
            for k in _parse_key(match.get('v_scope', 'base'), usage, **match):
                yield scope, '.'.join(k)
    elif scope == 'base':
        i = match['param']

        if i.lower() == 'version':
            yield 'flag', 'input_version'
        else:
            m = match.copy()
            for c in sh.stlp(_get_cycle(usage=usage, **match)):
                m['cycle'] = c
                stage = _get_default_stage(usage=usage, **m)
                yield scope, usage, stage, c, i


def _parse_values(data, default=None, where=''):
    default = default or {}
    for k, v in data.items():
        k = k.strip(' ')
        match = _re_params_name.match(k) if k is not None else None
        if not match and default.get('scope') == 'meta':
            # noinspection PyTypeChecker
            match = _re_params_name.match(
                '.'.join(filter(bool, ('meta', default.get('meta'), k)))
            )
        if not match:
            log.warning("Parameter '%s' %s cannot be parsed!", k, where)
            continue
        elif _isempty(v):
            continue
        match = {i: j.lower() for i, j in match.groupdict().items() if j}

        for key in _parse_key(**sh.combine_dicts(default, match)):
            yield key, v


def _parse_sheet(match, parent, sheet_name, res=None):
    if res is None:
        res = {}

    import pandas as pd
    sh_type = _get_sheet_type(**match)
    data = Rererence(_xl_ref[sh_type] % sheet_name, parent, parent.cache).values
    if sh_type == 'pl':
        try:
            data = pd.DataFrame(data[1:], columns=data[0])
        except IndexError:
            return None
        if 'id' not in data:
            data['id'] = data.index + 1
        else:
            data['id'] = data['id'].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

        data.set_index(['id'], inplace=True)
        data.dropna(how='all', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
    elif sh_type == 'ts':
        data = pd.DataFrame(data)
        data.dropna(how='all', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
        # noinspection PyProtectedMember
        mask = data.count(0) == len(data._get_axis(0))
        # noinspection PyUnresolvedReferences
        drop = [k for k, v in mask.items() if not v]
        if drop:
            msg = 'Columns {} in {} sheet contains nan.\n ' \
                  'Please correct the inputs!'
            raise ValueError(msg.format(drop, sheet_name))
        data = data.to_dict('list')
    else:
        import numpy as np
        data = {
            k: v for k, v in data.items()
            if k and not pd.isnull(np.ravel(v)).any()
        }

    for k, v in _parse_values(data, match, "in sheet '%s'" % sheet_name):
        sh.get_nested_dicts(res, *k[:-1])[k[-1]] = v
    return res


def _add_times_base(data, scope='base', usage='input', **match):
    if scope != 'base':
        return
    sh_type = _get_sheet_type(scope=scope, usage=usage, **match)
    n = (scope, 'target')
    if sh_type == 'ts' and sh.are_in_nested_dicts(data, *n):
        t = sh.get_nested_dicts(data, *n)
        for k, v in sh.stack_nested_keys(t, key=n, depth=2):
            if 'times' not in v:
                n = list(k + ('times',))
                n[1] = usage
                if sh.are_in_nested_dicts(data, *n):
                    v['times'] = sh.get_nested_dicts(data, *n)
                else:
                    for i, j in sh.stack_nested_keys(data, depth=4):
                        if 'times' in j:
                            v['times'] = j['times']
                            break


def _add_index_plan(plan, file_path):
    if 'base' not in plan:
        plan['base'] = file_path
    else:
        d = osp.dirname(file_path)
        plan['base'].fillna(osp.basename(file_path), inplace=True)
        plan['base'] = plan['base'].apply(
            lambda x: osp.isabs(x) and x or osp.join(d, x)
        )

    plan['base'] = plan['base'].apply(osp.normpath)

    if 'run_base' not in plan:
        plan['run_base'] = True
    else:
        plan['run_base'].fillna(True)

    plan['id'] = plan.index
    return plan


def _finalize_plan(res, plans, file_path):
    import pandas as pd
    if not plans:
        plans = (pd.DataFrame(),)

    for k, v in sh.stack_nested_keys(res.get('plan', {}), depth=4):
        n = '.'.join(k)
        m = '.'.join(k[:-1])
        for p in plans:
            if any(c.startswith(m) for c in p.columns):
                if n in p:
                    p[n].fillna(value=v, inplace=True)
                else:
                    p[n] = v

    plan = pd.concat(plans, axis=1, copy=False, verify_integrity=True)
    # noinspection PyTypeChecker
    return _add_index_plan(plan, file_path)


def parse_excel_file(input_file_name, input_file):
    """
    Reads cycle's data and simulation plans.

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :param input_file:
        Input file.
    :type input_file: io.BytesIO

    :return:
        Raw input data.
    :rtype: dict
    """
    import xlref
    import warnings
    import pandas as pd
    input_file.seek(0)
    warnings.filterwarnings(
        'ignore', 'Conditional Formatting', UserWarning, 'openpyxl'
    )
    res, plans = {'base': {}}, []
    ext = osp.splitext(input_file_name.lower())[1][1:]
    engine = xlref.Ref._engines.get(ext, xlref.Ref._engines[None])
    with pd.ExcelFile(io.BytesIO(input_file.read()), engine=engine) as xl:
        parent = Rererence('#A1')
        parent.ref['fpath'] = osp.abspath(input_file_name)
        parent.cache[parent.ref['fpath']] = parent.ref['xl_book'] = xl
        xl.sheet_indices = {k.lower(): i for i, k in enumerate(xl.sheet_names)}
        for sheet_name in xl.sheet_names:
            match = _re_input_sheet_name.match(sheet_name.strip(' '))
            if not match:
                log.debug("Sheet name '%s' cannot be parsed!", sheet_name)
                continue
            match = {k: v.lower() for k, v in match.groupdict().items() if v}
            is_plan = match.get('scope', None) == 'plan'
            if is_plan:
                r = {'plan': pd.DataFrame()}
            else:
                r = {}
            r = _parse_sheet(match, parent, sheet_name, res=r)
            if is_plan:
                plans.append(r['plan'])
            else:
                _add_times_base(r, **match)
                sh.combine_nested_dicts(r, depth=5, base=res)

    for k, v in sh.stack_nested_keys(res['base'], depth=3):
        if k[0] != 'target':
            v['cycle_type'] = v.get('cycle_type', k[-1].split('_')[0]).upper()
            v['cycle_name'] = v.get('cycle_name', k[-1]).upper()

    res['plan'] = _finalize_plan(res, plans, input_file_name).to_dict('records')

    return res
