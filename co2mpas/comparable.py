# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from binascii import crc32
from collections import abc
from schedula import Dispatcher, add_args
from schedula.utils import sol, dsp
import tempfile
import types
from typing import Tuple, Sequence, Mapping, Any
import operator

import contextvars
from numpy import ndarray
from pandas.core.generic import NDFrame

import functools as fnt
import toolz.dicttoolz as dtz
from co2mpas.model.physical.gear_box import GearBoxLosses, GearBoxModel
from co2mpas.model.physical.engine.co2_emission import IdleFuelConsumptionModel, FMEP
import logging
from co2mpas.model.physical.wheels import WheelsModel
from co2mpas.model.physical.final_drive import FinalDriveModel


log = logging.getLogger(__name__)

def _convert_fun(fun):
    "hide obj-address from functions."
    return ('fun: ', fun.__name__)

def _convert_meth(fun):
    "hide obj-address from functions."
    return ('fun: ', fun.__qualname__)


def _convert_dict(d):
    "Expand into a sequence, to allow partials and funcs to weed out."
    return [i for pair in d.items() for i in pair]


def _convert_partial(p):
    return ('partial', p.func.__name__,
            'args', p.args,
            'kw', _convert_dict(p.keywords))


def _convert_obj(obj):
    "Used explicetely only, bc maybe str(obj) gives a better hash."
    return (type(obj).__name__, *_convert_dict(vars(obj)))


def _remove_timestamp_from_plan(item):
    notstamp = lambda k: k != 'timestamp'
    item = dtz.keyfilter(notstamp, item)
    item['flag'] = dtz.keyfilter(notstamp, item['flag'])

    return item


def _convert_interp_partial_in_fmep(item):
    item = vars(item).copy()
    item['fbc'] = _convert_partial(item['fbc'])
    return item


def _convert_fmep_in_idle(item):
    item = vars(item).copy()
    item['fmep_model'] = _convert_interp_partial_in_fmep(item['fmep_model'])
    return item


###################################
## COMPARATOR EDITABLE CONTENT

#: Map
#:    {<funame}: (<xarg1>, ...)}
#: A `None` value exclude the whole function.
funs_to_exclude = {
    #'get_cache_fpath': None,
    'default_start_time': None,
    'default_timestamp': None,
}
funs_to_reset = {
}
args_to_exclude = {
    '_',  # make it a set, even when items below missing
    'output_folder',
    'vehicle_name',
    'output_file_name',
    'timestamp',
    'start_time',
    'output_file_name',     # contains timestamp
    'gear_filter',          # a function
    'tau_function',         # a function
    'k_factor_curve',       # a function
    'full_load_curve',      # an InterpolatedUnivariateSpline
}
args_to_print = {
    '_',
    'full_bmep_curve',
    'correct_gear',
    'error_function_on_emissions',
}
args_to_convert = {
    'base_data': _remove_timestamp_from_plan,
    'plan_data': _remove_timestamp_from_plan,
    #'gear_box_loss_model': _convert_obj,
    'correct_gear': _convert_obj,
    'idle_fuel_consumption_model': _convert_fmep_in_idle,
    'fmep_model': _convert_interp_partial_in_fmep,
}
objects_to_convert = (
    Dispatcher, add_args,
    GearBoxLosses,
    GearBoxModel,
    IdleFuelConsumptionModel,
    FMEP,
    WheelsModel,
    FinalDriveModel,
)
#
###################################

#: Collects the values for `args_to_print`.
args_printed: Tuple[str, Any] = []
checksum_stack = contextvars.ContextVar('checksum_stack', default=('', 0))


def _to_bytes(item) -> bytes:
    try:
        if item is None:
            return b'\0'

        if isinstance(item, ndarray):
            return item.tobytes()

        if isinstance(item, NDFrame):
            return item.values.tobytes()

        return str(item).encode(errors='ignore')
    except Exception as ex:
        print("CKerr: %s\n  %s" % (ex, item))
        raise


def _hash(item):
    if item is None:
        return 0

    if hasattr(item, 'co2hash'):
        return item.co2hash

#     if isinstance(item, str):
#         return crc32(_to_bytes(item))
#
#     if isinstance(item, abc.Sequence):
#         if not item:
#             return crc32(_to_bytes(item))
#         return fnt.reduce(operator.xor, (_hash(i) for i in item), 1)
#
#     if isinstance(item, abc.Mapping):
#         return fnt.reduce(operator.xor, (_hash(i) for i in item.items()
#                                          if i[0] not in args_to_exclude), 3)

    if isinstance(item, fnt.partial):
        return checksum(*_convert_partial(item))

    if isinstance(item, types.MethodType):
        return checksum(*_convert_meth(item))
    if isinstance(item, types.FunctionType):
        return checksum(*_convert_fun(item))

    if isinstance(item, objects_to_convert):
        return checksum(*_convert_obj(item))

    return crc32(_to_bytes(item))


def checksum(*items) -> int:
    return fnt.reduce(operator.xor, (_hash(i) for i in items), 1)


def args_cked(arg_pairs: Mapping, func_xargs: Sequence[Tuple[str, Any]], *,
              base_ck=0):
    d = {n: checksum(base_ck, [a])
         for n, a in arg_pairs
         if n not in args_to_exclude and
         n not in func_xargs}
    return d


def make_args_map(node, nattr, args, per_funx_xargs: Sequence):
    if nattr not in node:
        ## data-nodes have not `inputs`
        names = ['_%s_%i' % (nattr, i) for i in range(len(args))]
    else:
        names = node[nattr]
    assert len(names) >= len(args), (names, args)

    inp = dict(zip(names, args))
    # Check there were not any dupe keys.
    assert len(inp) == len(args), (inp, args)

    ## Process certain args.
    #
    to_proc = args_to_convert.keys() & inp.keys()
    if to_proc:
        for k in to_proc:
            inp[k] = args_to_convert[k](inp[k])

    ## Debug-print certain args.
    #
    for name in args_to_print & inp.keys():
        args_printed.append((name, inp[name]))

    ckmap = args_cked(inp.items(), per_funx_xargs)

    return ckmap


_ckfile = None


def eval_fun(self_sol, args, node_id, node_attr, attr):
    fun = node_attr['function']
    funame = dsp.parent_func(fun).__name__

    per_funx_xargs = funs_to_exclude.get(funame, ())
    if per_funx_xargs is None:
        return _original_eval_fun(self_sol, args, node_id, node_attr, attr)

    ckmap = make_args_map(node_attr, 'inputs', args, per_funx_xargs)

    funpath, base_ck = checksum_stack.get()
    funpath = '%s/%s' % (funpath, funame)
    if funame in funs_to_reset:
        base_ck = 0

    myck = checksum(base_ck, list(ckmap.values()))
    _ckfile.write('\n- %s: %s\n' % (funpath, myck))
    if ckmap:
        _ckfile.write('  ARGS:\n' + ''.join(
            '    - %s: %s\n' % (name, ck)
            for name, ck in ckmap.items()))

    token = checksum_stack.set((funpath, myck))
    res = FAIL = object()
    try:
        res = _original_eval_fun(self_sol, args, node_id, node_attr, attr)
        return res
    finally:
        checksum_stack.reset(token)

        ## warn when comparing, res might be dataframe.
        if res is not None and res is not FAIL:
            ## Checksum based on pre-func checksum `ck` (and not `myck`),
            #  to detect differences separately from input-args.
            myck = checksum(base_ck, res)

            names = node_attr.get('outputs', ())
            outname = '(%s)' % names[0] if len(names) == 1 else ''
            _ckfile.write('  OUT%s: %s\n' % (outname, myck))
            if len(names) > 1:
                ckmap = make_args_map(node_attr, 'outputs', res, per_funx_xargs)
                _ckfile.write('  RES:\n' + ''.join(
                    '    - %s: %s\n' % (name, ck)
                    for name, ck in ckmap.items()))

        if checksum_stack.get()[0] == '':
            if args_printed:
                _ckfile.write('- PRINTED:\n' + ''.join(
                    '  - %s: %s\n' % (k, v)
                    for k, v in args_printed))
                args_printed.clear()
            _ckfile.flush()


_original_eval_fun = sol.Solution._evaluate_function


def monkeypatch_schedula():
    global _ckfile
    _fd, fpath = tempfile.mkstemp(suffix='.yaml',
                                 prefix='co2funcsums-',
                                 text=False)
    log.info("Writting `comparable` at: %s", fpath)
    _ckfile = open(fpath, 'w')  # will close when process die...
    sol.Solution._evaluate_function = eval_fun
