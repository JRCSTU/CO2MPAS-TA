# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from abc import ABC, abstractmethod
from binascii import crc32
from co2mpas.model.physical.engine.co2_emission import IdleFuelConsumptionModel, FMEP
from co2mpas.model.physical.final_drive import FinalDriveModel
from co2mpas.model.physical.gear_box import GearBoxLosses, GearBoxModel
from co2mpas.model.physical.wheels import WheelsModel
from collections import abc
import logging
import operator
from schedula import Dispatcher, add_args
from schedula.utils import sol, dsp
import tempfile
import types
from typing import Tuple, Sequence, Mapping, Any, Callable

import contextvars
from numpy import ndarray
from pandas.core.generic import NDFrame

import functools as fnt
import toolz.dicttoolz as dtz


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


class ComparableHasher(ABC):
    @property
    @abstractmethod
    def funs_to_exclude(self) -> Mapping[str, Any]:
        """
        A Map ``{<funame}: (<xarg1>, ...)}``, where
        a `None` value exclude the whole function.
        """
        ...

    @property
    @abstractmethod
    def funs_to_reset(self) -> set:
        ...

    @property
    @abstractmethod
    def args_to_exclude(self) -> set:
        ...

    @property
    @abstractmethod
    def args_to_print(self) -> set:
        ...

    @property
    @abstractmethod
    def args_to_convert(self) -> Mapping[str, Callable]:
        ...

    @property
    @abstractmethod
    def objects_to_convert(self) -> set:
        pass

    #: Collects the values for `self.args_to_print`.
    _args_printed: Tuple[str, Any] = []

    def _hash(self, item):
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
    #         return fnt.reduce(operator.xor, (self._hash(i) for i in item), 1)
    #
    #     if isinstance(item, abc.Mapping):
    #         return fnt.reduce(operator.xor, (self._hash(i) for i in item.items()
    #                                          if i[0] not in self.args_to_exclude), 3)

        if isinstance(item, fnt.partial):
            return self.checksum(*_convert_partial(item))

        if isinstance(item, types.MethodType):
            return self.checksum(*_convert_meth(item))
        if isinstance(item, types.FunctionType):
            return self.checksum(*_convert_fun(item))

        if isinstance(item, self.objects_to_convert):
            return self.checksum(*_convert_obj(item))

        return crc32(_to_bytes(item))

    def checksum(self, *items) -> int:
        return fnt.reduce(operator.xor, (self._hash(i) for i in items), 1)

    def _args_cked(self,
                  arg_pairs: Mapping,
                  func_xargs: Sequence[Tuple[str, Any]],
                  *,
                  base_ck=0):
        d = {n: self.checksum(base_ck, [a])
             for n, a in arg_pairs
             if n not in self.args_to_exclude and
             n not in func_xargs}
        return d

    def _make_args_map(self, node, nattr, args, per_funx_xargs: Sequence):
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
        to_proc = self.args_to_convert.keys() & inp.keys()
        if to_proc:
            for k in to_proc:
                inp[k] = self.args_to_convert[k](inp[k])

        ## Debug-print certain args.
        #
        for name in self.args_to_print & inp.keys():
            self._args_printed.append((name, inp[name]))

        ckmap = self._args_cked(inp.items(), per_funx_xargs)

        return ckmap

    #: The schedula functions visited stored here along with
    #: their checksum of all their args, forming a tree-path.
    _checksum_stack = contextvars.ContextVar('checksum_stack', default=('', 0))

    def eval_fun(self_sol, self, args, node_id, node_attr, attr):  # @NoSelf
        fun = node_attr['function']
        funame = dsp.parent_func(fun).__name__

        ## Filter out Funcs or Args.
        #
        per_funx_xargs = self.funs_to_exclude.get(funame, ())
        if per_funx_xargs is None:
            return self._org_eval_fun(self_sol, args, node_id, node_attr, attr)

        funpath, base_ck = self._checksum_stack.get()
        funpath = '%s/%s' % (funpath, funame)

        ## Checksums
        #
        if funame in self.funs_to_reset:
            base_ck = 0
        ckmap = self._make_args_map(node_attr, 'inputs', args, per_funx_xargs)
        myck = self.checksum(base_ck, list(ckmap.values()))

        ## Write comparable lines.
        #
        self._ckfile.write('\n- %s: %s\n' % (funpath, myck))
        if ckmap:
            self._ckfile.write('  ARGS:\n' +
                               ''.join('    - %s: %s\n' % (name, ck)
                                       for name, ck in ckmap.items()))

        ## Do nested schedula call.
        #
        res = FAIL = object() # sentinel
        token = self._checksum_stack.set((funpath, myck))
        try:
            res = self._org_eval_fun(self_sol, args, node_id, node_attr, attr)
            return res
        finally:
            self._checksum_stack.reset(token)

            ## No == compares, res might be dataframe.
            if res is not None and res is not FAIL:
                ## Checksum based on pre-func checksum `base_ck` (and not `myck`),
                #  to detect differences separately from input-args.
                myck = self.checksum(base_ck, res)

                names = node_attr.get('outputs', ())
                outname = '(%s)' % names[0] if len(names) == 1 else ''
                self._ckfile.write('  OUT%s: %s\n' % (outname, myck))
                if len(names) > 1:
                    ckmap = self._make_args_map(node_attr, 'outputs', res, per_funx_xargs)
                    self._ckfile.write('  RES:\n' + ''.join(
                        '    - %s: %s\n' % (name, ck)
                        for name, ck in ckmap.items()))

            if self._checksum_stack.get()[0] == '':
                if self._args_printed:
                    self._ckfile.write('- PRINTED:\n' + ''.join(
                        '  - %s: %s\n' % (k, v)
                        for k, v in self._args_printed))
                    self._args_printed.clear()
                self._ckfile.flush()

    _ckfile = None
    _org_eval_fun = None

    def __init__(self):
        global _hasher

        if _hasher:
            raise AssertionError("Already patched dsp.Solution!")

        self._org_eval_fun = sol.Solution._evaluate_function
        _fd, fpath = tempfile.mkstemp(suffix='.yaml',
                                      prefix='co2funcsums-',
                                      text=False)
        log.info("Writing `comparable` at: %s", fpath)
        self._ckfile = open(fpath, 'w')  # will close when process die...
        sol.Solution._evaluate_function = fnt.partialmethod(
            ComparableHasher.eval_fun, self)  # `self` will pass as the 2nd arg
        _hasher = self


#: Global stored for :func:`checksum()` below, and to detect double-monkeypatches.
_hasher = None


def checksum(*items):
    "Call this to hash with active hasher, or a dummy one."
    return _hasher and _hasher.checksum(*items) or -1
