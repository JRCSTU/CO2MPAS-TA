# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"Generate co2mparable by intercepting all Schedula function-calls and hashing args/results."
import sys
from abc import ABC, abstractmethod
from binascii import crc32
from co2mpas.model.physical.engine.co2_emission import IdleFuelConsumptionModel, FMEP
from co2mpas.model.physical.final_drive import FinalDriveModel
from co2mpas.model.physical.gear_box import GearBoxLosses, GearBoxModel
from co2mpas.model.physical.wheels import WheelsModel
from collections import defaultdict, abc
import logging
import lzma
import operator
from schedula import Dispatcher, add_args
from schedula.utils import sol, dsp
import tempfile
import types
from typing import Tuple, Sequence, Mapping, Any, Callable, Optional, Union

import contextvars
from numpy import ndarray
from pandas.core.generic import NDFrame

import functools as fnt
import toolz.dicttoolz as dtz
import weakref
import time
import os


log = logging.getLogger(__name__)

FLUSH_INTERVAL_SEC = 3
#: Env-var that when true, co2mparable is generated and optionally compared
#: against an existing co2mparable given in --co2mparable=<old-yaml>
CO2MPARE_ENABLED = 'CO2MPARE_ENABLED'
#: Env-var that when true, generate compressed co2mparable (as '.xz' with LZMA).
CO2MPARE_ZIP = 'CO2MPARE_ZIP'
#: Env-var specifying an existing yaml(.xz) co2mparable to compare while executing.
CO2MPARE_WITH_FPATH = 'CO2MPARE_WITH_FPATH'

def bool_env(env_var, default):
    "A `true` is any non-null value except: 0, false, off"
    v = os.environ.get(env_var, default)
    return v and v.lower() not in ('0', 'false', 'off')


def _convert_fun(fun):
    "hide obj-address from functions."
    return ('fun: ', fun.__name__)


def _convert_meth(fun):
    "hide obj-address from functions."
    return ('fun: ', fun.__qualname__)


def _convert_dict(d):
    "Expand into a sequence` k1, v1, k2,     ...`, to allow partials and funcs to weed out."
    return [i for pair in d.items() for i in pair]


def _convert_partial(p):
    return ('partial', p.func.__name__,
            'args', p.args,
            'kw', _convert_dict(p.keywords))


def _convert_obj(obj):
    "Used explicetely only, bc maybe str(obj) gives a better hash."
    return (type(obj).__name__, *_convert_dict(vars(obj)))


def _convert_default_dict(d):
    return (d.default_factory, *d.items())


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
    except KeyboardInterrupt:
        raise
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

        if isinstance(item, types.MethodType):
            return self.checksum(*_convert_meth(item))
        if isinstance(item, types.FunctionType):
            return self.checksum(*_convert_fun(item))

        if isinstance(item, fnt.partial):
            return self.checksum(*_convert_partial(item))

        if isinstance(item, defaultdict):
            return self.checksum(*_convert_default_dict(item))

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
        d = {n: self.checksum(base_ck, a)
             for n, a in arg_pairs
             if n not in self.args_to_exclude and
             n not in func_xargs}
        return d

    def _make_args_map(self, names, args, per_func_xargs: Sequence, expandargs=True):
        if not names:
            names = ['_item_%i' % i for i in range(len(args))]
            inp = dict(zip(names, args))
        elif len(names) == 1 and not expandargs:  # missing or single RES in "output"
            inp = {names[0]: args}
        else:
            assert len(names) >= len(args), (len(names), len(args))
            inp = dict(zip(names, args))
            # Check there were not any dupe keys.
            assert len(inp) == len(args), (len(inp), len(args))


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

        ckmap = self._args_cked(inp.items(), per_func_xargs)

        return ckmap

    def _yield_old_non_print_lines(self):
        "Skip debug lines in old file, preserving line-numbering."
        for l in self._old_ckfile:
            if not l.startswith('- PRINT'):
                yield l
            self._old_nline += 1

    def _write(self, text, *, skip_compare=False):
        "Write text and compare it against any old-file, preserving line-numbering."
        new_lines = text.split('\n')
        self._ckfile.write(text)

        if self._old_ckfile and not skip_compare:
            old_lines = [oldl
                         for _newl, oldl in zip(new_lines,
                                                self._yield_old_non_print_lines())]
            if old_lines and new_lines != old_lines:
                log.info('Comparable missmatch: NEW#L%i != OLD#L%i',
                         self._ckfile_nline, self._old_nline)

        self._ckfile_nline += len(new_lines)

    #: The schedula functions visited stored here along with
    #: their checksum of all their args, forming a tree-path.
    _checksum_stack = contextvars.ContextVar('checksum_stack', default=('', 0))
    _last_flash = time.clock()

    def eval_fun(self_sol, self, args, node_id, node_attr, attr):  # @NoSelf
        fun = node_attr['function']
        funame = dsp.parent_func(fun).__name__
        funames = {node_id, funame}

        ## Filter out Funcs or Args.
        #
        per_func_xargs = set()
        for funame in funames:
            xargs = self.funs_to_exclude.get(funame, ())
            if xargs is None:
                return self._org_eval_fun(self_sol, args, node_id, node_attr, attr)
            per_func_xargs.update(xargs)

        funpath, base_ck = self._checksum_stack.get()
        funpath = '%s/%s' % (funpath, node_id)

        ## Checksums
        #
        # if funames & self.funs_to_reset.keys():
        #     base_ck = 0
        base_ck = myck = 0  # RESET ALWAYS!
        names = node_attr.get('inputs')
        ckmap = self._make_args_map(names, args, per_func_xargs)
        #myck = self.checksum(base_ck, list(ckmap.values()))

        ## Dump comparable lines form INP.
        #
        if ckmap:
            self._write('\n' +
                               ''.join('- INP, %s, %s, %i\n' % (funpath, name, ck)
                                       for name, ck in ckmap.items()))

        ## Do nested schedula call.
        #
        res = None
        token = self._checksum_stack.set((funpath, myck))
        try:
            res = self._org_eval_fun(self_sol, args, node_id, node_attr, attr)
            return res
        finally:
            self._checksum_stack.reset(token)

            ## Dump comparable lines form OUT.
            #
            ## No == compares, res might be dataframe.
            if res is not  None:
                names = node_attr.get('outputs', ('RES', ))
                assert not isinstance(names, str), names
                ckmap = self._make_args_map(names, res, per_func_xargs, expandargs=False)
                self._write('\n' +
                                   ''.join('- OUT, %s, %s, %i\n' % (funpath, name, ck)
                                           for name, ck in ckmap.items()))

            ## Dump any collected dubugged items to print.
            #
            if self._checksum_stack.get()[0] == '':
                if self._args_printed:
                    ## Note: not compared.
                    self._write('- PRINT:\n' + ''.join(
                        '  - %s: %s\n' % (k, v)
                        for k, v in self._args_printed),
                    skip_compare=True)
                    self._args_printed.clear()

            now = time.clock()
            if now - self._last_flash > FLUSH_INTERVAL_SEC:
                self._ckfile.flush()
                self._last_flash = now

    _ckfile = None
    _ckfile_nline = 0
    _old_ckfile = None
    _old_nline = 0
    _org_eval_fun = None

    def _open_file(self, fpath, mode, *args, **kw):
        if fpath.endswith('.xz'):
            if 'w' in mode:
                FLUSH_INTERVAL_SEC = sys.maxsize  # no fun flushing zip-archives
            return lzma.open(fpath, mode, *args, **kw)
        return open(fpath, mode, *args, **kw)

    def __init__(self, *,
                 compare_with_fpath: Union[str, Path, None] = None):
        """
        Intercept schedula and open new & old files if env[CO2MPARE_ENABLED] is true.

        :param co2mpare_enabled:
            if true/false, override env[CO2MPARE_ENABLED]
        :param co2mpare_with_fpath:
            if true/false, override env[CO2MPARE_WITH_FPATH]
        - must be called only once.
        """
        global _hasher

        if _hasher:
            raise AssertionError("Already patched dsp.Solution!")

        suffix = '.yaml%s' % ('.xz' if bool_env(CO2MPARE_ZIP, True) else '')
        _fd, fpath = tempfile.mkstemp(suffix=suffix,
                                      prefix='co2cksums-',
                                      text=False)
        log.info("Writing `comparable` at: %s", fpath)
        self._ckfile = self._open_file(fpath, 'wt')
        ## close before process dies...
        weakref.finalize(self, self._ckfile.close)

        if compare_with_fpath:
            self._old_ckfile = self._open_file(compare_with_fpath, 'rt')
            ## close before process dies...
            weakref.finalize(self, self._old_ckfile.close)

        ## Intercept Schedula.
        #
        self._org_eval_fun = sol.Solution._evaluate_function
        sol.Solution._evaluate_function = fnt.partialmethod(
            ComparableHasher.eval_fun, self)  # `self` will pass as the 2nd arg

        _hasher = self


#: Global stored for :func:`checksum()` below, and to detect double-monkeypatches.
_hasher = None


def checksum(*items) -> Optional[int]:
    "Call this to hash with active hasher, or a dummy one."
    return _hasher and _hasher.checksum(*items)
