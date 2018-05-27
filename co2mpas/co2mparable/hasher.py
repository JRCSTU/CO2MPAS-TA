# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"Generate co2mparable by intercepting all Schedula function-calls and hashing args/results."
from binascii import crc32
from collections import defaultdict
import logging
import lzma
import operator
from pathlib import Path  # @UnusedImport
import re
from schedula.utils import sol, dsp
import sys
import tempfile
import time
import types
from typing import Tuple, List, Sequence, Mapping, Any, Pattern, \
    Optional, Union  # @UnusedImport
import weakref

import contextvars
from numpy import ndarray
from pandas.core.generic import NDFrame
from toolz import dicttoolz as dtz

import functools as fnt

from . import CO2MPARE_DEBUG, bool_env


log = logging.getLogger(__name__)

FLUSH_INTERVAL_SEC = 3

def _convert_fun(fun):
    "hide obj-address from functions."
    return ('fun: ', fun.__name__)


def _convert_meth(fun):
    "hide obj-address from functions."
    return ('meth: ', fun.__qualname__)


def _convert_dict(d):
    "Expand into a sequence` k1, v1, k2,     ...`, to allow partials and funcs to weed out."
    return tuple(i for pair in d.items() for i in pair)


def _convert_partial(p):
    return ('partial', p.func.__name__,
            ## Don't delve nto args/kwds
            *p.args,
            _convert_dict(p.keywords))


def _convert_obj(obj):
    "Used explicetely only, bc maybe str(obj) gives a better hash."
    return (type(obj).__name__, *_convert_dict(vars(obj)))


def _convert_default_dict(d):
    return (d.default_factory, *d.items())  # don't delve into pairs


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
        from sklearn.pipeline import Pipeline
        ## Ex-msg: "scikit-learn estimators should always specify their parameters
        #           in the signature of their __init__ (no varargs)."
        if not isinstance(item, Pipeline):
            log.error("Cannot stringify instance from class `%s` due to: %s" %
                      type(item), ex)
            if bool_env(CO2MPARE_DEBUG, False):
                raise
        return b'\17'


def _match_regex_map(s: str, d: Mapping[Pattern, Any], default=None):
    "Return the value of the 1st matching regex key in the dict. "
    for regex, v in d.items():
        if regex.match(s):
            return v
    return default


class Hasher:
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
    #                                          if i[0] not in self.args_to_skip), 3)

        if isinstance(item, types.MethodType):
            return self.checksum(*_convert_meth(item))
        if isinstance(item, types.FunctionType):
            return self.checksum(*_convert_fun(item))

        if isinstance(item, fnt.partial):
            return self.checksum(*_convert_partial(item))

        if isinstance(item, defaultdict):
            return self.checksum(*_convert_default_dict(item))

        if isinstance(item, self.classes_to_convert):
            return self.checksum(*_convert_obj(item))
        if type(item) in self.classes_to_skip:
            return 13


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
             if n not in self.args_to_skip and
             n not in func_xargs}
        return d

    def dump_args(self, funpath: str,
                      names:List[str], args,
                      per_func_xargs: Sequence, expandargs=True):
        ## Checksum failures twice, to allow debugging them.
        #
        i = 0
        while i < 1 + int(bool_env(CO2MPARE_DEBUG, 0)):
            if not names:
                names = ['_item_%i' % i for i in range(len(args))]
                inp = dict(zip(names, args))
            elif len(names) == 1 and not expandargs:  # missing or single RES in "output"
                inp = {names[0]: args}
            else:
                assert len(names) >= len(args), (len(names), len(args))
                inp = dict(zip(names, args))
                # Check there were not any duplicate keys.
                assert len(inp) == len(args), (len(inp), len(args))

            ## Process before Checksum on certain args.
            #
            to_proc = self.args_to_convert.keys() & inp.keys()
            if to_proc:
                for k in to_proc:
                    inp[k] = self.checksum(*self.args_to_convert[k](inp[k]))

            ## Debug-print certain args.
            #
            for name in self.args_to_print & inp.keys():
                self._args_printed.append((name, inp[name]))

            ## Checksum:
            ckmap = self._args_cked(inp.items(), per_func_xargs)

            if ckmap and i == 0:  # compare & read old-file only once.
                self._write_and_compare(self._ckmap_to_text(funpath, ckmap))
            i += 1

        ## return for inspeaxtion, or to generate a global hash.
        return ckmap

    _ckfile = None
    #: current written-line number
    _ckfile_nline = 1
    _old_ckfile = None
    #: current read-line
    _old_nline = 1
    _org_eval_fun = None

    def _open_file(self, fpath, mode, *args, **kw):
        if fpath.endswith('.xz'):
            if 'w' in mode:
                FLUSH_INTERVAL_SEC = sys.maxsize  # no fun flushing zip-archives
            return lzma.open(fpath, mode, *args, **kw)
        fp = open(fpath, mode, *args, **kw)
        ## close before process dies...
        weakref.finalize(self, fp.close)

        return fp

    def _yield_old_non_print_lines(self):
        "Skip debug lines in old file, preserving line-numbering."
        for l in self._old_ckfile:
            if l.startswith('- PRINT:'):
                self._old_nline += 1
            else:
                ## nlines increased in self._write_and_compare()
                yield l.strip()

    def _write_and_compare(self, text, *, skip_compare=False) -> bool:
        "Write text and compare it against any old-file, preserving line-numbering."
        self._ckfile.write(text)

        same = None
        if self._old_ckfile:
            ## trim last \n or it consumes +1 from old-file.
            new_lines = text.rstrip().split('\n')
            nlines = len(new_lines)

            if not skip_compare:
                old_lines = [oldl
                             for _newl, oldl in zip(new_lines,
                                                    self._yield_old_non_print_lines())]
                same = old_lines and new_lines == old_lines
                if not same:
                    ## -{{{{-BREAKPOINT HERE TO DISCOVER PROBLEMS-}}}}-
                    log.debug('Comparable missmatch: NEW#L%i != OLD#L%i (%i)',
                             self._ckfile_nline, self._old_nline, nlines)

                self._old_nline += nlines
            self._ckfile_nline += nlines
        return same

    def _ckmap_to_text(self, funpath, ckmap):
        if self._dump_yaml:
            return '\n- %s:\n%s' % (
                funpath,
              ''.join('    %s: %i\n' % (name, ck)
                      for name, ck in ckmap.items()))
        else:
            return ('\n' +
                    ''.join('%s,%s,%i\n' % (funpath, name, ck)
                            for name, ck in ckmap.items()))

    #: The schedula functions visited stored here along with
    #: their checksum of all their args, forming a tree-path.
    _checksum_stack = contextvars.ContextVar('checksum_stack', default='')
    _last_flash = time.clock()

    def dump_args_to_debug(self):
        if self._args_printed:
            ## Note: not compared.
            dumpobj = ''.join('  - %s: %s\n' % (k, v)
                           for k, v in self._args_printed)
            self._write_and_compare('- PRINT: %s\n' % dumpobj.replace('\n', '\\n'),
                       skip_compare=True)
            self._args_printed.clear()

    def __init__(self, *,
                 compare_with_fpath: Union[str, Path, None] = None,
                 zip_output: bool = None,
                 dump_yaml: bool = None):
        "Intercept schedula and open new & old co2mparable files."
        ## TODO: more sanity checks on the subclass.
        self.funcs_to_exclude = dtz.keymap(re.compile, self.funcs_to_exclude)

        self._dump_yaml = bool(compare_with_fpath and
                               '.yaml' in compare_with_fpath.lower() or
                               dump_yaml)

        suffix = '%s%s' % ('.yaml' if self._dump_yaml else '.txt',
                           '.xz' if zip_output else '')
        _fd, fpath = tempfile.mkstemp(suffix=suffix,
                                      prefix='co2mparable-',
                                      text=False)
        log.info("Writing co2mparable at: %s", fpath)
        self._ckfile = self._open_file(fpath, 'wt')

        if compare_with_fpath:
            self._old_ckfile = self._open_file(compare_with_fpath, 'rt')

        ## Intercept Schedula.
        #
        self._org_eval_fun = sol.Solution._evaluate_function
        ## `self` will bind to 2nd arg, `solution` to 1st
        sol.Solution._evaluate_function = my_eval_fun


def my_eval_fun(solution: sol.Solution,
                args, node_id, node_attr, attr):
    from . import _hasher as hasher
    assert hasher

    fun = node_attr['function']
    funame = dsp.parent_func(fun).__name__
    funames = {node_id, funame}

    ## Filter out Funcs or Args.
    #
    per_func_xargs = set()
    for funame in funames:
        ## Exclude certain functions
        #
        exclude = (not funame[0].isidentifier() or     # formulas
                   funame.startswith('IFERROR'))       # formulas
        if not exclude:
            xargs = _match_regex_map(funame, hasher.funcs_to_exclude, ())
            if xargs is None:
                exclude = True
            else:
                per_func_xargs.update(xargs)    # btw prepare per-fun args exclusions
        if exclude:
            return hasher._org_eval_fun(solution, args, node_id, node_attr, attr)

    funpath = hasher._checksum_stack.get()
    funpath = '%s/%s' % (funpath, node_id)

    ## Checksum, Dump & Compare INPs.
    #
    inpnames = node_attr.get('inputs')
    hasher.dump_args('INP/' + funpath,
                         inpnames, args,
                         per_func_xargs)
    #myck = hasher.checksum(base_ck, list(ckmap.values()))

    ## Do nested schedula call.
    #
    res = None
    token = hasher._checksum_stack.set(funpath)
    try:
        res = hasher._org_eval_fun(solution, args, node_id, node_attr, attr)
        return res
    finally:
        hasher._checksum_stack.reset(token)

        ## Checksum, Dump & Compare OUTs.
        #
        ## No == comparisons, res might be dataframe.
        if res is not  None:
            outnames = node_attr.get('outputs')
            if outnames:
                assert not isinstance(outnames, str), outnames
                hasher.dump_args('OUT/' + funpath,
                                     outnames, res,
                                     per_func_xargs,
                                     expandargs=False)

        ## Dump any collected debugged items to print
        #  when funpath at root.
        #
        if hasher._checksum_stack.get() == '':
            hasher.dump_args_to_debug()

        now = time.clock()
        if now - hasher._last_flash > FLUSH_INTERVAL_SEC:
            hasher._ckfile.flush()
            hasher._last_flash = now
