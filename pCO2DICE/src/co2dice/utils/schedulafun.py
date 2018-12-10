#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Cloned from `schedula` last version containing it:

 https://github.com/vinci1it2000/schedula/commit/702a41aa
"""
from schedula.utils.dsp import selector


class DFun(object):
    """
     A 3-tuple ``(out, fun, **kwds)``, used to prepare a list of calls to
     :meth:`Dispatcher.add_function()`.

     The workhorse is the :meth:`addme()` which delegates to
     :meth:`Dispatcher.add_function()`:

       - ``out``: a scalar string or a string-list that, sent as `output` arg,
       - ``fun``: a callable, sent as `function` args,
       - ``kwds``: any keywords of :meth:`Dispatcher.add_function()`.
       - Specifically for the 'inputs' argument, if present in `kwds`, use them
         (a scalar-string or string-list type, possibly empty), else inspect
         function; in any case wrap the result in a tuple (if not already a
         list-type).

         .. note::
            Inspection works only for regular args, no ``*args, **kwds``
            supported, and they will fail late, on :meth:`addme()`, if no
            `input` or `inp` defined.

    **Example**:

    .. dispatcher:: dsp
       :opt: graph_attr={'ratio': '1'}
       :code:

        >>> dfuns = [
        ...     DFun('res', lambda num: num * 2),
        ...     DFun('res2', lambda num, num2: num + num2, weight=30),
        ...     DFun(out=['nargs', 'res22'],
        ...          fun=lambda *args: (len(args), args),
        ...          inputs=('res', 'res1')
        ...     )]
        >>> dfuns
        [DFun('res', <function <lambda> at 0x...>, ),
         DFun('res2', <function <lambda> at 0x...>, weight=30),
         DFun(['nargs', 'res22'], <function <lambda> at 0x...>,
              inputs=('res', 'res1'))]
        >>> from schedula import Dispatcher
        >>> dsp = Dispatcher()
        >>> DFun.add_dfuns(dfuns, dsp)

    """

    def __init__(self, out, fun, inputs=None, **kwds):
        self.out = out
        self.fun = fun
        if inputs is not None:
            kwds['inputs'] = inputs
        self.kwds = kwds
        assert 'outputs' not in kwds and 'function' not in kwds, self

    def __repr__(self, *args, **kwargs):
        kwds = selector(set(self.kwds) - {'fun', 'out'}, self.kwds)
        return 'DFun(%r, %r, %s)' % (
            self.out,
            self.fun,
            ', '.join('%s=%s' % (k, v) for k, v in kwds.items()))

    def copy(self):
        cp = DFun(**vars(self))
        cp.kwds = dict(self.kwds)
        return cp

    def inspect_inputs(self):
        import inspect
        fun_params = inspect.signature(self.fun).parameters
        assert not any(p.kind for p in fun_params.values()
                       if p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD), (
            "Found '*args or **kwds on function!", self)
        return tuple(fun_params.keys())

    def addme(self, dsp):
        kwds = self.kwds
        out = self.out
        fun = self.fun

        if not isinstance(out, (tuple, list)):
            out = (out,)
        else:
            pass

        inp = kwds.pop('inputs', None)
        if inp is None:
            inp = self.inspect_inputs()

        if not isinstance(inp, (tuple, list)):
            inp = (inp,)
        else:
            pass

        if 'description' not in kwds:
            kwds['function_id'] = '%s%s --> %s' % (fun.__name__, inp, out)

        return dsp.add_function(inputs=inp,
                                outputs=out,
                                function=fun,
                                **kwds)

    @classmethod
    def add_dfuns(cls, dfuns, dsp):
        for uf in dfuns:
            try:
                uf.addme(dsp)
            except Exception as ex:
                raise ValueError("Failed adding dfun %s due to: %s: %s"
                                 % (uf, type(ex).__name__, ex)) from ex
