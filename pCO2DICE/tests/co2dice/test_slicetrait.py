# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from polyvers._vendor.traitlets.traitlets import HasTraits, TraitError
from polyvers.cmdlet.slicetrait import _parse_slice, Slice

import pytest


test_slices = [
    (None, None),
    ('', None),
    (' ', None),

    ('_', False),
    ('_:_:_', False),
    ('l', False),
    ('-', False),
    ('_6::34', False),
    ('f:1:3', False),
    ('3-6:1:3', False),
    ('- 33', False),
    (':- 33', False),

    (': :', (None, None, None)),
    (':', (None, None, None)),

    ('0', (0, 1, None)),
    ('1', (1, 2, None)),
    (2, (2, 3, None)),
    ('-0100', (-100, -99, None)),
    ('0:', (0, None, None)),
    ('0: :', (0, None, None)),
    ('0: 0', (0, 0, None)),
    (' : 0 ', (None, 0, None)),
    (':0:0', (None, 0, 0)),
    (' :1:0', (None, 1, 0)),
    ('0:0:0 ', (0, 0, 0)),
    ('::0', (None, None, 0)),
    ('12 :0', (12, 0, None)),
    ('12:0 :', (12, 0, None)),
    ('-12: 0', (-12, 0, None)),
    ('12:-013:34', (12, -13, 34)),
    (' -0: -0 :-0', (0, 0, 0)),
    ('12: :34', (12, None, 34)),
    ('-12::-34', (-12, None, -34)),
    ('::34', (None, None, 34)),
]


@pytest.mark.parametrize('inp, exp', test_slices)
def test_slice_parsing(inp, exp):
    if exp is False:
        with pytest.raises(TraitError, match='Syntax-error'):
            _parse_slice(inp)
    else:
        if isinstance(exp, tuple):
            exp = slice(*exp)
        assert _parse_slice(inp) == exp


@pytest.mark.parametrize('inp, exp', test_slices)
def test_Slice_traitlet(inp, exp):
    class C(HasTraits):
        s = Slice()

    if exp in (False, None):
        with pytest.raises(TraitError,
                           match='Syntax-error' if exp is False
                           else 'not the NoneType'):
            C(s=inp)
    else:
        if isinstance(exp, tuple):
            exp = slice(*exp)

        assert C(s=inp).s == exp
