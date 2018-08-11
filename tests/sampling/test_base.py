#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.sampling.base import PFiles
import pytest

_ = []


@pytest.mark.parametrize('pfile, opts', [
    (PFiles(_, _, _), []),

    ## Missing & quoting
    (PFiles(['abc'], _, _), ['--inp', 'abc']),
    (PFiles(['ab\'c'], _, _), ['--inp', '"ab\'c"']),
    (PFiles(_, _, ['b&g']), ['"b&g"']),
    ## Mumtiple = missing
    (PFiles(_, ['a', 'b'], _), ['--out', 'a', '--out', 'b']),

    ## Other
    (PFiles(_, ['a', 'b g'], _), ['--out', 'a', '--out', '"b g"']),

    ## Multiple IO, balanced
    (PFiles(['a'], ['b g'], _), ['--inp', 'a', '--out', '"b g"']),
    (PFiles(['a', 'b'], ['aa', 'bb'], _), ['--inp', 'a', '--out', 'aa',
                                           '--inp', 'b', '--out', 'bb']),

    ## Multiple IO, non-balanced
    (PFiles(['a', 'b'], ['bb'], _), ['--inp', 'a', '--out', 'bb',
                                     '--inp', 'b']),
    (PFiles(['b'], ['aa', 'bb'], _), ['--inp', 'b', '--out', 'aa',
                                      '--out', 'bb']),
])
def test_pfiles_cli_opts(pfile: PFiles, opts):
    assert pfile.build_cmd_line(abs_path=False) == opts
