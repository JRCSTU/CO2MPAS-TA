#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from sampling.base import PFiles
import pytest


_ = []


@pytest.mark.parametrize('pfile, opts', [
    (PFiles(_), []),

    ## Missing & quoting
    (PFiles(['abc']), ['--inp', 'abc']),
    (PFiles(['ab\'c']), ['--inp', '"ab\'c"']),
    (PFiles(_, _, ['b&g']), ['"b&g"']),
    ## Mumtiple = missing
    (PFiles(_, ['a', 'b']), ['--out', 'a', '--out', 'b']),

    ## Other
    (PFiles(_, ['a', 'b g']), ['--out', 'a', '--out', '"b g"']),

    ## Multiple IO, balanced
    (PFiles(['a'], ['b g']), ['--inp', 'a', '--out', '"b g"']),
    (PFiles(['a', 'b'], ['aa', 'bb']), ['--inp', 'a', '--out', 'aa',
                                        '--inp', 'b', '--out', 'bb']),

    ## Multiple IO, non-balanced
    (PFiles(['a', 'b'], ['bb']), ['--inp', 'a', '--out', 'bb',
                                  '--inp', 'b']),
    (PFiles(['b'], ['aa', 'bb']), ['--inp', 'b', '--out', 'aa',
                                   '--out', 'bb']),
])
def test_pfiles_cli_opts(pfile: PFiles, opts):
    assert pfile.build_cmd_line(abs_path=False) == opts
