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


@pytest.mark.parametrize('many, some, exp', [
    (PFiles(), PFiles(_), None),

    (PFiles(['a']), PFiles(_), None),
    (PFiles(_, ['a']), PFiles(_), None),
    (PFiles(_, _, ['a']), PFiles(_), None),
    (PFiles(_, _, ['b', 'a']), PFiles(_), None),

    (PFiles(_), PFiles(['a']), ('inp', None, 'a')),
    (PFiles(_), PFiles(_, ['a']), ('out', None, 'a')),
    (PFiles(_), PFiles(_, _, ['a']), ('other', None, 'a')),
    (PFiles(_), PFiles(_, ['b', 'a']), ('out', None, 'b')),

    (PFiles(_), PFiles(['a'], _, ['c']), ('inp', None, 'a')),
    (PFiles(_), PFiles(_, ['b'], ['c']), ('out', None, 'b')),
    (PFiles(_), PFiles(['a'], ['b'], ['c']), ('inp', None, 'a')),

])
def test_compare_pfiles__only_fpath_misses(many: PFiles, some: PFiles, exp):
    assert many.compare(some) == exp


_fileset = [
    (PFiles(['a']), PFiles(), None),
    (PFiles(['a']), PFiles(['a']), None),
    (PFiles(['a', 'b']), PFiles(['a']), None),
    (PFiles(['a', 'b']), PFiles(['a', 'b']), None),

    (PFiles(['a', 'b'], ['a', 'b'], ['a', 'b']),
     PFiles(['a', 'b'], ['a', 'b'], ['a', 'b']), None),
    (PFiles(['a', 'b'], ['a', 'b'], ['a', 'b']),
     PFiles(['b'], ['a'], []), None),
]


@pytest.mark.parametrize('many, some, exp', _fileset)
def test_compare_pfiles_empty(many: PFiles, some: PFiles, exp,
                              tmpdir_factory):
    mdir = tmpdir_factory.mktemp('M')
    sdir = tmpdir_factory.mktemp('S')
    mdir.ensure('a')
    sdir.ensure('a')
    mdir.ensure('b')
    sdir.ensure('b')
    assert many.compare(some, mdir, sdir) == exp


@pytest.mark.parametrize('many, some, exp', _fileset)
def test_compare_pfiles_equals(many: PFiles, some: PFiles, exp,
                               tmpdir_factory):
    mdir = tmpdir_factory.mktemp('M')
    sdir = tmpdir_factory.mktemp('S')
    (mdir / 'a').write('abc')
    (sdir / 'a').write('abc')
    (mdir / 'b').write('123')
    (sdir / 'b').write('123')
    assert many.compare(some, mdir, sdir) == exp


@pytest.mark.parametrize('many, some, exp', [
    (PFiles(['a']), PFiles(), None),
    (PFiles(['a']), PFiles(['a', 'b']), ('inp', 'a', 'a')),
    (PFiles(['a', 'b']), PFiles(['a', 'b']), ('inp', 'a', 'a')),
    (PFiles(_, ['a', 'b']), PFiles(['a', 'b']), ('inp', None, 'a')),

    (PFiles(_, ['a', 'b']), PFiles(_, ['b']), ('out', 'b', 'b')),
])
def test_compare_pfiles_diff(many: PFiles, some: PFiles, exp,
                             tmpdir_factory):
    mdir = tmpdir_factory.mktemp('M')
    sdir = tmpdir_factory.mktemp('S')
    (mdir / 'a').write('abc')
    (sdir / 'a').write('ABC')
    (mdir / 'b').write('123')
    (sdir / 'b').ensure()
    assert many.compare(some, mdir, sdir) == exp
