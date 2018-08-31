# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
r"""
Shift and resample excel-tables; see https://co2mpas.io/usage.html#synchronizing-time-series

Usage:
  datasync template [-f] [--cycle <cycle>] <excel-file-path>...
  datasync          [-v | -q | --logconf=<conf-file>] [--force | -f]
                    [--interp <method>] [--no-clone] [--prefix-cols]
                    [-O <output>] <x-label> <y-label> <ref-table>
                    [<sync-table> ...] [-i=<label=interp> ...]
  datasync          [-v | -q | --logconf=<conf-file>] (--version | -V)
  datasync          (--interp-methods | -l)
  datasync          --help

Options:
  <x-label>              Column-name of the common x-axis (e.g. 'times') to be
                         re-sampled if needed.
  <y-label>              Column-name of y-axis cross-correlated between all
                         <sync-table> and <ref-table>.
  <ref-table>            The reference table, in *xl-ref* notation (usually
                         given as `file#sheet!`); synced columns will be
                         appended into this table.
                         The captured table must contain <x_label> & <y_label>
                         as column labels.
                         If hash(`#`) symbol missing, assumed as file-path and
                         the table is read from its 1st sheet .
  <sync-table>           Sheets to be synced in relation to <ref-table>, also in
                         *xl-ref* notation.
                         All tables must contain <x_label> & <y_label> as column
                         labels.
                         Each xlref may omit file or sheet-name parts; in that
                         case, those from the previous xlref(s) are reused.
                         If hash(`#`) symbol missing, assumed as sheet-name.
                         If none given, all non-empty sheets of <ref-table> are
                         synced against the 1st one.
  -O=<output>            Output folder or file path to write the results
                         [default: .]:

                         - Non-existent path: taken as the new file-path; fails
                           if intermediate folders do not exist, unless --force.
                         - Existent file: file-path to overwrite if --force,
                           fails otherwise.
                         - Existent folder: writes a new file
                           `<ref-file>.sync<.ext>` in that folder; --force
                           required if that file exists.

  -f, --force            Overwrite excel-file(s) and create any missing
                         intermediate folders.
  --prefix-cols          Prefix all synced column names with their source
                         sheet-names. By default, only clashing column-names are
                         prefixed.
  --no-clone             Do not clone excel-sheets contained in <ref-table>
                         workbook into output.
  --interp=<method>      Interpolation method used in the resampling for all
                         signals [default: linear]:
                         'linear', 'nearest', 'zero', 'slinear', 'quadratic',
                         'cubic' are passed to `scipy.interpolate.interp1d`.
                         'spline' and 'polynomial' require also to specify an
                         order (int), e.g. `--interp=spline3`.
                         'pchip' and 'akima' are wrappers around the scipy
                         interpolation methods of similar names.
                         'integral' is respecting the signal integral.

  -i=<label=interp>      Interpolation method used in the resampling for a
                         signal with a specific label
                         (e.g., `-i alternator_currents=integral`).
  -l, --interp-methods   List of all interpolation methods that can be used in
                         the resampling.
  --cycle=<cycle>        If set (e.g., --cycle=nedc.manual), the <ref-table> is
                         populated with the theoretical velocity profile.
                         Options: 'nedc.manual', 'nedc.automatic',
                         'wltp.class1', 'wltp.class2', 'wltp.class3a', and
                         'wltp.class3b'.

  <excel-file-path>      Output file.

Miscellaneous:
  -h, --help             Show this help message and exit.
  -V, --version          Print version of the program, with --verbose
                         list release-date and installation details.
  -v, --verbose          Print more verbosely messages - overridden by --logconf.
  -q, --quiet            Print less verbosely messages (warnings) - overridden by --logconf.
  --logconf=<conf-file>  Path to a logging-configuration file, according to:
                           https://docs.python.org/3/library/logging.config.html#configuration-file-format
                         If the file-extension is '.yaml' or '.yml', it reads a dict-schema from YAML:
                           https://docs.python.org/3/library/logging.config.html#logging-config-dictschema

* For xl-refs see: https://pandalone.readthedocs.org/en/latest/reference.html#module-pandalone.xleash

SUB-COMMANDS:
    template             Generate "empty" input-file for the `datasync` cmd as
                         <excel-file-path>.


Examples::

    ## Read the full contents from all `wbook.xlsx` sheets as tables and
    ## sync their columns using the table from the 1st sheet as reference:
    datasync times velocities folder/Book.xlsx

    ## Sync `Sheet1` using `Sheet3` as reference:
    datasync times velocities wbook.xlsx#Sheet3!  Sheet1

    ## The same as above but with integers used to index excel-sheets.
    ## NOTE that sheet-indices are zero based!
    datasync times velocities wbook.xlsx#2! 0

    ## Complex Xlr-ref example:
    ## Read the table in sheet2 of wbook-2 starting at D5 cell
    ## or more Down 'n Right if that was empty, till Down n Right,
    ## and sync this based on 1st sheet of wbook-1:
    datasync times velocities wbook-1.xlsx  wbook-2.xlsx#0!D5(DR):..(DR)

    ## Typical usage for CO2MPAS velocity time-series from Dyno and OBD
    ## (the ref sheet contains the theoretical velocity profile):
    datasync template --cycle wltp.class3b template.xlsx
    datasync -O ./output times velocities template.xlsx#ref! dyno obd -i alternator_currents=integral -i battery_currents=integral
"""

from collections import OrderedDict, Counter
import logging
import os
import sys
import functools
import regex
from boltons.setutils import IndexedSet
import docopt
import numpy.fft as fft
from pandalone import xleash
import scipy.integrate as sci_itg
import scipy.interpolate as sci_itp
import functools as fnt
import numpy as np
import os.path as osp
import pandas as pd
from co2mpas.__main__ import (CmdException, init_logging, build_version_string,
                              parse_overrides, exit_with_pride)
import openpyxl
import shutil
import schedula as sh
import collections


proj_name = 'datasync'

log = logging.getLogger(__name__)


synced_file_frmt = '%s.sync%s'


def cross_correlation_using_fft(x, y):
    f1 = fft.fft(x)
    f2 = fft.fft(np.flipud(y))
    cc = np.real(fft.ifft(f1 * f2))
    return fft.fftshift(cc)


# shift &lt; 0 means that y starts 'shift' time steps before x
# shift &gt; 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


def _interp_wrapper(func, x, xp, fp, **kw):
    x, xp, fp = np.asarray(x), np.asarray(xp), np.asarray(fp)
    if isinstance(kw.get('fill_value'), tuple) and not kw['fill_value']:
        kw['fill_value'] = fp[0], fp[-1]
    return np.nan_to_num(func(xp, fp, **kw)(x))


_re_interpolation_method = regex.compile(
    r"""
        ^(?P<kind>\D+)$
        |
        ^(?P<kind>(spline|polynomial))(?P<order>\d+)?$
    """, regex.IGNORECASE | regex.X | regex.DOTALL)


def polynomial_interpolation(x, xp, fp, order=1):
    return np.poly1d(np.polyfit(xp, fp, order))(x)


@functools.lru_cache(None)
def _interpolation_methods():
    methods = ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
    kw = dict(fill_value=(), copy=False, bounds_error=False)
    methods = {k: fnt.partial(_interp_wrapper, sci_itp.interp1d, kind=k, **kw)
               for k in methods}
    methods['spline'] = fnt.partial(_interp_wrapper, sci_itp.interp1d, **kw)
    methods['polynomial'] = polynomial_interpolation
    methods['pchip'] = fnt.partial(_interp_wrapper, sci_itp.PchipInterpolator)
    methods['akima'] = fnt.partial(_interp_wrapper, sci_itp.Akima1DInterpolator)
    methods['integral'] = integral_interpolation
    return methods


@functools.lru_cache(256)
def _get_interp_method(interpolation_method):
    methods = _interpolation_methods()
    try:
        kw = _re_interpolation_method.match(interpolation_method)
        if kw:
            kw = {k: v for k, v in kw.groupdict().items() if v is not None}
            if 'order' in kw:
                kw['order'] = int(kw['order'])
            kind = kw.pop('kind').lower()
            if kind == 'spline':
                kw['kind'] = kw.pop('order')
            return fnt.partial(methods[kind], **kw)
        else:
            raise KeyError
    except KeyError:
        msg = '`%s` is not implemented as re-sampling method!\n' \
              'Please choose one of: \n %s'
        msg %= interpolation_method, ', '.join(sorted(methods))
        raise CmdException(msg)


def _yield_synched_tables(ref, *data, x_label='times', y_label='velocities',
                          interpolation_method='linear',
                          interpolation_methods=None):
    """
    Yields the data re-sampled and synchronized respect to x axes (`x_id`) and
    the reference signal `y_id`.

    :param dict ref:
        Reference data.
    :param data:
        Data to  yield synched tables from.
    :type data: list[dict]
    :param str x_label:
        X label of the reference signal.
    :param str y_label:
        Y label of the reference signal.

    :return:
        The re-sampled and synchronized data, as types of original `data`
        (e.g. dicts or DataFrames).
    :rtype: generator
    """
    linear = _get_interp_method('linear')

    re_sampling = collections.defaultdict(
        lambda: _get_interp_method(interpolation_method)
    )

    if interpolation_methods:
        re_sampling.update(
            {k: _get_interp_method(v) for k, v in interpolation_methods.items()}
        )

    dx = float(np.median(np.diff(ref[x_label])) / 10)
    m, M = min(ref[x_label]), max(ref[x_label])

    for d in data:
        m, M = min(min(d[x_label]), m), max(max(d[x_label]), M)

    X = np.arange(m, M + dx, dx)
    Y = linear(X, ref[x_label], ref[y_label])

    x = ref[x_label]

    yield 0, ref
    for d in data:
        y = linear(X, d[x_label], d[y_label])
        shift = compute_shift(Y, y) * dx

        s = OrderedDict([(k, fnt.partial(re_sampling[k], xp=d[x_label], fp=v))
                         for k, v in d.items() if k != x_label])

        x_shift = x + shift

        r = [(k, v(x_shift)) for k, v in s.items()]

        yield shift, ref.__class__(OrderedDict(r))


def _cum_integral(x, xp, fp):
    X = np.unique(np.concatenate((x, xp)))
    Y = np.interp(X, xp, fp, left=0.0, right=0.0)
    return sci_itg.cumtrapz(Y, X, initial=0)[np.searchsorted(X, x)]


def integral_interpolation(x, xp, fp):
    """
    Re-samples data maintaining the signal integral.

    :param x:
        The x-coordinates of the re-sampled values.
    :type x: numpy.array

    :param xp:
        The x-coordinates of the data points.
    :type xp: numpy.array

    :param fp:
        The y-coordinates of the data points, same length as xp.
    :type fp: numpy.array

    :return:
        Re-sampled y-values.
    :rtype: numpy.array
    """

    x, fp = np.asarray(x, dtype=float), np.asarray(fp, dtype=float)
    xp = np.asarray(xp, dtype=float)
    n = len(x)
    X, dx = np.zeros(n + 1), np.zeros(n + 1)
    dx[1:-1] = np.diff(x)
    X[0], X[1:-1], X[-1] = x[0], x[:-1] + dx[1:-1] / 2, x[-1]
    I = np.diff(_cum_integral(X, xp, fp))

    dx /= 8.0
    # noinspection PyTypeChecker
    A = np.diag((dx[:-1] + dx[1:]) * 3.0)
    i, j = np.indices(A.shape)
    A[i == j - 1] = A[i - 1 == j] = dx[1:-1]

    return np.linalg.solve(A, I)


def synchronize(headers, tables, x_label, y_label, prefix_cols,
                interpolation_method='linear', interpolation_methods=None):
    res = _yield_synched_tables(*tables, x_label=x_label, y_label=y_label,
                                interpolation_method=interpolation_method,
                                interpolation_methods=interpolation_methods)
    res = list(res)

    if prefix_cols:
        ix = set()
        for sn, i, h in headers:
            ix.update(h.columns)
    else:
        ix = Counter()
        for sn, i, h in headers:
            ix.update(set(h.columns))
        ix = {k for k, v in ix.items() if v > 1}

    for sn, i, h in headers[1:]:
        for j in ix.intersection(h.columns):
            h[j].iloc[i] = '%s.%s' % (sn, h[j].iloc[i])

    frames = [h[df.columns].append(df)
              for (_, df), (sn, i, h) in zip(res, headers)]
    df = pd.concat(frames, axis=1)

    return df


def _guess_xlref_without_hash(xlref, bias_on_fragment):
    if not xlref:
        raise CmdException("An xlref cannot be empty-string!")
    if '#' not in xlref:
        xlref = ('#%s!' if bias_on_fragment else '%s#:') % xlref
    return xlref


def _get_rest_sheet_names(url_file, sheet, sheets_factory):
    # TODO: Move to pandalone.
    book = sheets_factory.fetch_sheet(url_file, sheet)._sheet.book
    return IndexedSet(book.sheet_names()) - [sheet]


def sheet_name(lasso):
    # TODO: Move to pandalone.
    return lasso.sheet.get_sheet_ids().ids[0]


class Tables(object):
    # Nice API, may adopt by pandalone.
    _sheets_factory = None

    def __init__(self, required_labels, sheets_factory=None):
        self.required_labels = required_labels
        if sheets_factory:
            self._sheets_factory = sheets_factory
        elif not self._sheets_factory:
            # Permit class-wide sheets-fact.
            self._sheets_factory = xleash.SheetsFactory()
        self.headers = []
        self.tables = []
        self.ref_fpath = None
        self.ref_sh_name = None

    def _consume_next_xlref(self, xlref, lasso):
        """
        :param str xlref:
                an xlref that may not contain hash(`#`); in that case,
                it is taken as *file-part* or as *fragment-part* depending
                on the existence of prev lasso's `url_file`.
        :param Lasso lasso:
                reuses `url_file` & `sheet` if missing from xlref
        """

        xlref = _guess_xlref_without_hash(xlref,
                                          bias_on_fragment=bool(lasso.url_file))
        lasso = xleash.lasso(xlref,
                             sheets_factory=self._sheets_factory,
                             url_file=lasso.url_file,
                             sheet=lasso.sheet,
                             return_lasso=True)
        values = lasso.values
        if values:  # Skip blank sheets.
            # TODO: Convert column monkeybiz into pure-pandas using xleash.
            str_row_indices = [i for i, r in enumerate(values)
                               if any(isinstance(v, str) for v in r)]

            req_labels = IndexedSet(self.required_labels)
            for k in str_row_indices:
                if set(values[k]) >= req_labels:
                    break
            else:
                raise CmdException(
                    "Columns %r not found in table of sheet(%r) in book(%r)!" %
                    (self.required_labels, lasso.sheet._sheet.name,
                     lasso.sheet.book_fname))
            ix = values[k]
            i = max(str_row_indices, default=0) + 1

            h = pd.DataFrame(values[:i], columns=ix)
            self.headers.append((sheet_name(lasso), k, h))

            values = pd.DataFrame(values[i:], columns=ix)
            values.dropna(how='all', inplace=True)
            values.dropna(axis=1, how='any', inplace=True)
            if values.empty:
                log.warning(
                    "Empty table of sheet(%r) in book (%r)!" %
                    (lasso.sheet._sheet.name, lasso.sheet.book_fname))
            else:
                self.tables.append(values)

        return lasso

    def consume_next_xlref(self, xlref, lasso):
        i = len(self.tables)
        try:
            return self._consume_next_xlref(xlref, lasso)
        except CmdException as ex:
            raise CmdException('Cannot read sync-sheet(%i: %s) due to: %s' %
                               (i, xlref, ex.args[0]))
        except Exception as ex:
            log.error('Failed reading sync-sheet(%i: %s) due to: %s',
                      i, xlref, ex)
            raise

    def collect_tables(self, ref_xlref, *sync_xlrefs):
        """
        Extract tables from ref and sync xlrefs.

        Each xlref may omit file or sheet-name parts; in that case, those from
        the previous xlref(s) are reused.
        """

        lasso = self.consume_next_xlref(ref_xlref, xleash.Lasso())
        self.ref_fpath = lasso.url_file
        self.ref_sh_name = sheet_name(lasso)
        assert lasso.url_file and self.ref_sh_name, (lasso.url_file,
                                                     self.ref_sh_name)
        if not sync_xlrefs:
            sync_xlrefs = _get_rest_sheet_names(
                lasso.url_file, self.ref_sh_name, self._sheets_factory
            )
        for xlref in sync_xlrefs:
            lasso = self.consume_next_xlref(xlref, lasso)


def _ensure_out_file(out_path, inp_path, force, out_frmt):
    """
    :param str out_path:
            If `None`, same folder as `inp_path` assumed.
    """
    basename = osp.basename(inp_path)

    if not osp.exists(out_path):
        out_file = out_path
        folders = osp.dirname(out_path)
        if not osp.isdir(folders):
            if force:
                log.info('Creating intermediate folders: %r...', folders)
                os.makedirs(folders)
            else:
                raise CmdException(
                    "Intermediate folders %r do not exist! \n"
                    "Tip: specify --force to create them." % out_path)
    elif osp.isfile(out_path):
        out_file = out_path
    elif osp.isdir(out_path):
        out_file = osp.join(out_path, out_frmt % osp.splitext(basename))
    else:
        assert False, 'Unexpected file-type: %r' % out_path
    assert out_file, (out_path, inp_path, force, out_frmt)

    out_file = osp.abspath(osp.expanduser(osp.expandvars(out_file)))
    if osp.isfile(out_file):
        if force:
            log.info('Overwriting datasync-file: %r...', out_file)
        else:
            raise CmdException("Output file exists! \n"
                               "\n To overwrite add '-f' option!")
    return out_file


def do_datasync(x_label, y_label, ref_xlref, *sync_xlrefs,
                out_path=None, prefix_cols=False, force=False,
                sheets_factory=None, no_clone=False,
                interpolation_method='linear', interpolation_methods=None):
    """

    :param str x_label:
            `x` column label.
    :param str y_label:
            `y` column label.
    :param str ref_xlref:
            The `xl-ref` capturing a table from a workbook-sheet to use as *reference*.
            The table must contain `x_label`, `y_label` column labels.
    :param [str] sync_xlrefs:
            A list of `xl-ref` capturing tables from workbook-sheets,
            to be *synced* in relation to *reference*.
            All tables must contain `x_label`, `y_label` column labels.
            Each xlref may omit file or sheet-name parts; in that case,
            those from the previous xlref(s) are reused.
    :param bool prefix_cols:
            Prefix all synced column names with their source sheet-names.
            If not true, only clashing column-names are prefixed.
    :param str out_path:
            Output folder or file path to write synchronized results:

            - Non-existent path: taken as the new file-path; fails
              if intermediate folders do not exist, unless --force.
            - Existent file: fails, unless --force.
            - Existent folder: writes a new file `<ref-file>.sync<.ext>`
              in that folder; --force required if that file exists.

            If not true, use folder of the <ref-table>.
    :param bool force:
            When true, overwrites excel-file(s) and/or create missing folders.
    :param bool no_clone:
            When true, do not clone excel-sheets contained in <ref-table> workbook
            into output.
    :param xleash.SheetsFactory sheets_factory:
            cache of workbook-sheets
    :param str interpolation_method:
            Interpolation method.
    :param dict interpolation_methods:
            Interpolation methods specified for specific signals.
    """
    tables = Tables((x_label, y_label), sheets_factory)
    tables.collect_tables(ref_xlref, *sync_xlrefs)
    df = synchronize(tables.headers, tables.tables, x_label, y_label,
                     prefix_cols, interpolation_method=interpolation_method,
                     interpolation_methods=interpolation_methods)

    if no_clone:
        writer_fact = pd.ExcelWriter
    else:
        from co2mpas.io.excel import clone_excel
        writer_fact = fnt.partial(clone_excel, tables.ref_fpath)

    out_file = _ensure_out_file(out_path, tables.ref_fpath, force,
                                synced_file_frmt)
    with writer_fact(out_file) as writer:
        # noinspection PyUnresolvedReferences
        df.to_excel(writer, tables.ref_sh_name, header=False, index=False)
        writer.save()
        log.info(
            'Data are synchronized and written into the sheet (%r) of xl-file '
            '(%r) !\n', tables.ref_sh_name, out_file
        )
    return out_file


def _get_input_template_fpath():
    import pkg_resources

    fname = 'datasync_template.xlsx'
    return pkg_resources.resource_filename(__name__, fname)  # @UndefinedVariable


_re_template = regex.compile(
    r"""
    ^(?P<cycle_type>nedc)(.(?P<gear_box_type>(manual|automatic)))$
    |
    ^(?P<cycle_type>wltp)(.(?P<wltp_class>(class([12]|3[ab]))))$
    """, regex.IGNORECASE | regex.X | regex.DOTALL)


def _get_theoretical(profile):
    defaults = {
        'cycle_type': 'WLTP',
        'gear_box_type': 'manual',
        'wltp_class': 'class3b',
        'downscale_factor': 0
    }
    profile = {k: v for k, v in profile.items() if v}
    profile = sh.combine_dicts(defaults, profile)
    profile['cycle_type'] = profile['cycle_type'].upper()
    profile['wltp_class'] = profile['wltp_class'].lower()
    profile['gear_box_type'] = profile['gear_box_type'].lower()
    from co2mpas.model.physical.cycle import cycle
    res = cycle().dispatch(
        inputs=profile, outputs=['times', 'velocities'], shrink=True
    )
    data = sh.selector(['times', 'velocities'], res, output_type='list')
    return pd.DataFrame(data).T


def _cmd_template(opts):
    dst_fpaths = opts.get('<excel-file-path>', None)

    if not dst_fpaths:
        raise CmdException('Missing destination filepath for INPUT-TEMPLATE!')
    if opts['--cycle'] is not None:

        profile = _re_template.match(opts['--cycle'])
        if profile is None:
            raise CmdException('Cycle %s not allowed' % opts['--cycle'])
        df = _get_theoretical(profile.groupdict())

        def overwrite_ref(fpath):
            book = openpyxl.load_workbook(fpath)
            writer = pd.ExcelWriter(fpath, engine='openpyxl')
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            df.to_excel(writer, "ref", index=False, startrow=2, header=False)
            writer.save()
    else:
        def overwrite_ref(fpath):
            pass
    template = _get_input_template_fpath()
    force = opts['--force']
    for fpath in dst_fpaths:
        if not fpath.endswith('.xlsx'):
            fpath = '%s.xlsx' % fpath
        if osp.exists(fpath) and not force:
            raise CmdException(
                "File '%s', already exists! "
                "Use '-f' to overwrite it." % fpath)

        if not osp.splitext(fpath)[-1]:
            raise CmdException(
                "Expecting a file-name instead of directory '%s'!" % fpath)

        dir_name = osp.dirname(fpath)
        if dir_name and not osp.isdir(dir_name):
            if force:
                os.makedirs(dir_name)
            else:
                raise CmdException(
                    "Directory '%s' does not exists! "
                    "Use '-f' to create it." % dir_name)

        log.info("Creating INPUT-TEMPLATE file '%s'...", fpath)
        shutil.copy(template, fpath)
        overwrite_ref(fpath)


def _main(*args):
    """Throws any exception or (optionally) return an exit-code."""
    opts = docopt.docopt(__doc__, argv=args or sys.argv[1:])

    verbose = opts['--verbose']
    quiet = opts['--quiet']
    level = None  # Let `init_logging()` decide.
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    init_logging(level=level, logconf_file=opts.get('--logconf'))

    if opts['--version']:
        v = build_version_string(verbose)
        # noinspection PyBroadException
        try:
            sys.stdout.buffer.write(v.encode() + b'\n')
            sys.stdout.buffer.flush()
        except Exception:
            print(v)
    elif opts['--interp-methods']:
        msg = 'List of all interpolation methods:\n%s\n'
        msg %= ', '.join(sorted(_interpolation_methods()))
        # noinspection PyBroadException
        try:
            sys.stdout.buffer.write(msg)
            sys.stdout.buffer.flush()
        except Exception:
            print(msg)
    elif opts['template']:
        _cmd_template(opts)
    else:
        do_datasync(
            opts['<x-label>'], opts['<y-label>'],
            opts['<ref-table>'], *opts['<sync-table>'],
            out_path=opts['-O'],
            prefix_cols=opts['--prefix-cols'],
            force=opts['--force'],
            no_clone=opts['--no-clone'],
            interpolation_method=opts['--interp'],
            interpolation_methods=parse_overrides(opts['-i'], option_name='-i'))


def main(*args):
    """Handles some exceptions politely and returns the exit-code."""
    if sys.version_info < (3, 5):
        return exit_with_pride(
            "Sorry, Python >= 3.5 is required, found: %s" % sys.version_info,
            logger=log)

    try:
        return _main(*args)
    except CmdException as ex:
        log.debug('App exited due to: %r', ex, exc_info=1)
        ## Suppress stack-trace for "expected" errors but exit-code(1).
        return exit_with_pride(str(ex), logger=log)
    except Exception as ex:
        ## Log in DEBUG not to see exception x2, but log it anyway,
        #  in case log has been redirected to a file.
        log.debug('App failed due to: %r', ex, exc_info=1)
        ## Print stacktrace to stderr and exit-code(-1).
        return exit_with_pride(ex, logger=log)


if __name__ == '__main__':
    sys.exit(main())
