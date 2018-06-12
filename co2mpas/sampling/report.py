#!/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""
A *report* contains the co2mpas-run values to time-stamp and disseminate to TA authorities.
"""

from collections import (
    defaultdict, OrderedDict, namedtuple, Mapping)  # @UnusedImport
from typing import (
    List, Sequence, Iterable, Text, Tuple, Dict, Callable)  # @UnusedImport

import os.path as osp
import pandalone.utils as pndlu

from . import baseapp, project, CmdException, PFiles
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from .._vendor import traitlets as trt


def clip_and_asciify(s, fname_clip_len=64):
    import unidecode

    if len(s) > fname_clip_len:
        s = s[:fname_clip_len - 3] + '...'
    return unidecode.unidecode(s)


###################
##     Specs     ##
###################
def _report_tuple_2_dict(fpath, iokind, report) -> dict:
    """
    Converts tuples from :meth:`_make_report_tuples_from_iofiles()` into stuff YAML-able.
    """
    import pandas as pd

    d = OrderedDict([
        ('file', clip_and_asciify(osp.basename(fpath))),
        ('iokind', iokind)])

    if isinstance(report, pd.DataFrame):
        decs_rounding = 4  # Keep report below 76 QuotedPrintable length-limit.

        def fmt_row_as_pair(i, k, v):
            try:
                v = round(v.astype(float), decs_rounding)
            except:  # @IgnorePep8
                pass
            v = v.tolist()

            return ('%i.%s' % (i, k), v)

        ## Enumerate for duplicate-items, see #396.
        report = OrderedDict(fmt_row_as_pair(i, k, v)
                             for i, (k, v) in
                             enumerate(report.T.items()))
    elif isinstance(report, pd.Series):
        report = OrderedDict(report.items())
    elif report is not None:
        assert isinstance(report, dict)

    d['report'] = report

    return d


class Report(baseapp.Spec):
    """Mines reported-parameters from co2mpas excel-files and serves them as a pandas dataframes."""

    input_head_xlref = trt.Unicode(
        '#Inputs!B1:D5:{"func": "df", "kwds": {"index_col": 0}}',
        help="The *xlref* extracting 5-10 lines from ``Inputs`` sheets "
        "of the input-file as a dataframe."
    ).tag(config=True)
    input_vfid_coords = trt.Unicode(
        'flag.vehicle_family_id',
        help="the dot-separated keys of  ``vehicle_family_id`` into parsed excel file."
    ).tag(config=True)

    dice_report_xlref = trt.Unicode(
        '#dice_report!:{"func": "df", "kwds": {"index_col": 0}}',
        help="The *xlref* extracting the dice-report from the output-file as a dataframe."
    ).tag(config=True)
    output_vfid_coords = trt.Tuple(
        trt.Unicode(), trt.Unicode(),
        default_value=('vehicle_family_id', 'vehicle-H'),
        help="the (row, col) names of the ``vehicle_family_id`` value in the extracted dataframe."
    ).tag(config=True)

    def _parse_input_xlsx(self, inp_xlsx_fpath):
        from co2mpas.io.excel import parse_excel_file

        data = parse_excel_file(inp_xlsx_fpath)

        file_vfid = data
        for k in self.input_vfid_coords.split('.'):
            file_vfid = file_vfid[k]

        return file_vfid, data

    def _extract_dice_report_from_output(self, fpath):
        import pandas as pd
        from pandalone import xleash

        df = xleash.lasso(self.dice_report_xlref, url_file=fpath)
        assert isinstance(df, pd.DataFrame), (
            "The *dice_report* xlref(%s) must resolve to a DataFrame, not type(%r): %s" %
            (self.dice_report_xlref, type(df), df))

        df = df.where((pd.notnull(df)), None)
        vfid = df.at[self.output_vfid_coords]

        return vfid, df

    def is_excel_true_or_null(self, val):
        return val is None or val is True or str(val).lower() == 'true'

    def _check_is_ta(self, fpath, report):
        ta_flags = report.ix['TA_mode', :]
        self.log.debug("TA flags for file('%s'): %s" % (fpath, ta_flags))
        is_ta_mode = all(self.is_excel_true_or_null(f) for f in ta_flags)
        if not is_ta_mode:
            return "file is NOT in TA mode: %s" % ta_flags

    def _check_deviations_are_valid(self, fpath, report):
        import numpy as np

        deviations = report.ix['TA_mode', :]
        is_ok = False
        self.log.debug("Deviations for file('%s'): %s" % (fpath, deviations))
        try:
            is_ok = np.isfinite(deviations)
        except Exception as ex:
            self.log.warning(
                "Ignored error while checking deviations(%s) for file('%s'): %s" %
                (deviations, fpath, ex))

        if not is_ok:
            return "invalid deviations: %s" % deviations

    def _make_report_tuples_from_iofiles(self, iofiles: PFiles,
                                         expected_vfid=None):
        """
        Parses input/output files and yields their *unique* vehicle-family-id and any dice-reports.

        :param expected_vfid:
            raise
        :return:
            A generator that begins by yielding the following 3-tuple
            for each input/output file::

                ('inp' | 'out', <abs-fpath>, <report-df>)

            - `<report>` is series/data-frame, because that's extracted from excel.
            - For *input* files, the ``<report>`` has this index:
              ``['vehicle_family_id': <expectec_vfid>}``;
            - For *output* files, the ``<report>`` is a pandas data-frame.
            - For *other* files, the ``<report>`` is None.

        :raise:
            CmdException if *vehicle_family_id* not matching among each other,
            and `expected_vfid` when provided, unless --force.

        """
        def check_vfid_missmatch(fpath, file_vfid):
            nonlocal expected_vfid

            if expected_vfid is None:
                expected_vfid = file_vfid
            elif expected_vfid != file_vfid:
                return ("mismatch `vehicle_family_id` between this file(%s) and the rest: "
                        "'%s' != expected('%s')'" %
                        (fpath, file_vfid, expected_vfid))

        rtuples = []
        for fpath in iofiles.inp:
            fpath = pndlu.convpath(fpath)
            file_vfid, inp_data = self._parse_input_xlsx(fpath)
            msg = check_vfid_missmatch(fpath, file_vfid)
            if msg:
                msg = "File('%s') %s!" % (fpath, msg)
                if self.force:
                    self.log.warning(msg)
                else:
                    raise CmdException(msg)
            rtuples.append((fpath, 'inp', OrderedDict([
                ('vehicle_family_id', file_vfid),
                ('timestamp', osp.getmtime(fpath)),
            ])))

        for fpath in iofiles.out:
            fpath = pndlu.convpath(fpath)
            file_vfid, dice_report = self._extract_dice_report_from_output(fpath)
            msg1 = self._check_is_ta(fpath, dice_report)
            msg2 = check_vfid_missmatch(fpath, file_vfid)
            msg3 = check_vfid_missmatch(fpath, file_vfid)
            msgs = [m for m in [msg1, msg2, msg3] if m]
            if any(msgs):
                msg = ';\n  also '.join(msgs)
                msg = "File('%s') %s!" % (fpath, msg)
                if self.force:
                    self.log.warning(msg)
                else:
                    raise CmdException(msg)

            rtuples.append((fpath, 'out', dice_report))

        for fpath in iofiles.other:
            fpath = pndlu.convpath(fpath)
            rtuples.append((fpath, 'other', None))

        return rtuples

    ## TODO: Rename Report to `extract_file_infos()`.
    def get_dice_report(self, iofiles: PFiles, expected_vfid=None):
        tuples = self._make_report_tuples_from_iofiles(iofiles, expected_vfid)
        report = OrderedDict((file_tuple[0], _report_tuple_2_dict(*file_tuple))
                             for file_tuple
                             in tuples)
        return report


###################
##    Commands   ##
###################


class ReportCmd(baseapp.Cmd):
    """
    Extract dice-report from the input/output/other files, or from *current-project*.

    SYNTAX
        %(cmd_chain)s [OPTIONS] ( --inp <co2mpas-file> | --out <co2mpas-file> | <other-file> ) ...
        %(cmd_chain)s [OPTIONS] --project

    - The *report parameters* will be time-stamped and disseminated to
      TA authorities & oversight bodies with an email, to receive back
      the sampling decision.
    - If multiple files given from a kind (inp/out), later ones overwrite any previous.
    - Note that any file argument not given with `--inp`, `--out`, will end-up as "other".
    """

    examples = trt.Unicode("""
        - To extract the report-parameters from an INPUT co2mpas file, try::
              %(cmd_chain)s --inp co2mpas_input.xlsx

        - To extract the report from both INPUT and OUTPUT files, try::
              %(cmd_chain)s --inp co2mpas_input.xlsx --out co2mpas_results.xlsx

        - To view the report of the *current-project*, try::
              %(cmd_chain)s --project
    """)

    inp = trt.List(
        trt.Unicode(),
        help="Specify co2mpas INPUT files; use this option one or more times."
    ).tag(config=True)
    out = trt.List(
        trt.Unicode(),
        help="Specify co2mpas OUTPUT files; use this option one or more times."
    ).tag(config=True)

    project = trt.Bool(
        False,
        help="""
        Whether to extract report from files present already in the *current-project*.
        """).tag(config=True)

    vfids_only = trt.Bool(
        help=""""Prints `- fpath: vehicle_family_id` YAML lines."""
    ).tag(config=True)

    __report = None

    @property
    def projects_db(self):
        p = project.ProjectsDB.instance(config=self.config)
        p.update_config(self.config)
        return p

    def __init__(self, **kwds):
        dkwds = {
            'conf_classes': [project.ProjectsDB, project.Project, Report],
            'cmd_aliases': {
                ('i', 'inp'): ('ReportCmd.inp', type(self).inp.help),
                ('o', 'out'): ('ReportCmd.out', type(self).out.help),
            },
            'cmd_flags': {
                'project': ({
                    'ReportCmd': {'project': True},
                }, ReportCmd.project.help),
                'vfids': ({
                    'ReportCmd': {'vfids_only': True},
                }, ReportCmd.vfids_only.help),
            }
        }
        dkwds.update(kwds)
        super().__init__(**dkwds)

    def _build_io_files_from_project(self, args) -> PFiles:
        project = self.projects_db.current_project()
        pfiles = project.list_pfiles(*PFiles._fields, _as_index_paths=True)  # @UndefinedVariable
        if not pfiles:
            raise CmdException(
                "Current %s contains no input/output files!" % project)

        return pfiles

    def run(self, *args):
        nargs = len(args)
        infos = self.vfids_only and '`vehicle_family_id`' or 'report infos'
        if self.project:
            if nargs > 0:
                raise CmdException(
                    "Cmd '%s --project' takes no arguments, received %d: %r!"
                    % (self.name, len(args), args))

            self.log.info("Extracting %s from current-project...", infos)
            pfiles = self._build_io_files_from_project(args)
        else:
            ## TODO: Support heuristic inp/out classification
            pfiles = PFiles(inp=self.inp, out=self.out, other=args)
            if not pfiles.nfiles():
                raise CmdException(
                    "Cmd %r must be given at least one file argument, received %d: %r!"
                    % (self.name, pfiles.nfiles(), pfiles))
            pfiles.check_files_exist(self.name)
            self.log.info("Extracting %s from files...\n  %s", infos, pfiles)

        import yaml

        repspec = Report(config=self.config)
        if self.vfids_only:
            repspec.force = True  # Irrelevant to check for mismatching VFids.
            for fpath, data in repspec.get_dice_report(pfiles).items():
                if not self.verbose:
                    fpath = osp.basename(fpath)

                rep = data['report']
                yield '- %s: %s' % (fpath, rep and rep.get('vehicle_family_id'))

        else:
            for rtuple in repspec._make_report_tuples_from_iofiles(pfiles):
                drep = _report_tuple_2_dict(*rtuple)
                fpath = rtuple[0]
                if not self.verbose:
                    fpath = osp.basename(fpath)

                yield yaml.dump({fpath: drep}, indent=2)

## test CMDS:
#    co2dice report -i ./co2mpas/demos/co2mpas_demo-7.xlsx -o 20170207_192057-* && \
#    co2dice report  --vfids --project && co2dice report   --project && \
#    co2dice report   --project -v &&  co2dice report   --project --vfids -v
