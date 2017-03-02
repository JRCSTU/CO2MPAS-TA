#!/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""A *report* contains the co2mpas-run values to time-stamp and disseminate to TA authorities & oversight bodies."""

from collections import (
    defaultdict, OrderedDict, namedtuple, Mapping)  # @UnusedImport
from typing import (
    List, Sequence, Iterable, Text, Tuple, Dict, Callable)  # @UnusedImport

import os.path as osp
import pandalone.utils as pndlu
import pandas as pd
import traitlets as trt

from . import baseapp, project, CmdException, PFiles
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport


###################
##     Specs     ##
###################
def _report_tuple_2_dict(fpath, iokind, report) -> dict:
    """Converts tuples produced by :meth:`_yield_report_tuples_from_iofiles()` into stuff YAML-able. """
    d = OrderedDict([
        ('file', osp.basename(fpath)),
        ('iokind', iokind)])

    if isinstance(report, pd.DataFrame):
        report = OrderedDict((k, list(v)) for k, v in report.T.items())
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
        help="The *xlref* extracting 5-10 lines from ``Inputs`` sheets of the input-file as a dataframe."
    ).tag(config=True)
    input_vfid_coords = trt.Tuple(
        trt.Unicode(), trt.Unicode(),
        default_value=('flag.vehicle_family_id', 'Value'),
        help="the (row, col) names of the ``vehicle_family_id`` value in the extracted dataframe."
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

    def _extract_vfid_from_input(self, fpath):
        from pandalone import xleash
        df = xleash.lasso(self.input_head_xlref, url_file=fpath)
        assert isinstance(df, pd.DataFrame), (
            "The *inputs* xlref(%s) must resolve to a DataFrame, not type(%r): %s" %
            (self.input_head_xlref, type(df), df))

        return df.at[self.input_vfid_coords]

    def _extract_dice_report_from_output(self, fpath):
        from pandalone import xleash
        df = xleash.lasso(self.dice_report_xlref, url_file=fpath)
        assert isinstance(df, pd.DataFrame), (
            "The *dice_report* xlref(%s) must resolve to a DataFrame, not type(%r): %s" %
            (self.dice_report_xlref, type(df), df))

        df = df.where((pd.notnull(df)), None)
        vfid = df.at[self.output_vfid_coords]

        return vfid, df

    def _yield_report_tuples_from_iofiles(self, iofiles: PFiles, expected_vfid=None):
        """
        Parses input/output files and yields their *unique* vehicle-family-id and any dice-reports.

        :param expected_vfid:
            raise
        :return:
            A generator that begins by yielding the following 3-tuple
            for each input/output file::

                ('inp' | 'out', <abs-fpath>, <report-df>)

            - `<report>` is series/data-frame, because that's extracted from excel.
            - For *input* files, the ``<report>`` has this index: ``['vehicle_family_id': <expectec_vfid>}``;
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
                return ("mismatch `vehicle_family_id` between this file the rest: "
                        "'%s' != expected('%s')'" %
                        (file_vfid, expected_vfid))

        def is_excel_true_or_null(val):
            return val is None or val is True or str(val).lower() == 'true'

        def check_is_ta(fpath, report):
            ta_flags = report.ix['TA_mode', :]
            self.log.debug("TA flags for file('%s'): %s" % (fpath, ta_flags))
            is_ta_mode = all(is_excel_true_or_null(f) for f in ta_flags)
            if not is_ta_mode:
                return "file is NOT in TA mode: %s" % ta_flags

        for fpath in iofiles.inp:
            fpath = pndlu.convpath(fpath)
            file_vfid = self._extract_vfid_from_input(fpath)
            msg = check_vfid_missmatch(fpath, file_vfid)
            if msg:
                msg = "File('%s') %s!" % (fpath, msg)
                if self.force:
                    self.log.warning(msg)
                else:
                    raise CmdException(msg)

            yield (fpath, 'inp', OrderedDict([
                ('report_type', 'input_report'),
                ('vehicle_family_id', file_vfid),
            ]))

        for fpath in iofiles.out:
            fpath = pndlu.convpath(fpath)
            file_vfid, dice_report = self._extract_dice_report_from_output(fpath)
            msg1 = check_is_ta(fpath, dice_report)
            msg2 = check_vfid_missmatch(fpath, file_vfid)
            msgs = [m for m in [msg1, msg2] if m]
            if any(msgs):
                msg = ';\n  also '.join(msgs)
                msg = "File('%s') %s!" % (fpath, msg)
                if self.force:
                    self.log.warning(msg)
                else:
                    raise CmdException(msg)

            yield (fpath, 'out', dice_report)

        for fpath in iofiles.other:
            fpath = pndlu.convpath(fpath)
            yield (fpath, 'other', None)

    def get_dice_report(self, iofiles: PFiles, expected_vfid=None):
        tuples = self._yield_report_tuples_from_iofiles(iofiles, expected_vfid)
        report = OrderedDict((file_tuple[0], _report_tuple_2_dict(*file_tuple))
                             for file_tuple
                             in tuples)
        return report


###################
##    Commands   ##
###################


class ReportCmd(baseapp.Cmd):
    """
    Extract dice-report from the given co2mpas input/output/other files, or from those in *current-project*.

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
        To extract the report-parameters from an INPUT co2mpas file, try:

            %(cmd_chain)s --inp co2mpas_input.xlsx

        To extract the report from both INPUT and OUTPUT files, try:

            %(cmd_chain)s --inp co2mpas_input.xlsx --out co2mpas_results.xlsx

        To view the report of the *current-project*, try:

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
    def repspec(self):
        if not self.__report:
            self.__report = Report(config=self.config)
        return self.__report

    @property
    def projects_db(self):
        p = project.ProjectsDB.instance(config=self.config)
        p.config = self.config
        return p

    def __init__(self, **kwds):
        dkwds = {
            'conf_classes': [project.ProjectsDB, project.Project, Report],
            'cmd_aliases': {
                ('i', 'inp'): ('ReportCmd.inp', pndlu.first_line(type(self).inp.help)),
                ('o', 'out'): ('ReportCmd.out', pndlu.first_line(type(self).out.help)),
            },
            'cmd_flags': {
                'project': ({
                    'ReportCmd': {'project': True},
                }, pndlu.first_line(ReportCmd.project.help)),
                'vfids': ({
                    'ReportCmd': {'vfids_only': True},
                }, pndlu.first_line(ReportCmd.vfids_only.help)),
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
            self.log.info("Extracting %s from files...\n  %s", infos, pfiles)
            if not pfiles.nfiles():
                raise CmdException(
                    "Cmd %r must be given at least one file argument, received %d: %r!"
                    % (self.name, pfiles.nfiles(), pfiles))

        import yaml

        repspec = self.repspec
        if self.vfids_only:
            repspec.force = True  # Irrelevant to check for mismatching VFids.
            for fpath, data in repspec.get_dice_report(pfiles).items():
                if not self.verbose:
                    fpath = osp.basename(fpath)

                rep = data['report']
                yield '- %s: %s' % (fpath, rep and rep.get('vehicle_family_id'))

        else:
            for rtuple in repspec._yield_report_tuples_from_iofiles(pfiles):
                drep = _report_tuple_2_dict(*rtuple)
                fpath = rtuple[0]
                if not self.verbose:
                    fpath = osp.basename(fpath)

                yield yaml.dump({fpath: drep}, indent=2)

## test CMDS:
#    co2dice report -i ../../../compas.vinz/co2mpas/demos/co2mpas_demo-7.xlsx -o 20170207_192057-* && \
#    co2dice report  --vfids --project && co2dice report   --project && \
#    co2dice report   --project -v &&  co2dice report   --project --vfids -v
