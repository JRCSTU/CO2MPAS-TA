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

from . import baseapp, base, project, CmdException
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from .._vendor import traitlets as trt
from .base import PFiles


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
            except Exception:
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
        assert isinstance(report, (list, dict))

    d['report'] = report

    return d


class ReporterSpec(baseapp.Spec):
    """Mines reported-parameters from co2mpas excel-files and serves them as a pandas dataframes."""

    input_head_xlref = trt.Unicode(
        '#Inputs!B1:D5:{"func": "df", "kwds": {"index_col": 0}}',
        help="The *xlref* extracting 5-10 lines from ``Inputs`` sheets "
        "of the input-file as a dataframe."
    ).tag(config=True)
    input_vfid_path = trt.Unicode(
        'flag/vehicle_family_id',
        help="the slash-separated keys of  `vehicle_family_id` into parsed excel file."
    ).tag(config=True)

    include_input_in_dice = trt.Bool(
        help="If true, inputs are included in dice report encrypted."
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
        "Return VehId & raw_data if include-in-dice flag is true"
        from co2mpas.io.excel import parse_excel_file
        from pandalone.pandata import resolve_path

        data = parse_excel_file(inp_xlsx_fpath)

        file_vfid = resolve_path(data, self.input_vfid_path)

        return file_vfid, data if self.include_input_in_dice else None

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

    def _check_deviations_are_valid(self, fpath, report: "pd.DataFrame"):
        import numpy as np
        import pandas as pd

        deviations = report.ix['CO2MPAS_deviation', :]
        is_ok = False
        self.log.debug("Deviations for file('%s'): %s" % (fpath, deviations))
        try:
            is_ok = np.isfinite(pd.to_numeric(deviations, 'coerce')).any()
        except Exception as ex:
            self.log.warning(
                "Ignored error while checking deviations(%s) for file('%s'): %s" %
                (deviations, fpath, ex))

        if not is_ok:
            return "invalid deviations: %s" % deviations

    def _encrypt_data(self, data, width=72) -> List[str]:
        """
        :param data:
             json-encodable object
        :return:
            a list of base64 lines of the encrypted & compressed msgpack of `data`
        """
        from . import crypto
        import base64
        import msgpack
        import lzma

        enc = crypto.get_encrypter(self.config)

        plainbytes = msgpack.packb(data, use_bin_type=True,
                                   unicode_errors='surrogateescape')
        ## NOTE: compression:
        #      - 270kb with GPG's `lzma` algo,
        #      - 225kb with python-zlib(EXTREME, 9)
        lzmabytes = lzma.compress(plainbytes, preset=9 | lzma.PRESET_EXTREME)
        cipherbytes = enc.encryptobj('input-report', lzmabytes,
                                     no_armor=True,
                                     no_pickle=True,
                                     extra_args=['--compress-algo', '0'])
        b64bytes = base64.b64encode(cipherbytes)
        b64str = b64bytes.decode()
        first_width = width - 7  # len('port: [')
        atext = [b64str[:first_width]] + [
            b64str[i:i + width]
            for i in range(first_width, len(b64str), width)]

        return atext

    def _decrypt_b32_lines(self, b64_lines) -> List[str]:
        from . import crypto
        import base64
        import msgpack
        import lzma

        enc = crypto.get_encrypter(self.config)

        cipher = ''.join(b64_lines)
        cipherbytes = base64.b64decode(cipher)
        plainbytes = enc.decryptobj('input-report', cipherbytes,
                                    no_pickle=True)
        plainbytes = lzma.decompress(plainbytes)

        data = msgpack.unpackb(plainbytes,
                               encoding='utf-8',
                               unicode_errors='surrogateescape')
        return data

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
        import pandalone.utils as pndlu

        def check_vfid_missmatch(fpath, file_vfid):
            nonlocal expected_vfid

            if expected_vfid is None:
                expected_vfid = file_vfid
            elif expected_vfid != file_vfid:
                return ("mismatch `vehicle_family_id` between this file(%s) and the rest: "
                        "'%s' != expected('%s')'" %
                        (fpath, file_vfid, expected_vfid))

        rtuples = []

        input_report = []  # a list of dicts {file: ..., input_data: raw_data}
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

            if inp_data:
                input_report.append({'file': fpath, 'input_data': inp_data})

        for fpath in iofiles.out:
            fpath = pndlu.convpath(fpath)
            file_vfid, dice_report = self._extract_dice_report_from_output(fpath)
            msg1 = self._check_is_ta(fpath, dice_report)
            msg2 = check_vfid_missmatch(fpath, file_vfid)
            msg3 = self._check_deviations_are_valid(fpath, dice_report)
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

        if input_report:
            self.log.info("Attaching input-report...")
            rtuples.append(('inputs.yaml', 'cipher', self._encrypt_data(input_report)))

        return rtuples

    def extract_dice_report(self, iofiles: PFiles, expected_vfid=None):
        tuples = self._make_report_tuples_from_iofiles(iofiles, expected_vfid)
        report = OrderedDict((file_tuple[0], _report_tuple_2_dict(*file_tuple))
                             for file_tuple
                             in tuples)
        return report

    def _collect_ciphers(self, records):
        return [rec
                for rec in records
                if rec['iokind'] == 'cipher']

    def unlock_report_records(self, records: str):
        """
        Unlock any encrypted data-records, if attached in the dice-report

        :param dreport_text:
            the dice report (not stamp) as a list of 3-item dicts::

                {'file: 'inputs',  # can have different name
                 'kind: 'cipher',
                 'report: ...  # an array with base64 lines
                }

        :return:
            The same record with plain text in `report` key, replaced.

        """
        ciphered_recs = self._collect_ciphers(records)
        if not ciphered_recs:
            raise CmdException('No encrypted records in dice-report!')

        for rec in ciphered_recs:
            cipher = rec['report']
            rec['report'] = self._decrypt_b32_lines(cipher)
            rec['iokind'] = 'plaintext'

        return ciphered_recs


###################
##    Commands   ##
###################


class ReportCmd(baseapp.Cmd):
    """
    Subcommands to extract dice-reports from co2mpas files or unlock encrypted reports.
    """

    def __init__(self, **kwds):
        kwds.setdefault('subcommands', baseapp.build_sub_cmds(*all_subcmds))
        super().__init__(**kwds)


class ExtractCmd(baseapp.Cmd):
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
            'conf_classes': [project.ProjectsDB, project.Project, ReporterSpec],
            'cmd_aliases': {
                ('i', 'inp'): ('ExtractCmd.inp', type(self).inp.help),
                ('o', 'out'): ('ExtractCmd.out', type(self).out.help),
            },
            'cmd_flags': {
                'project': ({
                    'ExtractCmd': {'project': True},
                }, ExtractCmd.project.help),
                'vfids': ({
                    'ExtractCmd': {'vfids_only': True},
                }, ExtractCmd.vfids_only.help),
                'with-inputs': ({
                    'ReporterSpec': {'include_input_in_dice': True},
                }, ReporterSpec.include_input_in_dice.help),
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

        repspec = ReporterSpec(config=self.config)
        if self.vfids_only:
            repspec.force = True  # Irrelevant to check for mismatching VFids.
            for fpath, data in repspec.extract_dice_report(pfiles).items():
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

                yield yaml.dump([drep], indent=2, width=76)


class UnlockCmd(baseapp.Cmd, base._StampParsingCmdMixin):
    """
    Decrypt data attached in dice-reports (also when wrapped in stamps) into STDOUT.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<dice-or-stamp-file>...]

    - If no file or '-' given, read STDIN.
      Use the PYTHONIOENCODING envvar to change its encoding.
      See: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONIOENCODING
    """

    examples = trt.Unicode("""
    $ co2dice report unlock tests/sampling/cipherdice.txt
    2018-06-25 22:43:32: INFO:co2mpas.sampling.report.UnlockCmd:Parsing file '/path/to/cipherdice.txt' as TAG...
    2018-06-25 22:43:34: INFO:co2mpas.sampling.report.UnlockCmd:Unlocking '/path/to/cipherdice.txt' as TAG
    - /path/to/cipherdice.txt:
      - file: inputs.yaml
        iokind: cipher
        report:
        - file: /path/to/original/co2mpas_demo-1.xlsx
          input_data:
            base:
              input:
                calibration:
                  wltp_h:
                    active_cylinder_ratios: [1]
                    alternator_efficiency: 0.67
                    ...
    """)

    def __init__(self, **kwds):
        from . import crypto

        dkwds = {'conf_classes': [ReporterSpec, crypto.EncrypterSpec]}
        dkwds.update(kwds)
        super().__init__(**dkwds)

    _reporter: ReporterSpec = None

    @property
    def reporter(self) -> ReporterSpec:
        if not self._reporter:
            self._reporter = ReporterSpec(config=self.config)
        return self._reporter

    def run(self, *args):
        import yaml
        from pandalone.pandata import resolve_path

        for fpath, is_tag, verdict in self.yield_verdicts(*args):
            inp_type = 'TAG' if is_tag else 'STAMP'
            self.log.info("Unlocking '%s' as %s", fpath, inp_type)
            if 'commit_msg' in verdict:
                dpath = 'commit_msg/data'
            elif 'report' in verdict:
                dpath = 'report/commit_msg/data'
            else:
                self.warning("Skipping %s from '%s' due to unexpected keys: %s",
                             inp_type, fpath, list(verdict))
                continue

            records = resolve_path(verdict, dpath, None)
            plain_recs = self.reporter.unlock_report_records(records)

            yield yaml.dump({fpath: plain_recs})


all_subcmds = (
    ExtractCmd,
    UnlockCmd,
)

## test CMDS:
#    co2dice report extract  -i ./co2mpas/demos/co2mpas_demo-7.xlsx -o 20170207_192057-* && \
#    co2dice report extract  --vfids --project && co2dice report   --project && \
#    co2dice report extract  --project -v &&  co2dice report   --project --vfids -v
