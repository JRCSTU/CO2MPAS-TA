# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Common code and classes shared among various Cmd-lets and Specs.
"""

from collections import namedtuple
from typing import Optional, Text, Tuple
import re
import sys

import os.path as osp

from . import CmdException, slicetrait
from ._vendor.traitlets import config as trc
from ._vendor.traitlets import traitlets as trt


#: Define VehicleFamilyId (aka ProjectId) pattern here not to import the world on use.
#: Referenced by :meth:`.sampling.tstamp.TstampReceiver.extract_dice_tag_name()`.
#:
#: NOTE: keep synced with ``in pCO2SIM/src/co2sim/__init__.py``!
vehicle_family_id_pattern = r'''
    (?:
        (IP|RL|RM|PR) - (\d{2}) - ([A-Z0-9_]{2,3}) - (\d{4}) - (\d{4})
    )
    |
    (?:
        IP - ([A-Z0-9_]{2,15}) - ([A-Z0-9_]{3}) - ([01])
    )
'''


def convpath(fpath, abs_path=None, exp_user=True, exp_vars=True):
    """
    Override `abs_path` functioning..

    :param abs_path:
        3-state, None: expand if it exists
        Useful to preserve POSIX fpaths under Windows, e.g. ``/dev/null``.
    """
    from pandalone.utils import convpath

    if abs_path is None:
        fpath = convpath(fpath, False, exp_user, exp_vars)
        if osp.exists(fpath):
            afpath = convpath(fpath, True, False, False)
            if osp.exists(afpath):
                fpath = afpath
    else:
        fpath = convpath(fpath, abs_path, exp_user, exp_vars)

    return fpath


_file_arg_regex = re.compile('(inp|out)=(.+)', re.IGNORECASE)

all_io_kinds = tuple('inp out other'.split())


class PFiles(namedtuple('PFiles', all_io_kinds)):
    """
    Holder of project-files stored in the repository.

    :ivar inp:   ``[fname1, ...]``
    :ivar out:   ``[fname1, ...]``
    :ivar other: ``[fname1, ...]``
    """
    ## INFO: Defined here to avoid circular deps between report.py <-> project.py,
    #  because it is used in their function declarations.

    @staticmethod
    def io_kinds_list(*io_kinds) -> Tuple[Text]:
        """
        :param io_kinds:
            if none specified, return all kinds,
            otherwise, validates and converts everything into a string.
        """
        if not io_kinds:
            io_kinds = all_io_kinds
        else:
            assert not (set(io_kinds) - set(all_io_kinds)), (
                "Invalid io-kind(s): ", set(io_kinds) - set(all_io_kinds))
        return tuple(set(io_kinds))

    def nfiles(self):
        return sum(len(f) for f in self._asdict().values())

    ## CARE needed e.g. when vali-getting them fromm transitions events.
    #def __bool__(self):
    #    return self.nfiles() == 0

    def find_nonfiles(self):
        import itertools as itt

        return [fpath for fpath in
                itt.chain(self.inp, self.out, self.other)
                if not osp.isfile(fpath)]

    def check_files_exist(self, name):
        from .utils import joinstuff

        badfiles = self.find_nonfiles()
        if badfiles:
            raise CmdException("%s: cannot find %i file(s): %s" %
                               (name, len(badfiles),
                                joinstuff((convpath(f, abs_path=True) for f in badfiles),
                                          '', '\n  %s')))

    def build_cmd_line(self, **convpath_kwds):
        "Build cli-options for `project append` preserving pair-order."
        import string
        import itertools as itt

        ok_file_chars = set(string.ascii_letters + string.digits + '-')

        def quote(s):
            "best-effort that works also on Windows (no single-quote)"
            if set(s) - ok_file_chars:
                s = '"%s"' % s
            return s

        def append_opt(l, opt, fpath):
            if fpath:
                l.append(opt)
                l.append(quote(convpath(fpath, **convpath_kwds)))

        args = []
        for inp, out in itt.zip_longest(self.inp, self.out, fillvalue=()):
            append_opt(args, '--inp', inp)
            append_opt(args, '--out', out)

        args.extend(quote(convpath(f, **convpath_kwds))
                    for f in self.other)

        return args


#: Allow creation of PFiles with partial arguments.
PFiles.__new__.__defaults__ = ([], ) * len(all_io_kinds)


class EmailStamperWarning(trc.Configurable):
    mute = trt.Bool().tag(config=True)

    def __init__(self, **kwds):
        import logging

        super().__init__(**kwds)

        if not self.mute:
            logging.getLogger('EmailStamping').warning(
                "SINCE co2mpas-1.9.x (summer 2018) THIS COMMAND IS DEPRECATED."
                "\n  Please migrate to WebStamper.")


class FileReadingMixin(metaclass=trt.MetaHasTraits):
    """
    Facilitates commands reading input files given in the command-line.

    If no file or '-' given, read STDIN.
    Use the PYTHONIOENCODING envvar to change its encoding.
    See: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONIOENCODING
    """

    def yield_files(self, *fpaths):
        """
        :return:
            a 2 tuple `(fpath, file_text)`
        """

        import io
        import os
        from boltons.setutils import IndexedSet as iset

        fpaths = iset(fpaths) or ['-']
        for fpath in fpaths:
            if fpath == '-':
                msg = "Reading STDIN."
                if getattr(sys.stdin, 'isatty', lambda: False)():
                    msg += ("..paste text, then [Ctrl+%s] to exit!" %
                            'Z' if sys.platform == 'win32' else 'D')
                self.log.info(msg)
                text = sys.stdin.read()
                yield "<STDIN: %i-chars>" % len(text), text
            else:
                fpath = convpath(fpath, abs_path=False)
                if osp.exists(fpath):
                    afpath = convpath(fpath, abs_path=True)
                    if osp.exists(afpath):
                        fpath = afpath
                else:
                    self.log.error("File to read '%s' not found!"
                                   "\n  CWD: %s", fpath, os.curdir)
                    continue

                try:
                    with io.open(fpath, 'rt') as fin:
                        text = fin.read()

                    yield fpath, text
                except Exception as ex:
                    self.log.error(
                        "Reading file-path '%s' failed due to: %r",
                        fpath, ex,
                        exc_info=self.verbose)  # WARN: from `cmdlets.Spec`
                    continue


class StampParsingCmdMixin(FileReadingMixin):
    """
    Parses dice-reports & stamps from input files.
    """
    parse_as_tag = trt.Bool(
        default_value=None, allow_none=True,
        help="""
        true: tag given, false: stamp given, None: guess based on '"' chars.
        """
    ).tag(config=True)

    def _is_parse_tag(self, ftext):
        if self.parse_as_tag is None:
            ## tstamper produces 57 x '*'.
            return ('#' * 50) not in ftext
        else:
            return bool(self.parse_as_tag)

    def __init__(self, **kwds):
        flags = kwds.pop('cmd_flags', {})
        flags.update({
            'tag': (
                {'StampParsingCmdMixin': {'parse_as_tag': True}},
                "Parse input as tag."
            ),
            'stamp': (
                {'StampParsingCmdMixin': {'parse_as_tag': False}},
                "Parse input as stamp."
            ),
        })
        super().__init__(cmd_flags=flags, **kwds)

    _stamp_parser = None

    @property
    def stamp_parser(self):
        from . import tstamp

        if not self._stamp_parser:
            self._stamp_parser = tstamp.TstampReceiver(config=self.config)

        return self._stamp_parser

    def yield_verdicts(self, *fpaths, ex_handler=None):
        """
        :param ex_handler:
            a ``callable(fpath, ex)`` to handle any exceptions
        :return:
            a 3 tuple `(fpath, is_tag, verdict)` where `verdict` is
            a dict with these keys if parsed *stamp* (``is_tag == False``)::

                report:
                    creation_date, data, expire_timestamp, fingerprint, key_id, key_status,
                    parts, pubkey_fingerprint, sig_timestamp, signature_id, status,
                    stderr, timestamp, trust_level, trust_text, username, valid, commit_msg,
                    project, project_source
                tstamp:
                    valid, fingerprint, creation_date, timestamp, signature_id, key_id,
                    username, key_status, status, pubkey_fingerprint, expire_timestamp,
                    sig_timestamp, trust_text, trust_level, data, stderr, mail_text,
                    parts, stamper_id
                dice:
                    tag, issuer, issue_date, stamper, dice_date, hexnum, percent, decision

            ... and the `report` keys from above, if it parsed a dice-report.
            :raise:
                any exception
        """
        for fpath, ftext in self.yield_files(*fpaths):
            try:
                is_tag = self._is_parse_tag(ftext)
                self.log.info("Parsing file '%s' as %s...",
                              fpath, 'TAG' if is_tag else 'STAMP')

                if is_tag:
                    verdict = self.stamp_parser.parse_signed_tag(ftext)
                else:
                    verdict = self.stamp_parser.parse_tstamp_response(ftext)

                if fpath != '-':
                    verdict['fpath'] = fpath

                yield fpath, is_tag, verdict
            except Exception as ex:
                if ex_handler:
                    ex_handler(fpath, ex)
                else:
                    raise


class ReportsKeeper(trc.Configurable):
    """Manages a list of loggers to write content into."""

    default_reports_fpath = trt.Unicode(
        '+~/.co2dice/reports.txt',
        allow_none=True,
        help="""
        The log-file where to keep a record of all intermediate Dices & Stamps exchanged.

        This value gets appended in the `write_fpaths` list by certain dice commands.
        If set to defined/empty, no log is kept.
        """

    ).tag(config=True, envvar='CO2DICE_REPORTS_FPATH')

    write_fpaths = trt.List(
        trt.Unicode(),
        help="""
        Filepaths where to log stuff (Dices, Stamps & Decisions).

        - For paths starting with '+', stuff is appended, separated with a timestamp
          and a title.
        - Otherwise (files without a '+' prefix), any existing files are overwritten,
          and stuff gets written without separator lines.
        """
    ).tag(config=True)

    @trt.validate('write_fpaths')
    def _ensure_default_reports_fpath(self, p):
        v = p.value
        default_fpath = self.default_reports_fpath
        if default_fpath and default_fpath not in v:
            v.append(default_fpath)

        return v

    report_log_format = trt.Unicode(
        "%(asctime)s:%(module)s.%(funcName)s:%(msg)s",
        help="""
        The logging formatting of a message written (`log_format` was taken by traits).

        https://docs.python.org/3/library/logging.html#logrecord-attributes

        WARN: DON'T USE ``%(message)s`` (interpolated)!!
        """,
    ).tag(config=True)

    report_date_format = trt.Unicode(
        None, allow_none=True,
        help="""
        The logging formatting for date-times for the messages written.

        https://docs.python.org/3/howto/logging.html#displaying-the-date-time-in-messages
        """
    ).tag(config=True)

    rotating_handler_cstor_kwds = trt.Dict(
        {
            "maxBytes": 2 * 1024 * 1024,
            "backupCount": 0,
            "encoding": 'utf-8',
            #"delay": False  # may leave it to default
        },
        help="""
        Cstor kwds for `RotatingFileHandler`, except the `filename` and `mode`.

        https://docs.python.org/3/library/logging.handlers.html#logging.handlers.RotatingFileHandler"  # noqa
        """
    ).tag(config=True)

    def _collect_wfpaths(self, *extra_fpaths):
        wfpaths = list(self.write_fpaths)
        wfpaths.extend(extra_fpaths)

        return [f.strip() for f in wfpaths]

    def get_non_default_fpaths(self, *extra_fpaths):
        """Return any other write-fpaths given, except :field:`default_reports_fpath`."""
        wfpaths = set(self._collect_wfpaths(*extra_fpaths))
        wfpaths.discard(self.default_reports_fpath)

        return wfpaths

    def _logger_configed(self, log):
        """The ``log.name`` becomes the (expanded) filename to write into."""
        from logging import Formatter

        wfpath = log.name
        if wfpath.startswith('+'):
            append = True
            wfpath = wfpath[1:]
        else:
            append = False
        wfpath = convpath(wfpath, True)

        if append:
            from logging.handlers import RotatingFileHandler

            handler = RotatingFileHandler(
                wfpath, 'a', **self.rotating_handler_cstor_kwds)

            formatter = Formatter(self.report_log_format, self.report_date_format)
            handler.setFormatter(formatter)
        else:
            from logging import FileHandler

            handler = FileHandler(wfpath, 'w', encoding='utf-8')

        handler.setLevel(0)

        log.addHandler(handler)
        log.setLevel(0)
        log.propagate = False

        return log

    def _get_logger(self, wfpath):
        import logging

        if wfpath in logging.Logger.manager.loggerDict:  # @UndefinedVariable
            return logging.getLogger(wfpath)

        return self._logger_configed(logging.getLogger(wfpath))

    def store_report(self, txt, title=None, *extra_fpaths):
        """
        Append text in all `write_fpaths` using customized file-rolling loggers.

        :param title:
            Without it, write-notification is not written in main-program's logs.
        :return:
            the number of written files

        - By default, in the customized loggers it is always added
          the :field:`default_reports_fpath` (if defined).
        """
        from .utils import joinstuff

        wfpaths = self._collect_wfpaths(*extra_fpaths)
        if not txt:
            if title:
                self.log.warning(
                    "Given empty '%s' report to store in filepaths: %s",
                    title, joinstuff(wfpaths))
            return

        titled_txt = "  ----((%s))----  \n%s" % (title, txt) if title else txt

        written = 0
        for wfpath in wfpaths:
            if not wfpath:
                self.log.warning(
                    "Cannot store report '%s', missing filepaths!",
                    title, txt and txt[:32])
                continue

            log = self._get_logger(wfpath)
            log.info(titled_txt if wfpath.startswith('+') else txt)
            written += 1

        ## Don't inform about title-lines.
        #
        if title:
            self.log.info(
                "Stored %i-lines %s into %i file(s): %s",
                txt.count('\n'), title, len(wfpaths), joinstuff(wfpaths))

        return written


## TODO: enforce --write aliase from mixin-constructor.
reports_keeper_alias_kwd = {
    ('W', 'write-fpath'): ('ReportsKeeper.write_fpaths', ReportsKeeper.write_fpaths.help)
}


class ShrinkingOutputMixin(trc.Configurable):
    shrink = trt.Bool(
        None,
        allow_none=True,
        help="""
        A 3-state bool, deciding whether to shrink output according to `shrink_slices` param.

        - If none, shrinks if STDOUT is interactive (console).
        - Does not affect results written in `write-fpath` param.
        """
    ).tag(config=True)

    shrink_nlines_threshold = trt.Int(
        128,
        help="The maximum number of lines allowed to print without shrinking."
    ).tag(config=True)

    shrink_slices = trt.Union(
        (slicetrait.Slice(), trt.List(slicetrait.Slice())),
        default_value=[':48', '-32:'],
        help="""
        A slice or a list-of-slices applied when shrinking results printed.

        Examples:
            ':100'
            [':100', '-64:']
        """
    ).tag(config=True)

    def should_shrink_text(self, txt_lines):
        return (len(txt_lines) > self.shrink_nlines_threshold and
                (self.shrink or
                (self.shrink is None and sys.stdout.isatty())))

    def shrink_text(self, txt: Optional[str], shrink_slices=None) -> Optional[str]:
        shrink_slices = shrink_slices or self.shrink_slices
        if not (shrink_slices and txt):
            return txt

        txt_lines = txt.splitlines()

        if self.should_shrink_text(txt_lines):
            shrinked_txt_lines = slicetrait._slice_text_lines(txt_lines,
                                                              shrink_slices)
            self.log.warning("Shrinked result text-lines from %i --> %i."
                             "\n  ATTENTION: result is not valid for stamping/validation!"
                             "\n  Write it to a file with `--write-fpath`(`-W`).",
                             len(txt_lines), len(shrinked_txt_lines))
            txt = '\n'.join(shrinked_txt_lines)

        return txt


## TODO: enforcethe --shrink flags from mixin-constructor.
shrink_flags_kwd = {
    'shrink': (
        {'ShrinkingOutputMixin': {'shrink': True}},
        "Omit lines of the report to facilitate console reading."
    ),
    'no-shrink': (
        {'ShrinkingOutputMixin': {'shrink': False}},
        "Print full report - don't omit any lines."
    ),
}
