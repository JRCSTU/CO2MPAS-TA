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
from .._vendor.traitlets import config as trc
from .._vendor.traitlets import traitlets as trt


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
        badfiles = self.find_nonfiles()
        if badfiles:
            raise CmdException("%s: cannot find %i file(s): %s" %
                               (name, len(badfiles), badfiles))


#: Allow creation of PFiles with partial arguments.
PFiles.__new__.__defaults__ = ([], ) * len(all_io_kinds)


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
        from boltons.setutils import IndexedSet as iset
        from pandalone.utils import convpath

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
                fpath = convpath(fpath)
                if not osp.exists(fpath):
                    self.log.error("File to read '%s' not found!", fpath)
                    continue

                try:
                    with io.open(fpath, 'rt') as fin:
                        text = fin.read()

                    yield fpath, text
                except Exception as ex:
                    self.log.error(
                        "Reading file-path '%s' failed due to: %r",
                        fpath, ex,
                        exc_info=self.verbose)  # WARN: from `baseapp.Spec`
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


class FileOutputMixin(trc.Configurable):
    write_fpath = trt.Unicode(
        help="Write report into this file, if given; overwriten if it already exists."
    ).tag(config=True)

    write_append = trt.Bool(
        help="If true, do not overwrite existing files - append into them."
    ).tag(config=True)

    write_kwds = trt.Dict(
        {'encoding': 'utf-8}'},
        help="Keywords sent to `open()`, like encoding and encoding-errors, etc."
    ).tag(config=True)

    def _open_file_mode(self):
        return 'at' if self.write_append else 'wt'

    def write_file(self, txt, wfpath=None):
        if not wfpath:
            wfpath = self.write_fpath
        if not wfpath:
            self.log.warning('Cannot write output file when no fpath given!')
            return

        wfpath = pndlu.convpath(wfpath)
        self.log.info('%s report into: %s',
                      'Appending' if self.write_append else 'Writing', wfpath)
        file_mode = self._open_file_mode()
        with open(wfpath, file_mode, **self.write_kwds) as fd:
            fd.write(txt)


## TODO: enforce --write aliase from mixin-constructor.
write_fpath_alias_kwd = {
    ('W', 'write-fpath'): ('FileOutputMixin.write_fpath', FileOutputMixin.write_fpath.help)
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
        default_value=[':64', '-32:'],
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
