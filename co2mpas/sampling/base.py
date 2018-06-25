# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Common code and classes shared among various Cmd-lets and Specs.
"""
from collections import namedtuple
from typing import Text, Tuple
import re

from . import CmdException
from .._vendor import traitlets as trt


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

    def find_nonfiles(self):
        import os.path as osp
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


class _FileReadingMixin(metaclass=trt.MetaHasTraits):
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
        import sys
        import os.path as osp
        from boltons.setutils import IndexedSet as iset
        from pandalone.utils import convpath

        fpaths = iset(fpaths) or ['-']
        for fpath in fpaths:
            if fpath == '-':
                msg = "Reading STDIN."
                if getattr(sys.stdin, 'isatty', lambda: False)():
                    msg += "..paste text, then [Ctrl+Z] to exit!"
                self.log.info(msg)
                yield '<STDIN>', sys.stdin.read()
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


class _StampParsingCmdMixin(_FileReadingMixin):
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
                {'_StampParsingCmdMixin': {'parse_as_tag': True}},
                "Parse input as tag."
            ),
            'stamp': (
                {'_StampParsingCmdMixin': {'parse_as_tag': False}},
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
