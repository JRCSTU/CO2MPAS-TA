#!/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""A *project* stores all CO2MPAS files for a single vehicle, and tracks its sampling procedure. """
from co2mpas._vendor.traitlets.traitlets import TraitError
from collections import (defaultdict, OrderedDict, namedtuple)  # @UnusedImport
from typing import (
    Any, Union, List, Dict, Sequence, Iterable, Optional, Text, Tuple, Callable)  # @UnusedImport
import contextlib
import copy
import io
import os
import re
import sys

from boltons.setutils import IndexedSet as iset
from toolz import itertoolz as itz
from transitions.core import MachineError
import transitions
import yaml  # TODO: Upgrade unaintained yaml to ruamel

import functools as fnt
import os.path as osp
import pandalone.utils as pndlu
import textwrap as tw

from . import baseapp, dice, CmdException
from .. import (__version__, __updated__, __uri__, __copyright__, __license__,  # @UnusedImport
                __dice_report_version__)
from .._vendor import traitlets as trt
from .._vendor.traitlets import config as trtc
from .base import PFiles


git_project_regex = re.compile(r'^\w[\w-]+$')

_git_messaged_obj = re.compile(r'^(:?object|tag) ')
_after_first_empty_line_regex = re.compile(r'\n\r?\n')


def _mydump(obj, indent=2, **kwds):
    return yaml.dump(obj, indent=indent, **kwds)


_CommitMsgVer_regex = re.compile(r'(?<!\w)v:[ \t\r\n]+(\d+)\.(\d+)\.(\d+)(?!\d)')


class _CommitMsg(namedtuple('_CommitMsg', 'v a p s data')):
    """
    A commit-message is a list like ``[headline, dataline, ...]``.

    For this version(:data:`__dice_report_version__`) the format is:

    - The `headline` is a dictionary with this ordered fields:
      - mver
      - action
      - proj
      - state
    - Only one `dataline` expected: report
    - The first 4 fields are smallest possible to fit headline in 78 chars:
      - v: mesage-version
      - a: action
      - p: project
      - s: status
    """

    @classmethod
    def _check_commit_msg_version(cls, msg_starting_txt):
        ## TODO: Parse msg-version from raw text first.
        prog_ver = __dice_report_version__.split('.')
        m = _CommitMsgVer_regex.search(msg_starting_txt)
        if not m:
            raise ValueError(
                "incompatible message, cannot parse its version, "
                "expected version %s', message header: \n%s)," %
                (__dice_report_version__, msg_starting_txt))

        major, minor, micro = m.group(1, 2, 3)
        if int(major) != int(prog_ver[0]) or int(minor) > int(prog_ver[1]):
            raise ValueError(
                "incompatible message version '%s', expected '%s'" %
                ('.'.join((major, minor, micro)), __dice_report_version__))

    def dump_commit_msg(self, **kwds):
        cdic = self._asdict()
        data = cdic.pop('data')
        msg = _mydump([cdic], **kwds)
        if data:
            if not isinstance(data, list):
                data = [data]
                kwds['default_flow_style'] = False
            msg += '\n' + _mydump(data, **kwds)

        return msg

    @classmethod
    def parse_commit_msg(cls, cmsg_txt: Text):
        """
        :return: a :class:`_CommitMsg` instance, or fails if cannot parse.
        """
        try:
            ## Are we parsing `git cat-object tag foo`?
            #
            if _git_messaged_obj.match(cmsg_txt):
                m = _after_first_empty_line_regex.search(cmsg_txt)
                cmsg_txt = cmsg_txt[m.end():]

            _CommitMsg._check_commit_msg_version(cmsg_txt[:30])
            m = yaml.load(cmsg_txt)
            if not isinstance(m, list) or not m:
                raise ValueError("expected a non-empty list")

            headline = m[0]
            cmsg = _CommitMsg(data=m[1:], **headline)

            return cmsg
        except Exception as ex:
            import logging

            msg = "Failed parsing commit message due to: %r\nmsg:\n%s" % (
                ex, tw.indent(cmsg_txt, "  "))
            logging.getLogger(cls.__name__).debug(msg, exc_info=1)
            raise CmdException(msg)


_PROJECTS_PREFIX = 'projects/'
_HEADS_PREFIX = 'refs/heads/'
_PROJECTS_FULL_PREFIX = _HEADS_PREFIX + _PROJECTS_PREFIX


def _is_project_ref(ref: 'git.Reference') -> bool:
    return bool(ref) and ref.name.startswith(_PROJECTS_PREFIX)


def _ref2pname(ref: 'git.Reference') -> Text:
    return ref.path[len(_PROJECTS_FULL_PREFIX):]


def _pname2ref_path(pname: Text) -> Text:
    if pname.startswith(_HEADS_PREFIX):
        pass
    elif not pname.startswith(_PROJECTS_PREFIX):
        pname = '%s%s' % (_PROJECTS_FULL_PREFIX, pname)
    return pname


def _pname2ref_name(pname: Text) -> Text:
    if pname.startswith(_HEADS_PREFIX):
        pname = pname[len(_HEADS_PREFIX):]
    elif not pname.startswith(_PROJECTS_PREFIX):
        pname = '%s%s' % (_PROJECTS_PREFIX, pname)
    return pname


def _get_ref(refs, refname: Text, default: 'git.Reference'=None) -> 'git.Reference':
    return refname and refname in refs and refs[refname] or default


_DICES_PREFIX = 'dices/'


def _is_dice_tag(ref: 'git.Reference') -> bool:
    return ref.name.startswith(_DICES_PREFIX)


def _tname2ref_name(tname: Text) -> Text:
    if not tname.startswith(_DICES_PREFIX):
        tname = '%s%s' % (_DICES_PREFIX, tname)
    return tname


def _yield_project_refs(repo, *pnames: Text):
    """Yields given pnames are git-python refs, or all if no pname given."""
    if pnames:
        pnames = [_pname2ref_path(p) for p in pnames]
    for ref in repo.heads:
        if _is_project_ref(ref) and not pnames or ref.path in pnames:
            yield ref


def _yield_dices_tags(repo, *pnames: Text):
    if pnames:
        pnames = [_tname2ref_name(p) for p in pnames]
    for ref in repo.tags:
        if (_is_dice_tag(ref) and not pnames or
                any(ref.name.startswith(p) for p in pnames)):
            yield ref


def _find_dice_tag(repo, pname, max_dices_per_project,
                   fetch_next=False) -> Union[Text, 'git.TagReference']:
    """Return None if no tag exists yet."""
    tref = _tname2ref_name(pname)
    tags = repo.tags
    for i in range(max_dices_per_project):
        tagname = '%s/%d' % (tref, i)
        if tagname not in tags:
            if fetch_next:
                return tagname
            else:
                if i == 0:
                    return None
                else:
                    tagname = '%s/%d' % (tref, i - 1)

                    return tags[tagname]

    raise CmdException("Too many dices(at least %d) for project '%s'!"
                       "\n  Maybe delete project and start over"
                       "(or use `max_dices_per_project`)?" %
                       (i + 1, pname))


def _read_dice_tag(repo, tag: Union[Text, 'git.TagReference']) -> Text:
    ## TODO: Attempt parsing dice-report when reading tag.
    if isinstance(tag, str):
        tag = repo.tags[tag]
    return tag.tag.data_stream.read().decode('utf-8')


#transitions.logger.level = 50 ## FSM logs annoyingly high.
def _evarg(event, dname, dtype=None, none_ok=False, missing_ok=False):
    """
    :param dtype:
        A single or tuple of types, passed to `isinstance()`.
    """
    kwargs = event.kwargs

    _ = object()
    data = kwargs.get(dname, _)
    if data is _:
        assert missing_ok, (
            "Missing event-data(%r) from event: %s" % (dname, vars(event)))
        return

    if dtype:
        assert none_ok and data is None or isinstance(data, dtype), (
            "Expected TYPE of event-data(%r) is %r, but was %r!"
            "\n  data: %s\n  event: %s" %
            (dname, dtype, type(data), data, vars(event)))
    return data


###################
##     Specs     ##
###################

class ProjectSpec(dice.DiceSpec):
    """Common configurations for both ProjectsDB & ProjectFSM."""

    max_dices_per_project = trt.Int(
        5,
        help="""Number of dice-attempts allowed to be forced for a project."""
    ).tag(config=True)

    def extract_uid_from_report(self, report: Text) -> Text:
        from . import crypto

        git_auth = crypto.get_git_auth(self.config)
        verdict = git_auth.verify_git_signed(report.encode())
        assert verdict, _mydump(vars(verdict))

        return crypto.uid_from_verdict(verdict)


class Project(transitions.Machine, ProjectSpec):
    """The Finite State Machine for the currently checked-out project."""

    dry_run = trt.Bool(
        help="Process actions but do not actually commit/tag results in the project."
    ).tag(config=True)

    git_desc_width = trt.Int(
        76,  # NOTE: not respected from report.ExtractCmd, number duplicated.
        allow_none=False,
        help="""
        The width of the textual descriptions when committing and tagging Git objects.

        The email sent for timestamping is generated from tag-descriptions.

        According to RFC5322, 78 is the maximum width for textual emails;
        mails with width > 78 may be sent as HTML-encoded and/or mime-multipart.
        QuotedPrintable has 76 as limit, probably to account for CR+NL
        end-ofline chars
        """
    ).tag(config=True)

    recertify = trt.Bool(
        help="When true, allow to `append` files in a project "
        "on `sampe/nosample` states."
    ).tag(config=True)

    commit_msg_lines_limit = trt.Int(
        30,
        config=True,
        help="""
            Clip commit-msg to those lines due to GitPython lib's technical limitation.

            GitPython cannot receive the message from a file, and in case
            the dice-report contains the encrypted input, it reaches the limit
            of characters in command-line when launching the `git` process.
        """
    )

    @classmethod
    @fnt.lru_cache()
    def _project_zygote(cls) -> 'Project':
        """Cached Project FSM used by :meth:`Project.new_instance()` to speed-up construction."""
        return cls('<zygote>', None)

    @classmethod
    def new_instance(cls, pname, repo, config) -> 'Project':
        """
        Avoid repeated FSM constructions by forking :meth:`Project._project_zygote()`.

        For an example, see ::meth:`ProjectsDB._conceive_new_project()`.

        INFO: set here any non-serializable fields for :func:`fnt.lru_cache()` to work.
        """
        p = Project._project_zygote()

        clone = copy.deepcopy(p)
        clone.pname = pname
        clone.id = pname + ": "
        clone.repo = repo
        clone.update_config(config)

        return clone

    #: The commit/tag of the recent transition
    #: as stored by :meth:`_cb_commit_or_tag()`.
    result = None

    #: Any problems when state 'INVALID'.
    error = None

    def __str__(self, *args, **kwargs):
        #TODO: Obey verbosity on project-str.
        if self.error:
            s = 'Project(%s: %s, error: %s)' % (self.pname, self.state, self.error)
        else:
            s = 'Project(%s: %s)' % (self.pname, self.state)
        return s

    def _report_spec(self):
        from . import report
        return report.ReporterSpec(config=self.config)

    def _tstamp_sender_spec(self):
        from . import tstamp
        return tstamp.TstampSender(config=self.config)

    def _tstamp_receiver_spec(self):
        from . import tstamp
        return tstamp.TstampReceiver(config=self.config)

    def _is_force(self, event):
        accepted = event.kwargs.get('force', self.force)
        if not accepted:
            self.log.warning('Transition %s-->%s denied!\n  Use force if you must.',
                             event.transition.source, event.transition.dest)
        return accepted

    def _is_recertify(self, event):
        accepted = (event.kwargs.get('recertify', self.recertify))
        if not accepted:
            self.log.warning("Transition %s-->%s denied!\n  "
                             "Use `--recertify` if you must.",
                             event.transition.source, event.transition.dest)
        return accepted

    def _is_dry_run(self, event):
        return self.dry_run

    def _is_inp_files(self, event):
        pfiles = _evarg(event, 'pfiles', PFiles)
        accepted = bool(pfiles and pfiles.inp and
                        not (pfiles.out))

        return accepted

    def _is_out_files(self, event):
        pfiles = _evarg(event, 'pfiles', PFiles)
        accepted = bool(pfiles and pfiles.out and
                        not (pfiles.inp))

        return accepted

    def _is_inp_out_files(self, event):
        pfiles = _evarg(event, 'pfiles', PFiles)
        accepted = bool(pfiles and pfiles.inp and pfiles.out)

        return accepted

    def _is_other_files(self, event):
        pfiles = _evarg(event, 'pfiles', PFiles)
        accepted = bool(pfiles and pfiles.other and
                        not (pfiles.inp or pfiles.out))

        if not accepted:
            self.log.debug('Transition %s-->%s denied, had `out` files',
                           event.transition.source, event.transition.dest)
        return accepted

    def _is_decision_yes(self, event):
        return _evarg(event, 'decision', bool)

    def __init__(self, pname, repo, **kwds):
        """DO NOT INVOKE THIS; use performant :meth:`Project.new_instance()` instead."""
        self.pname = pname
        self.rpo = repo
        states = [
            'BORN', 'INVALID', 'empty', 'wltp_out', 'wltp_inp', 'wltp_iof', 'tagged',
            'mailed', 'nosample', 'sample', 'nedc',
        ]
        trans = yaml.load(
            # Trigger        Source     Dest-state    Conditions? unless before after prepare
            """
            - {trigger: do_invalidate, source: '*', dest: INVALID, before: _cb_invalidated}

            - [do_createme,  BORN,    empty]

            - [do_addfiles,  empty,      wltp_iof,     _is_inp_out_files]

            - [do_addfiles,  empty,      wltp_inp,     _is_inp_files]
            - [do_addfiles,  empty,      wltp_out,     _is_out_files]

            - [do_addfiles,  [wltp_inp,
                              wltp_out,
                              tagged],   wltp_iof,     [_is_inp_out_files, _is_force]]

            - [do_addfiles,  wltp_inp,   wltp_inp,     [_is_inp_files, _is_force]]
            - [do_addfiles,  wltp_inp,   wltp_iof,     _is_out_files]

            - [do_addfiles,  wltp_out,   wltp_out,     [_is_out_files, _is_force]]
            - [do_addfiles,  wltp_out,   wltp_iof,     _is_inp_files]

            - [do_addfiles,  wltp_iof,   wltp_iof,     _is_force        ]

            - [do_addfiles,  [sample,
                              nosample], wltp_iof,     [_is_inp_out_files, _is_recertify]]
            - [do_addfiles,  [sample,
                              nosample], wltp_inp,     [_is_inp_files, _is_recertify]]
            - [do_addfiles,  [sample,
                              nosample], wltp_out,     [_is_out_files, _is_recertify]]

            - [do_report,  wltp_iof,   tagged]
            - [do_report,  tagged,     tagged]
            - [do_report,  mailed,     tagged,       _is_force        ]

            - [do_sendmail,  tagged,     mailed                         ]
            - [do_sendmail,  mailed,     mailed,     _is_dry_run        ]

            - trigger:    do_storedice
              source:     [tagged, mailed]
              dest:       nosample
              prepare:    _parse_response
              conditions:     [_is_not_decision_sample, _is_not_dry_run_dicing]

            - trigger:    do_storedice
              source:     [tagged, mailed]
              dest:       sample
              prepare:    _parse_response
              conditions: [_is_decision_sample, _is_not_dry_run_dicing]

            - [do_addfiles,  [diced,
                              nosample,
                              sample,
                              nedc],   nedc,         _is_other_files  ]
            """)

        super().__init__(states=states,
                         initial=states[0],
                         transitions=trans,
                         send_event=True,
                         prepare_event=['_cb_clear_result'],
                         before_state_change=['_cb_check_my_index'],
                         after_state_change='_cb_commit_or_tag',
                         auto_transitions=False,
                         name=pname,
                         **kwds
                         )
        self.on_enter_empty('_cb_stage_new_project_content')
        self.on_enter_tagged('_cb_pepare_email')
        self.on_enter_wltp_inp('_cb_stage_pfiles')
        self.on_enter_wltp_out('_cb_stage_pfiles')
        self.on_enter_wltp_iof('_cb_stage_pfiles')
        self.on_enter_nedc('_cb_stage_pfiles')
        self.on_enter_mailed('_cb_send_email')

    def attempt_repair(self, force=None):
        if force is None:
            force = self.force
        ## TODO: IMPL REPAIR CUR PROJECT
        self.log.warning('TODO: IMPL REPAIR CUR PROJECT')

    def _cb_invalidated(self, event):
        """
        Triggered by `do_invalidate(error=<ex>)` on BEFORE transition, and raises the `error`.

        :param Exception error:
                The invalidation exception to be stored on :attr:`Project.error`
                as ``(<prev-state>, error)`` for future reference.
        """
        self.log.error('Invalidating current %s with event: %s',
                       self, event.kwargs)
        ex = _evarg(event, 'error')
        self.error = (self.state, ex)
        raise ex

    def _make_commitMsg(self, action, data=None) -> Text:
        assert data is None or isinstance(data, (list, dict)), (
            "Data not a (list|dict): %s" % data)
        cmsg = _CommitMsg(__dice_report_version__, action, self.pname, self.state, data)

        return cmsg

    def _cb_clear_result(self, event):
        """
        Executed on GLOBAL PREPARE, and clears any results from previous transitions.

        TODO: REQUIRES ankostis transitions!!
        """
        self.result = None

    def _cb_check_my_index(self, event):
        """ Executed BEFORE exiting any state, to compare my `pname` with checked-out ref. """
        active_branch = self.repo.active_branch
        if self.pname != _ref2pname(active_branch):
            ex = MachineError("Expected current project to be %r, but was %r!"
                              % (self.pname, active_branch))
            self.do_invalidate(error=ex)

    def _prep_env_vars(self, gnupghome):
        env = {'GNUPGHOME': gnupghome}
        if self.verbose:
            env['GIT_TRACE'] = '2'

        return env

    def _commit(self, repo, report_txt: str, project: str):
        index = repo.index
        cmsg_txt = '\n'.join(report_txt.split('\n')[:self.commit_msg_lines_limit])
        try:
            index.commit(cmsg_txt)
        except ModuleNotFoundError as ex:
            if "No module named 'pwd'" in str(ex):
                raise CmdException(
                    "Cannot derive user while committing dice-project: %s"
                    "\n  Please set USERNAME env-var." %
                    project)

    @contextlib.contextmanager
    def _make_msg_tmpfile(self, bcontents: bytes,
                          *tempfile_args, prefix='co2tag-', **tempfile_kwds):
        import tempfile

        msg_fd, msg_fpath = tempfile.mkstemp(prefix=prefix,
                                             *tempfile_args, **tempfile_kwds)
        try:
            os.write(msg_fd, bcontents)
            os.close(msg_fd)
            yield msg_fpath
        finally:
            os.unlink(msg_fpath)

    def _cb_commit_or_tag(self, event):
        """Executed AFTER all state changes, and commits/tags into repo. """
        from . import crypto

        state = self.state
        ## No action wne, reporting on already tagged project.
        action = _evarg(event, 'action', (str, dict), missing_ok=True)
        ## Exclude transient/special cases (BIRTH/DEATH).
        if state.isupper() or not action:
            return

        self.log.debug('Committing: %s', event.kwargs)

        git_auth = crypto.get_git_auth(self.config)
        ## WARN: Without next cmd, GpgSpec `git_auth` only lazily creates GPG
        #  which imports keys/trust.
        git_auth.GPG

        repo = self.repo
        report = _evarg(event, 'report', (list, dict), missing_ok=True)
        is_tagging = state in 'tagged sample nosample'.split() and report
        cmsg = self._make_commitMsg(action, report)
        report_txt = cmsg.dump_commit_msg(width=self.git_desc_width)

        self.log.info('Committing %s: %s', self, action)
        self._commit(repo, report_txt, cmsg.p)

        ## Update result if not any cb previous has done it first
        #
        if not self.result:
            self.result = cmsg._asdict()

        if is_tagging:
            ## Note: No meaning to enable env-vars earlier,
            #  *GitPython* commis without invoking `git` cmd.
            #
            env_vars = self._prep_env_vars(git_auth.gnupghome_resolved)
            with repo.git.custom_environment(**env_vars):

                ok = False
                try:
                    tagname = _find_dice_tag(repo, self.pname,
                                             self.max_dices_per_project,
                                             fetch_next=True)
                    self.log.info('Tagging %s: %s', self, tagname)
                    assert isinstance(tagname, str), tagname

                    with self._make_msg_tmpfile(report_txt.encode()) as msg_fpath:
                        tagref = repo.create_tag(
                            tagname,
                            file=msg_fpath,
                            sign=True,
                            local_user=git_auth.master_key_resolved)

                    self.result = _read_dice_tag(repo, tagref)

                    ok = True
                finally:
                    if not ok:
                        self.log.warning(
                            "New status('%s') failed, REVERTING to prev-status('%s').",
                            state, event.transition.source)
                        repo.active_branch.commit = 'HEAD~'

    def _make_readme(self):
        from datetime import datetime

        return tw.dedent("""
        This is the CO2MPAS-project %r (see https://co2mpas.io/ for more).

        - created: %s
        """ % (self.pname, datetime.now()))

    def _cb_stage_new_project_content(self, event):
        """Triggered by `do_createme()` on ENTER 'empty' state."""
        repo = self.repo
        index = repo.index

        ## Cleanup any files from old project.
        #
        old_fpaths = [e[0] for e in index.entries]
        if old_fpaths:
            index.remove(old_fpaths, working_tree=True, r=True, force=True)

        state_fpath = osp.join(repo.working_tree_dir, 'CO2MPAS')
        with io.open(state_fpath, 'wt') as fp:
            fp.write(self._make_readme())
        index.add([state_fpath])

        ## Commit/tag callback expects `action` on event.
        event.kwargs['action'] = 'init'

    def _cb_stage_pfiles(self, event):
        """
        Triggered by `do_addfiles(pfiles=<PFiles>)` on ENTER for all `wltp_XX` & 'nedc' states.

        :param PFiles pfiles:
            what to import
        """
        import shutil

        pfiles = _evarg(event, 'pfiles', PFiles)
        self.log.info('Importing files: %s...', pfiles)

        ## Check extraction of report works ok,
        #  and that VFids match.
        #
        try:
            rep = self._report_spec()
            rep.extract_dice_report(pfiles, expected_vfid=self.pname)
            ## TODO: reuse these findos later, if --report given.
        except CmdException as ex:
            msg = "Failed extracting report from %s, due to: %s"
            if self.force:
                msg += "  BUT FORCED to import them!"
                self.log.warning(msg, pfiles, ex, exc_info=1)
            else:
                raise CmdException(msg % (pfiles, ex)) from ex

        if self.dry_run:
            self.log.warning('DRY-RUN: Not actually committed %d files.',
                             pfiles.nfiles())
            return

        repo = self.repo
        index = repo.index
        for io_kind, fpaths in pfiles._asdict().items():
            for ext_fpath in fpaths:
                self.log.debug('Importing %s-file: %s', io_kind, ext_fpath)
                assert ext_fpath, "Import none as %s file!" % io_kind

                ext_fname = osp.basename(ext_fpath)
                index_fpath = osp.join(repo.working_tree_dir, io_kind, ext_fname)
                pndlu.ensure_dir_exists(osp.dirname(index_fpath))
                shutil.copy(ext_fpath, index_fpath)
                index.add([index_fpath])

        ## Commit/tag callback expects `action` on event.
        event.kwargs['action'] = 'add'

    def list_pfiles(self, *io_kinds, _as_index_paths=False) -> PFiles or None:
        """
        List project's imported files.

        :param io_kinds:
            What files to fetch; by default if none specified,
            fetches all: inp,  out, other
            Use this to fetch some::

                self.list_io_files('inp', 'out')

        :param _as_index_paths:
            When true, filepaths are prefixed with repo's working-dir
            like ``~/.co2dice/repo/inp/inp1.xlsx``.

        :return:
            A class:`PFiles` containing list of working-dir paths
            for any WLTP files, or none if none exists.
        """
        io_kinds = PFiles.io_kinds_list(*io_kinds)
        repo = self.repo

        def collect_kind_files(io_kind):
            wd_fpath = osp.join(repo.working_tree_dir, io_kind)
            io_pathlist = os.listdir(wd_fpath) if osp.isdir(wd_fpath) else []
            if _as_index_paths:
                io_pathlist = [osp.join(wd_fpath, f) for f in io_pathlist]
            return io_pathlist

        iofpaths = {io_kind: collect_kind_files(io_kind) for io_kind in io_kinds}
        if any(iofpaths.values()):
            return PFiles(**iofpaths)

    def _cb_pepare_email(self, event):
        """
        Triggered by `do_report()` on ENTER of `tagged` state.

        If already on `tagged`, just sets the :data:`result` and exits,
        unless --force, in which case it generates another tag.

        Uses the :class:`ReporterSpec` to build the tag-msg.

        :return:
            Setting :attr:`result` to string.
        """
        repo = self.repo
        tagref = _find_dice_tag(repo, self.pname,
                                self.max_dices_per_project)
        # FIXME: Use state-condition to capture `wltp_iof`
        gen_report = event.transition.source == 'wltp_iof' or not tagref or self.force
        if gen_report:
            self.log.info('Preparing %s report: %s...',
                          'ANEW' if self.force else '', event.kwargs)
            repspec = self._report_spec()
            pfiles = self.list_pfiles(*PFiles._fields, _as_index_paths=True)  # @UndefinedVariable
            report = list(repspec.extract_dice_report(pfiles).values())

            if self.dry_run:
                self.log.warning("DRY-RUN: Not actually committed the report, "
                                 "and it is not yet signed!")
                self.result = _mydump(report, width=self.git_desc_width)

                return

            ## Commit/tag callback expects `report` on event.
            event.kwargs['action'] = 'drep'
            event.kwargs['report'] = report
        else:
            assert tagref
            self.log.debug("Report already generated as '%s'.", tagref.path)
            self.result = _read_dice_tag(repo, tagref)

    def _cb_send_email(self, event):
        """
        Triggered by `do_sendmail()` on ENTER of `sendmail` state.

        Parses last tag and uses class:`SMTP` to send its message as email.

        :return:
            Setting :attr:`result` to string.
        """
        repo = self.repo
        dry_run = self.dry_run
        self.log.info('%s email for tstamping...', 'Printing' if dry_run else 'Sending')
        tstamp_sender = self._tstamp_sender_spec()

        tagref = _find_dice_tag(repo, self.pname,
                                self.max_dices_per_project)
        assert tagref, (tw.dedent("""\
            Project corrupted! state is `%s` but cannot find any dice-report tag!
              Try the following:
              - re-run `project report` command;
              - run `export --erase-afterwards` and restart project;
              - run `project backup --erase-afterwards` to delete you projects-DB
                and restart project.
        """ % self.state))
        signed_dice_report = _read_dice_tag(repo, tagref)
        assert signed_dice_report

        dice_mail_mime = tstamp_sender.send_timestamped_email(
            signed_dice_report, tagref.name, dry_run=dry_run)

        if dry_run:
            HEADER = '\n'.join('%7s: %s' % (k, dice_mail_mime.get(k))
                               for k in ('Subject To Cc').split())
            self.log.info("""
DRY-RUN: Now you must send the email your self!
================================================================================
- Start a new email and set EMAIL-DELIVERY option  as 'plain-text'  before(!)
  pasting text (visit ://goo.gl/jwR5Hz for examples on popular email-clients)
- set EMAIL-FIELDS as shown below (you may also set 'Bcc').
%s
- copy all text from the 1st line below starting with 'X-Stamper-To:', and
  paste it in the email body;
- send the email, and wait for "Proof of Posting" reply-back to `tparse` it.
================================================================================
""", tw.indent(HEADER, '  - '))
            self.result = dice_mail_mime.get_payload()

        if event.transition.source != 'mailed':
            ## Don't repeat your self...
            event.kwargs['action'] = '%s stamp' % ('FAKED' if dry_run else 'sent')

    def _validate_stamp_verdict(self, verdict):
        err_msgs = []

        stamp_pname = verdict.get('report', {}).get('project')
        if stamp_pname != self.pname:
            err_msgs.append("Stamp's project('%s') is different from current one('%s')!"
                            % (stamp_pname, self.pname))

        stamp_tag = verdict.get('dice', {}).get('tag')
        tag = _find_dice_tag(self.repo, self.pname, self.max_dices_per_project)
        last_tag = tag and '%s: %s' % (tag.name, tag.commit.hexsha)
        if stamp_tag != last_tag:
            err_msgs.append("Stamp's tag('%s') is different "
                            "from last tag on current project('%s')!"
                            % (stamp_tag, last_tag))

        return '\n'.join(err_msgs)

    def _parse_response(self, event) -> bool:
        """
        Triggered by `do_storedice(verdict=<dict> | tstamp_txt=<str>)` in PREPARE `sample/nosample`.

        :param verdict:
            The result of verifying timestamped-response.
        :return:
            Setting :attr:`result` to ODict.

        .. Note:
            It needs an already verified tstamp-response because to select which project
            it belongs to, it needs to parse the dice-report contained within the response.
        """
        ## FIXME: executed twice for SAMPLE/NOSAMPLE!!
        from toolz import dicttoolz as dtz
        from . import tstamp

        tstamp_txt = _evarg(event, 'tstamp_txt', str)
        verdict = _evarg(event, 'verdict', dict, missing_ok=True)

        if verdict is None:
            recv = tstamp.TstampReceiver(config=self.config)
            verdict = recv.parse_tstamp_response(tstamp_txt)

        err_msg = self._validate_stamp_verdict(verdict)
        if err_msg:
            if self.force:
                self.log.error("%s  \n  But forced to accept it.", err_msg)
            else:
                raise CmdException(err_msg)

        ## Store DICE-decision under email-response!
        #
        short_verdict = dtz.keyfilter(lambda k: k == 'dice', verdict)
        tstamp_txt += '\n' + _mydump(short_verdict, default_flow_style=False)

        event.kwargs['verdict'] = self.result = verdict

        ## TODO: **On commit, set arbitrary files to store (where? name?)**.
        repo = self.repo
        index = repo.index
        new_files = [('tstamp.txt', tstamp_txt),
                     ('verdict.txt', _mydump(verdict))]
        for new_path, text in new_files:
            new_fpath = osp.join(repo.working_tree_dir, new_path)
            with io.open(new_fpath, 'wt') as fp:
                fp.write(text)
            index.add([new_fpath])

        ## Store in new TAG the decision only;
        #  the rest are stored in `verdict.txt` file.
        report = dtz.keyfilter(lambda k: k not in ('tstamp', ), verdict)

        ## Notify to create new tag!
        event.kwargs['report'] = report

    def _is_decision_sample(self, event) -> bool:
        verdict = _evarg(event, 'verdict', dict)

        decision = verdict.get('dice', {}).get('decision', 'SAMPLE')
        event.kwargs['action'] = "diced %s" % decision

        return decision == 'SAMPLE'

    def _is_not_decision_sample(self, event) -> bool:
        return not self._is_decision_sample(event)

    def _is_not_dry_run_dicing(self, event):
        if self.dry_run:
            self.log.warning('DRY-RUN: Not actually registering decision.')
        return not self.dry_run


class ProjectsDB(trtc.SingletonConfigurable, ProjectSpec):
    r"""A git-based repository storing the TA projects (containing signed-files and sampling-responses).

    It handles checkouts but delegates index modifications to `Project` spec.

    ### Git Command Debugging and Customization:

    - :envvar:`GIT_PYTHON_TRACE`: If set to non-0,
      all executed git commands will be shown as they happen
      If set to full, the executed git command _and_ its entire output on stdout and stderr
      will be shown as they happen

      NOTE: All logging is done through a Python logger, so make sure your program is configured
      to show INFO-level messages.

    - :envvar:`GIT_PYTHON_GIT_EXECUTABLE`: If set, it should contain the full path
      to the git executable, e.g. ``c:\Program Files (x86)\Git\bin\git.exe on windows``
      or ``/usr/bin/git`` on linux.
    """

    repo_path = trt.Unicode(
        osp.join(baseapp.default_config_dir(), 'repo'),
        help="""
        The path to the Git repository to store TA files (signed and exchanged).
        If relative, it joined against default config-dir: '{confdir}'
        """.format(confdir=baseapp.default_config_dir())
    ).tag(config=True, envvar='CO2DICE_REPO_PATH')

    preserved_git_settings = trt.List(
        trt.CRegExp(),
        help="""
        Overwritte all git-repo's settings on start-up except those mathcing this list of regexes.

        Git settings include user-name and email address, so this option might be usefull
        when the regular owner running the app has changed but old-user should be preserved.
        """).tag(config=True)

    ## Useless, see https://github.com/ipython/traitlets/issues/287
    # @trt.validate('repo_path')
    # def _normalize_path(self, proposal):
    #     repo_path = proposal.value
    #     if not osp.isabs(repo_path):
    #         repo_path = osp.join(default_config_dir(), repo_path)
    #     repo_path = pndlu.convpath(repo_path)
    # return repo_path

    allow_foreign_dice = trt.Bool(
        help="Storing \"foreign\ dices\" is an advanced operation, "
        "corrupting git.")

    __repo = None

    def __del__(self):
        ## TODO: use weakref for proj/gitpython (recent)
        if self.__repo:
            self.__repo.git.clear_cache()

    @property
    def repopath_resolved(self):
        """Used internally AND for printing configurations."""
        repo_path = self.repo_path
        if not osp.isabs(repo_path):
            repo_path = osp.join(baseapp.default_config_dir(), repo_path)
        repo_path = pndlu.convpath(repo_path)

        return repo_path

    def _setup_repo(self):
        import git  # From: pip install gitpython

        repo_path = self.repopath_resolved
        pndlu.ensure_dir_exists(repo_path)
        try:
            self.log.debug('Opening repo %r...', repo_path)
            self.__repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError as ex:
            self.log.info("...failed opening repo '%s',\n  initializing a new repo %r instead...",
                          ex, repo_path)
            self.__repo = git.Repo.init(repo_path)

        self._write_repo_configs(preserved_git_settings=self.preserved_git_settings)

    @trt.observe('repo_path')
    def _cleanup_old_repo(self, change):
        self.log.debug('CHANGE repo %r-->%r...', change['old'], change['new'])
        repo_path = change['new']
        if self.__repo:
            if self.__repo.working_dir == repo_path:
                self.log.debug('Reusing repo %r...', repo_path)
                return
            else:
                ## Clean up old repo,
                #  or else... https://github.com/gitpython-developers/GitPython/issues/508
                self.__repo.close()
                ## Xmm, nai...
                self._current_project = None
            self.__repo = None

    @property
    def repo(self):
        if not self.__repo:
            self._setup_repo()
        return self.__repo

    def _write_repo_configs(self, preserved_git_settings=()):
        r"""
        :type preserved_git_settings:
            list-of-regex expressions fully matching git-configs,
            e.g ``user\..+`` matches both ``user.name`` and ``user.email``.
        """
        from . import crypto

        log = self.log
        repo = self.repo
        git_auth = crypto.get_git_auth(self.config)
        gnupgexe = git_auth.gnupgexe_resolved
        if repo.git.is_cygwin:
            from git.util import cygpath
            gnupgexe = cygpath(gnupgexe)

        gconfigs = [
            ('core.filemode', False),
            ('core.ignorecase', False),
            ('user.email', self.user_email),
            ('user.name', self.user_name),
            ('gc.auto', 0),                 # To salvage user-mistakes.
            ('alias.lg',                    # Famous alias for inspecting history.
                r"log --graph --abbrev-commit --decorate --date=relative "
                r"--format=format:'%C(bold blue)%h%C(reset) "
                r"- %C(bold green)(%ar)%C(reset) %C(white)%s%C(reset) %C(dim white)- "
                r"%an%C(reset)%C(bold yellow)%d%C(reset)' --all"),
            ('gpg.program', gnupgexe),
            ('user.signingkey', git_auth.master_key_resolved),
        ]

        preserved_git_settings = [re.compile(r) for r in preserved_git_settings]
        unexpected_kvalues = OrderedDict()
        overwritten_count = 0
        with repo.config_writer('repository') as cw:
            for key, val in gconfigs:
                sec, prop = key.split('.')

                ## Check if differrent.
                #
                try:
                    old_val = cw.get_value(sec, prop)

                    ## Git strips them, so comparison would fail
                    #  if it had trailing spaces (hard to catch)
                    #
                    if isinstance(val, str):
                        val = val.strip()

                    if old_val == val:
                        continue

                    diff_value = [old_val, val]
                except Exception:
                    diff_value = ['<missing>', val]

                is_preserved = any(r.match(key) for r in preserved_git_settings)

                ## Update record of preserve/overwrite actions.
                #
                diff_value.append('<preserved>' if is_preserved else '<overwritten>')
                unexpected_kvalues[key] = diff_value

                if is_preserved:
                    continue

                ## Write setting.
                #
                overwritten_count += 1
                ok = False
                try:
                    ## gitpython-developers/GitPython#578
                    if isinstance(val, bool):
                        val = str(val).lower()

                    cw.set_value(sec, prop, val)
                    ok = True
                finally:
                    if not ok:
                        log.error("Failed to write git-setting '%s': %s!",
                                  key, val)

            if unexpected_kvalues:
                log.warning("Overwritten %d out of %d missmatched value in git-settings('%s'):"
                            "\n  %s",
                            overwritten_count, len(unexpected_kvalues),
                            osp.join(repo.git_dir, 'config'),
                            tw.indent(_mydump(unexpected_kvalues), '    '))

    def read_git_settings(self, prefix: Text=None, config_level: Text=None):  # -> List(Text):
        """
        :param prefix:
            prefix of all settings.key (without a dot).
        :param config_level:
            One of: ( system | global | repository )
            If None, all applicable levels will be merged.
            See :meth:`git.Repo.config_reader`.
        :return: a list with ``section.setting = value`` str lines
        """
        settings = defaultdict()
        settings.default_factory = defaultdict
        sec = '<not-started>'
        cname = '<not-started>'
        try:
            with self.repo.config_reader(config_level) as conf_reader:
                for sec in conf_reader.sections():
                    for cname, citem in conf_reader.items(sec):
                        s = settings
                        if prefix:
                            s = s[prefix]
                        s[sec][cname] = citem
        except Exception as ex:
            self.log.error('Failed reading git-settings on %s.%s due to: %r',
                           sec, cname, ex, exc_info=self.verbose)
            raise
        return settings

    def repo_backup(self, folder: Text='.', repo_name: Text='co2mpas_repo',
                    erase_afterwards=False, force: bool=None) -> Text:
        """
        :param folder:
            The path to the folder to store the repo-archive in.
        :return:
            the path of the repo-archive
        """
        import tarfile
        from datetime import datetime

        if force is None:
            force = self.force

        now = datetime.now().strftime('%Y%m%d-%H%M%S%Z')
        repo_name = '%s-%s' % (now, repo_name)
        repo_name = pndlu.ensure_file_ext(repo_name, '.txz')
        repo_name_no_ext = osp.splitext(repo_name)[0]
        archive_fpath = pndlu.convpath(osp.join(folder, repo_name))
        basepath, _ = osp.split(archive_fpath)
        if not osp.isdir(basepath) and not force:
            raise FileNotFoundError(basepath)
        pndlu.ensure_dir_exists(basepath)

        self.log.debug('Archiving repo into %r...', archive_fpath)
        with tarfile.open(archive_fpath, "w:xz") as tarfile:
            tarfile.add(self.repo.working_dir, repo_name_no_ext)

        if erase_afterwards:
            from git.util import rmtree

            self.log.info("Erasing Repo '%s'..."
                          "\n  Tip: if it fails, restart and retry :-)",
                          self.repo_path)
            try:
                rmtree(self.repo_path)
            except Exception as ex:
                self.log.error("Failed erasing Repo '%s'due to: %r",
                               self.repo_path, ex, exc_info=self.verbose)

        return archive_fpath

    @fnt.lru_cache()  # x6(!) faster!
    def _infos_dsp(self, fallback_value='<invalid>'):
        from schedula import Dispatcher
        from schedula.utils.dsp import DFun

        ## see _info_fields()
        P = 'project'

        dfuns = [
            DFun('_repo', lambda _rinfos: self.repo),
            DFun('git_cmds', lambda _rinfos: pndlu.where('git')),
            DFun('exec_path', lambda _repo: getattr(_repo.git, '--exec-path')()),
            DFun('is_dirty', lambda _repo: _repo.is_dirty()),
            DFun('is_bare', lambda _repo: _repo.bare),
            #DFun('is_empty', lambda _repo: _repo.is_empty), pygit2!
            DFun('untracked', lambda _repo: _repo.untracked_files),
            DFun('wd_files', lambda _repo: os.listdir(_repo.working_dir)),
            DFun('_heads', lambda _repo: _repo.heads),
            DFun('heads', lambda _heads: [r.name for r in _heads]),
            DFun('heads_count', lambda _heads: len(_heads)),
            DFun('_projects', lambda _repo: list(_yield_project_refs(_repo))),
            DFun('projects', lambda _projects: [p.name for p in _projects]),
            DFun('projects_count', lambda projects: len(projects)),
            DFun('_all_dices', lambda _repo: list(_yield_dices_tags(_repo))),
            DFun('all_dices', lambda _all_dices: [t.name for t in _all_dices]),
            DFun('all_dices_count', lambda all_dices: len(all_dices)),
            DFun('git.settings', lambda _repo: self.read_git_settings()),

            DFun('git.version', lambda _repo: '.'.join(str(v) for v in _repo.git.version_info)),

            DFun('_head', lambda _repo: _repo.head),
            #DFun('head_unborn', lambda _repo: _repo.head_is_unborn()), pygit2
            DFun('head', lambda _head: _head.path),
            DFun('head_valid', lambda _head: _head.is_valid()),
            DFun('head_detached', lambda _head: _head.is_detached),
            DFun('_head_ref', lambda _head: _head.ref),
            DFun('head_ref', lambda _head_ref: _head_ref.path),

            DFun('_index_entries', lambda _repo: list(_repo.index.entries)),
            DFun('index_count', lambda _index_entries: len(_index_entries)),
            DFun('index_entries', lambda _index_entries: [e[0] for e in _index_entries]),
            DFun('_index', lambda _repo: _repo.index),

            ## Project-infos
            #
            DFun('_projref', lambda _repo, _pname:
                 _get_ref(_repo.heads, _pname2ref_name(_pname)), inf=P),
            DFun('_cmt', lambda _projref: _projref.commit, inf=P),
            DFun('_tree', lambda _cmt: _cmt.tree, inf=P),
            DFun('is_current', lambda _repo, _projref: _projref == _repo.active_branch, inf=P),
            DFun('author', lambda _cmt: '%s <%s>' % (_cmt.author.name, _cmt.author.email), inf=P),
            DFun('last_cdate', lambda _cmt: str(_cmt.authored_datetime), inf=P),
            DFun('_last_dice', lambda _repo, _pname: _find_dice_tag(
                _repo, _pname, self.max_dices_per_project), inf=P),
            DFun('last_dice', lambda _last_dice: _last_dice and '%s: %s' % (
                _last_dice.name, _last_dice.commit.hexsha), inf=P),
            DFun('last_dice_msg', lambda _last_dice: _last_dice and _last_dice.tag.message, inf=P),
            DFun('last_commit', lambda _cmt: _cmt.hexsha, inf=P),
            DFun('last_tree', lambda _tree: _tree.hexsha, inf=P),
            DFun('_dices', lambda _repo, _pname: list(_yield_dices_tags(_repo, _pname)), inf=P),
            DFun('dices', lambda _dices: ['%s: %s' % (t.name, t.commit.hexsha)
                                          for t in _dices], inf=P),
            DFun('dices_count', lambda _dices: len(_dices), inf=P),
            DFun('_revs', lambda _cmt: list(_cmt.iter_parents()), inf=P),
            DFun('revs', lambda _revs: [c.hexsha for c in _revs], inf=P),
            DFun('revs_count', lambda _revs: len(_revs), inf=P),
            DFun('cmsg', lambda _cmt: _cmt.message, inf=P),
            DFun('cmsg', lambda _cmt: '<invalid: %s>' % _cmt.message, weight=10, inf=P),

            DFun(['msg.%s' % f for f in _CommitMsg._fields],
                 lambda cmsg: _CommitMsg.parse_commit_msg(cmsg), inf=P),

            DFun('_objects', lambda _tree: list(_tree.list_traverse()), inf=P),
            DFun('objects_count', lambda _objects: len(_objects), inf=P),
            DFun('objects', lambda _objects: ['%s: %s' % (b.type, b.path)
                                              for b in _objects], inf=P),
            DFun('files', lambda _objects: [b.path for b in _objects if b.type == 'blob'], inf=P),
            DFun('files_count', lambda files: len(files), inf=P),
        ]
        dsp = Dispatcher()
        DFun.add_dfuns(dfuns, dsp)
        return dsp

    @fnt.lru_cache()
    def _info_fields(self, level, want_project=None, want_repo=False):
        """
        :param level:
            If ''> max-level'' then max-level assumed, negatives fetch no fields.
        """
        dsp = self._infos_dsp()

        ## see _infos_dsp() :-)
        P = 'project'
        R = 'repo'

        ## TODO: Make vlevels them configurable.
        verbose_levels = [
            [
                'head',
                'head_valid',
                'head_ref',
                'heads_count',
                'projects_count',
                'projects',
                'all_dices_count',
                'all_dices',
                #'is_empty',
                'wd_files',
                'untracked',
                'index_count',
                'index_entries',

                'msg.s',
                'msg.a',
                'last_dice',
                'last_commit',
                'last_tree',
                'files',
                'dices',
                'dices_count',
                'revs_count',
                'files_count',
                'last_cdate',
                'author',
                'is_current',
            ],
            [
                'git_cmds',
                'git.version',
                'exec_path',
                'is_dirty',
                'is_bare',
                'head_detached',
                #'head_unborn',
                'heads',

                'objects_count',
                'revs',
                'last_dice_msg',
                'cmsg',
                #('msg.data',
            ],
            [f for f in dsp.data_nodes
             if not f.startswith('_')]
        ]

        ## Extract `inf` attributes from functions
        #  and pair them with info-fields, above
        #
        ftype_map = {}  # outfield -> ( P | R )
        for f in dsp.function_nodes.values():
            outputs = f['outputs']
            inf = f.get('inf', R)  # "project" funcs are unmarked
            for outfield in outputs:
                assert outfield not in ftype_map or ftype_map[outfield] == inf, (
                    outfield, inf, ftype_map)
                ftype_map[outfield] = inf

        if level >= len(verbose_levels):
            return None  # meaning all

        wanted_ftypes = set()
        if want_project:
            wanted_ftypes.add(P)
        if want_repo:
            wanted_ftypes.add(R)

        ## Fetch all kinds if unspecified
        #
        if not wanted_ftypes:
            wanted_ftypes = set([P, R])

        sel_fields = itz.concat(verbose_levels[:level + 1])
        return list(field for field
                    in sel_fields
                    if ftype_map[field] in wanted_ftypes)

    def _scan_infos(self, *, pname: Text=None,
                    fields: Sequence[Text]=None,
                    inv_value=None) -> List[Tuple[Text, Any]]:
        """Runs repo examination code returning all requested fields (even failed ones)."""
        dsp = self._infos_dsp()
        inputs = {'_rinfos': 'boo    '}
        if pname:
            inputs['_pname'] = pname

        infos = dsp.dispatch(inputs=inputs,
                             outputs=fields)
        fallbacks = {d: inv_value for d in dsp.data_nodes.keys()}
        fallbacks.update(infos)
        infos = fallbacks

        ## FIXME: stack-nested-jeys did what??
        #from schedula import utils
        #infos = dict(utils.stack_nested_keys(infos))
        #infos = dtz.keymap(lambda k: '.'.join(k), infos)

        if fields:
            infos = [(f, infos.get(f, inv_value))
                     for f in fields]
        else:
            infos = sorted((f, infos.get(f, inv_value))
                           for f in dsp.data_nodes.keys())

        return infos

    def repo_status(self, verbose=None, as_text=False):
        """
        Examine infos about the projects-db.

        :retun: text message with infos.
        """

        if verbose is None:
            verbose = self.verbose
        verbose_level = int(verbose)

        fields = self._info_fields(verbose_level, want_repo=True)
        infos = self._scan_infos(fields=fields)

        if as_text:
            infos = _mydump(OrderedDict(infos), default_flow_style=False)

        return infos

    def _conceive_new_project(self, pname):  # -> Project:
        """Returns a "BORN" :class:`Project`; its state must be triggered immediately."""
        return Project.new_instance(pname, self.repo, self.config)

    _current_project = None

    def current_project(self) -> Project:
        """
        Returns the current :class:`Project`, or raises a help-msg if none exists yet.

        - Trait exceptions pass through (e.g. test-key).
        - The project returned is appropriately configured according to its
          recorded state.
        - The git-repo is not touched.
        """
        if not self._current_project:
            try:
                headref = self.repo.active_branch
                if _is_project_ref(headref):
                    pname = _ref2pname(headref)
                    p = self._conceive_new_project(pname)
                    cmsg = _CommitMsg.parse_commit_msg(headref.commit.message)
                    p.set_state(cmsg.s)

                    self._current_project = p
            except TraitError:
                raise
            except Exception as ex:
                self.log.warning("Failure while getting current-project: %r",
                                 ex, exc_info=self.verbose)

        if not self._current_project:
                raise CmdException(tw.dedent("""
                        No current-project exists yet!
                          Try opening one with: co2mpas project open  <project-name>
                        """))

        return self._current_project

    def validate_project_name(self, pname: Text) -> Project:
        from ..io import schema

        if not pname or not git_project_regex.match(pname):
            raise CmdException(schema.invalid_vehicle_family_id_msg % pname)
        if not self.force:
            schema.vehicle_family_id().validate(pname)

    def proj_add(self, pname: Text) -> Project:
        """
        Creates a new project and sets it as the current one.

        :param pname:
            the project name (without prefix)
        :return:
            the current :class:`Project` or fail
        """
        self.log.info('Creating project %r...', pname)
        self.validate_project_name(pname)

        prefname = _pname2ref_name(pname)
        if prefname in self.repo.heads:
            raise CmdException('Project %r already exists!' % pname)

        p = self._conceive_new_project(pname)
        self.repo.git.checkout(prefname, orphan=True, force=self.force)
        self._current_project = p
        try:
            ## Trigger ProjectFSM methods that will modify Git-index & commit.
            ok = p.do_createme()
            assert ok, "Refused adding new project %r!" % pname

            return p
        except Exception as ex:
            p.do_invalidate(error=ex)

    def proj_open(self, pname: Text=None) -> Project:
        """
        :param pname:
            the project name (without prefix); auto-deduced if missing and
            a single project exists.
        :return:
            the current :class:`Project`
        """
        if not pname:
            plist = list(self.proj_list())
            if len(plist) != 1:
                raise CmdException(
                    'Cannot deduce which project to open from: %s', plist)
            pname = plist[0]

        repo = self.repo
        if pname == '.':
            ## FIXME: fails if no project open; `repo.head.ref` might work.
            pname = _ref2pname(repo.head.name)

        prefname = _pname2ref_name(pname)
        if prefname not in repo.heads:
            raise CmdException('Project %r not found!' % pname)
        repo.heads[_pname2ref_name(pname)].checkout(force=self.force)

        self._current_project = None
        return self.current_project()

    def append_foreign_dice(self, infos: dict):
        import gitdb
        from git.refs import tag

        tag_body = infos['report']['parts']['whole']
        assert isinstance(tag_body, bytes), infos
        ## Git stores tags without `\r`, and starting/ending with `\n`.
        tag_body = b'\n%s\n' % tag_body.strip().replace(b'\r', b'')
        tag_name = infos['dice']['tag']
        repo = self.repo

        ## Create tag onject.
        #
        ins = gitdb.IStream('tag', len(tag_body), io.BytesIO(tag_body))
        odb = repo.odb  # type: gitdb.ObjectDBW
        odb.store(ins)
        new_hexsha = ins.hexsha.decode()

        ## Create tag reference (if not already-there).
        #
        tag_path = re.match('^[^:]+', tag_name).group()
        tag_path = osp.join(repo.git_dir,
                            tag.TagReference._common_path_default,
                            tag_path)
        if osp.exists(tag_path):
            with io.open(tag_path, 'rt') as tag_fp:
                old_hexsha = tag_fp.read()
            if old_hexsha == new_hexsha:
                self.log.info("Dice '%s' --> '%s' already stored.",
                              tag_name, new_hexsha)
            else:
                ## TODO: print more Tag-collission infos.
                raise CmdException(
                    "Different dice '%s' --> '%s' already exists, "
                    "cannot overwrite with '%s'!" %
                    (tag_name, old_hexsha, new_hexsha)) from None
        else:
            os.makedirs(osp.dirname(tag_path), exist_ok=True)
            with io.open(tag_path, 'wt') as tag_fp:
                tag_fp.write(new_hexsha)

            self.log.info("Created foreign project tag: %s", tag_name)

    def proj_list(self, *pnames: Text, verbose=None,
                  as_text=False, fields=None):
        """
        :param pnames:
            some project name, or none for all
        :param verbose:
            return infos based on :meth:`_info_fields()`
        :param fields:
            If defined, takes precendance over `verbose`.
        :param as_text:
            If true, return YAML, otherwise, strings or dicts if verbose
        :retun:
            yield any matched projects, or all if `pnames` were empty.
        """
        repo = self.repo
        if verbose is None:
            verbose = self.verbose

        if fields:
            verbose = True  # Othrewise, hand-crafted infos ignore fields.
        else:
            verbose_level = int(verbose) - 1  # V0 print hand-crafted infos.
            if verbose:
                fields = self._info_fields(verbose_level, want_project=True)
            else:
                fields = ['is_current', 'msg.s']

        def cpname():
            try:
                return self.current_project().pname
            except CmdException as ex:
                self.log.warning('%s', ex)

        pnames = iset(cpname() if p == '.' else p
                      for p in iset(pnames))
        pnames = [p for p in pnames if p]
        for ref in _yield_project_refs(repo, *pnames):
            pname = _ref2pname(ref)
            if not as_text and not verbose:
                to_yield = pname
            else:
                infos = self._scan_infos(pname=pname, fields=fields,
                                         inv_value='<invalid>')
                if verbose:
                    infos = OrderedDict(infos)
                    to_yield = {pname: infos}
                    if as_text:
                        to_yield = _mydump(to_yield, default_flow_style=False)
                else:
                    i = dict(infos)
                    to_yield = '%s %s: %s' % (i['is_current'] and '*' or ' ',
                                              pname, i['msg.s'])

            yield to_yield


###################
##    Commands   ##
###################

class _SubCmd(baseapp.Cmd):
    @property
    def projects_db(self) -> ProjectsDB:
        p = ProjectsDB.instance(config=self.config)
        p.update_config(self.config)  # Above is not enough, if already inited.

        return p

    @property
    def current_project(self) -> Project:
        p = self.projects_db.current_project()
        p.update_config(self.config)  # TODO: drop when project de-zygotized.

        return p

    def _format_result(self, concise, long, *, is_verbose=None, **kwds):
        is_verbose = self.verbose if is_verbose is None else is_verbose
        result = long if is_verbose else concise

        return isinstance(result, str) and result or _mydump(result, **kwds)


class ProjectCmd(_SubCmd):
    """
    Commands to administer the storage repo of TA *projects*.

    A *project* stores all CO2MPAS files for a single vehicle,
    and tracks its sampling procedure.

    TIP:
      If you bump into blocking errors, please use the `co2dice project backup` command and
      send the generated archive-file back to "CO2MPAS-Team <JRC-CO2MPAS@ec.europa.eu>",
      for examination.
    """

    examples = trt.Unicode("""
        - To list all existing projects, try::
              %(cmd_chain)s  ls

        - To make a new project, type::
              %(cmd_chain)s  init RL-77-AAA-2016-0000

        - or to open an existing one (to become the *current* one)::
              %(cmd_chain)s  open IP-10-AAA-2017-1006

        - To see more infos about the current project, use::
              %(cmd_chain)s  ls . -v

        - A typical workflow is this::
              %(cmd_chain)s  init RL-12-BM3-2016-0000
              %(cmd_chain)s  append  --inp input.xlsx  --out output.xlsx   summary.xlsx  co2mpas.log
              %(cmd_chain)s  report
              %(cmd_chain)s  tstamp
              cat <tstamp-response-text> | %(cmd_chain)s  tparse
              %(cmd_chain)s  export

        - You may enquiry the status the projects database::
              %(cmd_chain)s  status --vlevel 2
    """)

    def __init__(self, **kwds):
        dkwds = {
            'conf_classes': [ProjectsDB, Project],
            'subcommands': baseapp.build_sub_cmds(*all_subcmds),
        }
        dkwds.update(kwds)
        super().__init__(**dkwds)

    def run(self, *args):
        """Just to ensure project-repo created."""
        self.projects_db.repo
        super().run(*args)


class StatusCmd(_SubCmd):
    """
    Print various information about the projects-repo.

    - Use `--verbose` or `--vlevel (2|3)` to view more infos.

    SYNTAX
        %(cmd_chain)s [OPTIONS]
    """
    def run(self, *args):
        if len(args) > 0:
            raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                               % (self.name, len(args), args))
        return self.projects_db.repo_status(as_text=True)


class LsCmd(_SubCmd):
    """
    List specified projects, or all, if none specified.

    - Use `--verbose` or `--vlevel (2|3|4)` to view more infos about the projects.
    - Use '.' to denote current project.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<project-1>] ...
    """
    def run(self, *args):
        self.log.info('Listing %s projects...', args or 'all')
        return self.projects_db.proj_list(*args, as_text=True)


class OpenCmd(_SubCmd):
    """
    Make an existing project as *current*.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<project>]

    - Auto-deduced, if no <project> is given, but there is only one
      in the database.
    """
    def run(self, *args):
        if len(args) > 1:
            raise CmdException(
                "Cmd %r takes one optional argument as project-name, received %r!"
                % (self.name, args))
        self.log.info('Opening project %r...', args)

        projDB = self.projects_db
        proj = projDB.proj_open(args and args[0] or None)

        return projDB.proj_list(proj.pname, as_text=True) if self.verbose else proj


class AppendCmd(_SubCmd):
    """
    Import the specified input/output co2mpas files into the *current project*.

    SYNTAX
        %(cmd_chain)s [OPTIONS] ( --inp <co2mpas-input> | --out <co2mpas-output> ) ...
                                [<any-other-file>] ...

    - To report and tstamp a project, one file (at least) from *inp* & *out* must be given.
    - If an input/output are already present in the current project, use --force.
    - Note that any file argument not given with `--inp`, `--out`, will end-up as "other".
    - If `--report` given, generates report if no file is missing.
    - Use `--recertify` to re-certify a vehicle-family
      (when a project has reached is `sample/nosample` state).
    """

    examples = trt.Unicode("""
        - To import an INPUT co2mpas file, try::
              %(cmd_chain)s --inp co2mpas_input.xlsx

        - To import both INPUT and OUTPUT files, and overwrite any already imported try::
              %(cmd_chain)s --force --inp co2mpas_input.xlsx --out co2mpas_results.xlsx
    """)

    inp = trt.List(
        trt.Unicode(),
        help="Specify co2mpas INPUT files; use this option one or more times."
    ).tag(config=True)
    out = trt.List(
        trt.Unicode(),
        help="Specify co2mpas OUTPUT files; use this option one or more times."
    ).tag(config=True)

    report = trt.Bool(
        help="When True, proceed to generate report; will fail if files missing"
    ).tag(config=True)

    def __init__(self, **kwds):
        from . import report

        kwds.setdefault('cmd_aliases', {
            ('i', 'inp'): ('AppendCmd.inp', AppendCmd.inp.help),
            ('o', 'out'): ('AppendCmd.out', AppendCmd.out.help),
        })
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {'Project': {'dry_run': True}},
                "Parse files but do not actually store them in the project."
            ),
            'report': (
                {AppendCmd.__name__: {'report': True}},
                AppendCmd.report.help
            ),
            'recertify': (
                {Project.__name__: {'recertify': True}},
                Project.recertify.help
            ),
            'with-inputs': (
                {
                    'ReporterSpec': {'include_input_in_dice': True},
                }, report.ReporterSpec.include_input_in_dice.help),
        })
        super().__init__(**kwds)

    def run(self, *args):
        from . import report

        ## TODO: Support heuristic inp/out classification
        pfiles = PFiles(inp=self.inp, out=self.out, other=args)
        if not pfiles.nfiles():
            raise CmdException(
                "Cmd %r must be given at least one file argument, received %d: %r!"
                % (self.name, pfiles.nfiles(), pfiles))

        if not self.report and \
                report.ReporterSpec(config=self.config).include_input_in_dice:
            raise CmdException(
                "Command %r received a --with-inputs flag but without --report!"
                % (self.name))

        pfiles.check_files_exist(self.name)
        self.log.info("Importing report files...\n  %s", pfiles)

        yield from self.append_and_report(pfiles)

    def append_and_report(self, pfiles):
        proj = self.current_project
        ok = proj.do_addfiles(pfiles=pfiles)

        if not self.report:
            yield self._format_result(ok, proj.result)
        else:
            ok = proj.do_report()

            assert isinstance(proj.result, str)
            yield ok and proj.result or ok

            key_uid = proj.extract_uid_from_report(proj.result)
            self.log.info("Report has been signed by '%s'.", key_uid)


class InitCmd(AppendCmd):
    """
    Create a new project, and optionally append files and generate report.

    SYNTAX
        %(cmd_chain)s [OPTIONS] <project>
        %(cmd_chain)s [OPTIONS] ( --inp <co2mpas-input> | --out <co2mpas-output> ) ...
                                [<any-other-file>] ...

    - The 1st form, the project-id is given explicetely.
    - The 2nd form, the project-id gets derrived from the files, and
      it must be identical.
    - If both files given, use `--report` to advance immediately to `tagged` state.
    """

    examples = trt.Unicode("""
        - In the simplest case, just create a new project like this::
              %(cmd_chain)s XX-12-YYY-2017-0000

        - To import both INPUT and OUTPUT files and generate report::
              %(cmd_chain)s --inp co2mpas_input.xlsx --out co2mpas_results.xlsx --report
    """)

    def run(self, *args):
        pfiles = PFiles(inp=self.inp, out=self.out, other=args)

        if not (len(args) == 1) ^ bool(pfiles.inp or pfiles.out):
            raise CmdException(
                "Cmd %r takes either a project-name or extracts it "
                "from --inp or --out files given; received args(%s), %s!"
                % (self.name, args, pfiles))

        if self.report and not (pfiles.inp and pfiles.out):
            raise CmdException(
                "Cmd %r needs BOTH --inp and --out files when --report given; "
                "received args(%s), %s!"
                % (self.name, args, pfiles))

        if len(args) == 1:
            yield self.projects_db.proj_add(args[0])
        else:
            pfiles.check_files_exist(self.name)

            from . import report

            repspec = report.ReporterSpec(config=self.config)
            if repspec.include_input_in_dice and not self.report:
                raise CmdException(
                    "Command %r received a --with-inputs flag but without --report!"
                    % (self.name))

            finfos = repspec.extract_dice_report(pfiles)
            for fpath, data in finfos.items():
                iokind = data['iokind']
                if iokind in ('inp', 'out'):
                    project = data['report']['vehicle_family_id']
                    self.log.info("Project '%s' derived from '%s' file: %s",
                                  project, iokind, fpath)
                    break
            else:
                assert False, "Failed derriving project-id from: %s" % finfos

            self.projects_db.proj_add(project)

            yield from self.append_and_report(pfiles)


class ReportCmd(_SubCmd):
    """
    Prepares or re-prints the signed dice-report that can be sent for timestamping.

    - Use --force to generate a new report.
    - Use --dry-run to see its rough contents without signing and storing it.

    SYNTAX
        %(cmd_chain)s [OPTIONS]

    - Eventually the *Dice Report* parameters will be time-stamped and disseminated to
      TA authorities & oversight bodies with an email, to receive back
      the sampling decision.
    - To send the report to the stamper, use `tsend` sub-command.
    - To get report ready for sending it MANUALLY, use `tsend --dry-run`
      instead.

    """

    examples = trt.Unicode("""
        - Create or view existing report of the *current* project::
              %(cmd_chain)s

        - Or the same command using `git` primitives (in Bash)::
              git -C ~/.codice/repo cat-file tag dices/RL-12-BM3-2016-000/1 | %(cmd_chain)s send
    """)

    def __init__(self, **kwds):
        from . import crypto
        from . import report

        kwds.setdefault('conf_classes', [report.ReporterSpec, crypto.GitAuthSpec])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    'Project': {'dry_run': True},
                },
                "Verify dice-report do not actually store it in the project."
            ),
            'with-inputs': (
                {
                    'ReporterSpec': {'include_input_in_dice': True},
                }, report.ReporterSpec.include_input_in_dice.help),
        })
        super().__init__(**kwds)

    def run(self, *args):
        self.log.info('Tagging project %r...', args)
        if len(args) > 0:
            raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                               % (self.name, len(args), args))

        ok = None

        ## TODO: move code in project and simplify `do_report()`.
        #
        proj = self.current_project
        repo = proj.repo
        tagref = _find_dice_tag(repo, proj.pname,
                                proj.max_dices_per_project)
        gen_report = proj.state == 'wltp_iof' or not tagref or self.force
        if gen_report:
            ok = proj.do_report()
            assert isinstance(proj.result, str)
            result = ok and proj.result or ok
        else:
            self.log.debug("Report already generated as '%s'.", tagref.path)
            result = _read_dice_tag(repo, tagref)
            ok = True

        if ok:
            key_uid = proj.extract_uid_from_report(result)
            self.log.info("Report has been signed by '%s'.", key_uid)

        yield result


class TstampCmd(_SubCmd):
    """Deprecated: renamed as `tsend`!"""
    def run(self, *args):
        raise CmdException("Cmd %r has been renamed to %r!"
                           % (self.name, 'tsend'))


class TsendCmd(_SubCmd):
    """
    IRREVOCABLY send report to time-stamp service, or print it for sending it manually (--dry-run).

    SYNTAX
        %(cmd_chain)s [OPTIONS]

    - THIS COMMAND IS IREVOCABLE!
    - Use --dry-run if you want to send the email yourself.
      Remember to set the 'To', 'Subject' & 'Cc` fields.
    - The --dry-run option prints the email as it would have been sent; you may
      copy-paste this lient and send it, formatted as 'plain-text' (not 'HTML').
    - Stage transitions: tagged --> mailed
    """

    examples = trt.Unicode("""\
        - Send report (project must be in `taged` state)::
              %(cmd_chain)s

        - Get dice-email for "manual" procedure::
              %(cmd_chain)s  --dry-run

         see https://github.com/JRCSTU/CO2MPAS-TA/wiki/8.-The-DICE.#steps-to-proceed-manually

        - See `co2dice tstamp` command for server configurations.
    """)

    def __init__(self, **kwds):
        from . import crypto
        from . import tstamp

        kwds.setdefault('conf_classes', [
            tstamp.TstampSender, crypto.GitAuthSpec])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    'Project': {'dry_run': True},
                },
                "Print dice-report and bump `mailed` but do not actually send tstamp-email."
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        if len(args) > 0:
            raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                               % (self.name, len(args), args))

        proj = self.current_project
        ok = proj.do_sendmail()

        return self._format_result(ok, proj.result,
                                   is_verbose=self.verbose or proj.dry_run)


class TparseCmd(_SubCmd):
    """
    Derives *decision* OK/SAMPLE flag from tstamped-response, and store it in current-project.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<tstamped-file-1> ...]

    - If '-' is given or no file at all, it reads from STDIN.
    - If --force, ignores most verification/parsing errors.
      that is, when you don't have the files of the projects in the repo.
      With this option, tstamp-response get, it extracts the dice-repot and adds it
      as a "broken" tag referring to projects that might not exist in the repo,
      assuming they don't clash with pre-existing dice-reponses.
    - Fails if dice refers to different project.
    """
    examples = trt.Unicode("""\
        - Parse `dice_tstamp.txt` file, as received manually from timestamper::
              %(cmd_chain)s  dice_tstamp.txt

        - Parse response copied from *clipboard*::
              %(cmd_chain)s
              [CTRL+Insert]
              [CTRL+Z]

        - See `co2dice tstamp` command for server configurations.
    """)

    def __init__(self, **kwds):
        from . import tstamp
        from . import crypto

        kwds.setdefault('conf_classes', [
            tstamp.TstampReceiver, crypto.GitAuthSpec, crypto.StamperAuthSpec])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    'Project': {'dry_run': True},
                },
                "Parse the tstamped response without storing it in the project."
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        if len(args) > 1:
            raise CmdException('Cmd %r takes one optional filepath, received %d: %r!'
                               % (self.name, len(args), args))

        file = '-' if not args else args[0]
        ## Fail early if cur-project not open.
        proj = self.current_project

        if file == '-':
            self.log.info("Reading STDIN; paste message verbatim!")
            mail_text = sys.stdin.read()
        else:
            if not osp.exists(file):
                raise CmdException("File to parse '%s' not found!" % file)

            self.log.info("Reading '%s'...", pndlu.convpath(file))
            with io.open(file, 'rt') as fin:
                mail_text = fin.read()

        proj.do_storedice(tstamp_txt=mail_text)  # Ignoring ok/false.
        report = proj.result

        if isinstance(report, str):
            ## That's parsed decision.
            return report

        from toolz import dicttoolz as dtz

        short = dtz.keyfilter(lambda k: k == 'dice', report)

        return self._format_result(short, report, default_flow_style=False)


class TrecvCmd(TparseCmd):
    """
    Fetch tstamps from IMAP server, derive *decisions* OK/SAMPLE flags and store them.


    SYNTAX
        %(cmd_chain)s [OPTIONS] [<search-term-1> ...]


    - The fetching of emails can happen in one-shot or waiting mode.
    - For terms are searched in the email-subject - tip: use the project name(s).
    - If --force, ignores most verification/parsing errors.
      that is, when you don't have the files of the projects in the repo.
      With this option, tstamp-response get, it extracts the dice-repot and adds it
      as a "broken" tag referring to projects that might not exist in the repo,
      assuming they don't clash with pre-existing dice-reponses.
    """
    #- The --build-registry is for those handling "foreign" dices (i.e. TAAs),

    examples = trt.Unicode("""
        - To search emails in one-shot::
              %(cmd_chain)s --after today "IP-10-AAA-2017-1003"
              %(cmd_chain)s --after "last week"
              %(cmd_chain)s --after "1 year ago" --before "18 March 2017"

        - To wait for new mails arriving (and not to block console),
          - on Linux::
                %(cmd_chain)s --wait &
                ## wait...
                kill %%1  ## Asumming this was the only job started.

          - On Windows::
                START \\B %(cmd_chain)s --wait

            and kill with one of:
              - `[Ctrl+Beak]` or `[Ctrl+Pause]` keystrokes,
              - `TASKLIST/TASKKILL` console commands, or
              - with the "Task Manager" GUI.

        - See `co2dice tstamp` command for server configurations.
    """)

    wait = trt.Bool(
        False,
        help="""
        Whether to wait reading IMAP for any email(s) satisfying the criteria and report them.

        WARN:
          Process must be killed afterwards, so start it in the background.
          e.g. `START /B co2dice ...` or append the `&` character in Bash.
        """
    ).tag(config=True)

    email_preview_nchars = trt.Int(500).tag(config=True)

    def __init__(self, **kwds):
        from . import tstamp
        from . import crypto

        ## Note here cannot update kwds-defaults,
        #  or would cancel baseclass's choices.
        self.conf_classes.extend([
            tstamp.TstampSender, tstamp.TstampReceiver,
            crypto.GitAuthSpec, crypto.StamperAuthSpec])
        self.cmd_aliases.update(tstamp.recv_cmd_aliases)
        self.cmd_flags.update({
            ('n', 'dry-run'): (
                {'Project': {'dry_run': True}},
                "Pase the tstamped response without storing it in the project."
            ),
            'wait': (
                {type(self).__name__: {'wait': True}},
                type(self).wait.help
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        from . import tstamp

        warn = self.log.warning
        info = self.log.info
        error = self.log.error

        info("Receiving emails for projects(s) %s: ...", args)

        projDB = self.projects_db
        default_flow_style = False
        rcver = tstamp.TstampReceiver(config=self.config)

        ## IMAP & CmdException raised here.
        emails = rcver.receive_timestamped_emails(self.wait, args, read_only=False)
        for uid, mail in emails:
            mid = mail.get('Message-Id')
            mail_text = mail.get_payload()
            try:
                tag_name = rcver.extract_dice_tag_name(mail['Subject'], mail_text)
                verdict = rcver.parse_tstamp_response(mail_text, tag_name)
            except CmdException as ex:
                verdict = ex
                error("[%s]%s: parsing tstamp-email stopped due to: %s"
                      "\n  Use `--verbose ` or command `tstamp recv --raw` "
                      "to view email.",
                      uid, mid, ex, exc_info=self.verbose)
            except Exception as ex:
                verdict = ex
                error("[%s]%s: parsing tstamp-email failed due to: %r",
                      "\n  Use `--verbose ` or command `tstamp recv --raw`"
                      "to view email.",
                      uid, mid, ex, exc_info=self.verbose)

            ## Store full-verdict (verbose).
            all_infos = rcver.get_recved_email_infos(
                mail, verdict, verbose=True, email_infos=None)
            pname = all_infos.get('project')

            if pname is None:
                ## Must have already warn.
                preview = ('\n%s\n' % mail_text
                           if self.verbose else
                           '\n%s\n...\n' % mail_text[:self.email_preview_nchars])
                info("[%s]%s: skipping unparseable tstamp-email.\n%s",
                     uid, mid, preview)
                continue

            try:
                proj = projDB.proj_open(pname)
                is_foreign = False
            except CmdException:
                ## Build registry.
                info("[%s]%s: tstamp-email from foreign project '%s'.",
                     uid, mid, pname)
                is_foreign = True

            try:
                if is_foreign:
                    if projDB.allow_foreign_dice:
                        projDB.append_foreign_dice(all_infos)
                    else:
                        self.log.warning(
                            "Storing a dice from \"foreign\" project '%s'"
                            "is an advanced operation, which is not enabled.",
                            pname)
                else:
                    proj.do_storedice(tstamp_txt=mail_text, verdict=verdict)  # Ignoring ok/false.

                ## Respect --verbose and --email-infos for print-outs.
                infos = rcver.get_recved_email_infos(mail, verdict)
                yield _mydump({'[%s]%s' % (uid, mid): infos},
                              default_flow_style=default_flow_style)

            except CmdException as ex:
                email_dump = _mydump({'[%s]%s' % (uid, mid): all_infos},
                                     default_flow_style=default_flow_style)
                warn('[%s]%s: Cannot store %ststamp due to: %s\n  email: \n%s',
                     uid, mid, "foreign " if is_foreign else '', ex,
                     tw.indent(email_dump, '    '))

            except Exception as ex:
                email_dump = _mydump({'[%s]%s' % (uid, mid): all_infos},
                                     default_flow_style=default_flow_style)
                error('[%s]%s: storing %ststamp failed due to: %s\n  email: \n%s',
                      uid, mid, "foreign " if is_foreign else '', ex,
                      tw.indent(email_dump, '    '), exc_info=1)


class ExportCmd(_SubCmd):
    """
    Archives given project or all if none given.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<project-1>] ...

    - The archive created is named `CO2MPAS_projects-<timestamp>`.
    - If the `--erase-afterwards` is given on the *current-project*,
      you must then select another one, with `project open` command.
      For that, if '.' is given, it deletes the *current*, and if no args given,
      it DELETES ALL projects.
    """
    erase_afterwards = trt.Bool(
        help="Will erase all archived projects from repo."
    ).tag(config=True)

    out = trt.CUnicode(
        help="""
        The filepath of the resulting zip archive to create.

        - If undefined, a "standard" filename is generated based on
          current date & time, like `CO2MPAS_projects-<timestamp>`.
        - If the given path resolves to a folder, the "standard" filename
          is appended.
        - The '.zip' extension is not needed.
        """
    ).tag(config=True)

    def __init__(self, **kwds):
        self.cmd_aliases.update({('o', 'out'): 'ExportCmd.out'})
        self.cmd_flags.update({
            'erase-afterwards': (
                {type(self).__name__: {'erase_afterwards': True}},
                type(self).erase_afterwards.help
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        ## TODO: Move Export/Import code to a Spec.
        from datetime import datetime
        import shutil
        import tempfile
        import git
        from git.util import rmtree
        from ..utils import chdir

        arch_format = 'zip'
        repo = self.projects_db.repo
        cur_pname = repo.active_branch and _ref2pname(repo.active_branch)

        ## Resolve '.', ALL or specific project-names.
        #
        if not args:
            pnames = [_ref2pname(ref) for ref in _yield_project_refs(repo)]
        else:
            pnames = [pn == '.' and cur_pname or pn
                      for pn in iset(args)]
        pnames = iset(pnames)

        now = datetime.now().strftime('%Y%m%d-%H%M%S%Z')
        standard_fname = '%s-%s' % ("CO2MPAS_projects", now)
        zip_name = self.out or standard_fname
        zip_name = pndlu.convpath(zip_name)
        if osp.isdir(zip_name):
            zip_name = osp.join(zip_name, standard_fname)
        zip_name = pndlu.ensure_file_ext(zip_name, '.%s' % arch_format)

        if not self.force and osp.exists(zip_name):
            raise CmdException("File to export '%s' already exists!"
                               "\n  Use force to append into it."
                               % zip_name)
        dst_folder = osp.dirname(zip_name)
        if dst_folder:
            if not osp.exists(dst_folder):
                if self.force:
                    os.makedirs(dst_folder)
                else:
                    raise CmdException(
                        "Archive destination folder '%s' does not exist!  "
                        "Use --force to create it." % dst_folder)
            elif not osp.isdir(dst_folder):
                raise CmdException(
                    "Archive's parent '%s' already exists "
                    "but is not a folder!" % dst_folder)

        self.log.info("Will export %s project(s) %s --> %s...",
                      len(pnames), tuple(pnames), zip_name)

        ## NOTE: Create arch-repo clone next to project-repo,
        #  because local-paths for *remotes* in CYGWIN/MSYS2 DO NOT WORK!!
        arch_repo_parentdir = osp.join(repo.working_dir, '..')
        with chdir(arch_repo_parentdir), tempfile.TemporaryDirectory(
                prefix='co2mpas_export-', dir='.') as tdir:
            arch_dir = osp.join(tdir, 'repo')
            arch_repo = git.Repo.init(arch_dir, bare=True)
            rem_url = osp.join('..', '..', osp.basename(repo.working_dir))
            any_exported = False
            try:
                rem = arch_repo.create_remote('origin', rem_url)
                try:
                    ## `rem` pointing to my (.co2dice) repo.

                    for pi, p in enumerate(pnames):
                        self.log.info("Exporting project %s: '%s' out of %s...",
                                      pi, p, len(pnames))
                        pp = _pname2ref_name(p)
                        if pp not in repo.heads:
                            self.log.warning(
                                "Ignoring branch(%s), not a co2mpas project.", p)
                            continue
                        any_exported = True

                        ## Note: All tags pointsing to branches are fetched.
                        fetch_infos = rem.fetch('%s:%s' % (pp, pp))

                        ## Create local branches in arch_repo
                        #
                        for fi in fetch_infos:
                            path = fi.remote_ref_path
                            #if fi.flags == fi.NEW_HEAD:  ## 0 is new branch!!
                            if fi.flags == 0:
                                arch_repo.create_head(path, fi.ref)
                            yield 'packed: %s' % path

                    if not any_exported:
                        raise CmdException(
                            "Nothing exported for these arguments: {}"
                            .format(args))

                finally:
                    arch_repo.delete_remote(rem)

                root_dir, base_dir = osp.split(arch_repo.working_dir)
                yield 'Archive: %s' % shutil.make_archive(
                    base_name=osp.splitext(zip_name)[0],
                    format=arch_format,
                    base_dir=base_dir,
                    root_dir=root_dir)

                if self.erase_afterwards:
                    for p in pnames:
                        tref = _tname2ref_name(p)
                        for t in list(repo.tags):
                            if t.name.startswith(tref):
                                yield "del tag: %s" % t.name
                                repo.delete_tag(t)

                        pbr = repo.heads[_pname2ref_name(p)]
                        yield "del branch: %s" % pbr.name

                        ## Cannot del checked-out branch!
                        #
                        ok = False
                        try:
                            if pbr == repo.active_branch:
                                if 'tmp' not in repo.heads:
                                    repo.create_head('tmp')
                                repo.heads.tmp.checkout(force=True)

                            repo.delete_head(pbr, force=True)
                            ok = True
                        finally:
                            if not ok:
                                pbr.checkout(pbr)

            finally:
                arch_repo.__del__()
                del arch_repo
                rmtree(arch_dir)


class ImportCmd(_SubCmd):
    """
    Import the specified zipped project-archives into repo; reads STDIN if non specified.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<zip-file-1> ...]

    - If '-' is given or no file at all, it reads from STDIN.
    """
    def run(self, *args):
        import tempfile
        import zipfile

        files = iset(pndlu.convpath(pndlu.ensure_file_ext(a, '.zip'))
                     for a in args) or ['-']

        self.log.info("Will import from %s archive(s): %s...",
                      len(files), tuple(files))

        repo = self.projects_db.repo
        with tempfile.TemporaryDirectory(prefix='co2mpas_import-') as tdir:
            for f in files:
                if f == '-':
                    f = sys.stdin.buffer
                    remname = 'stdin'
                else:
                    remname, _ = osp.splitext(osp.basename(f))
                exdir = osp.join(tdir, remname)

                self.log.info("Importing '%s'...", f)
                try:
                    with zipfile.ZipFile(f, "r") as zip_ref:
                        zip_ref.extractall(exdir)

                    arch_remote = repo.create_remote(
                        remname, osp.join(exdir, 'repo'))
                    try:
                        fetch_infos = arch_remote.fetch(force=self.force, tags=True)

                        for fi in fetch_infos:
                            path = fi.remote_ref_path
                            if fi.flags == fi.NEW_HEAD:
                                repo.create_head(path, fi.ref)
                            yield 'unpacked: %s' % path

                    finally:
                        repo.delete_remote(remname)

                except Exception as ex:
                    if 'missing object referenced by' in str(ex):
                        self.log.warning(
                            "Imported only PARTIALLY objects from '%s: %s(%s)"
                            "'n  Note: importing archives with \"foreign dices\" "
                            "is not supported yet!",
                            f, type(ex).__name__, ex,
                            exc_info=self.verbose)
                    else:
                        self.log.error("Error while importing from '%s': %s(%s)",
                                       f, type(ex).__name__, ex,
                                       exc_info=self.verbose)

        plist = list(self.projects_db.proj_list())
        if len(plist) == 1:
            alone_project = plist[0]
            self.log.info("Projects-db now contains a single project '%s'; "
                          "auto-opening it...", alone_project)
            self.projects_db.proj_open(alone_project)


class BackupCmd(_SubCmd):
    """
    Backup projects repo into the archive filepath specified (or current-directory).

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<archive-path>]
    """
    erase_afterwards = trt.Bool(
        help="Will erase the whole repository and ALL PROJECTS contained fter backing them up."
    ).tag(config=True)

    def __init__(self, **kwds):
        self.cmd_flags.update({
            'erase-afterwards': (
                {type(self).__name__: {'erase_afterwards': True}},
                type(self).erase_afterwards.help
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        if len(args) > 1:
            raise CmdException('Cmd %r takes one optional filepath, received %d: %r!'
                               % (self.name, len(args), args))
        self.log.info('Archiving repo into %r...', args)

        archive_fpath = args and args[0] or None
        kwds = {}
        if archive_fpath:
            base, fname = osp.split(archive_fpath)
            if base:
                kwds['folder'] = base
            if fname:
                kwds['repo_name'] = fname
        try:
            return self.projects_db.repo_backup(
                erase_afterwards=self.erase_afterwards,
                **kwds)
        except FileNotFoundError as ex:
            raise baseapp.CmdException(
                "Folder '%s' to store archive does not exist!"
                "\n  Use --force to create it." % ex)


all_subcmds = (LsCmd, InitCmd, OpenCmd,
               AppendCmd, ReportCmd,
               TstampCmd,  # TODO: delete deprecated `projext tsend` cmd
               TsendCmd,
               TrecvCmd, TparseCmd,
               StatusCmd,
               ExportCmd, ImportCmd, BackupCmd)
