#!/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""A *project* stores all CO2MPAS files for a single vehicle, and tracks its sampling procedure. """
from collections import (defaultdict, OrderedDict, namedtuple)  # @UnusedImport
import copy
import io
import os
import re
import sys
from typing import (
    Any, Union, List, Dict, Sequence, Iterable, Optional, Text, Tuple, Callable)  # @UnusedImport

from boltons.setutils import IndexedSet as iset
from toolz import itertoolz as itz
import transitions
from transitions.core import MachineError
import yaml

import functools as fnt
import os.path as osp
import pandalone.utils as pndlu
import textwrap as tw
import traitlets as trt
import traitlets.config as trtc

from . import baseapp, dice, CmdException, PFiles
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from .._version import __dice_report_version__


vehicle_family_id_regex = re.compile(r'^(?:IP|RL|RM|PR)-\d{2}-\w{2,3}-\d{4}-\d{4}$')
git_project_regex = re.compile('^\w[\w-]+$')

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
                "incompatible message, cannot parse its version'" %
                (msg_starting_txt, prog_ver[0], prog_ver[1]))

        major, minor, micro = m.group(1, 2, 3)
        if int(major) != int(prog_ver[0]) or int(minor) > int(prog_ver[1]):
            raise ValueError(
                "incompatible message version '%s', expected '%s'" %
                ('.'.join((major, minor, micro)), __dice_report_version__))

    def dump_commit_msg(self, **kwds):
        cdic = self._asdict()
        del cdic['data']
        clist = [cdic]
        if self.data:
            clist.extend(self.data)
        msg = _mydump(clist, **kwds)

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
            l = yaml.load(cmsg_txt)
            if not isinstance(l, list) or not l:
                raise ValueError("expected a non-empty list")

            headline = l[0]
            cmsg = _CommitMsg(data=l[1:], **headline)

            return cmsg
        except Exception as ex:
            raise CmdException(
                "Failed parsing commit message due to: %r\nmsg:\n%s" %
                (ex, tw.indent(cmsg_txt, "  ")))


_PROJECTS_PREFIX = 'projects/'
_HEADS_PREFIX = 'refs/heads/'
_PROJECTS_FULL_PREFIX = _HEADS_PREFIX + _PROJECTS_PREFIX


def _is_project_ref(ref: 'git.Reference') -> bool:
    return ref.name.startswith(_PROJECTS_PREFIX)


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

    raise CmdException("Too many dices(%d) for project '%s'!"
                       "\n  Maybe delete project and start all over?" %
                       (i + 1, pname))


def _read_dice_tag(repo, tag: Union[Text, 'git.TagReference']):
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
        3,
        help="""Number of dice-attempts allowed to be forced for a project."""
    ).tag(config=True)


class Project(transitions.Machine, ProjectSpec):
    """The Finite State Machine for the currently checked-out project."""

    dry_run = trt.Bool(
        help="Process actions but do not actually commit/tag results in the project."
    ).tag(config=True)

    git_desc_width = trt.Int(
        78, allow_none=False,
        help="""
        The width of the textual descriptions when committing and tagging Git objects.

        The email sent for timestamping is generated from tag-descriptions.

        According to RFC5322, 78 is the maximum width for textual emails;
        mails with width > 78 may be sent as HTML-encoded and/or mime-multipart.
        """
    ).tag(config=True)

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
        return report.Report(config=self.config)

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

            - [do_report,  wltp_iof,   tagged]
            - [do_report,  tagged,     tagged]
            - [do_report,  mailed,     tagged,       _is_force        ]

            - [do_sendmail,  tagged,     mailed                         ]
            - [do_sendmail,  mailed,     mailed,     _is_dry_run        ]

            - trigger:    do_storedice
              source:     mailed
              dest:       sample
              prepare:    _parse_response
              conditions: [_is_decision_sample, _is_not_dry_run_dicing]

            - trigger:    do_storedice
              source:     mailed
              dest:       nosample
              prepare:    _parse_response
              conditions:     [_is_not_decision_sample, _is_not_dry_run_dicing]

            - [do_addfiles,  [diced,
                              nosample,
                              sample],   nedc,         _is_other_files  ]
            """)

        super().__init__(states=states,
                         initial=states[0],
                         transitions=trans,
                         send_event=True,
                         global_prepare=['_cb_clear_result'],
                         before_state_change=['_cb_check_my_index'],
                         after_state_change='_cb_commit_or_tag',
                         auto_transitions=False,
                         name=pname,
                         **kwds)
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
        assert data is None or isinstance(data, list), "Data not a list: %s" % data
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

    def _cb_commit_or_tag(self, event):
        """Executed AFTER al state changes, and commits/tags into repo. """
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
        report = _evarg(event, 'report', list, missing_ok=True)
        is_tagging = state == 'tagged' and report
        cmsg = self._make_commitMsg(action, report)
        cmsg_txt = cmsg.dump_commit_msg(width=self.git_desc_width)

        self.log.info('Committing %s: %s', self, action)
        index = repo.index
        index.commit(cmsg_txt)

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
                                             self.max_dices_per_project, fetch_next=True)
                    self.log.info('Tagging %s: %s', self, tagname)
                    assert isinstance(tagname, str), tagname

                    tagref = repo.create_tag(tagname, message=cmsg_txt,
                                             sign=True, local_user=git_auth.master_key)
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
        event.kwargs['action'] = 'new project'

    def _cb_stage_pfiles(self, event):
        """
        Triggered by `do_addfiles(pfiles=<PFiles>)` on ENTER for all `wltp_XX` & 'nedc' states.

        :param PFiles pfiles:
            what to import
        """
        import shutil

        self.log.info('Importing files: %s...', event.kwargs)
        pfiles = _evarg(event, 'pfiles', PFiles)

        ## Check extraction of report works ok,
        #  and that VFids match.
        #
        try:
            rep = self._report_spec()
            rep.get_dice_report(pfiles, expected_vfid=self.pname)
        except CmdException as ex:
            msg = "Failed extracting report from %s, due to: %s"
            if self.force:
                msg += "  BUT FORCED to import them!"
                self.log.warning(msg, pfiles, ex, exc_info=1)
            else:
                raise CmdException(msg % (pfiles, ex))

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
        event.kwargs['action'] = 'add %s files' % pfiles.nfiles()

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
        io_kinds = PFiles._io_kinds_list(*io_kinds)
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

        Uses the :class:`Report` to build the tag-msg.

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
            report = list(repspec.get_dice_report(pfiles).values())

            if self.dry_run:
                self.log.warning("DRY-RUN: Not actually committed the report, "
                                 "and it is not yet signed!")
                self.result = _mydump(report)

                return

            ## Commit/tag callback expects `report` on event.
            event.kwargs['action'] = 'drep %s files' % pfiles.nfiles()
            event.kwargs['report'] = report
        else:
            assert tagref
            self.log.info("Report already generated  as '%s'.", tagref.path)
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
        assert tagref
        signed_dice_report = _read_dice_tag(repo, tagref)
        assert signed_dice_report

        dice_mail_mime = tstamp_sender.send_timestamped_email(
            signed_dice_report, tagref.name, dry_run=dry_run)

        if dry_run:
            self.log.warning(tw.dedent("""\
                DRY-RUN: Now you must send the email your self!
                ==========================================================================
                  - Copy from the 1st line starting with 'X-Stamper-To:', below;
                  - set 'Subject' and 'To' exactly as shown (you may also set Cc & Bcc);
                  - remember to set the email as 'plain-text' (not 'HTML'!) right before
                    clicking `send`!
                ==========================================================================
                """))
            self.result = str(dice_mail_mime)

        if event.transition.source != 'mailed':
            ## Don't repeat your self...
            event.kwargs['action'] = '%s stamp-email' % ('FAKED' if dry_run else 'sent')

    def _parse_response(self, event) -> bool:
        """
        Triggered by `do_storedice(verdict=<dict> | tstamp_txt=<str>)` in PREPARE for `sample/nosample` states.

        :param verdict:
            The result of verifying timestamped-response.
        :return:
            Setting :attr:`result` to ODict.

        .. Note:
            It needs an already verified tstamp-response because to select which project
            it belongs to, it needs to parse the dice-report contained within the response.
        """
        from . import tstamp

        verdict = _evarg(event, 'verdict', dict, missing_ok=True)
        tstamp_txt = _evarg(event, 'tstamp_txt', str, missing_ok=True)
        # TODO: assert for future, when single prep/ trans.
        ##assert (verdict is None) ^ (tstamp_txt is None), (verdict, tstamp_txt)

        if verdict is None:
            recv = tstamp.TstampReceiver(config=self.config)
            verdict = recv.parse_tstamp_response(tstamp_txt)  # FIXME: Bad tstamps brake parsing in there!!!

        pname = verdict.get('report', {}).get('project')
        if pname != self.pname and not self.force:
            raise CmdException(
                "Current project('%s') is different from tstamp('%s')!" %
                (self.pname, pname))

        event.kwargs['verdict'] = self.result = verdict

        ## TODO: **On commit, set arbitrary files to store (where? name?)**.
        repo = self.repo
        index = repo.index
        tstamp_fpath = osp.join(repo.working_tree_dir, 'tstamp.txt')
        with io.open(tstamp_fpath, 'wt') as fp:
            res = _mydump(verdict)
            fp.write(res)
        index.add([tstamp_fpath])

        event.kwargs['report'] = list(verdict.get('dice', {}).items())

    def _is_decision_sample(self, event) -> bool:
        verdict = _evarg(event, 'verdict', dict)

        decision = verdict.get('dice', {}).get('decision', 'SAMPLE')
        event.kwargs['action'] = "diced as %s" % decision

        return decision == 'SAMPLE'

    def _is_not_decision_sample(self, event) -> bool:
        return not self._is_decision_sample(event)

    def _is_not_dry_run_dicing(self, event):
        if self.dry_run:
            self.log.warning('DRY-RUN: Not actually registering decision.')
        return not self.dry_run


class ProjectsDB(trtc.SingletonConfigurable, ProjectSpec):
    """A git-based repository storing the TA projects (containing signed-files and sampling-responses).

    It handles checkouts but delegates index modifications to `Project` spec.

        ### Git Command Debugging and Customization:

        - :envvar:`GIT_PYTHON_TRACE`: If set to non-0,
          all executed git commands will be shown as they happen
      If set to full, the executed git command _and_ its entire output on stdout and stderr
      will be shown as they happen

      NOTE: All logging is done through a Python logger, so make sure your program is configured
      to show INFO-level messages. If this is not the case, try adding the following to your program:

    - :envvar:`GIT_PYTHON_GIT_EXECUTABLE`: If set, it should contain the full path to the git executable, e.g.
      ``c:\Program Files (x86)\Git\bin\git.exe on windows`` or ``/usr/bin/git`` on linux.
    """

    repo_path = trt.Unicode(
        osp.join(baseapp.default_config_dir(), 'repo'),
        help="""
        The path to the Git repository to store TA files (signed and exchanged).
        If relative, it joined against default config-dir: '{confdir}'
        """.format(confdir=baseapp.default_config_dir())
    ).tag(config=True)

    preserved_git_settings = trt.List(
        trt.Unicode(),
        help="""
        On app start up, re-write git's all config-settings except those mathcing this list of regexes.
        Git settings include user-name and email address, so this option might be usefull
        when the regular owner running the app has changed.
        """).tag(config=True)

    ## TODO: Delete `reset_git_settings` in next big release after 1.5.5.
    reset_git_settings = trt.Bool(
        False,
        help="Deprecated and non-functional!  Replaced by `--ProjectsDB.preserved_git_settings` list."
    ).tag(config=True)

    @trt.validate('reset_git_settings')
    def _warn_deprecated(self, proposal):
        self.log.warning(type(self).reset_git_settings.help)

    ## Useless, see https://github.com/ipython/traitlets/issues/287
    # @trt.validate('repo_path')
    # def _normalize_path(self, proposal):
    #     repo_path = proposal['value']
    #     if not osp.isabs(repo_path):
    #         repo_path = osp.join(default_config_dir(), repo_path)
    #     repo_path = pndlu.convpath(repo_path)
    # return repo_path

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
                self.__repo.git.clear_cache()
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
                except:
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
            self.log.info('Failed reading git-settings on %s.%s due to: %s',
                          sec, cname, ex, exc_info=1)
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
                self.log.error("Failed erasing Repo '%s'due to: %s",
                               self.repo_path, ex, exc_info=1)

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
        Examine infos bout the projects-db.

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

        The project returned is appropriately configured according to its recorded state.
        The git-repo is not touched.
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
            except Exception as ex:
                self.log.warning("Failure while getting current-project: %s",
                                 ex, exc_info=1)

        if not self._current_project:
                raise CmdException(tw.dedent("""
                        No current-project exists yet!"
                        Try opening an existing project, with:
                            co2mpas project open <project-name>
                        or create a new one, with:
                            co2mpas project init <project-name>
                        """))

        return self._current_project

    def validate_project_name(self, pname: Text) -> Project:
        return pname and (self.force and
                          git_project_regex.match(pname) or
                          vehicle_family_id_regex.match(pname))

    def proj_add(self, pname: Text) -> Project:
        """
        Creates a new project and sets it as the current one.

        :param pname:
            the project name (without prefix)
        :return:
            the current :class:`Project` or fail
        """
        self.log.info('Creating project %r...', pname)
        if not self.validate_project_name(pname):
            raise CmdException(
                "Invalid name %r for a project!\n  Expected('FT-ta-WMI-yyyy-nnnn'), "
                "where ta, yyy, nnn are numbers." % pname)

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

    def proj_open(self, pname: Text) -> Project:
        """
        :param pname:
            the project name (without prefix)
        :return:
            the current :class:`Project`
        """
        repo = self.repo
        if pname == '.':
            pname = _ref2pname(repo.head.name)

        prefname = _pname2ref_name(pname)
        if prefname not in repo.heads:
            raise CmdException('Project %r not found!' % pname)
        repo.heads[_pname2ref_name(pname)].checkout(force=self.force)

        self._current_project = None
        return self.current_project()

    def proj_parse_stamped_and_assign_project(self, mail_text: Text):
        from . import tstamp

        recv = tstamp.TstampReceiver(config=self.config)
        verdict = recv.parse_tstamp_response(mail_text)
        report = verdict.get('report', {})
        pname = report.get('project')
        if not pname:
            raise CmdException(
                'Cannot identify which project tstamped-response belongs to!\n%s',
                _mydump(verdict))

        repo = self.repo
        refname = _pname2ref_name(pname)
        if refname in repo.refs:
            ## TODO: Check if dice moved!!
            proj = self.proj_open(pname)
            return proj.do_storedice(verdict=verdict)
        else:
            ## TODO: build_registry
            self.log.warning("Registration of arbitrary Dice-reports is not implemented yet!")
            return verdict

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

        pnames = iset(self.current_project().pname if p == '.' else p
                      for p in pnames)
        for ref in _yield_project_refs(repo, *pnames):
            pname = _ref2pname(ref)
            if not as_text and not verbose:
                yield pname

            infos = self._scan_infos(pname=pname, fields=fields, inv_value='<invalid>')
            if verbose:
                infos = OrderedDict(infos)
                to_yield = {pname: infos}
                if as_text:
                    to_yield = _mydump(to_yield, default_flow_style=False)
            else:
                if as_text:
                    i = dict(infos)
                    to_yield = '%s %s: %s' % (i['is_current'] and '*' or ' ',
                                              pname, i['msg.s'])
                else:
                    to_yield = pname

            yield to_yield


###################
##    Commands   ##
###################

class ProjectCmd(baseapp.Cmd):
    """
    Commands to administer the storage repo of TA *projects*.

    A *project* stores all CO2MPAS files for a single vehicle,
    and tracks its sampling procedure.
    """

    examples = trt.Unicode("""
        To list all existing projects, try:
            %(cmd_chain)s  ls

        To make a new project, type:
            %(cmd_chain)s  init RL-77-AAA-2016-0000

        or to open an existing one (to become the *current* one):
            %(cmd_chain)s  open IP-10-AAA-2017-1006

        To see more infos about the current project, use:
            %(cmd_chain)s  ls. -v

        A typical workflow is this:
            %(cmd_chain)s  init RL-12-BM3-2016-0000
            %(cmd_chain)s  append  --inp input.xlsx  --out output.xlsx   summary.xlsx  co2mpas.log
            %(cmd_chain)s  report
            %(cmd_chain)s  tstamp
            cat <tstamp-response-text> | %(cmd_chain)s  tparse
            %(cmd_chain)s  export

        You may enquiry the status the projects database:
            %(cmd_chain)s  status --vlevel 2
        """)

    def __init__(self, **kwds):
        dkwds = {
            'conf_classes': [ProjectsDB, Project],
            'cmd_flags': {
                'reset-git-settings': (
                    {
                        'ProjectsDB': {'reset_git_settings': True},
                    }, pndlu.first_line(ProjectsDB.reset_git_settings.help)
                )
            },
            'subcommands': baseapp.build_sub_cmds(*all_subcmds),
        }
        dkwds.update(kwds)
        super().__init__(**dkwds)

    @property
    def projects_db(self) -> ProjectsDB:
        p = ProjectsDB.instance(config=self.config)
        p.config = self.config
        return p

    @property
    def current_project(self) -> Project:
        return self.projects_db.current_project()

    def _format_result(self, concise, long, *, is_verbose=None):
        is_verbose = self.verbose if is_verbose is None else is_verbose
        result = long if is_verbose else concise

        return isinstance(result, str) and result or _mydump(result)

    def run(self, *args):
        """Just to ensure project-repo created."""
        self.projects_db.repo
        super().run(*args)


class StatusCmd(ProjectCmd):
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


class LsCmd(ProjectCmd):
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


class OpenCmd(ProjectCmd):
    """
    Make an existing project as *current*.

    SYNTAX
        %(cmd_chain)s [OPTIONS] <project>
    """
    def run(self, *args):
        self.log.info('Opening project %r...', args)
        if len(args) != 1:
            raise CmdException(
                "Cmd %r takes exactly one argument as the project-name, received %r!"
                % (self.name, args))

        projDB = self.projects_db
        proj = projDB.proj_open(args[0])

        return projDB.proj_list(proj.pname, as_text=True) if self.verbose else str(proj)


class InitCmd(ProjectCmd):
    """
    Create a new project.

    SYNTAX
        %(cmd_chain)s [OPTIONS] <project>
    """
    def run(self, *args):
        if len(args) != 1:
            raise CmdException(
                "Cmd %r takes exactly one argument as the project-name, received %r!"
                % (self.name, args))

        return self.projects_db.proj_add(args[0])


class AppendCmd(ProjectCmd):
    """
    Import the specified input/output co2mpas files into the *current project*.

    SYNTAX
        %(cmd_chain)s [OPTIONS] ( --inp <co2mpas-file> |
                                  --out <co2mpas-file> |
                                  <any-file> ) ...

    - To report and tstamp a project, one file (at least) from *inp* & *out* must be given.
    - If an input/output are already present in the current project, use --force.
    - Note that any file argument not given with `--inp`, `--out`, will end-up as "other".
    """

    examples = trt.Unicode("""
        To import an INPUT co2mpas file, try:

            %(cmd_chain)s --inp co2mpas_input.xlsx

        To import both INPUT and OUTPUT files, and overwrite any already imported try:

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

    def __init__(self, **kwds):
        kwds.setdefault('cmd_aliases', {
            ('i', 'inp'): ('AppendCmd.inp', pndlu.first_line(type(self).inp.help)),
            ('o', 'out'): ('AppendCmd.out', pndlu.first_line(type(self).out.help)),
        })
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    'Project': {'dry_run': True},
                },
                "Parse files but do not actually store them in the project."
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        ## TODO: Support heuristic inp/out classification
        pfiles = PFiles(inp=self.inp, out=self.out, other=args)
        self.log.info("Importing report files...\n  %s", pfiles)
        if not pfiles.nfiles():
            raise CmdException(
                "Cmd %r must be given at least one file argument, received %d: %r!"
                % (self.name, pfiles.nfiles(), pfiles))

        proj = self.current_project
        ok = proj.do_addfiles(pfiles=pfiles)

        return self._format_result(ok, proj.result)


class ReportCmd(ProjectCmd):
    """
    Prepares or re-prints the signed dice-report that can be sent for timestamping.

    - Use --force to generate a new report.
    - Use --dry-run to see its rough contents without signing and storing it.

    SYNTAX
        %(cmd_chain)s [OPTIONS]

    - Eventually the *Dice Report* parameters will be time-stamped and disseminated to
      TA authorities & oversight bodies with an email, to receive back
      the sampling decision.
    - To get report ready for sending it MANUALLY, use tstamp` sub-command.

    """

    #examples = trt.Unicode(""" """)

    def __init__(self, **kwds):
        from . import crypto
        from . import report

        kwds.setdefault('conf_classes', [report.Report, crypto.GitAuthSpec])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    'Project': {'dry_run': True},
                },
                "Verify dice-report do not actually store it in the project."
            )
        })
        super().__init__(**kwds)

    def run(self, *args):
        self.log.info('Tagging project %r...', args)
        if len(args) > 0:
            raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                               % (self.name, len(args), args))

        proj = self.current_project
        ok = proj.do_report()

        assert isinstance(proj.result, str)
        return ok and proj.result or ok


class TstampCmd(ProjectCmd):
    """
    IRREVOCABLY send report to the time-stamp service, or print it for sending it manually (--dry-run).

    SYNTAX
        %(cmd_chain)s [OPTIONS]

    - THIS COMMAND IS IIREVOCABLE!
    - Use --dry-run if you want to send the email yourself.
      Remember to use the appropriate 'Subject'.
    - The --dry-run option prints the email as it would have been sent; you may
      copy-paste this lient and send it, formatted as 'plain-text' (not 'HTML').
    """

    #examples = trt.Unicode(""" """)

    def __init__(self, **kwds):
        from . import crypto
        from . import tstamp

        kwds.setdefault('conf_classes', [tstamp.TstampSender, crypto.GitAuthSpec])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    'Project': {'dry_run': True},
                },
                "Print dice-report and bump `mailed` but do not actually send tstamp-email."
            )
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


class TparseCmd(ProjectCmd):
    """
    Derives *decision* OK/SAMPLE flag from tstamped-response, and store it (or compare with existing).

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<tstamped-file-1> ...]

    - If '-' is given or no file at all, it reads from STDIN.
    - If --force, ignores most verification/parsing errors.
    - The --build-registry is for those handling "foreign" dices (i.e. TAAs),
      that is, when you don't have the files of the projects in the repo.
      With this option, tstamp-response get, it extracts the dice-repot and adds it
      as a "broken" tag referring to projects that might not exist in the repo,
      assuming they don't clash with pre-existing dice-reponses.
    """

    #examples = trt.Unicode(""" """)

    build_registry = trt.Bool(
        help="When true, store stamp-response to project referenced, instead of *current*."
    ).tag(config=True)

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
                "Pase the tstamped response without storing it in the project."
            ),
            'build-registry': (
                {
                    'TparseCmd': {'build_registry': True},
                },
                pndlu.first_line(type(self).build_registry.help)
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        if len(args) > 1:
            raise CmdException('Cmd %r takes one optional filepath, received %d: %r!'
                               % (self.name, len(args), args))

        file = '-' if not args else args[0]

        if file == '-':
            self.log.info("Reading STDIN; paste message verbatim!")
            mail_text = sys.stdin.read()
        else:
            self.log.debug("Reading '%s'...", pndlu.convpath(file))
            with io.open(file, 'rt') as fin:
                mail_text = fin.read()

        if self.build_registry:
            report = self.projects_db.proj_parse_stamped_and_assign_project(mail_text)
            ok = False
        else:
            proj = self.current_project
            ok = proj.do_storedice(tstamp_txt=mail_text)
            report = proj.result

        short, long = report.get('dice', ok), report

        return self._format_result(short, long)


class ExportCmd(ProjectCmd):
    """
    Archives projects.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<project-1>] ...

    - The archive created is named `CO2MPAS_projects-<timestamp>`.
    - If the `--ExportCmd.erase_afterwards` flag  is given on the *current-project*,
      you must then select another one, with `project open` command.
      For that, f '.' is given, it deletes the *current*, and if f no args given,
      it DELETES ALL projects.
    """
    erase_afterwards = trt.Bool(
        help="Will erase all archived projects from repo."
    ).tag(config=True)

    def run(self, *args):
        ## TODO: Move Export/Import code to a Spec.
        from datetime import datetime
        import shutil
        import tempfile
        import git
        from git.util import rmtree

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
        self.log.info('Exporting %s --> %s...', args, tuple(pnames))

        now = datetime.now().strftime('%Y%m%d-%H%M%S%Z')
        zip_name = '%s-%s' % ("CO2MPAS_projects", now)
        with tempfile.TemporaryDirectory(prefix='co2mpas_export-') as tdir:
            exdir = osp.join(tdir, 'repo')
            exrepo = git.Repo.init(exdir, bare=True)
            remname = osp.join(repo.working_dir, '.git')
            try:
                rem = exrepo.create_remote('origin', remname)
                try:
                    ## `rem` pointing to my (.co2dice) repo.

                    for p in pnames:
                        pp = _pname2ref_name(p)
                        if pp not in repo.heads:
                            self.log.info("Ignoring branch(%s), not a co2mpas project.", p)
                            continue

                        ## FIXME: Either ALL TAGS (--tags) or NONE without it!
                        fetch_infos = rem.fetch(pp, tag=True)

                        ## Create local branches in exrepo
                        #
                        for fi in fetch_infos:
                            path = fi.remote_ref_path
                            #if fi.flags == fi.NEW_HEAD:  ## 0 is new branch!!
                            if fi.flags == 0:
                                exrepo.create_head(path, fi.ref)
                            yield 'packed: %s' % path

                    root_dir, base_dir = osp.split(exrepo.working_dir)
                    yield 'Archive: %s' % shutil.make_archive(
                        base_name=zip_name, format='zip',
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
                    exrepo.delete_remote(rem)
            finally:
                exrepo.__del__()
                del exrepo
                rmtree(exdir)


class ImportCmd(ProjectCmd):
    """
    Import the specified zipped project-archives into repo; reads SDIN if non specified.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<zip-file-1> ...]

    - If '-' is given or no file at all, it reads from STDIN.
    """
    def run(self, *args):
        ## TODO: Mve ziproject code to Spec.
        import tempfile
        import zipfile

        files = iset(args) or ['-']
        self.log.info('Importing %s...', tuple(files))

        repo = self.projects_db.repo
        with tempfile.TemporaryDirectory(prefix='co2mpas_import-') as tdir:
            for f in files:
                if f == '-':
                    f = sys.stdin
                    remname = 'stdin'
                else:
                    remname, _ = osp.splitext(osp.basename(f))
                exdir = osp.join(tdir, remname)

                with zipfile.ZipFile(f, "r") as zip_ref:
                    zip_ref.extractall(exdir)

                try:
                    rem = repo.create_remote(remname, osp.join(exdir, 'repo'))
                    fetch_infos = rem.fetch(force=self.force)

                    for fi in fetch_infos:
                        path = fi.remote_ref_path
                        if fi.flags == fi.NEW_HEAD:
                            repo.create_head(path, fi.ref)
                        yield 'unpacked: %s' % path

                except Exception as ex:
                    self.log.error("Error while importing from '%s: %s",
                                   ex, exc_info=1)
                else:
                    repo.delete_remote(remname)


class BackupCmd(ProjectCmd):
    """
    Backup projects repository into the archive filepath specified, or current-directory, if none specified.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<archive-path>]
    """
    erase_afterwards = trt.Bool(
        help="Will erase the whole repository and ALL PROJECTS contained fter backing them up."
    ).tag(config=True)

    def run(self, *args):
        self.log.info('Archiving repo into %r...', args)
        if len(args) > 1:
            raise CmdException('Cmd %r takes one optional filepath, received %d: %r!'
                               % (self.name, len(args), args))
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
               TstampCmd, TparseCmd,
               StatusCmd,
               ExportCmd, ImportCmd, BackupCmd)
