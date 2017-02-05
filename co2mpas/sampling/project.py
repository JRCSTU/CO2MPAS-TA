#!/usr/bin/env python
#
# Copyright 2014-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""A *project* stores all CO2MPAS files for a single vehicle, and tracks its sampling procedure. """
from collections import (defaultdict, OrderedDict, namedtuple)  # @UnusedImport
import copy
from datetime import datetime
import io
import os
import re
import sys
import textwrap as tw
from typing import (
    Any, Union, List, Dict, Sequence, Iterable, Optional, Text, Tuple, Callable)  # @UnusedImport

from toolz import itertoolz as itz, dicttoolz as dtz
import transitions
from transitions.core import MachineError
import yaml

import functools as fnt
import os.path as osp
import pandalone.utils as pndlu
import traitlets as trt
import traitlets.config as trtc

from . import baseapp, dice, CmdException, PFiles
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from .._version import __dice_report_version__


###################
##     Specs     ##
###################
vehicle_family_id_regex = re.compile(r'^(?:IP|RL|RM|PR)-\d{2}-\w{2,3}-\d{4}-\d{4}$')
git_project_regex = re.compile('^\w[\w-]+$')

_git_messaged_obj = re.compile(r'^(:?object|tag) ')
_after_first_empty_line_regex = re.compile(r'\n\r?\n')


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
    def _check_commit_msg_version(cls, msg_ver_txt):
        prog_ver = __dice_report_version__.split('.')
        msg_ver = msg_ver_txt.split('.')
        if (len(msg_ver) != 3 or
                msg_ver[0] != prog_ver[0] or
                msg_ver[1] > prog_ver[1]):
            raise ValueError(
                "incompatible message version '%s', expected '%s.%s-.x'" %
                (msg_ver_txt, prog_ver[0], prog_ver[1]))

    def dump_commit_msg(self, indent=2, **kwds):
        cdic = self._asdict()
        del cdic['data']
        clist = [cdic]
        if self.data:
            clist.extend(self.data)
        msg = yaml.dump(clist, indent=indent, **kwds)

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

            l = yaml.load(cmsg_txt)
            if not isinstance(l, list) or not l:
                raise ValueError("expected a non-empty list")

            headline = l[0]
            cmsg = _CommitMsg(data=l[1:], **headline)
            cmsg._check_commit_msg_version(str(cmsg.v))

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


def _tname2ref_name(tname: Text) -> Text:
    if not tname.startswith(_DICES_PREFIX):
        tname = '%s%s' % (_DICES_PREFIX, tname)
    return tname


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


class Project(transitions.Machine, dice.DiceSpec):
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

    def __init__(self, pname, repo, **kwds):
        """DO NOT INVOKE THIS; use performant :meth:`Project.new_instance()` instead."""
        self.pname = pname
        self.rpo = repo
        states = [
            'UNBORN', 'INVALID', 'empty', 'wltp_out', 'wltp_inp', 'wltp_iof', 'tagged',
            'mailed', 'diced', 'nedc',
        ]
        trans = yaml.load(
            # Trigger        Source     Dest-state    Conditions? unless before after prepare
            """
            - [do_invalidate, '*',      INVALID,      None, None,        _cb_invalidated]

            - [do_createme,  UNBORN,    empty]

            - [do_addfiles,  empty,      wltp_iof,     _is_inp_out_files]
            - [do_addfiles,  empty,      wltp_inp,     _is_inp_files    ]
            - [do_addfiles,  empty,      wltp_out,     _is_out_files    ]

            - [do_addfiles,  [wltp_inp,
                              wltp_out,
                              tagged],   wltp_iof,     [_is_inp_out_files,
                                                        _is_force]      ]

            - [do_addfiles,  wltp_inp,   wltp_inp,     [_is_inp_files,
                                                        _is_force]      ]
            - [do_addfiles,  wltp_inp,   wltp_iof,     _is_out_files]

            - [do_addfiles,  wltp_out,   wltp_out,     [_is_out_files,
                                                        _is_force]      ]
            - [do_addfiles,  wltp_out,   wltp_iof,     _is_inp_files]

            - [do_addfiles,  wltp_iof,   wltp_iof,     _is_force        ]

            - [do_prepmail,  wltp_iof,   tagged]
            - [do_prepmail,  tagged,     tagged]
            - [do_prepmail,  mailed,     tagged,       _is_force        ]

            - [do_sendmail,  tagged,     mailed]

            - [do_storedice, mailed,    diced,         _cond_is_diced]

            # - [do_noting,  diced,    diced_ok]
            # - [do_sample,  diced,    diced_ok]

            - [do_addfiles,  diced,      nedc,         _is_other_files  ]
            - [do_addfiles,  nedc,       nedc,         [_is_other_files,
                                                       _is_force]      ]
            """)

        super().__init__(states=states,
                         initial=states[0],
                         transitions=trans,
                         send_event=True,
                         before_state_change=['_cb_check_my_index', '_cb_clear_result'],
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

    def _make_commit_msg(self, action, data=None) -> Text:
        assert data is None or isinstance(data, list), "Data not a list: %s" % data
        cmsg = _CommitMsg(__dice_report_version__, action, self.pname, self.state, data)

        return cmsg.dump_commit_msg(width=self.git_desc_width)

    def _cb_clear_result(self, event):
        """ Executed BEFORE exiting any state, and clears any results from previous transitions. """
        self.result = None

    def _cb_check_my_index(self, event):
        """ Executed BEFORE exiting any state, to compare my `pname` with checked-out ref. """
        active_branch = self.repo.active_branch
        if self.pname != _ref2pname(active_branch):
            ex = MachineError("Expected current project to be %r, but was %r!"
                              % (self.pname, active_branch))
            self.do_invalidate(error=ex)

    max_dices_per_project = trt.Int(
        3,
        help="""Number of dice-attempts allowed to be forced for a project."""
    ).tag(config=True)

    def _find_dice_tag(self, fetch_next=False) -> Union[Text, 'git.TagReference']:
        """Return None if no tag exists yet."""
        repo = self.repo
        tref = _tname2ref_name(self.pname)
        tags = repo.tags
        for i in range(self.max_dices_per_project):
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

        raise CmdException("Too many dices for this project '%s'!"
                           "\n  Maybe delete project and start all over?" % self.pname)

    def read_dice_tag(self, tag: Union[Text, 'git.TagReference']):
        if isinstance(tag, str):
            tag = self.repo.tags[tag]
        return tag.tag.data_stream.read().decode('utf-8')

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
        ## GpgSpec `git_auth` only lazily creates GPG
        #  which imports keys/trust.
        git_auth.GPG
        repo = self.repo
        is_tagging = state == 'tagged'
        report = _evarg(event, 'report', (list, dict), missing_ok=True)
        cmsg_txt = self._make_commit_msg(action, report)

        with repo.git.custom_environment(GNUPGHOME=git_auth.gnupghome_resolved):
            index = repo.index
            index.commit(cmsg_txt)

            self.result = cmsg_txt

            if is_tagging:
                ok = False
                try:
                    self.log.debug('Tagging: %s', event.kwargs)
                    tagname = self._find_dice_tag(fetch_next=True)
                    assert isinstance(tagname, str), tagname

                    tagref = repo.create_tag(tagname, message=cmsg_txt,
                                             sign=True, local_user=git_auth.master_key)
                    self.result = self.read_dice_tag(tagref)

                    ok = True
                finally:
                    if not ok:
                        self.log.warning(
                            "New status('%s') failed, REVERTING to prev-status('%s').",
                            state, event.transition.source)
                        repo.active_branch.commit = 'HEAD~'

    def _make_readme(self):
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
        except Exception as ex:
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

                ext_fname = osp.split(ext_fpath)[1]
                index_fpath = osp.join(repo.working_tree_dir, io_kind, ext_fname)
                pndlu.ensure_dir_exists(osp.split(index_fpath)[0])
                shutil.copy(ext_fpath, index_fpath)
                index.add([index_fpath])

        ## Commit/tag callback expects `action` on event.
        event.kwargs['action'] = 'imp %s files' % pfiles.nfiles()

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
        Triggered by `do_prepmail()` on ENTER of `tagged` state.

        If already on `tagged`, just sets the :data:`result` and exits,
        unless --force, in which case it generates another tag.

        Uses the :class:`Report` to build the tag-msg.
        """
        tagref = self._find_dice_tag()
        gen_report = not tagref or self.force
        if gen_report:
            self.log.info('Preparing %s dice-email: %s...',
                          'ANEW' if self.force else '', event.kwargs)
            repspec = self._report_spec()
            pfiles = self.list_pfiles('inp', 'out', _as_index_paths=True)
            report = list(repspec.get_dice_report(pfiles).values())

            if self.dry_run:
                self.log.warning("DRY-RUN: Not actually committed the report, "
                                 "and it is not yet signed!")
                # TODO: Add X_recipients!!
                self.result = yaml.dump(report, indent=2)

                return

            ## Commit/tag callback expects `report` on event.
            event.kwargs['action'] = 'drep %s files' % pfiles.nfiles()
            event.kwargs['report'] = report
        else:
            assert tagref
            self.result = self.read_dice_tag(tagref)

    def _cb_send_email(self, event):
        """
        Triggered by `do_sendmail()` on ENTER of `sendmail` state.

        Parses last tag and uses class:`SMTP` to send its message as email.
        """
        self.log.info('Sending email...')
        dry_run = self.dry_run
        tstamp_sender = self._tstamp_sender_spec()

        tagref = self._find_dice_tag()
        assert tagref
        signed_dice_report = self.read_dice_tag(tagref)
        assert signed_dice_report

        dice_mail_mime = tstamp_sender.send_timestamped_email(
            signed_dice_report, self.pname, dry_run=dry_run)

        if dry_run:
            self.log.warning(
                "DRY-RUN: Now you must send the email your self!"
                "\n  'Copy from the 1st line starting with 'X-Stamper-To:', and "
                "\n  remember to set 'Subject' and 'To' as shown.")
            self.result = str(dice_mail_mime)
        else:
            event.kwargs['action'] = '%s stamp-email' % ('FAKED' if dry_run else 'sent')

    def _cond_is_diced(self, event) -> bool:
        """
        Triggered by `do_storedice(verdict=<dict>)` as CONDITION before `diced` state.

        :param verdict:
            The result of verifying timestamped-response.

        .. Note:
            It needs an already verified tstamp-response because to select which project
            it belongs to, it needs to parse the dice-report contained within the response.
        """
        res = _evarg(event, 'verdict', dict)

        dice = res['dice']
        decision = dice['decision']
        resp_txt = yaml.dump(decision, indent=2, wdith=None, default_flow_style=False)

        if self.dry_run:
            self.log.warning('DRY-RUN: Not actually registering decision.')

            self.result = resp_txt

            return False

        event.kwargs['action'] = "diced as %s" % decision
        ## TODO: **On commit, set arbitrary files to store (where? name?)**.
        event.kwargs['report'] = res

        return True


class ProjectsDB(trtc.SingletonConfigurable, dice.DiceSpec):
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

    reset_settings = trt.Bool(
        False,
        help="""
        When enabled, re-writes default git's config-settings on app start up.
        Git settings include user-name and email address, so this option might be usefull
        when the regular owner running the app has changed.
        """).tag(config=True)

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
            if self.reset_settings:
                self.log.info("Resetting to default settings of repo(%s)...",
                              self.__repo.git_dir)
                self._write_repo_configs()
        except git.InvalidGitRepositoryError as ex:
            self.log.info("...failed opening repo '%s',\n  initializing a new repo %r instead...",
                          ex, repo_path)
            self.__repo = git.Repo.init(repo_path)
            self._write_repo_configs()

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

    def _write_repo_configs(self):
        from . import crypto

        repo = self.repo
        git_auth = crypto.get_git_auth(self.config)
        gnupgexe = git_auth.gnupgexe_resolved
        if repo.git.is_cygwin:
            from git.util import cygpath
            gnupgexe = cygpath(gnupgexe)

        gconfigs = [
            ('core.filemode', 'false'),
            ('core.ignorecase', 'false'),
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

        with repo.config_writer() as cw:
            for key, val in gconfigs:
                sec, prop = key.split('.')
                ok = False
                try:
                    cw.set_value(sec, prop, val)
                    ok = True
                finally:
                    if not ok:
                        self.log.error("Failed to write git-seeting '%s'=%s!",
                                       key, val)

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
                    force: bool=None) -> Text:
        """
        :param folder:
            The path to the folder to store the repo-archive in.
        :return:
            the path of the repo-archive
        """
        import tarfile

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

        return archive_fpath

    @fnt.lru_cache()  # x6(!) faster!
    def _infos_dsp(self, fallback_value='<invalid>'):
        from schedula import Dispatcher
        from schedula.utils.dsp import DFun

        dfuns = [
            DFun('repo', lambda _infos: self.repo),
            DFun('git_cmds', lambda _infos: pndlu.where('git')),
            DFun('dirty', lambda repo: repo.is_dirty()),
            DFun('untracked', lambda repo: repo.untracked_files),
            DFun('wd_files', lambda repo: os.listdir(repo.working_dir)),
            DFun('branch', lambda repo, _inp_prj:
                 _inp_prj and _get_ref(repo.heads, _pname2ref_path(_inp_prj)) or repo.active_branch),
            DFun('head', lambda repo: repo.head),
            DFun('heads_count', lambda repo: len(repo.heads)),
            DFun('projects_count', lambda repo: itz.count(self._yield_project_refs())),
            DFun('tags_count', lambda repo: len(repo.tags)),
            DFun('git.settings', lambda repo: self.read_git_settings()),

            DFun('git.version', lambda repo: '.'.join(str(v) for v in repo.git.version_info)),

            DFun('head_ref', lambda head: head.reference),
            DFun('head_valid', lambda head: head.is_valid()),
            DFun('head_detached', lambda head: head.is_detached),

            DFun('cmt', lambda branch: branch.commit),
            DFun('head', lambda branch: branch.path),
            DFun('branch_valid', lambda branch: branch.is_valid()),
            DFun('branch_detached', lambda branch: branch.is_detached),

            DFun('tre', lambda cmt: cmt.tree),
            DFun('author', lambda cmt: '%s <%s>' % (cmt.author.name, cmt.author.email)),
            DFun('last_cdate', lambda cmt: str(cmt.authored_datetime)),
            DFun('commit', lambda cmt: cmt.hexsha),
            DFun('revs_count', lambda cmt: itz.count(cmt.iter_parents())),
            DFun('cmsg', lambda cmt: cmt.message),
            DFun('cmsg', lambda cmt: '<invalid: %s>' % cmt.message, weight=10),

            DFun(['msg.%s' % f for f in _CommitMsg._fields],
                 lambda cmsg: _CommitMsg.parse_commit_msg(cmsg)),

            DFun('tree', lambda tre: tre.hexsha),
            DFun('blobs_count', lambda tre: itz.count(tre.list_traverse())),
        ]
        dsp = Dispatcher()
        DFun.add_dfuns(dfuns, dsp)
        return dsp

    @fnt.lru_cache()
    def _out_fields_by_verbose_level(self, level):
        """
        :param level:
            If ''> max-level'' then max-level assumed, negatives fetch no fields.
        """
        verbose_levels = {
            0: [
                'msg.project',
                'msg.state',
                'msg.action',
                'revs_count',
                'blobs_count',
                'last_cdate',
                'author',
            ],
            1: [
                'infos',
                'cmsg',
                'head',
                'dirty',
                'commit',
                'tree',
                'repo',
            ],
            2: None,  # null signifies "all fields".
        }
        max_level = max(verbose_levels.keys())
        if level > max_level:
            level = max_level
        fields = []
        for l in range(level + 1):
            fs = verbose_levels[l]
            if not fs:
                return None
            fields.extend(fs)
        return fields

    def _infos_fields(self, pname: Text=None, fields: Sequence[Text]=None, inv_value=None) -> List[Tuple[Text, Any]]:
        """Runs repo examination code returning all requested fields (even failed ones)."""
        from schedula import utils

        dsp = self._infos_dsp()
        inputs = {'_infos': 'ok', '_inp_prj': pname}
        infos = dsp.dispatch(inputs=inputs,
                             outputs=fields)
        fallbacks = {d: inv_value for d in dsp.data_nodes.keys()}
        fallbacks.update(infos)
        infos = fallbacks

        infos = dict(utils.stack_nested_keys(infos))
        infos = dtz.keymap(lambda k: '.'.join(k), infos)
        if fields:
            infos = [(f, infos.get(f, inv_value))
                     for f in fields]
        else:
            infos = sorted((f, infos.get(f, inv_value))
                           for f in dsp.data_nodes.keys())

        return infos

    def proj_examine(self, pname: Text=None, verbose=None, as_text=False):
        """
        Does not validate project, not fails, just reports situation.

        :param pname:
            Use current branch if unspecified; otherwise, DOES NOT checkout pname.
        :retun: text message with infos.
        """

        if verbose is None:
            verbose = self.verbose
        verbose_level = int(verbose)

        fields = self._out_fields_by_verbose_level(verbose_level)
        infos = self._infos_fields(pname, fields)

        if as_text:
            infos = yaml.dump(OrderedDict(infos), indent=2, default_flow_style=False)

        return infos

    def _conceive_new_project(self, pname):  # -> Project:
        """Returns a "UNBORN" :class:`Project`; its state must be triggered immediately."""
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
        prefname = _pname2ref_name(pname)
        if prefname not in self.repo.heads:
            raise CmdException('Project %r not found!' % pname)
        self.repo.heads[_pname2ref_name(pname)].checkout(force=self.force)

        self._current_project = None
        return self.current_project()

    def _yield_project_refs(self, *pnames: Text):
        if pnames:
            pnames = [_pname2ref_path(p) for p in pnames]
        for ref in self.repo.heads:
            if _is_project_ref(ref) and not pnames or ref.path in pnames:
                yield ref

    def proj_parse_stamped(self, mail_text: Text):
        from . import tstamp

        recv = tstamp.TstampReceiver(config=self.config)
        verdict = recv.parse_tstamp_response(mail_text)
        pname = verdict['report']['project']
        proj = self.proj_open(pname)
        proj.do_storedice(verdict)
        return proj.result

    def proj_list(self, *pnames: Text, verbose=None,
                  as_text=False, fields=None):
        """
        :param pnames:
            some project name, or none for all
        :param verbose:
            return infos based on :meth:`_out_fields_by_verbose_level()`
        :param fields:
            If defined, takes precendance over `verbose`.
        :param as_text:
            If true, return YAML, otherwise, strings or dicts if verbose
        :retun:
            yield any matched projects, or all if `pnames` were empty.
        """
        if verbose is None:
            verbose = self.verbose

        if fields:
            verbose = True  # Othrwise it would ignore fields.
        else:
            verbose_level = int(verbose) - 1  # V0 prints no infos.
            fields = self._out_fields_by_verbose_level(verbose_level)

        ap = self.repo.active_branch
        ap = ap and ap.path
        for ref in self._yield_project_refs(*pnames):
            pname = _ref2pname(ref)
            isactive = _pname2ref_path(pname) == ap

            if verbose:
                infos = self._infos_fields(pname, fields, inv_value='<invalid>')
                infos = OrderedDict(infos)
                infos['active'] = isactive
                to_yield = {pname: infos}
                if as_text:
                    to_yield = yaml.dump(to_yield, default_flow_style=False)
            else:
                if as_text:
                    to_yield = ('* %s' if isactive else '  %s') % pname
                else:
                    to_yield = pname

            yield to_yield


###################
##    Commands   ##
###################

class _PrjCmd(baseapp.Cmd):
    @property
    def projects_db(self):
        p = ProjectsDB.instance(config=self.config)
        p.config = self.config
        return p

    @property
    def current_project(self):
        return self.projects_db.current_project()


class ProjectCmd(_PrjCmd):
    """
    Commands to administer the storage repo of TA *projects*.

    A *project* stores all CO2MPAS files for a single vehicle,
    and tracks its sampling procedure.
    """

    examples = trt.Unicode("""
        To get the list with the status of all existing projects, try:
            %(cmd_chain)s list

        To see the current project, use one of those:
            %(cmd_chain)s current

        A typical workflow is this:
            %(cmd_chain)s init RL-12-BM3-2016-0000
            %(cmd_chain)s append inp=input.xlsx out=output.xlsx other=co2mpas.log
            %(cmd_chain)s report
            %(cmd_chain)s tstamp
            cat <mail-text> | %(cmd_chain)s dice

        You may enquiry the status of the project at any time :
            %(cmd_chain)s examine -v
        """)

    class ListCmd(_PrjCmd):
        """
        List specified projects, or all, if none specified.

        - Use --verbose to view more infos about the projects, or use the `examine` cmd
          to view even more details for a specific project.

        SYNTAX
            %(cmd_chain)s [OPTIONS] [<project-1>] ...
        """
        def run(self, *args):
            self.log.info('Listing %s projects...', args or 'all')
            return self.projects_db.proj_list(*args, as_text=True)

    class CurrentCmd(_PrjCmd):
        """Prints the currently open project."""
        def run(self, *args):
            if len(args) != 0:
                raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                                   % (self.name, len(args), args))

            return self.current_project

    class OpenCmd(_PrjCmd):
        """
        Make an existing project as *current*.

        SYNTAX
            %(cmd_chain)s [OPTIONS] <project>
        """
        def run(self, *args):
            self.log.info('Opening project %r...', args)
            if len(args) != 1:
                raise CmdException("Cmd %r takes a SINGLE project-name as argument, received: %r!"
                                   % (self.name, args))

            proj = self.projects_db.proj_open(args[0])

            return proj.result if self.verbose else proj

    class InitCmd(_PrjCmd):
        """
        Create a new project.

        SYNTAX
            %(cmd_chain)s [OPTIONS] <project>
        """
        def run(self, *args):
            if len(args) != 1:
                raise CmdException('Cmd %r takes a SINGLE project-name as argument, received %r!'
                                   % (self.name, args))

            return self.projects_db.proj_add(args[0])

    class AppendCmd(_PrjCmd):
        """
        Import the specified input/output co2mpas files into the *current project*.

        - One file from each kind (inp/out) may be given.
        - If an input/output is already present in the current project, use --force.

        SYNTAX
            %(cmd_chain)s [OPTIONS] ( inp=<co2mpas-file-1> | out=<co2mpas-file-1> ) ...
        """

        examples = trt.Unicode("""
            To import an INPUT co2mpas file, try:

                %(cmd_chain)s inp=co2mpas_input.xlsx

            To import both INPUT and OUTPUT files, and overwrite any already imported try:

                %(cmd_chain)s --force inp=co2mpas_input.xlsx out=co2mpas_results.xlsx
            """)

        def __init__(self, **kwds):
            kwds.setdefault('cmd_flags', {
                ('n', 'dry-run'): (
                    {
                        'Project': {'dry_run': True},
                    },
                    "Parse files but do not actually store them in the project."
                )
            })
            super().__init__(**kwds)

        def run(self, *args):
            self.log.info('Importing report files %s...', args)
            if len(args) < 1:
                raise CmdException('Cmd %r takes at least one argument, received %d: %r!'
                                   % (self.name, len(args), args))
            pfiles = PFiles.parse_io_args(*args)

            proj = self.current_project
            ok = proj.do_addfiles(pfiles=pfiles)

            return proj.result if self.verbose else ok

    class ReportCmd(_PrjCmd):
        """
        Prepares or re-prints the signed dice-report that can be sent for timestamping.

        - Use --force to generate a new report.
        - Use --dry-run to see its rough contents without signing and storing it.

        SYNTAX
            %(cmd_chain)s [OPTIONS]

        Eventually the *Dice Report* parameters will be time-stamped and disseminated to
        TA authorities & oversight bodies with an email, to receive back
        the sampling decision.

        """

        #examples = trt.Unicode(""" """)

        def __init__(self, **kwds):
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
            ok = proj.do_prepmail()

            return ok and proj.result or ok

    class TstampCmd(_PrjCmd):
        """
        Sends the prepared tag to be timestamped, or prints it for sending manually (--dry-run).

        SYNTAX
            %(cmd_chain)s [OPTIONS]

        - Use --dry-run if you want to send the email yourself.
          Remember to use the appropriate 'Subject'.
        - The --dry-run option prints the email as it would have been sent; you may
          copy-paste this lient and send it, formatted as 'plain-text' (not 'HTML').
        """

        #examples = trt.Unicode(""" """)

        def __init__(self, **kwds):
            kwds.setdefault('cmd_flags', {
                ('n', 'dry-run'): (
                    {
                        'Project': {'dry_run': True},
                    },
                    "Verify dice-report but not actually send tstamp email."
                )
            })
            super().__init__(**kwds)

        def run(self, *args):
            if len(args) > 0:
                raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                                   % (self.name, len(args), args))

            proj = self.current_project
            ok = proj.do_sendmail()

            return proj.result if self.verbose or proj.dry_run else ok

    class TparseCmd(_PrjCmd):
        """
        Derives *decision* OK/SAMPLE flag from tstamped-response, and store it (or compare with existing).

        SYNTAX
            %(cmd_chain)s [OPTIONS] [<tstamped-file-1> ...]

        - If '-' is given or no file at all, it reads from STDIN.
        - If --force is given to overcome any verification/parsing errors,
          then --project might be needed, to set the project the response belongs to .
        """

        #examples = trt.Unicode(""" """)

        project = trt.Unicode(
            help="Which project to store/compare the tstamp-response given"
        ).tag(config=True)

        def __init__(self, **kwds):
            from . import tstamp
            kwds.setdefault('conf_classes', [tstamp.TstampReceiver])
            kwds.setdefault('cmd_flags', {
                ('n', 'dry-run'): (
                    {
                        'Project': {'dry_run': True},
                    },
                    "Pase the tstamped response without storing it in the project."
                )
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
                with io.open(file, 'rt') as fin:
                    mail_text = fin.read()

            res = self.projects_db.proj_parse_stamped(mail_text)

            return res if self.verbose else ok

    class ExamineCmd(_PrjCmd):
        """
        Print various information about the specified project, or the current-project, if none specified.

        - Use --verbose to view more infos, including about the repository as a whole.

        SYNTAX
            %(cmd_chain)s [OPTIONS] [<project>]
        """
        def run(self, *args):
            if len(args) > 1:
                raise CmdException('Cmd %r takes one optional argument, received %d: %r!'
                                   % (self.name, len(args), args))
            pname = args and args[0] or None
            return self.projects_db.proj_examine(pname, as_text=True)

    class BackupCmd(_PrjCmd):
        """
        Backup projects repository into the archive filepath specified, or current-directory, if none specified.

        SYNTAX
            %(cmd_chain)s [OPTIONS] [<archive-path>]
        """
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
                return self.projects_db.repo_backup(**kwds)
            except FileNotFoundError as ex:
                raise baseapp.CmdException(
                    "Folder '%s' to store archive does not exist!"
                    "\n  Use --force to create it." % ex)

    def __init__(self, **kwds):
        dkwds = {
            'conf_classes': [ProjectsDB, Project],
            'subcommands': baseapp.build_sub_cmds(*all_subcmds),
            'cmd_flags': {
                'reset-git-settings': (
                    {

                        'ProjectsDB': {'reset_settings': True},
                    }, pndlu.first_line(ProjectsDB.reset_settings.help)
                )
            }
        }
        dkwds.update(kwds)
        super().__init__(**dkwds)

all_subcmds = (ProjectCmd.ListCmd, ProjectCmd.CurrentCmd, ProjectCmd.OpenCmd, ProjectCmd.InitCmd,
               ProjectCmd.AppendCmd, ProjectCmd.ReportCmd,
               ProjectCmd.TstampCmd,
               ProjectCmd.ExamineCmd, ProjectCmd.BackupCmd)
