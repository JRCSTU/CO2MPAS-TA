#!/usr/bin/env python
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
New base sub-cmd to simplify Stamping through WebStamper in one step.
"""
import textwrap as tw

from . import CmdException, base, cmdlets
from ._vendor import traitlets as trt


class DicerSpec(cmdlets.Spec, base.ShrinkingOutputMixin, base.ReportsKeeper):
    """A sequencer for dicing new or existing projects through WebStamper."""

    help_in_case_of_failure = trt.Unicode(
        tw.dedent("""\
            Dicing '%(vfid)s' will abort in state '%(state)s'.

              -> Intermediate Dices & Stamps can be found in your '%(reports_fpath)s' file.

              -> Use console commands to examine the situation and continue::

                     ## Examine the current project
                     co2dice project ls  [-v]  %(vfid)s

                     ## Add IO-files (if not added)
                     co2dice project append  %(iofiles)s

                     ## Generate Dice (or Decision)
                     co2dice project report

                 and then visit WebStamper with your browser to submit the Dice.

              -> Powerusers can use commands like this to web-stamp through the console:

                     co2dice project report  -W dice.txt    # generate dice if not done yet
                     cat dice.txt | co2dice tstamp wstamp  -W stamp.txt
                     co2dice project parse tparse  stamp.txt

                 or stamp without intermediate files::

                     co2dice project report | co2dice tstamp wstamp | co2dice project tparse
        """)).tag(config=True)

    @property
    def projects_db(self) -> 'project.ProjectsDB':
        from . import project
        p = project.ProjectsDB.instance(config=self.config)  # @UndefinedVariable
        p.update_config(self.config)  # Above is not enough, if already inited.

        return p

    _http_session = None

    def _check_ok(self, ok, project):
        if not ok:
            raise CmdException(
                "Bailing out (probably) due to forbidden state-transition from '%s'!"
                "\n  (look above in the logs)" % project.state)

    def _derrive_vfid(self, pfiles: 'PFiles') -> str:
        from . import report

        repspec = report.ReporterSpec(config=self.config)
        finfos = repspec.extract_dice_report(pfiles)
        for fpath, data in finfos.items():
            iokind = data['iokind']
            if iokind in ('inp', 'out'):
                vfid = data['report']['vehicle_family_id']
                self.log.info("Project '%s' derived from '%s' file: %s",
                              vfid, iokind, fpath)

                return vfid
        else:
            raise CmdException("Failed derriving project-id from: %s" % finfos)

    def do_dice_in_one_step(self, pfiles: 'PFiles',
                            observer=None,
                            http_session=None):
        '''
        Run all dice-steps in one run.

        :param observer:
            a callable like that::

                def progress_updated(msg=None: str, step=1, nsteps=None):
                    """
                    :param step:
                        >0: increment step, <=0: set absolute step, None: ignored
                    :param nsteps:
                        0 disables progressbar, negatives, set `indeterminate` mode, None ignored.
                    """

        :param http_session:
            remember to close it at some point
            if not used, a session is created and utilized within this method
            for the 2 calls (check & stamp)
        '''
        from . import project, tstamp
        import requests
        from .utils import joinstuff

        if observer:
            def notify(msg: str, step=1, max_step=None):
                observer(msg, step, max_step)
        else:
            notify = lambda *_, **__: None  # noqa: E731

        nsteps = 8

        notify("checking configurations and files...", max_step=nsteps)
        if not (pfiles.inp and pfiles.out):
            raise CmdException(
                "At least one INP & OUT file needed for single-step Dicing! "
                "\n  Received: %s!" % (pfiles, ))
        pfiles.check_files_exist("dicer")
        wstamper = tstamp.WstampSpec(config=self.config)
        wstamper.recipients  # trait-validations scream on access

        notify("extracting project-id from files...", max_step=nsteps)
        vfid = self._derrive_vfid(pfiles)

        if not http_session:
            http_session = self._http_session
            if not http_session:
                http_session = self._http_session = requests.Session()
        try:
            notify("checking WebStamper is live before modifying stuff....", max_step=nsteps)
            wstamper = tstamp.WstampSpec(config=self.config)
            wstamper.stamp_dice(None,
                                dry_run=True,
                                http_session=http_session)
            notify("preparing project...", max_step=nsteps)
            pdb = self.projects_db
            try:
                proj = pdb.proj_add(vfid)
            except project.ProjectExistError as ex:
                err = str(ex)[:-1]  # clip the last '!' of ex-text.
                self.log.info("%s, opening it." % err)
                proj = pdb.proj_open(vfid)

            ok = False
            try:
                notify("processing project files...", max_step=nsteps)
                if proj.state in ('wltp_iof', 'tagged'):
                    diffs = pdb.diff_wdir_pfiles(pfiles)
                    if diffs:
                        raise CmdException(
                            "Project %s files mismatched with new ones!%s" %
                            (proj.pname, joinstuff(diffs, '', '\n    %s')))
                    self.log.info("Project '%s' already contained files: %s", vfid, pfiles)
                else:
                    self._check_ok(proj.do_addfiles(pfiles=pfiles), proj)
                    self.log.info("Initiated '%s' with files: %s", vfid, pfiles)

                notify("creating or retrieving dice-report...", max_step=nsteps)
                self._check_ok(proj.do_report(), proj)
                dice = proj.result
                assert isinstance(dice, str)
                self.store_report(dice, 'dice for %s' % proj.pname)

                notify("stamping Dice through WebStamper...", max_step=nsteps)
                stamp = wstamper.stamp_dice(dice, http_session=http_session)
                self.log.info("Stamp was: \n%s", self.shrink_text(stamp))
                self.store_report(stamp, 'stamp for %s' % proj.pname)

                notify("storing Stamp in project, and creating & signing Decision-report...",
                       max_step=nsteps)
                self._check_ok(proj.do_storestamp(tstamp_txt=stamp), proj)
                decision = proj.result
                assert isinstance(decision, str), decision
                self.log.info("Imported Decision: \n%s'.",
                              self.shrink_text(stamp))
                ok = True

                self.store_report(decision, 'decision for %s' % proj.pname)

                return self.shrink_text(decision)
            finally:
                if not ok:
                    self.log.warning(self.help_in_case_of_failure % {
                        'vfid': vfid,
                        'state': proj.state,
                        'iofiles': pfiles.build_cmd_line(),
                        'reports_fpath': self.default_reports_fpath,
                    })
        finally:
            if self._http_session:
                self._http_session.close()
