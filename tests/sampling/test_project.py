#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp, crypto, dice, project, PFiles
from co2mpas.sampling.baseapp import pump_cmd, collect_cmd
from co2mpas.utils import chdir

import logging
import os
import shutil
import tempfile
from tests.sampling import (test_inp_fpath, test_out_fpath, test_vfid,
                            test_pgp_fingerprint, test_pgp_keys, test_pgp_trust)
import unittest

import ddt

import itertools as itt
import os.path as osp
import pandas as pd
from co2mpas._vendor.traitlets import config as trtc


mydir = osp.dirname(__file__)
init_logging(level=logging.DEBUG)
log = logging.getLogger(__name__)

proj1 = 'IP-12-WMI-1234-5678'
proj2 = 'IP-12-WMI-1111-2222'



@ddt.ddt
class TApp(unittest.TestCase):

    @ddt.data(*list(itt.product((
        dice.Co2diceCmd.document_config_options,
        dice.Co2diceCmd.print_alias_help,
        dice.Co2diceCmd.print_flag_help,
        dice.Co2diceCmd.print_options,
        dice.Co2diceCmd.print_subcommands,
        dice.Co2diceCmd.print_examples,
        dice.Co2diceCmd.print_help,
    ),
        project.all_subcmds))
    )
    def test_app(self, case):
        meth, cmd_cls = case
        c = trtc.get_config()
        c.Co2diceCmd.raise_config_file_errors = True
        cmd = cmd_cls(config=c)
        meth(cmd)


@ddt.ddt
class Tproject(unittest.TestCase):

    @ddt.data(
        (1, 'a', int),
        (1, 'a', int, True),
        (1, 'a', int, True, True),
        (1, 'a', int, False, True),

        (1, 'a', (str, int)),
        (1, 'a', (int, str)),
        (1, 'a', (int, str), True),
        (1, 'a', (int, str), True, True),
        (1, 'a', (int, str), False, True),

        ('s', 'b', str),
        ('s', 'b', str, True),
        ('s', 'b', str, True, True),
        ('s', 'b', str, False, True),

        ('s', 'b', (str, int)),
        ('s', 'b', (int, str)),
        ('s', 'b', (int, str), True),
        ('s', 'b', (int, str), True, True),
        ('s', 'b', (int, str), False, True),

        (None, 'c', int, True, False),
        (None, 'c', str, True, False),
        (None, 'c', (int, str), True, False),
        (None, 'c', (str, int), True, True),

        (None, 'd', int, True, True),
        (None, 'd', str, False, True),
        (None, 'd', (int, str), True, True),
        (None, 'd', (str, int), False, True),
    )
    def test_evarg(self, case):
        class E:
            def __init__(self, *args, **kwds):
                self.kwargs = dict(*args, **kwds)

        e = E(a=1, b='s', c=None)
        exp, *args = case
        self.assertEqual(project._evarg(e, *args), exp)

    @ddt.data(
        ('v1.1.1', None),
        ('v1.1.1', None),
        ('sdffasdfasda v 1.1.1', None),
        ('v-1.1.1', None),
        ('v -\n%1.1.1', None),
        ('v: \n1.1.01', ('1', '1', '01')),
        (' v: \n1.1.01', ('1', '1', '01')),
        ('sdffasdfasda v:\n 0.0.000\n', ('0', '0', '000')),
    )
    def test_cmt_version_regex(self, case):
        cmt_msg, exp_ver = case
        m = project._CommitMsgVer_regex.search(cmt_msg)
        if not exp_ver:
            self.assertIsNone(m, (m, cmt_msg, exp_ver))
        else:
            self.assertIsNotNone(m, (cmt_msg, exp_ver))


@ddt.ddt
class TProjectsDBStory(unittest.TestCase):
    ## INFO: Must run a whole, ordering of TCs matter.

    @classmethod
    def setUpClass(cls):
        cls._project_repo = tempfile.TemporaryDirectory()
        log.debug('Temp-repo: %s', cls._project_repo)

        cls.cfg = c = trtc.get_config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_keys
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.GpgSpec.allow_test_key = True
        c.DiceSpec.user_name = "Test Vase"
        c.DiceSpec.user_email = "test@vase.com"
        c.Project.force = True

        crypto.GpgSpec(config=c)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

    @classmethod
    def tearDownClass(cls):
        cls._project_repo.cleanup()
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)

    @property
    def _config(self):
        c = self.cfg.copy()
        c.ProjectsDB.repo_path = self._project_repo.name
        c.Spec.verbose = c.ProjectsDB.verbose = 0
        return c

    def _check_infos_shapes(self, proj):
        res = proj.repo_status(verbose=0)
        self.assertEqual(len(res), 12, res)

        res = proj.repo_status(verbose=1)
        self.assertEqual(len(res), 19, res)

        res = proj.repo_status(verbose=2)
        self.assertEqual(len(res), 39, res)

    def test_1a_empty_list(self):
        cfg = self._config
        exp_log_msg = r"No current-project exists yet!"

        cmd = project.LsCmd(config=cfg)
        with self.assertLogs(level=logging.WARNING) as log_rec:
            pump_cmd(cmd.run())
        self.assertIn(exp_log_msg, str(log_rec.output))
        self.assertIsNone(cmd.projects_db._current_project)

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)

        with self.assertLogs(level=logging.WARNING) as log_rec:
            pump_cmd(pdb.proj_list(verbose=1))
        self.assertIn(exp_log_msg, str(log_rec.output))
        self.assertIsNone(pdb._current_project)

        with self.assertLogs(level=logging.WARNING) as log_rec:
            pump_cmd(pdb.proj_list(verbose=2))
        self.assertIn(exp_log_msg, str(log_rec.output))
        self.assertIsNone(pdb._current_project)

    def test_1b_empty_infos(self):
        cfg = self._config
        cmd = project.StatusCmd(config=cfg)
        res = collect_cmd(cmd.run())
        self.assertIsNotNone(res)
        self.assertIsNone(cmd.projects_db._current_project)

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        self._check_infos_shapes(pdb)
        self.assertIsNone(pdb._current_project)

    def test_2a_add_project(self):
        cfg = self._config
        cmd = project.InitCmd(config=cfg)
        pname = proj1
        res = collect_cmd(cmd.run(pname))
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

        cmd = project.LsCmd(config=cfg)
        res = collect_cmd(cmd.run())
        self.assertEqual(str(res), '* %s: empty' % proj1)

    def test_2b_list(self):
        cfg = self._config
        cmd = project.LsCmd(config=cfg)
        res = collect_cmd(cmd.run())
        self.assertEqual(res, '* %s: empty' % proj1)

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)

        res = collect_cmd(pdb.proj_list(verbose=1))
        self.assertIsInstance(res, dict)
        self.assertIn(proj1, res)
        self.assertEqual(len(next(iter(res.values()))), 13, res)

        res = collect_cmd(pdb.proj_list(verbose=2))
        self.assertIsInstance(res, dict)
        self.assertIn(proj1, res)
        self.assertEqual(len(next(iter(res.values()))), 17, res)

    def test_2c_default_infos(self):
        cfg = self._config
        cmd = project.StatusCmd(config=cfg)
        res = collect_cmd(cmd.run())
        self.assertIn(proj1, res)

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        self._check_infos_shapes(pdb)

    def test_3a_add_same_project__fail(self):
        cfg = self._config
        cmd = project.InitCmd(config=cfg)
        with self.assertRaisesRegex(baseapp.CmdException,
                                    r"Project '%s' already exists!" % proj1):
            pump_cmd(cmd.run(proj1))

        cmd = project.LsCmd(config=cfg)
        res = list(cmd.run('.'))
        self.assertEqual(res, ['* %s: empty' % proj1])

    def test_4a_add_another_project(self):
        cfg = self._config
        pname = proj2
        cmd = project.InitCmd(config=cfg)
        res = collect_cmd(cmd.run(pname))
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

        cmd = project.LsCmd(config=cfg)
        res = collect_cmd(cmd.run('.'))
        self.assertEqual(res, '* %s: empty' % pname)

    def test_4b_list_projects(self):
        cfg = self._config
        cmd = project.LsCmd(config=cfg)
        res = collect_cmd(cmd.run('.'))
        self.assertEqual(res, '* %s: empty' % proj2)

        cmd = project.LsCmd(config=cfg)
        res = collect_cmd(cmd.run())
        self.assertSequenceEqual(res, [
            '* %s: empty' % proj2,
            '  %s: empty' % proj1])

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)

        res = collect_cmd(pdb.proj_list(verbose=1))
        for ires in res:
            self.assertIsInstance(ires, dict)
            self.assertTrue(proj1 in ires or proj2 in ires, ires)
            self.assertEqual(len(next(iter(ires.values()))), 13, ires)

        res = collect_cmd(pdb.proj_list(verbose=2))
        for ires in res:
            self.assertIsInstance(ires, dict)
            self.assertTrue(proj1 in ires or proj2 in ires, ires)
            self.assertEqual(len(next(iter(ires.values()))), 17, ires)

    def test_5_open_other(self):
        cfg = self._config
        pname = proj1

        cmd = project.OpenCmd(config=self._config)
        res = collect_cmd(cmd.run(pname))
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        res = collect_cmd(pdb.proj_list(pname))
        self.assertEqual(res, pname)

        cmd = project.LsCmd(config=cfg)
        res = collect_cmd(cmd.run('.'))
        self.assertEqual(str(res), '* %s: empty' % pname)

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        self._check_infos_shapes(pdb)

    def test_6_open_non_existing(self):
        cmd = project.OpenCmd(config=self._config)
        with self.assertRaisesRegex(baseapp.CmdException, "Project 'who' not found!"):
            cmd.run('who')


@ddt.ddt
class TStraightStory(unittest.TestCase):
    ## INFO: Must run a whole, ordering of TCs matter.

    @classmethod
    def setUpClass(cls):
        cls._project_repo = tempfile.TemporaryDirectory()
        log.debug('Temp-repo: %s', cls._project_repo)

        cls.cfg = c = trtc.get_config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_keys
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.GpgSpec.allow_test_key = True
        c.DiceSpec.user_name = "Test Vase"
        c.DiceSpec.user_email = "test@vase.com"
        c.TstampSender.tstamper_address = 'bar@foo.com'
        c.TstampSender.tstamp_recipients = ['foo@bar.com']

        crypto.GpgSpec(config=c)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

    @classmethod
    def tearDownClass(cls):
        project.ProjectsDB.clear_instance()
        cls._project_repo.cleanup()
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)

    @property
    def _config(self):
        c = self.cfg.copy()
        c.ProjectsDB.repo_path = self._project_repo.name
        c.Spec.verbose = c.ProjectsDB.verbose = 0
        return c

    def test_0_show_paths(self):
        from co2mpas.sampling import cfgcmd
        cmd = cfgcmd.PathsCmd(config=self._config)
        pump_cmd(cmd.run())

    def test_1_add_project(self):
        cmd = project.InitCmd(config=self._config)
        pname = test_vfid
        res = collect_cmd(cmd.run(pname))
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

    def test_2a_import_io(self):
        cfg = self._config
        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        p = pdb.current_project()

        cmd = project.AppendCmd(config=cfg,
                                inp=[test_inp_fpath],
                                out=[test_out_fpath])
        res = collect_cmd(cmd.run())
        self.assertTrue(res)

        p2 = pdb.current_project()
        self.assertIs(p, p2)

    def test_3_list_iofiles(self):
        cfg = self._config
        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        p = pdb.current_project()

        iof = p.list_pfiles()
        self.assertIsNotNone(iof)
        self.assertEqual(len(iof.inp), 1)
        self.assertEqual(len(iof.out), 1)
        self.assertFalse(iof.other)

    def test_4_tag(self):
        ## FIXME: Del tmp-repo denied with old pythingit.
        cfg = self._config
        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        p = pdb.current_project()

        res = p.do_report()
        self.assertTrue(res)
        self.assertEqual(p.state, 'tagged')

        p2 = pdb.current_project()
        self.assertIs(p, p2)

    def test_5_send_email(self):
        cfg = self._config

        persist_path = os.environ.get('TEST_TSTAMP_CONFIG_FPATH')
        cfg.Project.dry_run = not bool(persist_path)
        if persist_path:
            cfg.Cmd.persist_path = persist_path

        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        p = pdb.current_project()
        p.update_config(cfg)

        pretend = not bool(persist_path)
        res = p.do_sendmail(pretend=pretend)
        self.assertTrue(res)
        self.assertEqual(p.state, 'mailed')

        p2 = pdb.current_project()
        self.assertIs(p, p2)

        if pretend:
            raise unittest.SkipTest("No smtp-server credentials & tstamp config file "
                                    "found in 'TEST_TSTAMP_CONFIG_FPATH' env-var.")

    def test_6_receive_email(self):
        from . import test_tstamp

        cfg = self._config
        cfg.Project.dry_run = False  # modifed by prev TC.
        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        p = pdb.current_project()
        p.update_config(cfg)

        res = p.do_storedice(tstamp_txt=test_tstamp.tstamp_responses[-1][-1])
        self.assertTrue(res)
        self.assertEqual(p.state, 'nosample')

        p2 = pdb.current_project()
        self.assertIs(p, p2)

    def test_7_add_nedc_files(self):
        cfg = self._config
        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        p = pdb.current_project()
        p.update_config(cfg)

        pfiles = PFiles(other=[__file__])
        res = p.do_addfiles(pfiles=pfiles)
        self.assertTrue(res)
        self.assertEqual(p.state, 'nedc')

        p2 = pdb.current_project()
        self.assertIs(p, p2)


class TInitCmd(unittest.TestCase):

    def setUp(self):
        self._project_repo = tempfile.TemporaryDirectory()
        log.debug('Temp-repo: %s', self._project_repo)

    def tearDown(self):
        self._project_repo.cleanup()

    @property
    def _config(self):
        c = trtc.get_config()
        c.ProjectsDB.repo_path = self._project_repo.name
        c.Spec.verbose = c.ProjectsDB.verbose = 0
        c.GpgSpec.master_key = 'ali baba'
        c.DiceSpec.user_name = 'ali baba'
        c.DiceSpec.user_email = 'ali@baba.com'
        return c

    def get_cwp(self):
        cfg = self._config
        pdb = project.ProjectsDB.instance(config=cfg)
        pdb.update_config(cfg)
        p = pdb.current_project()

        return pdb, p

    def test_init_without_files(self):
        cmd = project.InitCmd(config=self._config)
        pump_cmd(cmd.run(test_vfid))

        pdb, p = self.get_cwp()

        self.assertEqual(len(list(pdb.repo.head.commit.tree.traverse())), 1, list(pdb.repo.head.commit.tree.traverse()))

        p.do_addfiles(pfiles=PFiles(inp=[test_inp_fpath], out=[test_out_fpath]))
        self.assertEqual(len(list(pdb.repo.head.commit.tree.traverse())), 5,
                         list(pdb.repo.head.commit.tree.traverse()))

    def test_init_with_files(self):
        cmd = project.InitCmd(config=self._config,
                              inp=[test_inp_fpath], out=[test_out_fpath])
        pump_cmd(cmd.run())

        pdb, _ = self.get_cwp()

        self.assertEqual(len(list(pdb.repo.head.commit.tree.traverse())), 5,
                         list(pdb.repo.head.commit.tree.traverse()))


class TBackupCmd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = c = trtc.get_config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_keys
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.GpgSpec.allow_test_key = True
        c.DiceSpec.user_name = "Test Vase"
        c.DiceSpec.user_email = "test@vase.com"

        crypto.GpgSpec(config=c)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)

    def setUp(self):
        self._project_repo = tempfile.TemporaryDirectory()
        log.debug('Temp-repo: %s', self._project_repo)

    def tearDown(self):
        self._project_repo.cleanup()

    @property
    def _config(self):
        c = self.cfg.copy()
        c.ProjectsDB.repo_path = self._project_repo.name
        c.Spec.verbose = c.ProjectsDB.verbose = 0
        return c

    def test_backup_cwd(self):
        cfg = self._config
        pump_cmd(project.InitCmd(config=cfg).run(proj1))
        cmd = project.BackupCmd(config=cfg)
        with tempfile.TemporaryDirectory() as td:
            with chdir(td):
                res = collect_cmd(cmd.run())
                self.assertIn(td, res)
                self.assertIn(os.getcwd(), res)
                self.assertTrue(osp.isfile(res), (res, os.listdir(osp.split(res)[0])))

    def test_backup_fullpath(self):
        cfg = self._config
        pump_cmd(project.InitCmd(config=cfg).run(proj1))
        cmd = project.BackupCmd(config=cfg)
        with tempfile.TemporaryDirectory() as td:
            archive_fpath = osp.join(td, 'foo')
            res = collect_cmd(cmd.run(archive_fpath))
            self.assertIn(td, res)
            self.assertIn('foo.txz', res)
            self.assertNotIn('co2mpas', res)
            self.assertTrue(osp.isfile(res), (res, os.listdir(osp.split(res)[0])))

    def test_backup_folder_only(self):
        cfg = self._config
        pump_cmd(project.InitCmd(config=cfg).run(proj2))
        cmd = project.BackupCmd(config=cfg)
        with tempfile.TemporaryDirectory() as td:
            archive_fpath = td + '\\'
            res = collect_cmd(cmd.run(archive_fpath))
            self.assertIn(archive_fpath, res)
            self.assertIn('co2mpas', res)
            self.assertTrue(osp.isfile(res), (res, os.listdir(osp.split(res)[0])))

    def test_backup_no_dir(self):
        cfg = self._config
        pump_cmd(project.InitCmd(config=cfg).run(proj1))
        cmd = project.BackupCmd(config=cfg)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(baseapp.CmdException,
                                        r"Folder '.+__BAD_FOLDER' to store archive does not exist!"):
                pump_cmd(cmd.run(osp.join(td, '__BAD_FOLDER', 'foo')))

    def test_export_import(self):
        import git
        import subprocess as sbp

        cfg = self._config
        pump_cmd(project.InitCmd(config=cfg,
                                 inp=[test_inp_fpath],
                                 out=[test_out_fpath],
                                 ).run())
        r = git.Repo(cfg.ProjectsDB.repo_path)
        r.create_tag('new_tag', message="just to be real tag")

        with tempfile.TemporaryDirectory() as td:
            archive_fpath = osp.join(td, 'proj.zip')

            pump_cmd(project.ExportCmd(config=cfg, out=archive_fpath,
                                       erase_afterwards=True).run())
            self.assertIsNone(collect_cmd(project.LsCmd(config=cfg).run()))

            file_list = sbp.check_output(['unzip', '-t', archive_fpath],
                                         universal_newlines=True)
            self.assertIn('refs/heads/projects/%s' % test_vfid, file_list)
            self.assertIn('refs/tags/new_tag', file_list)
            self.assertNotIn('refs/remotes/projects', file_list)

            pump_cmd(project.ImportCmd(config=cfg).run(archive_fpath))
            self.assertIn(test_vfid,
                          collect_cmd(project.LsCmd(config=cfg).run()))
