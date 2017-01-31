#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp, crypto, dice, project, PFiles
import logging
import os
import shutil
import tempfile
from tests._tutils import chdir
from tests.sampling import (test_inp_fpath, test_out_fpath,
                            test_pgp_fingerprint, test_pgp_key, test_pgp_trust)
from traitlets.config import get_config
import unittest

import ddt

import itertools as itt
import os.path as osp
import pandas as pd
import traitlets.config as trtc


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


@ddt.ddt
class TApp(unittest.TestCase):

    @ddt.data(*list(itt.product((
        dice.Co2dice.document_config_options,
        dice.Co2dice.print_alias_help,
        dice.Co2dice.print_flag_help,
        dice.Co2dice.print_options,
        dice.Co2dice.print_subcommands,
        dice.Co2dice.print_examples,
        dice.Co2dice.print_help,
    ),
        project.all_subcmds))
    )
    def test_app(self, case):
        meth, cmd_cls = case
        c = get_config()
        c.Co2dice.raise_config_file_errors = True
        cmd = cmd_cls(config=c)
        meth(cmd)


@ddt.ddt
class TProjectsDBStory(unittest.TestCase):
    ## INFO: Must run a whole, ordering of TCs matter.

    @classmethod
    def setUpClass(cls):
        cls._project_repo = tempfile.TemporaryDirectory()
        log.debug('Temp-repo: %s', cls._project_repo)

        cls.cfg = cfg = trtc.get_config()

        cfg.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.GpgSpec.keys_to_import = test_pgp_key
        cfg.GpgSpec.trust_to_import = test_pgp_trust
        cfg.GpgSpec.master_key = test_pgp_fingerprint
        cfg.DiceSpec.user_name = "Test Vase"
        cfg.DiceSpec.user_email = "test@vase.com"

        crypto.GpgSpec(config=cfg)

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

    def _check_infos_shapes(self, proj, pname=None):
        res = proj.proj_examine(pname=pname, verbose=0)
        self.assertEqual(len(res), 7, res)

        res = proj.proj_examine(pname=pname, verbose=1)
        self.assertEqual(len(res), 14, res)

        res = proj.proj_examine(pname=pname, verbose=2)
        self.assertEqual(len(res), 32, res)

    def test_1a_empty_list(self):
        cmd = project.ProjectCmd.ListCmd(config=self._config)
        res = cmd.run()
        self.assertIsNone(res)
        self.assertIsNone(cmd.projects_db._current_project)

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)

        res = pdb.proj_list(verbose=1)
        self.assertIsNone(res)
        self.assertIsNone(pdb._current_project)

        res = pdb.proj_list(verbose=2)
        self.assertIsNone(res)
        self.assertIsNone(pdb._current_project)

    def test_1b_empty_infos(self):
        cmd = project.ProjectCmd.ExamineCmd(config=self._config)
        res = cmd.run()
        self.assertIsNotNone(res)
        self.assertIsNone(cmd.projects_db._current_project)

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        self._check_infos_shapes(pdb)
        self.assertIsNone(pdb._current_project)

    def test_1c_empty_cwp(self):
        cmd = project.ProjectCmd.CurrentCmd(config=self._config)
        with self.assertRaisesRegex(baseapp.CmdException, r"No current-project exists yet!"):
            cmd.run()
        self.assertIsNone(cmd.projects_db._current_project)

    def test_2a_add_project(self):
        cmd = project.ProjectCmd.InitCmd(config=self._config)
        pname = 'foo'
        res = cmd.run(pname)
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

        cmd = project.ProjectCmd.CurrentCmd(config=self._config)
        res = cmd.run()
        self.assertEqual(str(res), 'Project(foo: empty)')

    def test_2b_list(self):
        cmd = project.ProjectCmd.ListCmd(config=self._config)
        res = cmd.run()
        self.assertEqual(res, ['* foo'])

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)

        res = pdb.proj_list(verbose=1)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(res.shape, (1, 7), res)
        self.assertIn('* foo', str(res))

        res = pdb.proj_list(verbose=2)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(res.shape, (1, 7), res)
        self.assertIn('* foo', str(res))

    def test_2c_default_infos(self):
        cmd = project.ProjectCmd.ExamineCmd(config=self._config)
        res = cmd.run()
        self.assertRegex(res, 'msg.project += foo')

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        self._check_infos_shapes(pdb)

    def test_3a_add_same_project__fail(self):
        cmd = project.ProjectCmd.InitCmd(config=self._config)
        with self.assertRaisesRegex(baseapp.CmdException, r"Project 'foo' already exists!"):
            cmd.run('foo')

        cmd = project.ProjectCmd.ListCmd(config=self._config)
        res = list(cmd.run())
        self.assertEqual(res, ['* foo'])

    @ddt.data('sp ace', '%fg', '1ffg&', 'kung@fu')
    def test_3b_add_bad_project__fail(self, pname):
        cmd = project.ProjectCmd.InitCmd(config=self._config)
        with self.assertRaisesRegex(baseapp.CmdException, "Invalid name '%s' for a project!" % pname):
            cmd.run(pname)

        cmd = project.ProjectCmd.ListCmd(config=self._config)
        res = list(cmd.run())
        self.assertEqual(res, ['* foo'])

    def test_4a_add_another_project(self):
        pname = 'bar'
        cmd = project.ProjectCmd.InitCmd(config=self._config)
        res = cmd.run(pname)
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

        cmd = project.ProjectCmd.CurrentCmd(config=self._config)
        res = cmd.run()
        self.assertEqual(str(res), 'Project(%s: empty)' % pname)

    def test_4b_list_projects(self):
        cmd = project.ProjectCmd.ListCmd(config=self._config)
        res = cmd.run()
        self.assertSequenceEqual(res, ['* bar', '  foo'])

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)

        res = pdb.proj_list(verbose=1)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(res.shape, (2, 7), res)
        self.assertIn('* bar', str(res))

        res = pdb.proj_list(verbose=2)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(res.shape, (2, 7), res)
        self.assertIn('* bar', str(res))

    def test_4c_default_infos(self):
        cmd = project.ProjectCmd.ExamineCmd(config=self._config)
        res = cmd.run()
        self.assertRegex(res, 'msg.project += bar')

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        self._check_infos_shapes(pdb)

    def test_4d_forced_infos(self):
        cmd = project.ProjectCmd.ExamineCmd(config=self._config)
        res = cmd.run('foo')
        self.assertRegex(res, 'msg.project += bar')

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        self._check_infos_shapes(pdb, 'foo')

    def test_5_open_other(self):
        pname = 'foo'
        cmd = project.ProjectCmd.OpenCmd(config=self._config)
        res = cmd.run(pname)
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

        cmd = project.ProjectCmd.CurrentCmd(config=self._config)
        res = cmd.run()
        self.assertEqual(str(res), 'Project(%s: empty)' % pname)

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        self._check_infos_shapes(pdb, pname)

    def test_6_open_non_existing(self):
        cmd = project.ProjectCmd.OpenCmd(config=self._config)
        with self.assertRaisesRegex(baseapp.CmdException, "Project 'who' not found!"):
            cmd.run('who')


@ddt.ddt
class TStraightStory(unittest.TestCase):
    ## INFO: Must run a whole, ordering of TCs matter.

    @classmethod
    def setUpClass(cls):
        cls._project_repo = tempfile.TemporaryDirectory()
        log.debug('Temp-repo: %s', cls._project_repo)

        cls.cfg = cfg = trtc.get_config()

        cfg.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.GpgSpec.keys_to_import = test_pgp_key
        cfg.GpgSpec.trust_to_import = test_pgp_trust
        cfg.GpgSpec.master_key = test_pgp_fingerprint
        cfg.DiceSpec.user_name = "Test Vase"
        cfg.DiceSpec.user_email = "test@vase.com"

        crypto.GpgSpec(config=cfg)

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

    def _check_infos_shapes(self, proj, pname=None):
        res = proj.proj_examine(pname=pname, verbose=0)
        self.assertEqual(len(res), 7, res)

        res = proj.proj_examine(pname=pname, verbose=1)
        self.assertEqual(len(res), 14, res)

        res = proj.proj_examine(pname=pname, verbose=2)
        self.assertEqual(len(res), 32, res)

    def test_1_add_project(self):
        cmd = project.ProjectCmd.InitCmd(config=self._config)
        pname = 'foo'
        res = cmd.run(pname)
        self.assertIsInstance(res, project.Project)
        self.assertEqual(res.pname, pname)
        self.assertEqual(res.state, 'empty')

    def test_2a_import_io(self):
        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        p = pdb.current_project()

        cmd = project.ProjectCmd.AddFileCmd(config=self._config)
        res = cmd.run('inp=%s' % test_inp_fpath, 'out=%s' % test_out_fpath)
        self.assertTrue(res)

        p2 = pdb.current_project()
        self.assertIs(p, p2)

    def test_3_list_iofiles(self):
        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        p = pdb.current_project()

        iof = p.list_pfiles()
        self.assertIsNotNone(iof)
        self.assertEqual(len(iof.inp), 1)
        self.assertEqual(len(iof.out), 1)
        self.assertFalse(iof.other)

    def test_4_tag(self):
        ## FIXME: Del tmp-repo denied with old pythingit.
        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        p = pdb.current_project()

        res = p.do_tagreport()
        self.assertTrue(res)
        self.assertEqual(p.state, 'tagged')

        p2 = pdb.current_project()
        self.assertIs(p, p2)

    def test_5_send_email(self):
        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        p = pdb.current_project()

        res = p.do_sendmail()
        self.assertTrue(res)
        self.assertEqual(p.state, 'mailed')

        p2 = pdb.current_project()
        self.assertIs(p, p2)

    def test_6_receive_email(self):
        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        p = pdb.current_project()

        res = p.do_mailrecv(mail='Hi there')
        self.assertIsInstance(res, bool)
        self.assertIn(p.state, ('dice_yes', 'dice_no'))

        p2 = pdb.current_project()
        self.assertIs(p, p2)

    def test_7_add_nedc_files(self):
        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        p = pdb.current_project()

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
        c = get_config()
        c.ProjectsDB.repo_path = self._project_repo.name
        c.Spec.verbose = c.ProjectsDB.verbose = 0
        return c

    def make_new_project(self, proj):
        cmd = project.ProjectCmd.InitCmd(config=self._config)
        cmd.run(proj)

        pdb = project.ProjectsDB.instance(config=self._config)
        pdb.update_config(self._config)
        p = pdb.current_project()
        self.assertEqual(len(list(pdb.repo.head.commit.tree.traverse())), 1, list(pdb.repo.head.commit.tree.traverse()))

        p.do_addfiles(pfiles=PFiles(inp=[test_inp_fpath], out=[test_out_fpath]))
        self.assertEqual(len(list(pdb.repo.head.commit.tree.traverse())), 5, list(pdb.repo.head.commit.tree.traverse()))

    def test_init_does_in_new_project(self):
        self.make_new_project('foobar')
        self.make_new_project('barfoo')


class TBackupCmd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = trtc.get_config()

        cfg.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.GpgSpec.keys_to_import = test_pgp_key
        cfg.GpgSpec.trust_to_import = test_pgp_trust
        cfg.GpgSpec.master_key = test_pgp_fingerprint
        cfg.DiceSpec.user_name = "Test Vase"
        cfg.DiceSpec.user_email = "test@vase.com"

        crypto.GpgSpec(config=cfg)

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
        project.ProjectCmd.InitCmd(config=self._config).run('foobar')
        cmd = project.ProjectCmd.BackupCmd(config=self._config)
        with tempfile.TemporaryDirectory() as td:
            with chdir(td):
                res = cmd.run()
                self.assertIn(td, res)
                self.assertIn(os.getcwd(), res)
                self.assertTrue(osp.isfile(res), (res, os.listdir(osp.split(res)[0])))

    def test_backup_fullpath(self):
        project.ProjectCmd.InitCmd(config=self._config).run('foobar')
        cmd = project.ProjectCmd.BackupCmd(config=self._config)
        with tempfile.TemporaryDirectory() as td:
            archive_fpath = osp.join(td, 'foo')
            res = cmd.run(archive_fpath)
            self.assertIn(td, res)
            self.assertIn('foo.txz', res)
            self.assertNotIn('co2mpas', res)
            self.assertTrue(osp.isfile(res), (res, os.listdir(osp.split(res)[0])))

    def test_backup_folder_only(self):
        project.ProjectCmd.InitCmd(config=self._config).run('barfoo')
        cmd = project.ProjectCmd.BackupCmd(config=self._config)
        with tempfile.TemporaryDirectory() as td:
            archive_fpath = td + '\\'
            res = cmd.run(archive_fpath)
            self.assertIn(archive_fpath, res)
            self.assertIn('co2mpas', res)
            self.assertTrue(osp.isfile(res), (res, os.listdir(osp.split(res)[0])))

    def test_backup_no_dir(self):
        project.ProjectCmd.InitCmd(config=self._config).run('foobar')
        cmd = project.ProjectCmd.BackupCmd(config=self._config)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(baseapp.CmdException,
                                        r"Folder '.+__BAD_FOLDER' to store archive does not exist!"):
                cmd.run(osp.join(td, '__BAD_FOLDER', 'foo'))
