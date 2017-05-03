#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import CmdException, report, project, crypto
import logging
import numpy as np
import re
import yaml
import shutil
import tempfile
import types
import unittest

import ddt

import os.path as osp
import pandas as pd
from co2mpas._vendor.traitlets import config as trtc

from . import (test_inp_fpath, test_out_fpath, test_vfid,
               test_pgp_fingerprint, test_pgp_key, test_pgp_trust)


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


@ddt.ddt
class TApp(unittest.TestCase):

    @ddt.data(
        report.ReportCmd.document_config_options,
        report.ReportCmd.print_alias_help,
        report.ReportCmd.print_flag_help,
        report.ReportCmd.print_options,
        report.ReportCmd.print_subcommands,
        report.ReportCmd.print_examples,
        report.ReportCmd.print_help,
    )
    def test_app(self, meth):
        c = trtc.get_config()
        c.ReportCmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c)
        meth(cmd)


class TReportBase(unittest.TestCase):
    def check_report_tuple(self, k, vfid, fpath, iokind, dice_report=None):
        self.assertEqual(len(k), 3, k)
        self.assertTrue(k['file'].endswith(osp.basename(fpath)), k)
        self.assertEqual(k['iokind'], iokind, k)
        dr = k.get('report')
        if dice_report is True:
            self.assertIsInstance(dr, dict, k)
            self.assertEqual(dr['report_type'][0], 'dice_report', k)
            self.assertEqual(dr['vehicle_family_id'][0], vfid, k)
        elif dice_report is False:
            self.assertIn('report_type', dr)
        else:
            self.assertIsNone(dr, k)


@ddt.ddt
class TReportArgs(TReportBase):

    def test_extract_input(self):
        c = trtc.get_config()
        c.ReportCmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c)
        res = cmd.run('inp=%s' % test_inp_fpath)
        self.assertIsInstance(res, types.GeneratorType)
        res = list(res)
        self.assertEqual(len(res), 1)
        rpt = yaml.load('\n'.join(res))
        f, rec = next(iter(rpt.items()))
        self.assertTrue(f.endswith("tests\sampling\input.xlsx"), rpt)
        self.check_report_tuple(rec, test_vfid, test_inp_fpath, 'inp', False)

    def test_extract_output(self):
        c = trtc.get_config()
        c.ReportCmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c)
        res = cmd.run('out=%s' % test_out_fpath)
        self.assertIsInstance(res, types.GeneratorType)
        res = list(res)
        self.assertEqual(len(res), 1)
        rpt = yaml.load('\n'.join(res))
        f, rec = next(iter(rpt.items()))
        self.assertTrue(f.endswith("tests\sampling\output.xlsx"), rpt)
        self.check_report_tuple(rec, test_vfid, test_out_fpath, 'out', True)

    def test_extract_both(self):
        c = trtc.get_config()
        c.ReportCmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c)
        res = cmd.run('inp=%s' % test_inp_fpath, 'out=%s' % test_out_fpath)
        self.assertIsInstance(res, types.GeneratorType)
        res = list(res)
        self.assertEqual(len(res), 2)
        rpt = yaml.load('\n'.join(res))
        for f, rec in rpt.items():
            if f.endswith('input.xlsx'):
                path, iokind, exp_rpt = "tests\sampling\input.xlsx", 'inp', False
            elif f.endswith('output.xlsx'):
                path, iokind, exp_rpt = "tests\sampling\output.xlsx", 'out', True
            self.assertTrue(f.endswith(path), rpt)
            self.check_report_tuple(rec, test_vfid, path, iokind, exp_rpt)


class TReportProject(TReportBase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = c = trtc.get_config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_key
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.ReportCmd.raise_config_file_errors = True
        c.ReportCmd.project = True
        c.DiceSpec.user_name = "Test Vase"
        c.DiceSpec.user_email = "test@vase.com"

        crypto.GpgSpec(config=c)

        ## Clean memories from past tests
        #
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)

    def test_fails_with_args(self):
        c = self.cfg
        with self.assertRaisesRegex(CmdException, "--project' takes no arguments, received"):
            list(report.ReportCmd(config=c).run('EXTRA_ARG'))

    def test_fails_when_no_project(self):
        c = self.cfg
        with tempfile.TemporaryDirectory() as td:
            c.ProjectsDB.repo_path = td
            cmd = report.ReportCmd(config=c)
            with self.assertRaisesRegex(CmdException, r"No current-project exists yet!"):
                list(cmd.run())

    def test_fails_when_empty(self):
        c = self.cfg
        with tempfile.TemporaryDirectory() as td:
            c.ProjectsDB.repo_path = td
            project.ProjectCmd.InitCmd(config=c).run('proj1')
            cmd = report.ReportCmd(config=c)
            with self.assertRaisesRegex(
                CmdException, re.escape(
                    r"Current Project(proj1: empty) contains no input/output files!")):
                list(cmd.run())

    def test_input_output(self):
        c = self.cfg
        with tempfile.TemporaryDirectory() as td:
            c.ProjectsDB.repo_path = td
            project.ProjectCmd.InitCmd(config=c).run('proj1')

            project.ProjectCmd.AppendCmd(config=c).run('inp=%s' % test_inp_fpath)
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 1)
            rpt = yaml.load('\n'.join(res))
            f, rec = next(iter(rpt.items()))
            self.assertTrue(f.endswith("tests\sampling\input.xlsx"), rpt)
            self.check_report_tuple(rec, test_vfid, test_inp_fpath, 'inp')

            project.ProjectCmd.AppendCmd(config=c).run('out=%s' % test_out_fpath)
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 2)
            rpt = yaml.load('\n'.join(res))
            for f, rec in rpt.items():
                if f.endswith('input.xlsx'):
                    path, iokind, rpt = "tests\sampling\input.xlsx", 'inp', None
                elif f.endswith('output.xlsx'):
                    path, iokind, rpt = "tests\sampling\output.xlsx", 'out', True
                self.assertTrue(f.endswith(path), rpt)
                self.check_report_tuple(rec, test_vfid, path, iokind, rpt)

    def test_output_input(self):
        c = self.cfg
        with tempfile.TemporaryDirectory() as td:
            c.ProjectsDB.repo_path = td
            project.ProjectCmd.InitCmd(config=c).run('proj1')

            project.ProjectCmd.AppendCmd(config=c).run('out=%s' % test_out_fpath)
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 1)
            rpt = yaml.load('\n'.join(res))
            f, rec = next(iter(rpt.items()))
            self.assertTrue(f.endswith("tests\sampling\output.xlsx"), rpt)
            self.check_report_tuple(rec, test_vfid, test_out_fpath, 'out', True)

            project.ProjectCmd.AppendCmd(config=c).run('inp=%s' % test_inp_fpath)
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 2)
            rpt = yaml.load('\n'.join(res))
            for f, rec in rpt.items():
                if f.endswith('input.xlsx'):
                    path, iokind, rpt = "tests\sampling\input.xlsx", 'inp', None
                elif f.endswith('output.xlsx'):
                    path, iokind, rpt = "tests\sampling\output.xlsx", 'out', True
                self.assertTrue(f.endswith(path), rpt)
                self.check_report_tuple(rec, test_vfid, path, iokind, rpt)

    def test_both(self):
        c = self.cfg
        with tempfile.TemporaryDirectory() as td:
            c.ProjectsDB.repo_path = td
            project.ProjectCmd.InitCmd(config=c).run('proj1')

            cmd = project.ProjectCmd.AppendCmd(config=c)
            cmd.run('out=%s' % test_out_fpath, 'inp=%s' % test_inp_fpath)
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 2)
            rpt = yaml.load('\n'.join(res))
            for f, rec in rpt.items():
                if f.endswith('input.xlsx'):
                    path, iokind, rpt = "tests\sampling\input.xlsx", 'inp', None
                elif f.endswith('output.xlsx'):
                    path, iokind, rpt = "tests\sampling\output.xlsx", 'out', True
                self.assertTrue(f.endswith(path), rpt)
                self.check_report_tuple(rec, test_vfid, path, iokind, rpt)
