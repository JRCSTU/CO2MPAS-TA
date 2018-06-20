#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas._vendor.traitlets import config as trtc
from co2mpas.sampling import CmdException, report, project, crypto
from co2mpas.sampling.baseapp import pump_cmd
import logging
import textwrap as tw
import os
import re
import shutil
import tempfile
import types
import unittest

import ddt
import yaml

import os.path as osp
import subprocess as sbp

from . import (test_inp_fpath, test_out_fpath, test_vfid,
               test_pgp_fingerprint, test_pgp_keys, test_pgp_trust)
import pytest


class FailingTempDir(tempfile.TemporaryDirectory):
    def __exit__(self, *args, **kwds):
        ## The process cannot access the file on windows
        try:
            super().__exit__(*args, **kwds)
        except Exception as ex:
            print('Ignored tempdir-failure: %s' % ex)


mydir = osp.dirname(__file__)
init_logging(level=logging.DEBUG)
log = logging.getLogger(__name__)

proj1 = 'IP-12-WMI-1234-5678'
proj2 = 'RL-99-BM3-2017-0001'


@ddt.ddt
class TApp(unittest.TestCase):

    @ddt.data(
        trtc.Application.document_config_options,
        trtc.Application.print_alias_help,
        trtc.Application.print_flag_help,
        trtc.Application.print_options,
        trtc.Application.print_subcommands,
        trtc.Application.print_examples,
        trtc.Application.print_help,
    )
    def test_app(self, meth):
        c = trtc.Config()
        c.Cmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c)
        meth(cmd)


@pytest.fixture
def dreport_df():
    """
    YAML file generated with::

        import io, json, pandas as pd, yaml
        from co2mpas.sampling.report import Report

        fpath = 'tests/sampling/output.xlsx'
        _vfid, dreport: pd.DataFrame = Report()._extract_dice_report_from_output(fpath)
        sink = io.StringIO()
        dreport.to_json(sink, 'columns')
        print(yaml.dump(json.loads(sink.getvalue()), default_flow_style=False))
    """
    import yaml
    import pandas as pd

    ytext = tw.dedent("""
        vehicle-H:
          CO2MPAS_deviation: -4.14
          CO2MPAS_version: 1.5.0.dev1
          Model_scores: vehicle-H
          TA_mode: true
          Vehicle: vehicle-H
          alternator_model: 4.56
          at_model: -0.95
          clutch_torque_converter_model: 4.71
          co2_params: 0
          datetime: 2017/01/29-23:42:41
          engine_capacity: 997
          engine_cold_start_speed_model: 18.74
          engine_coolant_temperature_model: 0.59
          engine_is_turbo: true
          engine_speed_model: 0.02
          fuel_type: diesel
          gear_box_type: automatic
          report_type: dice_report
          start_stop_model: -0.99
          vehicle_family_id: RL-99-BM3-2017-0001
        vehicle-L:
          CO2MPAS_deviation: null
          CO2MPAS_version: null
          Model_scores: vehicle-L
          TA_mode: null
          Vehicle: vehicle-L
          alternator_model: null
          at_model: null
          clutch_torque_converter_model: null
          co2_params: null
          datetime: null
          engine_capacity: 997
          engine_cold_start_speed_model: null
          engine_coolant_temperature_model: null
          engine_is_turbo: true
          engine_speed_model: 91.36
          fuel_type: diesel
          gear_box_type: automatic
          report_type: null
          start_stop_model: null
          vehicle_family_id: null
    """)
    return pd.DataFrame.from_dict(yaml.load(ytext), orient='columns')


def test_validate_deviations(dreport_df):
    msg = report.Report()._check_deviations_are_valid('<some path>', dreport_df)
    assert not msg

    dreport_df.loc['CO2MPAS_deviation', 'vehicle-L'] = 1.45
    msg = report.Report()._check_deviations_are_valid('<some path>', dreport_df)
    assert not msg

    dreport_df.loc['CO2MPAS_deviation', 'vehicle-H'] = None
    msg = report.Report()._check_deviations_are_valid('<some path>', dreport_df)
    assert not msg

    dreport_df.loc['CO2MPAS_deviation', 'vehicle-L'] = None
    msg = report.Report()._check_deviations_are_valid('<some path>', dreport_df)
    assert msg and "invalid deviations" in msg


class TReportBase(unittest.TestCase):
    def check_report_tuple(self, k, vfid, fpath, iokind, dice_report=None):
        self.assertEqual(len(k), 3, k)
        self.assertTrue(k['file'].endswith(osp.basename(fpath)), k)
        self.assertEqual(k['iokind'], iokind, k)
        dr = k.get('report')
        if dice_report is True:
            self.assertIsInstance(dr, dict, k)

            self.assertEqual(dr['0.vehicle_family_id'][0], vfid, k)
        elif dice_report is False:
            self.assertEqual(dr['vehicle_family_id'], vfid, k)
        else:
            self.assertIsNone(dr, k)


@ddt.ddt
class TReportArgs(TReportBase):

    @classmethod
    def setUpClass(cls):
        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()
        crypto.EncrypterSpec.clear_instance()

    def test_extract_input(self):
        c = trtc.Config()
        c.ReportCmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c, inp=[test_inp_fpath])
        res = cmd.run()
        self.assertIsInstance(res, types.GeneratorType)
        res = list(res)
        self.assertEqual(len(res), 1)
        rpt = yaml.load('\n'.join(res))
        f, rec = next(iter(rpt.items()))
        self.assertTrue(f.endswith("input.xlsx"), rpt)
        self.check_report_tuple(rec, test_vfid, test_inp_fpath, 'inp', False)

    def test_extract_output(self):
        c = trtc.Config()
        c.ReportCmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c, out=[test_out_fpath])
        res = cmd.run()
        self.assertIsInstance(res, types.GeneratorType)
        res = list(res)
        self.assertEqual(len(res), 1)
        rpt = yaml.load('\n'.join(res))
        f, rec = next(iter(rpt.items()))
        self.assertTrue(f.endswith("output.xlsx"), rpt)
        self.check_report_tuple(rec, test_vfid, test_out_fpath, 'out', True)

    def test_extract_both(self):
        c = trtc.Config()
        c.ReportCmd.raise_config_file_errors = True
        cmd = report.ReportCmd(config=c, inp=[test_inp_fpath], out=[test_out_fpath])
        res = cmd.run()
        self.assertIsInstance(res, types.GeneratorType)
        res = list(res)
        self.assertEqual(len(res), 2)
        rpt = yaml.load('\n'.join(res))
        for f, rec in rpt.items():
            if f.endswith('input.xlsx'):
                path, iokind, exp_rpt = "input.xlsx", 'inp', False
            elif f.endswith('output.xlsx'):
                path, iokind, exp_rpt = "output.xlsx", 'out', True
            self.assertTrue(f.endswith(path), rpt)
            self.check_report_tuple(rec, test_vfid, path, iokind, exp_rpt)

    def test_extract_both_input_in_dice(self):
        c = trtc.Config()
        c.ReportCmd.raise_config_file_errors = True
        c.Report.include_input_in_dice_override = True

        c.EncrypterSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.EncrypterSpec.keys_to_import = test_pgp_keys
        c.EncrypterSpec.trust_to_import = test_pgp_trust
        c.EncrypterSpec.master_key = test_pgp_fingerprint
        c.EncrypterSpec.allow_test_key = True

        cmd = report.ReportCmd(config=c, inp=[test_inp_fpath], out=[test_out_fpath])
        res = cmd.run()
        self.assertIsInstance(res, types.GeneratorType)
        res = list(res)
        self.assertEqual(len(res), 3)

        dreport = '\n'.join(res)
        data = report.Report(config=c).unlock_report(dreport)
        assert isinstance(data, dict)
        data = next(iter(data.values()))
        assert data['flag']['vehicle_family_id'] == 'RL-99-BM3-2017-0001'


class TReportProject(TReportBase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = c = trtc.Config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_keys
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.GpgSpec.allow_test_key = True
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

    def test_0_show_paths(self):
        from co2mpas.sampling import cfgcmd
        cmd = cfgcmd.PathsCmd(config=self.cfg)
        pump_cmd(cmd.run())

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
                pump_cmd(cmd.run())

    def test_fails_when_empty(self):
        c = self.cfg
        ## "The process cannot access the file" when deleting on Windows.
        with FailingTempDir() as td:
            c.ProjectsDB.repo_path = td
            pump_cmd(project.InitCmd(config=c).run(proj1))
            cmd = report.ReportCmd(config=c)
            with self.assertRaisesRegex(
                CmdException, re.escape(
                    r"Current Project(%s: empty) contains no input/output files!"
                    % proj1)):
                pump_cmd(cmd.run())

    def test_input_output(self):
        c = self.cfg
        ## "The process cannot access the file" when deleting on Windows.
        with FailingTempDir() as td:
            c.ProjectsDB.repo_path = td
            pump_cmd(project.InitCmd(config=c).run(test_vfid))

            pump_cmd(project.AppendCmd(config=c,
                                       inp=[test_inp_fpath]).run())
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 1)
            rpt = yaml.load('\n'.join(res))
            f, rec = next(iter(rpt.items()))
            self.assertNotIn('output.xlsx', rpt)
            self.assertIn("input.xlsx", rpt)
            self.check_report_tuple(rec, test_vfid, test_inp_fpath, 'inp', False)

            pump_cmd(project.AppendCmd(config=c, out=[test_out_fpath]).run())
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 2)
            rpt = yaml.load('\n'.join(res))
            self.assertIn('output.xlsx', rpt)
            self.assertIn('input.xlsx', rpt)
            for f, rec in rpt.items():
                if f.endswith('input.xlsx'):
                    path, iokind, rpt = "input.xlsx", 'inp', False
                elif f.endswith('output.xlsx'):
                    path, iokind, rpt = "output.xlsx", 'out', True
                self.assertTrue(f.endswith(path), rpt)
                self.check_report_tuple(rec, test_vfid, path, iokind, rpt)

    def test_output_input(self):
        c = self.cfg
        ## "The process cannot access the file" when deleting on Windows.
        with FailingTempDir() as td:
            c.ProjectsDB.repo_path = td
            pump_cmd(project.InitCmd(config=c).run(test_vfid))

            pump_cmd(project.AppendCmd(config=c, out=[test_out_fpath]).run())
            res = report.ReportCmd(config=c).run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 1)
            rpt = yaml.load('\n'.join(res))
            f, rec = next(iter(rpt.items()))
            self.assertNotIn('input.xlsx', rpt)
            self.assertIn("output.xlsx", rpt)
            self.check_report_tuple(rec, test_vfid, test_out_fpath, 'out', True)

            pump_cmd(project.AppendCmd(config=c, inp=[test_inp_fpath]).run())
            res = report.ReportCmd(config=c).run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 2)
            rpt = yaml.load('\n'.join(res))
            self.assertIn('output.xlsx', rpt)
            self.assertIn('input.xlsx', rpt)
            for f, rec in rpt.items():
                if f == 'input.xlsx':
                    path, iokind, rpt = "tests\sampling\input.xlsx", 'inp', False
                elif f == 'output.xlsx':
                    path, iokind, rpt = "tests\sampling\output.xlsx", 'out', True
                self.check_report_tuple(rec, test_vfid, path, iokind, rpt)

    def test_both(self):
        c = self.cfg
        ## "The process cannot access the file" when deleting on Windows.
        with FailingTempDir() as td:
            c.ProjectsDB.repo_path = td
            pump_cmd(project.InitCmd(config=c).run(proj2))

            cmd = project.AppendCmd(config=c, inp=[test_inp_fpath], out=[test_out_fpath])
            pump_cmd(cmd.run())
            cmd = report.ReportCmd(config=c)
            res = cmd.run()
            self.assertIsInstance(res, types.GeneratorType)
            res = list(res)
            self.assertEqual(len(res), 2)
            rpt = yaml.load('\n'.join(res))
            self.assertIn('output.xlsx', rpt)
            self.assertIn('input.xlsx', rpt)
            for f, rec in rpt.items():
                if f.endswith('input.xlsx'):
                    path, iokind, rpt = "input.xlsx", 'inp', False
                elif f.endswith('output.xlsx'):
                    path, iokind, rpt = "output.xlsx", 'out', True
                self.assertTrue(f.endswith(path), rpt)
                self.check_report_tuple(rec, test_vfid, path, iokind, rpt)


@ddt.ddt
class TReportShell(unittest.TestCase):
    def test_report_other_smoketest(self):
        fpath = osp.join(mydir, '..', '..', 'setup.py')
        ret = sbp.check_call('co2dice report %s' % fpath,
                             env=os.environ)
        self.assertEqual(ret, 0)

    def test_report_inp_smoketest(self):
        fpath = osp.join(mydir, 'input.xlsx')
        ret = sbp.check_call('co2dice report -i %s' % fpath,
                             env=os.environ)
        self.assertEqual(ret, 0)

    def test_report_out_smoketest(self):
        fpath = osp.join(mydir, 'output.xlsx')
        ret = sbp.check_call('co2dice report -o %s' % fpath,
                             env=os.environ)
        self.assertEqual(ret, 0)

    def test_report_io_smoketest(self):
        fpath1 = osp.join(mydir, 'input.xlsx')
        fpath2 = osp.join(mydir, 'output.xlsx')
        ret = sbp.check_call('co2dice report -i %s -o %s' %
                             (fpath1, fpath2),
                             env=os.environ)
        self.assertEqual(ret, 0)

    def test_report_iof_smoketest(self):
        fpath1 = osp.join(mydir, 'input.xlsx')
        fpath2 = osp.join(mydir, 'output.xlsx')
        fpath3 = osp.join(mydir, '__init__.py')
        ret = sbp.check_call('co2dice report -i %s -o %s %s' %
                             (fpath1, fpath2, fpath3),
                             env=os.environ)
        self.assertEqual(ret, 0)

