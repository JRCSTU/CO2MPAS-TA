#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import ddt
import filecmp
import unittest
import os.path as osp

fdir = osp.abspath(osp.dirname(__file__))


@ddt.ddt
class CLI(unittest.TestCase):
    def setUp(self):
        import functools
        from co2mpas.cli import cli
        from click.testing import CliRunner
        self.runner = CliRunner()
        self.invoke = functools.partial(self.runner.invoke, cli)

    @ddt.idata((
            (),
            ('temp.xlsx',),
            ('folder/temp.xlsx',),
            ('folder/temp.xlsx', '-TT', 'input'),
            ('folder/temp.xlsx', '-TT', 'output')
    ))
    def test_template(self, options):
        tt = len(options) <= 2 and 'input' or options[2]
        rfp = osp.join(fdir, '../co2mpas/templates/%s_template.xlsx' % tt)
        with self.runner.isolated_filesystem():
            result = self.invoke(('template',) + options)
            self.assertEqual(result.exit_code, 0)
            fp = len(options) == 0 and 'template.xlsx' or options[0]
            self.assertTrue(
                filecmp.cmp(fp, rfp), 'Template file is not as expected!'
            )

    @ddt.idata((
            (),
            ('demos',),
            ('demo/inputs',)
    ))
    def test_demo(self, options):
        import glob
        demos = {
            osp.basename(fp): fp
            for fp in glob.glob(osp.join(fdir, '../co2mpas/demos/*.xlsx'))
        }
        with self.runner.isolated_filesystem():
            result = self.invoke(('demo',) + options)
            self.assertEqual(result.exit_code, 0)
            it = glob.glob(osp.join(options and options[0] or '.', '*.xlsx'))
            for fpath in it:
                self.assertTrue(
                    filecmp.cmp(fpath, demos[osp.basename(fpath)]),
                    'Demo file (%s) is not as expected!' % fpath
                )

    @ddt.idata((
            (),
            ('temp.yaml',),
            ('conf/temp.yaml',),
            ('conf/temp.yaml', '-MC', osp.join(fdir, 'files/conf.yaml'))
    ))
    def test_conf(self, options):
        import yaml
        import schedula as sh
        from co2mpas.core.model.physical.defaults import dfl
        t = {k for k, _ in sh.stack_nested_keys(dfl.to_dict())}
        with self.runner.isolated_filesystem():
            result = self.invoke(('conf',) + options)
            self.assertEqual(result.exit_code, 0)
            fp = len(options) == 0 and 'conf.yaml' or options[0]
            with open(fp, 'rb') as f:
                r = dict(sh.stack_nested_keys(yaml.load(f)))
                self.assertSetEqual(set(r), t)

            if len(options) == 3:
                with open(options[2], 'rb') as f:
                    for k, v in sh.stack_nested_keys(yaml.load(f)):
                        self.assertEqual(r[k], v)
