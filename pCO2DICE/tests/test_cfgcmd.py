#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2dice import cfgcmd
from co2dice.cmdlets import collect_cmd
import logging
import os
import unittest

import ddt

import os.path as osp
import subprocess as sbp


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)

class T(unittest.TestCase):

    @unittest.expectedFailure
    def test_paths_valid_yaml(self):
        import yaml
        res = collect_cmd(cfgcmd.PathsCmd().run())
        ystr = '\n'.join(res)

        yaml.load(ystr)


@ddt.ddt
class TcfgcmdShell(unittest.TestCase):
    def test_paths_smoketest(self):
        ret = sbp.check_call('co2dice config paths', env=os.environ)
        self.assertEqual(ret, 0)

    @ddt.data('', 'proj', '-e proj', '-el proj')
    def test_show_smoketest(self, case):
        cmd = ('co2dice config show ' + case).strip()
        ret = sbp.check_call(cmd, env=os.environ)
        self.assertEqual(ret, 0)

    @ddt.data('', 'TstampReceiver', 'recv', '-l rec', '-e rec', '-le rec'
              '-ecl rec', '-cl rec')
    def test_desc_smoketest(self, case):
        cmd = ('co2dice config desc ' + case).strip()
        ret = sbp.check_call(cmd, env=os.environ)
        self.assertEqual(ret, 0)
