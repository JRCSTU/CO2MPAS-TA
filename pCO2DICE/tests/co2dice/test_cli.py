#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from sampling._vendor.traitlets import config as trtc
from sampling import cmdlets, cli, cfgcmd
import logging
import os
import tempfile
import unittest

import ddt

import os.path as osp


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


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
        c = trtc.get_config()
        c.Co2dice.raise_config_file_errors = True
        cmd = cli.Co2diceCmd(config=c)
        meth(cmd)

    def test_config_init(self):
        c = trtc.get_config()
        c.Co2dice.raise_config_file_errors = True
        cmd = cfgcmd.WriteCmd(config=c)
        cmd.initialize([])
        with tempfile.TemporaryDirectory() as td:
            conf_fpath = osp.join(td, 'cc.py')
            cmd.run(conf_fpath)
            self.assertTrue(osp.isfile(conf_fpath),
                            (conf_fpath, os.listdir(osp.split(conf_fpath)[0])))
            stat = os.stat(conf_fpath)
            self.assertGreater(stat.st_size, 7000, stat)

    def test_config_paths(self):
        c = trtc.get_config()
        c.Co2dice.raise_config_file_errors = True
        cmd = cmdlets.chain_cmds([cfgcmd.PathsCmd], [], config=c)
        res = cmd.start()
        res = list(res)
        self.assertGreaterEqual(len(res), 2, res)

    def test_config_show(self):
        c = trtc.get_config()
        c.Co2dice.raise_config_file_errors = True
        cmd = cmdlets.chain_cmds([cfgcmd.ShowCmd], [], config=c)
        res = cmd.start()
        res = list(res)
        ## Count Cmd-lines not starting with '  +--trait'.
        ncmdlines = sum(1 for r in res if r[0] != ' ')
        self.assertGreaterEqual(ncmdlines, 10, res)  # I counted at least 10...

    def test_config_show_verbose(self):
        c = trtc.get_config()
        c.ShowCmd.verbose = 1
        c.Co2dice.raise_config_file_errors = True
        cmd = cfgcmd.ShowCmd(config=c)
        cmd.initialize([])
        res = list(cmd.run())
        ## Count Cmd-lines not starting with '  +--trait'.
        ncmdlines = sum(1 for r in res if r[0] != ' ')
        self.assertGreaterEqual(ncmdlines, len(cmd.all_app_configurables()), res)

    @ddt.data(*cli.all_cmds())
    def test_all_cmds_help_smoketest(self, cmd: cmdlets.Cmd):
        cmd.class_get_help()
        cmd.class_config_section()
        cmd.class_config_rst_doc()

        c = cmd()
        c.print_help()
        c.document_config_options()
        c.print_alias_help()
        c.print_flag_help()
        c.print_options()
        c.print_subcommands()
        c.print_examples()
        c.print_help()
