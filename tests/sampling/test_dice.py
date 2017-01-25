#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp, dice, cfgcmd
import logging
import os
import tempfile
from traitlets.config import get_config
import unittest

import ddt

import os.path as osp


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


# def _make_test_cfg():
#     cfg = io.StringIO(_test_cfg)
#     return yaml.load(cfg)
#
# @ddt.ddt
# class TDice(unittest.TestCase):
#
#    def _has_repeatitive_prefix(self, word, limit, char=None):
#        c = word[0]
#        if not char or c == char:
#            for i  in range(1, limit):
#                if word[i] != c:
#                    break
#            else:
#                return True
#
#    def gpg_gen_interesting_keys(self, gpg, name_real, name_email, key_length,
#            predicate, nkeys=1, runs=0):
#        keys = []
#        for i in itt.count(1):
#            del_key = True
#            key = gpg.gen_key(gpg.gen_key_input(key_length=key_length,
#                    name_real=name_real, name_email=name_email))
#            try:
#                log.debug('Created-%i: %s', i, key.fingerprint)
#                if predicate(key.fingerprint):
#                    del_key = False
#                    keys.append(key.fingerprint)
#                    keyid = key.fingerprint[24:]
#                    log.info('FOUND-%i: %s-->%s', i, keyid, key.fingerprint)
#                    nkeys -= 1
#                    if nkeys == 0:
#                        break
#            finally:
#                if del_key:
#                    gpg_del_gened_key(gpg, key.fingerprint)
#        return keys
#
#     @unittest.skip('Enabled it to generate test-keys!!')
#     def test_gen_key_proof_of_work(self):
#         import gnupg
#         gpg_prog = 'gpg2.exe'
#         gpg2_path = dice.which(gpg_prog)
#         self.assertIsNotNone(gpg2_path)
#         gpg=gnupg.GPG(gpg2_path)
#
#         def key_predicate(fingerprint):
#             keyid = fingerprint[24:]
#             return self._has_repeatitive_prefix(keyid, limit=2)
#
#         def keyid_starts_repetitively(fingerprint):
#             keyid = fingerprint[24:]
#             return self._has_repeatitive_prefix(keyid, limit=2)
#         def keyid_n_fingerprint_start_repetitively(fingerprint):
#             keyid = fingerprint[24:]
#             return self._has_repeatitive_prefix(keyid, limit=2)
#
#         name_real='Sampling Test',
#         name_email='sampling@co2mpas.jrc.ec.europa.eu'
#         key_length=1024
#         dice.gpg_gen_interesting_keys(gpg, key_length, name_real, name_email,
#                 keyid_n_fingerprint_start_repetitively)


@ddt.ddt
class TApp(unittest.TestCase):

    @ddt.data(
        dice.MainCmd.document_config_options,
        dice.MainCmd.print_alias_help,
        dice.MainCmd.print_flag_help,
        dice.MainCmd.print_options,
        dice.MainCmd.print_subcommands,
        dice.MainCmd.print_examples,
        dice.MainCmd.print_help,
    )
    def test_app(self, meth):
        c = get_config()
        c.MainCmd.raise_config_file_errors = True
        cmd = dice.MainCmd(config=c)
        meth(cmd)

    def test_config_init(self):
        c = get_config()
        c.MainCmd.raise_config_file_errors = True
        cmd = baseapp.chain_cmds([cfgcmd.ConfigCmd.InitCmd], config=c)
        with tempfile.TemporaryDirectory() as td:
            conf_fpath = osp.join(td, 'cc.py')
            cmd.run(conf_fpath)
            self.assertTrue(osp.isfile(conf_fpath),
                            (conf_fpath, os.listdir(osp.split(conf_fpath)[0])))
            stat = os.stat(conf_fpath)
            self.assertGreater(stat.st_size, 7000, stat)

    def test_config_paths(self):
        c = get_config()
        c.MainCmd.raise_config_file_errors = True
        cmd = baseapp.chain_cmds([cfgcmd.ConfigCmd.PathsCmd], config=c)
        res = list(cmd.run())
        self.assertGreaterEqual(len(res), 2, res)

    def test_config_show(self):
        c = get_config()
        c.MainCmd.raise_config_file_errors = True
        cmd = baseapp.chain_cmds([cfgcmd.ConfigCmd.ShowCmd], config=c)
        res = list(cmd.run())
        ## Count Cmd-lines not starting with '  +--trait'.
        ncmdlines = sum(1 for r in res if r[0] != ' ')
        self.assertGreaterEqual(ncmdlines, 10, res)  # I counted at least 10...

    def test_config_show_verbose(self):
        c = get_config()
        c.ShowCmd.verbose = 1
        c.MainCmd.raise_config_file_errors = True
        cmd = baseapp.chain_cmds([cfgcmd.ConfigCmd.ShowCmd], config=c)
        res = list(cmd.run())
        ## Count Cmd-lines not starting with '  +--trait'.
        ncmdlines = sum(1 for r in res if r[0] != ' ')
        self.assertGreaterEqual(ncmdlines, len(cmd.all_app_configurables()), res)

    @ddt.data(*dice.all_cmds())
    def test_all_cmds_help_smoketest(self, cmd: baseapp.Cmd):
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
