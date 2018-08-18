#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import logging
import shutil
import tempfile
import unittest

from co2mpas.__main__ import init_logging
from co2dice._vendor.traitlets import config as trtc
from co2dice import crypto, tsigner
from co2dice.cmdlets import Cmd
import ddt

from git.util import rmtree as gutil_rmtree
import os.path as osp
import subprocess as sbp

from . import TEST_CONF_ENVAR, CONF_ENVAR, test_pgp_fingerprint, test_pgp_keys, test_pgp_trust
from .test_tstamp import signed_tag
import os


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)

pgp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
header_len = 400
default_sender = 'some body'
default_recipients = ['aa@foo', 'devnull']


@ddt.ddt
class TS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = c = trtc.Config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_keys
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.GpgSpec.allow_test_key = True
        crypto.GpgSpec(config=c)

        cls.tmp_chain_folder = tempfile.mkdtemp(prefix='stampchain-')
        c.SigChain.stamp_chain_dir = cls.tmp_chain_folder

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()     # @UndefinedVariable
        crypto.GitAuthSpec.clear_instance()         # @UndefinedVariable
        crypto.VaultSpec.clear_instance()           # @UndefinedVariable

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)
        gutil_rmtree(cls.tmp_chain_folder)

    def check_recipients(self, stamp):
        for i in default_recipients:
            self.assertIn(i, stamp[:header_len], stamp)

    def test_stamp_text(self):
        signer = tsigner.TsignerService(config=self.cfg)

        tag_verdict = signer.parse_signed_tag(signed_tag)
        sender = crypto.uid_from_verdict(tag_verdict)

        sign = signer.sign_text_as_tstamper(signed_tag,
                                            sender,
                                            '12345',
                                            full_output=True)
        stamp = str(sign)
        self.assertEqual(stamp[:len(pgp_prefix)], pgp_prefix, stamp)

        ts_verdict = vars(sign)
        tag = signer.extract_dice_tag_name(None, signed_tag)
        dice_decision = signer.make_dice_results(ts_verdict, tag_verdict, tag)

        stamp2 = signer.append_decision(stamp, dice_decision)
        self.assertIn(stamp, stamp2, stamp2)
        self.assertIn("dice:\n  tag:", stamp2, stamp2)

    def test_stamp_dreport(self):
        who = 'YummyMammy'
        signer = tsigner.TsignerService(config=self.cfg,
                                        recipients=default_recipients)
        stamp, _decision = signer.sign_dreport_as_tstamper(who, signed_tag)
        print(stamp)
        self.assertNotIn('\r\n', stamp)
        self.assertEqual(stamp[:len(pgp_prefix)], pgp_prefix, stamp)
        self.assertIn(who, stamp[:header_len], stamp)
        self.check_recipients(stamp)

    def test_stamp_dreport_suffix(self):
        who = 'YummyMammy'
        signer = tsigner.TsignerService(config=self.cfg,
                                        recipients=default_recipients)
        suffix = "Garbage"
        stamp, _decision = signer.sign_dreport_as_tstamper(who,
                                                           signed_tag + suffix)
        self.assertNotIn('\r\n', stamp)
        self.assertEqual(stamp[:len(pgp_prefix)], pgp_prefix, stamp)
        self.assertIn(who, stamp[:header_len], stamp)
        self.assertIn(suffix, stamp)
        self.check_recipients(stamp)

        signer.trim_dreport = True
        stamp, _decision = signer.sign_dreport_as_tstamper(who,
                                                           signed_tag + suffix)
        self.assertNotIn('\r\n', stamp)
        self.assertEqual(stamp[:len(pgp_prefix)], pgp_prefix, stamp)
        self.assertIn(who, stamp[:header_len], stamp)
        self.assertNotIn(suffix, stamp)
        self.check_recipients(stamp)


@ddt.ddt
class TSShell(unittest.TestCase):
    """Set ``TEST_CO2DICE_CONFIG_PATHS`` and optionally HOME env appropriately! to run"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_chain_folder = tempfile.mkdtemp(prefix='stampchain-')
        cls._old_envvar = os.environ.get(CONF_ENVAR)
        if TEST_CONF_ENVAR in os.environ:
            os.environ[CONF_ENVAR] = os.environ[TEST_CONF_ENVAR]

    @classmethod
    def tearDownClass(cls):
        if cls._old_envvar is not None:
            os.environ[CONF_ENVAR] = cls._old_envvar
        gutil_rmtree(cls.tmp_chain_folder)

    def _get_extra_options(self):
        """Allow cipher-traits encrypted with the GPG of the user running tests."""
        opts = []

        cmd = Cmd()
        cmd.initialize([])
        vault = crypto.VaultSpec(config=cmd.config)
        if vault.gnupghome:
            opts.append('--VaultSpec.gnupghome=%s' % vault.gnupghome)

        opts.extend(['--SigChain.stamp_chain_dir', self.tmp_chain_folder])
        opts.extend(['--GpgSpec.allow_test_key', 'True'])
        opts.extend(['--sender', default_sender])

        return opts

    def test_sign_smoketest(self):
        fpath = osp.join(mydir, 'tag.txt')
        stamp = sbp.check_output(['co2dice', 'tsigner', fpath] +
                                 self._get_extra_options(),
                                 universal_newlines=True)
        self.assertEqual(stamp[:len(pgp_prefix)], pgp_prefix, stamp)

        suffix = "Garbage"
        with open(fpath, 'rt') as fd:
            tag = fd.read()

        p = sbp.Popen(['co2dice', 'tsigner', '-'] + self._get_extra_options(),
                      universal_newlines=True,
                      stdin=sbp.PIPE, stdout=sbp.PIPE)
        stamp, _stderr = p.communicate(input=tag + suffix)
        assert p.returncode == 0
        self.assertEqual(stamp[:len(pgp_prefix)], pgp_prefix, stamp)
        self.assertIn(stamp[:len(pgp_prefix)], pgp_prefix, stamp)
        self.assertIn(default_sender, stamp)

        p = sbp.Popen(['co2dice', 'tsigner', '-',
                       '--TsignerService.trim_dreport=True'] +
                      self._get_extra_options(),
                      universal_newlines=True,
                      stdin=sbp.PIPE, stdout=sbp.PIPE)
        stamp, _stderr = p.communicate(input=tag + suffix)
        self.assertEqual(stamp[:len(pgp_prefix)], pgp_prefix, stamp)
        self.assertIn(default_sender, stamp)
        self.assertNotIn(suffix, stamp)
