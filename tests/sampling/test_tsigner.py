#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import logging
import shutil
import tempfile
import unittest

from co2mpas.__main__ import init_logging
from co2mpas._vendor.traitlets import config as trtc
from co2mpas.sampling import crypto, tsigner
from co2mpas.sampling.baseapp import Cmd
import ddt

from git.util import rmtree as gutil_rmtree
import os.path as osp
import subprocess as sbp

from . import test_pgp_fingerprint, test_pgp_keys, test_pgp_trust
from .test_tstamp import signed_tag


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


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
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)
        gutil_rmtree(cls.tmp_chain_folder)

    def test_stamp_text(self):
        signer = tsigner.TsignerService(config=self.cfg)

        tag_verdict = signer.parse_signed_tag(signed_tag)
        sender = crypto.uid_from_verdict(tag_verdict)

        sign = signer.sign_text_as_tstamper(signed_tag, sender,
                                            full_output=True)
        stamp = str(sign)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)

        ts_verdict = vars(sign)
        tag = signer.extract_dice_tag_name(None, signed_tag)
        dice_decision = signer.make_dice_results(ts_verdict, tag_verdict, tag)

        stamp2 = signer.append_decision(stamp, dice_decision)
        self.assertIn(stamp, stamp2, stamp2)
        self.assertIn("dice:\n  tag:", stamp2, stamp2)

    def test_stamp_dreport(self):
        signer = tsigner.TsignerService(config=self.cfg)
        stamp, _decision = signer.sign_dreport_as_tstamper(signed_tag)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)

        suffix = "Garbage"
        stamp, _decision = signer.sign_dreport_as_tstamper(signed_tag + suffix)
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)
        self.assertIn(suffix, stamp)

        signer.trim_dreport = True
        stamp, _decision = signer.sign_dreport_as_tstamper(signed_tag + suffix)
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)
        self.assertNotIn(suffix, stamp)


@ddt.ddt
class TSShell(unittest.TestCase):
    """Set ``CO2DICE_CONFIG_PATHS`` and optionally HOME env appropriately! to run"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_chain_folder = tempfile.mkdtemp(prefix='stampchain-')

    @classmethod
    def tearDownClass(cls):
        gutil_rmtree(cls.tmp_chain_folder)

    def _get_extra_options(self):
        """Allow cipher-traits encrypted with the GPG of the user running tests."""
        opts = []

        cmd = Cmd()
        cmd.initialize([])
        vault = crypto.VaultSpec(config=cmd.config)
        if vault.gnupghome:
            opts.append('--VaultSpec.gnupghome=%s' % vault.gnupghome)

        opts.append('--SigChain.stamp_chain_dir=%s' % self.tmp_chain_folder)

        return ' '.join(opts)

    def test_sign_smoketest(self):
        fpath = osp.join(mydir, 'tag.txt')
        stamp = sbp.check_output('co2dice tsigner %s %s' %
                                 (fpath, self._get_extra_options()),
                                 universal_newlines=True)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)

        suffix = "Garbage"
        with open(fpath, 'rt') as fd:
            tag = fd.read()

        p = sbp.Popen('co2dice tsigner - %s' % self._get_extra_options(),
                      universal_newlines=True,
                      stdin=sbp.PIPE, stdout=sbp.PIPE)
        stamp, _stderr = p.communicate(input=tag + suffix)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)
        self.assertIn(suffix, stamp)

        p = sbp.Popen('co2dice tsigner - --TsignerService.trim_dreport=True'
                      ' %s' % self._get_extra_options(),
                      universal_newlines=True,
                      stdin=sbp.PIPE, stdout=sbp.PIPE)
        stamp, _stderr = p.communicate(input=tag + suffix)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)
        self.assertNotIn(suffix, stamp)
