#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas._vendor.traitlets import config as trtc
from co2mpas.sampling import crypto, tsign
from co2mpas.sampling.baseapp import Cmd
import logging
import shutil
import tempfile
import unittest

import ddt

import os.path as osp
import subprocess as sbp

from . import test_pgp_fingerprint, test_pgp_keys, test_pgp_trust
from .test_tstamp import signed_tag


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


@ddt.ddt
class TStamper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = c = trtc.Config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_keys
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.GpgSpec.allow_test_key = True
        crypto.GpgSpec(config=c)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)
        shutil.rmtree(cls.tmp_chain_folder)

    def test_stamp_text(self):
        signer = tsign.TstamperService(config=self.cfg)

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
        signer = tsign.TstamperService(config=self.cfg)
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
class TstampShell(unittest.TestCase):
    """Set ``CO2DICE_CONFIG_PATHS`` and optionally HOME env appropriately! to run"""

    def _get_vault_gnupghome(self):
        """Allow cipher-traits encrypted with the GPG of the user running tests."""
        cmd = Cmd()
        cmd.initialize([])
        vault = crypto.VaultSpec(config=cmd.config)
        return ('--VaultSpec.gnupghome=%s' % vault.gnupghome
                if vault.gnupghome
                else '')

    def test_sign_smoketest(self):
        fpath = osp.join(mydir, 'tag.txt')
        stamp = sbp.check_output('co2dice sign %s' % fpath,
                                 universal_newlines=True)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)

        suffix = "Garbage"
        with open(fpath, 'rt') as fd:
            tag = fd.read()

        p = sbp.Popen('co2dice sign -',
                      universal_newlines=True,
                      stdin=sbp.PIPE, stdout=sbp.PIPE)
        stamp, _stderr = p.communicate(input=tag + suffix)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)
        self.assertIn(suffix, stamp)

        p = sbp.Popen('co2dice sign - --TstamperService.trim_dreport=True',
                      universal_newlines=True,
                      stdin=sbp.PIPE, stdout=sbp.PIPE)
        stamp, _stderr = p.communicate(input=tag + suffix)
        exp_prefix = '-----BEGIN PGP SIGNED MESSAGE'
        self.assertEqual(stamp[:len(exp_prefix)], exp_prefix, stamp)
        self.assertNotIn(suffix, stamp)
