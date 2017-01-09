#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp, crypto
import io
import json
import logging
import shutil
import tempfile
import unittest

import ddt

import itertools as itt
import os.path as osp
import pandalone.utils as pndlu
import traitlets as trt
import traitlets.config as trtc

from . import test_crypto as cryptotc


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


def prepare_persistent_config_file(pfile):
    j = {}
    if osp.isfile(pfile):
        with io.open(pfile, 'rt') as finp:
            j = json.load(finp)

    ## Add an arbitrary parameters in pfile to see if it is preserved.
    j['ANY'] = {'a': 1}

    ## Add an arbitrary parameters in pfile to see if it is preserved.
    j['MyCmd'] = {'ptrait': False}
    j['MySpec'] = {'ptrait': False}

    with io.open(pfile, 'wt') as fout:
        json.dump(j, fout)


def mix_dics(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


@ddt.ddt
class TBase(unittest.TestCase):

    def check_persistent_config_file(self, pfile, clsname=None, trait=None, value=None):
        with io.open(pfile, 'rt') as finp:
            j = json.load(finp)

        ## Check arbitrary params preserved.
        #
        self.assertIn('MySpec', j, j)
        self.assertIn('MyCmd', j, j)
        self.assertEqual(j['ANY']['a'], 1, j)

        if clsname is not None:
            self.assertEqual(j[clsname][trait], value, j)

        return j


@ddt.ddt
class TPTraits(TBase):

    def test_ptraits_spec(self):
        pfile = pndlu.ensure_file_ext(baseapp.default_config_fpath(), '.json')
        prepare_persistent_config_file(pfile)

        class MySpec(baseapp.Spec):
            "Ok Spec"
            ptrait = trt.Bool().tag(config=True, persist=True)

        c = MySpec()
        c.load_pconfig_file(pfile)  # Needed bc only final Cmds load ptraits.

        self.check_persistent_config_file(pfile, 'MySpec', 'ptrait', False)

        c.ptrait = True
        self.check_persistent_config_file(pfile, 'MySpec', 'ptrait', False)

        c.store_pconfig_file(pfile)
        self.check_persistent_config_file(pfile, 'MySpec', 'ptrait', True)

    def test_ptraits_cmd(self):
        pfile = pndlu.ensure_file_ext(baseapp.default_config_fpath(), '.json')
        prepare_persistent_config_file(pfile)

        class MyCmd(baseapp.Cmd):
            "Ok Cmd"
            ptrait = trt.Bool().tag(config=True, persist=True)

        c = MyCmd()
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', False)
        c.initialize([])
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', False)

        c.ptrait = True
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', False)

        c.store_pconfig_file(pfile)
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', True)

    @ddt.data(
        {}, {'config': False}, {'config': None}, {'config': 0}, {'config': 1}, {'config': -2})
    def test_invalid_ptraits_on_spec(self, tags):
        class MySpec(baseapp.Spec):
            "Spec with invalid ptrait"
            bad_ptrait = trt.Bool().tag(persist=True, **tags)

        with self.assertRaisesRegex(trt.TraitError,
                                    "Persistent trait 'bad_ptrait' not tagged as 'config'!"):
            MySpec()

    @ddt.data(
        {}, {'config': False}, {'config': None}, {'config': 0}, {'config': 1}, {'config': -2})
    def test_invalid_ptraits_on_cmd(self, tags):
        class MyCmd(baseapp.Cmd):
            "Cmd with invalid ptrait"
            bad_ptrait = trt.Bool().tag(persist=True, **tags)

        c = MyCmd()
        with self.assertLogs(c.log, logging.FATAL) as cm:
            try:
                c.initialize([])
            except SystemExit:
                pass
        exp_msg = "Persistent trait 'bad_ptrait' not tagged as 'config'!"
        self.assertTrue(any(exp_msg in m for m in cm.output), cm.output)


@ddt.ddt
class TCipherTraits(TBase):
    @classmethod
    def setUpClass(cls):
        cfg = trtc.get_config()
        cfg.SafeDepotSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        safedepot = crypto.SafeDepotSpec.instance(config=cfg)

        fingerprint = cryptotc.gpg_gen_key(
            safedepot.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')
        safedepot.master_key = fingerprint

    @classmethod
    def tearDownClass(cls):
        safedepot = crypto.SafeDepotSpec.instance()
        assert safedepot.gnupghome
        cryptotc.gpg_del_key(safedepot.GPG, safedepot.master_key)
        shutil.rmtree(safedepot.gnupghome)

    def test_chiphertraits_spec(self):
        pfile = pndlu.ensure_file_ext(baseapp.default_config_fpath(), '.json')
        prepare_persistent_config_file(pfile)

        class MySpec(baseapp.Spec):
            "OK Spec"

            ctrait = crypto.Cipher(allow_none=True).tag(config=True, persist=True)

        c = MySpec()
        c.load_pconfig_file(pfile)  # Needed bc only final Cmds load ptraits.

        val = c.ctrait = 'foo'
        cipher = c._trait_values['ctrait']
        self.assertEqual(c.ctrait, 'foo')  # Preserved among gets.
        self.assertEqual(c.ctrait, 'foo')  # Preserved among gets.
        self.assertNotEqual(cipher, val)
        self.assertTrue(crypto.is_pgp_encrypted(cipher), cipher)

        j = self.check_persistent_config_file(pfile)
        self.assertNotIn('ctrait', j['MySpec'], j)
        c.store_pconfig_file(pfile)
        self.check_persistent_config_file(pfile, 'MySpec', 'ctrait', cipher)

    def test_chiphertraits_cmd(self):
        pfile = pndlu.ensure_file_ext(baseapp.default_config_fpath(), '.json')
        prepare_persistent_config_file(pfile)

        class MyCmd(baseapp.Cmd):
            "OK Cmd"

            ctrait = crypto.Cipher(allow_none=True).tag(config=True, persist=True)

        c = MyCmd()
        c.initialize([])

        val = c.ctrait = 'foo'
        cipher = c._trait_values['ctrait']
        self.assertEqual(c.ctrait, 'foo')  # Preserved among gets.
        self.assertEqual(c.ctrait, 'foo')  # Preserved among gets.
        self.assertNotEqual(cipher, val)
        self.assertTrue(crypto.is_pgp_encrypted(cipher), cipher)

        j = self.check_persistent_config_file(pfile)
        self.assertNotIn('ctrait', j['MyCmd'], j)
        c.store_pconfig_file(pfile)
        self.check_persistent_config_file(pfile, 'MyCmd', 'ctrait', cipher)

    @ddt.idata(mix_dics(d1, d2) for d1, d2 in itt.product(
        [{}, {'config': False}, {'config': None}, {'config': 0}],
        [{}, {'persist': False}, {'persist': None}, {'persist': 0}],
    ))
    def test_invalid_enctraits_on_specs(self, tags):
        class MySpec(baseapp.Spec):
            "Spec with invalid cipher-trait"
            bad_ptrait = crypto.Cipher(allow_none=True).tag(**tags)

        with self.assertRaisesRegex(trt.TraitError,
                                    r"Cipher-trait 'MySpec.bad_ptrait' not tagged as 'config' \+ 'persist'!"):
            MySpec()

    @ddt.idata(mix_dics(d1, d2) for d1, d2 in itt.product(
        [{}, {'config': False}, {'config': None}, {'config': 0}],
        [{}, {'persist': False}, {'persist': None}, {'persist': 0}],
    ))
    def test_invalid_enctraits_on_cmds(self, tags):
        class MyCmd(baseapp.Cmd):
            "Cmd with invalid cipher-trait"
            bad_ptrait = crypto.Cipher(allow_none=True).tag(**tags)

        with self.assertRaisesRegex(trt.TraitError,
                                    r"Cipher-trait 'MyCmd.bad_ptrait' not tagged as 'config' \+ 'persist'!"):
            MyCmd()
