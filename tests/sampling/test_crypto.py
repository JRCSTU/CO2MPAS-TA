#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import crypto
import logging
import shutil
import tempfile
import unittest
import contextlib

import ddt

import itertools as itt
import os.path as osp
import traitlets.config as trtc


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)

_texts = ('', ' ', 'a' * 2048, '123', 'asdfasd|*(KJ|KL97GDk;')
_objs = ('', ' ', None, 'a' * 2048, 1244, b'\x22', {1: 'a', '2': {3, b'\x04'}})

_ciphertexts = set()


# class TestDoctest(unittest.TestCase):
#     def runTest(self):
#         failure_count, test_count = doctest.testmod(
#             crypto,
#             optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
#         self.assertGreater(test_count, 0, (failure_count, test_count))
#         self.assertEqual(failure_count, 0, (failure_count, test_count))


def _gpg_gen_key(GPG, key_length, name_real, name_email):
    key = GPG.gen_key(
        GPG.gen_key_input(key_length=key_length,
                          name_real=name_real,
                          name_email=name_email))
    assert key.fingerprint

    return key.fingerprint


def _gpg_del_key(GPG, fingerprint):
    log.debug('Deleting secret+pub: %s', fingerprint)
    d = GPG.delete_keys(fingerprint, secret=1)
    assert (d.status, d.stderr) == ('ok', ''), (
        "Failed DELETING pgp-secret: %s" % d.stderr)
    d = GPG.delete_keys(fingerprint)
    assert (d.status, d.stderr) == ('ok', ''), (
        "Failed DELETING pgp-public: %s" % d.stderr)


@contextlib.contextmanager
def _temp_master_key(safedepot, master_key):
    oldkey = safedepot.master_key
    safedepot.master_key = master_key
    try:
        yield safedepot
    finally:
        safedepot.master_key = oldkey


@ddt.ddt
class TSafeDepotSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = trtc.get_config()
        cfg.SafeDepotSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        safedepot = crypto.SafeDepotSpec.instance(config=cfg)

        key_fingerprint = _gpg_gen_key(
            safedepot.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')
        safedepot.master_key = key_fingerprint

    @classmethod
    def tearDownClass(cls):
        safedepot = crypto.SafeDepotSpec.instance()
        assert safedepot.gnupghome
        _gpg_del_key(safedepot.GPG, safedepot.master_key)
        shutil.rmtree(safedepot.gnupghome)

    @ddt.idata(itt.product(('user', '&^a09|*(K}'), _objs))
    def test_1_dencrypt(self, case):
        pswdid, obj = case
        safedepot = crypto.SafeDepotSpec.instance()

        ciphertext = safedepot.encryptobj('enc_test', obj)
        msg = ('CASE:', case, ciphertext)

        self.assertTrue(crypto.is_pgp_encrypted(ciphertext), msg)

        ## Check not generating indetical ciphers.
        #
        self.assertNotIn(ciphertext, _ciphertexts)
        _ciphertexts.add(ciphertext)

        plainbytes2 = safedepot.decryptobj(pswdid, ciphertext)
        self.assertEqual(obj, plainbytes2, msg)

    def test_2_many_master_keys(self):
        safedepot = crypto.SafeDepotSpec.instance()
        key_fingerprint = _gpg_gen_key(
            safedepot.GPG,
            key_length=1024,
            name_real='test user2',
            name_email='test2@test.com')
        try:
            with _temp_master_key(safedepot, None):
                with self.assertRaisesRegex(ValueError, 'Cannot guess master-key! Found 2 keys'):
                    safedepot.encryptobj('enc_test', b'')
        finally:
            _gpg_del_key(safedepot.GPG, key_fingerprint)

    def test_3_no_master_key(self):
        safedepot = crypto.SafeDepotSpec.instance()
        _gpg_del_key(safedepot.GPG, safedepot.master_key)
        try:
            with _temp_master_key(safedepot, None):
                with self.assertRaisesRegex(ValueError, 'Cannot guess master-key! Found 0 keys'):
                    safedepot.encryptobj('enc_test', b'')
        finally:
            safedepot.master_key = _gpg_gen_key(
                safedepot.GPG,
                key_length=1024,
                name_real='test user3',
                name_email='test2@test.com')
