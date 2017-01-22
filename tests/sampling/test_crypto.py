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


def gpg_gen_key(GPG, key_length, name_real, name_email):
    key = GPG.gen_key(
        GPG.gen_key_input(key_length=key_length,
                          name_real=name_real,
                          name_email=name_email))
    assert key.fingerprint

    return key.fingerprint


def gpg_del_key(GPG, fingerprint):
    log.debug('Deleting secret+pub: %s', fingerprint)
    d = GPG.delete_keys(fingerprint, secret=1)
    assert (d.status, d.stderr) == ('ok', ''), (
        "Failed DELETING pgp-secret: %s" % d.stderr)
    d = GPG.delete_keys(fingerprint)
    assert (d.status, d.stderr) == ('ok', ''), (
        "Failed DELETING pgp-public: %s" % d.stderr)


@contextlib.contextmanager
def _temp_master_key(vault, master_key):
    oldkey = vault.master_key
    vault.master_key = master_key
    try:
        yield vault
    finally:
        vault.master_key = oldkey


class TGnuPGSpecBinary(unittest.TestCase):
    def test_GPG_EXECUTABLE(self):
        from unittest.mock import patch

        with patch.dict('os.environ', {'GPG_EXECUTABLE': '/bad_path'}):  # @UndefinedVariable
            with self.assertRaisesRegex(OSError, 'Unable to run gpg - it may not be available.'):
                crypto.GnuPGSpec().GPG

            cfg = trtc.get_config()
            cfg.GnuPGSpec.gpgbinary = 'gpg'
            crypto.GnuPGSpec(config=cfg).GPG


@ddt.ddt
class TGnuPGSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = trtc.get_config()
        cfg.GnuPGSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        gpg_spec = crypto.GnuPGSpec(config=cfg)

        fingerprint = gpg_gen_key(
            gpg_spec.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')
        cfg.GnuPGSpec.master_key = fingerprint

    @classmethod
    def tearDownClass(cls):
        gpg_spec = crypto.GnuPGSpec(config=cls.cfg)
        assert gpg_spec.gnupghome
        gpg_del_key(gpg_spec.GPG, gpg_spec.master_key)
        shutil.rmtree(gpg_spec.gnupghome)

    def test_sign_verify(self):
        msg = 'Hi there'
        gpg_spec = crypto.GnuPGSpec(config=self.cfg)
        signed = gpg_spec.clearsign_text(msg)
        self.assertIsInstance(signed, str)

        verified = gpg_spec.GPG.verify(signed)
        print('\n'.join('%s = %s' % (k, v) for k, v in vars(verified).items()))
        self.assertTrue(verified.valid)

        import time
        time.sleep(1)  # Timestamp is the only differene.

        signed2 = gpg_spec.clearsign_text(msg)
        self.assertIsInstance(signed2, str)
        self.assertNotEqual(signed, signed2)

        verified2 = gpg_spec.GPG.verify(signed2)
        print('\n'.join('%s = %s' % (k, v) for k, v in vars(verified2).items()))

        self.assertEqual(verified.fingerprint, verified2.fingerprint)
        self.assertNotEqual(verified.signature_id, verified2.signature_id)


@ddt.ddt
class TVaultSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = trtc.get_config()
        cfg.VaultSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        vault = crypto.VaultSpec.instance(config=cfg)

        fingerprint = gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')
        vault.master_key = fingerprint

    @classmethod
    def tearDownClass(cls):
        vault = crypto.VaultSpec.instance()
        assert vault.gnupghome
        gpg_del_key(vault.GPG, vault.master_key)
        shutil.rmtree(vault.gnupghome)

    @ddt.idata(itt.product(('user', '&^a09|*(K}'), _objs))
    def test_1_dencrypt(self, case):
        pswdid, obj = case
        vault = crypto.VaultSpec.instance()

        ciphertext = vault.encryptobj('enc_test', obj)
        msg = ('CASE:', case, ciphertext)

        self.assertTrue(crypto.is_pgp_encrypted(ciphertext), msg)

        ## Check not generating indetical ciphers.
        #
        self.assertNotIn(ciphertext, _ciphertexts)
        _ciphertexts.add(ciphertext)

        plainbytes2 = vault.decryptobj(pswdid, ciphertext)
        self.assertEqual(obj, plainbytes2, msg)

    def test_2_many_master_keys(self):
        vault = crypto.VaultSpec.instance()
        fingerprint = gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user2',
            name_email='test2@test.com')
        try:
            with _temp_master_key(vault, None):
                with self.assertRaisesRegex(ValueError, 'Cannot guess master-key! Found 2 keys'):
                    vault.encryptobj('enc_test', b'')
        finally:
            gpg_del_key(vault.GPG, fingerprint)

    def test_3_no_master_key(self):
        vault = crypto.VaultSpec.instance()
        gpg_del_key(vault.GPG, vault.master_key)
        try:
            with _temp_master_key(vault, None):
                with self.assertRaisesRegex(ValueError, 'Cannot guess master-key! Found 0 keys'):
                    vault.encryptobj('enc_test', b'')
        finally:
            vault.master_key = gpg_gen_key(
                vault.GPG,
                key_length=1024,
                name_real='test user3',
                name_email='test2@test.com')

    def test_5_no_sec_key(self):
        vault = crypto.VaultSpec.instance()
        fingerprint = gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user2',
            name_email='test2@test.com')
        vault.GPG.delete_keys(fingerprint, secret=1)
        try:
            with _temp_master_key(vault, fingerprint):
                chiphered = vault.encryptobj('enc_test', b'foo')
                with self.assertRaisesRegex(ValueError, r"PswdId\('enc_test'\): decryption failed"):
                    vault.decryptobj('enc_test', chiphered)
        finally:
            vault.GPG.delete_keys(fingerprint, secret=0)


class TCipherTrait(unittest.TestCase):
    """See :class:`tests.sampling.test_baseapp`."""
