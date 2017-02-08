#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import crypto
import contextlib
import io
import logging
from pprint import pprint as pp
import re
import shutil
import tempfile
import unittest

import ddt

import itertools as itt
import os.path as osp
import textwrap as tw
import traitlets.config as trtc


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)
myproj = osp.join(mydir, '..', '..')

_texts = ('', ' ', 'a' * 2048, '123', 'asdfasd|*(KJ|KL97GDk;')
_objs = ('', ' ', None, 'a' * 2048, 1244, b'\x22', {1: 'a', '2': {3, b'\x04'}})

test_pgp_key = [tw.dedent(
    """
        -----BEGIN PGP PRIVATE KEY BLOCK-----

        lQHYBFiJ7q0BBAC7SXZw+XbnbT9QuB7mQUlMaw9NPtqj8iRRvJZOejV0PSY0F1Ub
        jNhLlmrBX+m4zoPtreEmjeGOa5uPDoqqiD1ft9kWf9Byr1Uq3L++NtDwIcetZzl1
        hHiG/wtY7kaWDZRgHXKMbf5TPjsFKyXS8lnyIRlD6nuU4xvMTzmiCdp4FQARAQAB
        AAP9Erl8SDzEvMwRG7igzDwEQEnm4H3zfZcotuxQKb3xqLKxZl1b0rKQ8HO0Liuw
        8hthmMp8224teh/7kECvr++JlSN8+EiXZ+DTffFdMRZKfAkB6uktfRNIuY98qH0h
        AUgRS0StXQEPgm3SzguzA+1TYTa2Khay8wVjIXCBU0M6EbkCAMO+T94VICGNlUXm
        mC/R1VdzO8o9XVFWhfGVUvVR0U4tzb5+izLSf4aau74OnICGqHnIQaEmU1DVit22
        ALQlgGkCAPTwvDKRkSfDAKhvOu5Flb7k0AsC5wdQMQrfs0m5lwXDhuojB2XB77zG
        ODwkhWIT46qGZZlvmcKPSQcOXnkpBM0B/ii7PsEWw7SNgVnRjOGeKpu/drpluUwa
        uT0B9x6sy+Fyx/IVZuNRsbG4Xetay7MeC+m7MaLAwe+ZezmgxVT4kn2gq7QwQ08y
        TVBBUyBUZXN0IDxzYW1wbGluZ0BjbzJtcGFzLmpyYy5lYy5ldXJvcGEuZXU+iL8E
        EwEIACkFAliJ7q0CGy8FCQDtTgAHCwkIBwMCAQYVCAIJCgsEFgIDAQIeAQIXgAAK
        CRCxJMmZy7tS/53UA/9G+m7bmn/HCKSRsH4fIkveq4jZRVmq1NEPmXm4pXCwROGK
        fRAw0pl/l+eGW3adMctTOxaX3lI/nz+g2QTEURgDHxLGaghDEGuy1VyjFFt0WXef
        2l5xTONpi4gs/G4M1+TY/MantEDRUJPh3EMgoEuT0H6gffhxsejI/YD1BH0RGQ==
        =AFgK
        -----END PGP PRIVATE KEY BLOCK-----
    """)
]
test_pgp_trust = tw.dedent("""\
    8922372A2983334307D7DA90FFBEC4A18C008403:4:
    """)

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

    return key


def gpg_del_key(GPG, fingerprint):
    log.debug('Deleting secret+pub: %s', fingerprint)
    d = GPG.delete_keys(fingerprint, secret=1)
    assert d.status == 'ok', (
        "Failed DELETING pgp-secret: %s" % d.stderr)
    d = GPG.delete_keys(fingerprint)
    assert d.status == 'ok', (
        "Failed DELETING pgp-public: %s" % d.stderr)


@contextlib.contextmanager
def _temp_master_key(vault, master_key):
    oldkey = vault.master_key
    vault.master_key = master_key
    try:
        yield vault
    finally:
        vault.master_key = oldkey


class TGpgSpecBinary(unittest.TestCase):
    ## Separate class for no classSetUp/ClassTearDown methods.

    def test_GPG_EXECUTABLE(self):
        from unittest.mock import patch
        import importlib

        with patch.dict('os.environ', {'GNUPGEXE': '/bad_path'}):  # @UndefinedVariable
            ## No-dynamic trait-defaults are loaded on import time...
            importlib.reload(crypto)

            with self.assertRaisesRegex(OSError, 'Unable to run gpg - it may not be available.'):
                crypto.GpgSpec().GPG

            cfg = trtc.get_config()
            cfg.GpgSpec.gnupgexe = 'gpg'
            crypto.GpgSpec(config=cfg).GPG

        ## Restore default.
        importlib.reload(crypto)
#

_clearsigned_msgs = [
    # MSG,     CLEARSIGNED
    ('hi gpg', tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----

        hi gpg
        -----BEGIN PGP SIGNATURE-----
        Version: GnuPG v2

        BAG SIGnature but valid format
        -----END PGP SIGNATURE-----""")),
    ('', tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----


        -----BEGIN PGP SIGNATURE-----

        BAG SIGnature but valid format
        -----END PGP SIGNATURE-----
    """)),
    ('One\r\n-TWO\r\n--abc\r\nTHREE', tw.dedent("""
        Blah Blah

        -----BEGIN PGP SIGNED MESSAGE-----
        Hash: SHA256

        One
        - -TWO\r
        --abc
        - THREE
        -----BEGIN PGP SIGNATURE-----
        Version: GnuPG v2

        =ctnm
        -----END PGP SIGNATURE-----
    """)),
    ('hi gpg\r\nLL\r\n', tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----

        hi gpg\r
        LL

        -----BEGIN PGP SIGNATURE-----

        =ctnm
        -----END PGP SIGNATURE-----

        Blqah Bklah
    """)),
    (None, tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----
        Hash: asfdf
        Hash

        hi gpg
        -----BEGIN PGP SIGNATURE-----

        =ctnm
        -----END PGP SIGNATURE-----
    """)),
    (None, tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----
        Hash:

        hi gpg
        -----BEGIN PGP SIGNATURE-----

        =ctnm
        -----END PGP SIGNATURE-----
    """)),
    (None, tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----

        hi gpg
        -----BEGIN PGP SIGNATURE-----

        -----END PGP SIGNATURE-----
    """)),
    (None, tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----

        -----BEGIN PGP SIGNATURE-----

        BAG SIG, no plaintext
        -----END PGP SIGNATURE-----
    """)),
    (None, tw.dedent("""
        -----BEGIN PGP SIGNED MESSAGE-----

        No0 SIG empty-line
        -----BEGIN PGP SIGNATURE-----
        BAG SIG
        BAG SIG
        -----END PGP SIGNATURE-----
    """)),
]


_signed_tag_old = tw.dedent("""\
        object 76b8bf7312770a488eaeab4424d080dea3272435
        type commit
        tag test_tag
        tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1485272439 +0100

        - Is bytes (utf-8 encodable);
        - all lines end with LF, and any trailing whitespace truncated;
        - any line can start with dashes;
        - any empty lines at the bottom are truncated,
        - apart from the last LF, which IS part of the msg.
        -----BEGIN PGP SIGNATURE-----
        Version: GnuPG v2

        iJwEAAEIAAYFAliHdXwACgkQ/77EoYwAhAMxDgQAhlqOjb0bHGxLcyYIpFg9kEmp
        4poL5eA7cdmq3eU1jXTfb5UXJV6BnP+DUsJ4TG+7KoUimgli0djG7ZisRvNYBWGD
        PNO2X5LqNx7tzgj/fQT5CzWcWMXfjUd337pfoj3K3kDroCNl7oQl/bSIR46z9l/3
        JS/kbngOONtzIkPbQvU=
        =bEkN
        -----END PGP SIGNATURE-----
    """).encode('utf-8')

_signed_tag = tw.dedent("""\
        object 3334bcde283480883f2fb209efcf84ae24da8335
        type commit
        tag tests/signed_by_CBBB52FF
        tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1485442847 +0100

        Test-tag for crypto TCs, signed by:
          CO2MPAS Test <sampling@co2mpas.jrc.ec.europa.eu>  1024R/B124C999CBBB52FF 2017-01-26 [expires: 2017-07-25]
        -----BEGIN PGP SIGNATURE-----
        Version: GnuPG v2

        iJwEAAEIAAYFAliKDx8ACgkQsSTJmcu7Uv9HsAP+KmK4+cSXvScwg5UHDq7VVj1B
        XjtEHZp6VwKndmMCQNIOsyR3F7o5qsleU2NympSVxQyOTL0WlFaJqdNMSLwqV/px
        oWZdPlYCw6lc1BFjRkYF5YVCb6E7dJG6WbUJTVys5lt3AIIN3l1WuO2JlhmXvubN
        021zAo8TJIn1aFQEkVw=
        =nxOG
        -----END PGP SIGNATURE-----
    """).encode('utf-8')


_splitted_signed_tag = [
    tw.dedent("""\
        object 3334bcde283480883f2fb209efcf84ae24da8335
        type commit
        tag tests/signed_by_CBBB52FF
        tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1485442847 +0100

        Test-tag for crypto TCs, signed by:
          CO2MPAS Test <sampling@co2mpas.jrc.ec.europa.eu>  1024R/B124C999CBBB52FF 2017-01-26 [expires: 2017-07-25]
        """).encode('utf-8'),
    tw.dedent("""\
        -----BEGIN PGP SIGNATURE-----
        Version: GnuPG v2

        iJwEAAEIAAYFAliKDx8ACgkQsSTJmcu7Uv9HsAP+KmK4+cSXvScwg5UHDq7VVj1B
        XjtEHZp6VwKndmMCQNIOsyR3F7o5qsleU2NympSVxQyOTL0WlFaJqdNMSLwqV/px
        oWZdPlYCw6lc1BFjRkYF5YVCb6E7dJG6WbUJTVys5lt3AIIN3l1WuO2JlhmXvubN
        021zAo8TJIn1aFQEkVw=
        =nxOG
        -----END PGP SIGNATURE-----
        """).encode('utf-8')
]


@ddt.ddt
class TGpgSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = trtc.get_config()
        cfg.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.GpgSpec.keys_to_import = test_pgp_key
        cfg.GpgSpec.trust_to_import = test_pgp_trust
        gpg_spec = crypto.GpgSpec(config=cfg)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

        key = gpg_gen_key(
            gpg_spec.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')
        cfg.GpgSpec.master_key = key.fingerprint

#    def test_verify_clearsigned(self):
#         return verified.valid
#             signature_id = IWLTrxduQKe1P7qGAUauyyNSpJ4
#             trust_text = TRUST_ULTIMATE
#             valid = True
#             key_status = None
#             expire_timestamp = 0
#             key_id = D720C846A2891883
#             trust_level = 4
#             stderr = [GNUPG:] NEWSIG
#                 gpg: Signature made 01/22/17 02:37:10 W. Europe Standard Time using RSA key ID A2891883
#                 [GNUPG:] SIG_ID IWLTrxduQKe1P7qGAUauyyNSpJ4 2017-01-22 1485049030
#                 gpg: checking the trustdb
#                 gpg: 3 marginal(s) needed, 1 complete(s) needed, PGP trust model
#                 gpg: depth: 0  valid:   1  signed:   0  trust: 0-, 0q, 0n, 0m, 0f, 1u
#                 [GNUPG:] GOODSIG D720C846A2891883 test user <test@test.com>
#                 gpg: Good signature from "test user <test@test.com>" [ultimate]
#                 [GNUPG:] VALIDSIG C0DE766CF516CB3CE2DDE616D720C846A2891883 2017-01-22 1
#                                485049030 0 4 0 1 8 01 C0DE766CF516CB3CE2DDE616D720C846A2891883
#                 [GNUPG:] TRUST_ULTIMATE
#             timestamp = 1485049030
#             data = b''
#             gpg = <gnupg.GPG object at 0x0000028DB4B1BBA8>
#             username = test user <test@test.com>
#             sig_timestamp = 1485049030
#             fingerprint = C0DE766CF516CB3CE2DDE616D720C846A2891883
#             status = signature valid
#             pubkey_fingerprint = C0DE766CF516CB3CE2DDE616D720C846A2891883
#             creation_date = 2017-01-22

    @classmethod
    def tearDownClass(cls):
        gpg_spec = crypto.GpgSpec(config=cls.cfg)
        assert gpg_spec.gnupghome
        shutil.rmtree(gpg_spec.gnupghome)

    @ddt.data(*_clearsigned_msgs)
    def test_parse_clearsigned(self, case):
        exp_msg, clearsigned = case

        csig = crypto.pgp_split_clearsigned(clearsigned)
        if isinstance(exp_msg, str):
            self.assertIsInstance(csig, dict)
            self.assertEqual(len(csig), 4)
            self.assertEqual(csig['msg'], exp_msg)
            self.assertIsNotNone(csig['sigarmor'])
        else:
            self.assertIsNone(csig)

        ## Check with \r\n at the end.
        #
        clearsigned = re.sub('$\n^', '\r\n', clearsigned, re.MULTILINE)
        csig = crypto.pgp_split_clearsigned(clearsigned)
        if isinstance(exp_msg, str):
            self.assertIsInstance(csig, dict)
            self.assertEqual(len(csig), 4)
            self.assertEqual(csig['msg'], re.sub('$\n^', '\r\n', exp_msg), re.MULTILINE)
            self.assertIsNotNone(csig['sigarmor'])
        else:
            self.assertIsNone(csig)

    def test_parse_git_tag_unknown_pubkey(self):
        import git

        repo = git.Repo(myproj)

        tagref = repo.tag('refs/tags/test_tag')
        tag = tagref.tag
        self.assertEqual(tag.hexsha, '0abf209dbf4c30370c1e2c7625f75a2aa0f0c9db')
        self.assertEqual(tagref.commit.hexsha, '76b8bf7312770a488eaeab4424d080dea3272435')

        bytes_sink = io.BytesIO()
        tag.stream_data(bytes_sink)
        tag_bytes = bytes_sink.getvalue()

        tag_csig = crypto.pgp_split_sig(tag_bytes)

        gpg_spec = crypto.GpgSpec(config=self.cfg)
        ver = gpg_spec.verify_detached(tag_csig['sigarmor'], tag_csig['msg'])
        verdict = vars(ver)
        pp(verdict)
        self.assertDictContainsSubset(
            {
                'valid': False,
                'status': 'no public key',
                'key_id': 'FFBEC4A18C008403',
                'key_status': None,
                'timestamp': '1485272444',
            },
            verdict)

    def test_parse_git_tag_ok(self):
        import git

        repo = git.Repo(myproj)

        tagref = repo.tag('refs/tags/tests/signed_by_CBBB52FF')
        tag = tagref.tag
        self.assertEqual(tag.hexsha, '1e28f8ffc717f407ff38d93d4cbad4e7a280d063')
        self.assertEqual(tagref.commit.hexsha, '3334bcde283480883f2fb209efcf84ae24da8335')

        bytes_sink = io.BytesIO()
        tag.stream_data(bytes_sink)
        tag_bytes = bytes_sink.getvalue()
        self.assertEqual(tag_bytes, _signed_tag)

        csig = crypto.pgp_split_sig(tag_bytes)
        self.assertEqual(csig['msg'], _splitted_signed_tag[0])
        self.assertEqual(csig['sigarmor'], _splitted_signed_tag[1])

        gpg_spec = crypto.GpgSpec(config=self.cfg)
        ver = gpg_spec.verify_detached(csig['sigarmor'], csig['msg'])
        pp(vars(ver))
        self.assertTrue(ver)

    def test_clearsign_verify(self):
        msg = 'Hi there'
        gpg_spec = crypto.GpgSpec(config=self.cfg)
        signed = gpg_spec.clearsign_text(msg)
        self.assertIsInstance(signed, str)

        verified = gpg_spec.verify_clearsigned(signed)
        print('\n'.join('%s = %s' % (k, v) for k, v in vars(verified).items()))
        self.assertTrue(verified.valid)

        import time
        time.sleep(1)  # Timestamp is the only differene.

        signed2 = gpg_spec.clearsign_text(msg)
        self.assertIsInstance(signed2, str)
        self.assertNotEqual(signed, signed2)

        verified2 = gpg_spec.verify_clearsigned(signed2)
        print('\n'.join('%s = %s' % (k, v) for k, v in vars(verified2).items()))

        self.assertEqual(verified.fingerprint, verified2.fingerprint)
        self.assertNotEqual(verified.signature_id, verified2.signature_id)

        ## Check parsing.
        #
        csig = crypto.pgp_split_clearsigned(signed2)
        self.assertIsInstance(csig, dict)
        self.assertEqual(csig['msg'], msg)
        self.assertIsNotNone(csig['sigarmor'])


_ciphertexts = set()


@ddt.ddt
class TVaultSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = trtc.get_config()
        cfg.VaultSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')

        ## Clean memories from past tests
        #
        crypto.VaultSpec.clear_instance()
        vault = crypto.VaultSpec.instance(config=cfg)

        key = gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')
        vault.master_key = key.fingerprint

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.VaultSpec.gnupghome)

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
        key = gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user2',
            name_email='test2@test.com')
        try:
            with _temp_master_key(vault, None):
                with self.assertRaisesRegex(
                        ValueError,
                        'Cannot guess master-key! Found 2 keys') as exmsg:
                    vault.encryptobj('enc_test', b'')
                self.assertIn(key.fingerprint[-8:], str(exmsg.exception))
        finally:
            gpg_del_key(vault.GPG, key.fingerprint)

    def test_3_no_master_key(self):
        vault = crypto.VaultSpec.instance()
        gpg_del_key(vault.GPG, vault.master_key)
        try:
            with _temp_master_key(vault, None):
                with self.assertRaisesRegex(
                        ValueError,
                        'Cannot guess master-key! Found 0 keys'):
                    vault.encryptobj('enc_test', b'')
        finally:
            vault.master_key = gpg_gen_key(
                vault.GPG,
                key_length=1024,
                name_real='test user3',
                name_email='test2@test.com').fingerprint

    def test_5_no_sec_key(self):
        vault = crypto.VaultSpec.instance()
        key = gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user2',
            name_email='test2@test.com')
        vault.GPG.delete_keys(key.fingerprint, secret=1)
        try:
            with _temp_master_key(vault, key.fingerprint):
                chiphered = vault.encryptobj('enc_test', b'foo')
                with self.assertRaisesRegex(
                        ValueError,
                        r"PswdId\('enc_test'\): decryption failed"):
                    vault.decryptobj('enc_test', chiphered)
        finally:
            vault.GPG.delete_keys(key.fingerprint, secret=0)


class TCipherTrait(unittest.TestCase):
    """See :class:`tests.sampling.test_baseapp`."""


#class TDice(unittest.TestCase):
#
#    def gpg_del_gened_key(self, gpg, fingerprint):
#        log.debug('Deleting secret+pub: %s', fingerprint)
#        d = gpg.delete_keys(fingerprint, secret=1)
#        assert d.status == 'ok', (
#            "Failed DELETING pgp-secret: %s" % d.stderr)
#        d = gpg.delete_keys(fingerprint)
#        assert d.status == 'ok', (
#            "Failed DELETING pgp-secret: %s" % d.stderr)
#
##    def _has_repeatitive_prefix(self, word, limit, char=None):
##        c = word[0]
##        if not char or c == char:
##            for i  in range(1, limit):
##                if word[i] != c:
##                    break
##            else:
##                return True
##
#
#    _repeatitive_regex3 = re.compile(r'(?P<l>.)(?P=l){2,}')
#    _repeatitive_regex4 = re.compile(r'(?P<l>.)(?P=l){3,}')
#
#    def _is_vanity_keyid(self, keyid):
#        """Search repeatitive letters"""
#        m1 = self._repeatitive_regex3.search(keyid[:8])
#        m2 = self._repeatitive_regex4.search(keyid[8:])
#        return m1 and m2
#
#    def gpg_gen_interesting_keys(self, gpg, nkeys=1, runs=0, **key_data):
#        predicate = self._is_vanity_keyid
#        keys = []
#        for i in itt.count(1):
#            del_key = True
#            key = gpg.gen_key(gpg.gen_key_input(**key_data))
#            try:
#                log.debug('Created-%i: %s', i, key.fingerprint)
#                if predicate(key.fingerprint[24:]):
#                    del_key = False
#                    keys.append(key.fingerprint)
#                    keyid = key.fingerprint[24:]
#                    log.info('FOUND-%i: %s-->%s', i, keyid, key.fingerprint)
#                    nkeys -= 1
#                    if nkeys == 0:
#                        break
#            finally:
#                if del_key:
#                    self.gpg_del_gened_key(gpg, key.fingerprint)
#            if runs > 0 and i >= runs:
#                break
#
#        return keys
#
#    #@unittest.skip('Enabled it to generate test-keys!!')
#    def test_gen_key_proof_of_work(self):
#        import os
#        import gnupg
#
#        gnupghome = 'temp.gpg'
#        try:
#            os.mkdir(gnupghome)
#        except FileExistsError:
#            pass
#
#        gpg = gnupg.GPG(gnupghome=gnupghome)
#
#        self.gpg_gen_interesting_keys(
#            gpg, runs=0,
#            name_real='CO2MPAS Test',
#            name_email='sampling@co2mpas.jrc.ec.europa.eu',
#            key_length=1024,
#            expire_date='6m',
#        )
