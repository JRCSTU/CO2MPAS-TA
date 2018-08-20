#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import contextlib
import io
import logging
from pprint import pprint as pp
import re
import shutil
import tempfile
import unittest

from co2dice.utils.logconfutils import init_logging
from co2dice._vendor.traitlets import config as trtc
from co2dice import crypto, CmdException
import ddt

import itertools as itt
import os.path as osp
import textwrap as tw

from . import test_pgp_keys, test_pgp_trust


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)
my_repo = osp.realpath(osp.join(mydir, '..', '..'))

_texts = ('', ' ', 'a' * 2048, '123', 'asdfasd|*(KJ|KL97GDk;')
_objs = ('', ' ', None, 'a' * 2048, 1244, b'\x22', {1: 'a', '2': {3, b'\x04'}})

test_pgp_key_id = 'CBBB52FF'
test_pgp_key = test_pgp_keys[0]

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


@ddt.ddt
class TFuncs(unittest.TestCase):

    @ddt.data(
        ((None, None), ''),
        (('', None), ''),
        (('', ''), ''),
        ((None, ''), ''),

        (('a', None), 'a'),
        (('a', 'b'), 'a: b'),
        ((None, 'b'), 'b'),
    )
    def test_uid_from_verdict(self, case):
        from toolz import dicttoolz as dtz

        inp, out = case

        class C:
            def __init__(self, d):
                self.__dict__ = d

        def _check(verdict):
            got = crypto.uid_from_verdict(verdict)
            self.assertEqual(got, out)

            got = crypto.uid_from_verdict(C(verdict))
            self.assertEqual(got, out)

            verdict = dtz.valfilter(lambda v: bool(v), verdict)

            got = crypto.uid_from_verdict(verdict)
            self.assertEqual(got, out)

            got = crypto.uid_from_verdict(C(verdict))
            self.assertEqual(got, out)

        _check(dict(zip(['key_id', 'username'], inp)))
        _check(dict(zip(['foo', 'key_id', 'username'], ('bar', ) + inp)))


class TGpgSpecBinary(unittest.TestCase):
    ## Separate class for no classSetUp/ClassTearDown methods.

    def test_GPG_EXECUTABLE(self):
        from unittest.mock import patch

        with patch.dict('os.environ',  # @UndefinedVariable
                        {'GNUPGEXE': '/bad_path'}):
            with self.assertRaisesRegex(
                    OSError,
                    r"Unable to run gpg \(/bad_path\) - it may not be available."):
                crypto.GpgSpec().GPG

            cfg = trtc.Config()
            cfg.GpgSpec.gnupgexe = 'gpg'
            with self.assertRaisesRegex(
                    OSError,
                    "Unable to run gpg \(/bad_path\) - it may not be available."):
                crypto.GpgSpec(config=cfg).GPG

        crypto.GpgSpec().GPG  # Ok.

    def test_GPGHOME(self):
        from unittest.mock import patch

        env_val = 'env_path'
        cfg_val = 'cfg_path'
        cfg = trtc.Config()
        cfg.GpgSpec.gnupghome = cfg_val
        with patch.dict('os.environ',  # @UndefinedVariable
                        {'GNUPGHOME': env_val}):
            self.assertEqual(crypto.GpgSpec().gnupghome, env_val)
            self.assertEqual(crypto.GpgSpec().gnupghome_resolved, env_val)

            cfg.GpgSpec.gnupghome = cfg_val
            self.assertEqual(crypto.GpgSpec(config=cfg).gnupghome,
                             env_val)
            self.assertEqual(crypto.GpgSpec(config=cfg).gnupghome_resolved,
                             env_val)

        with patch.dict('os.environ',  # @UndefinedVariable
                        clear=True):
            self.assertEqual(crypto.GpgSpec(config=cfg).gnupghome, cfg_val)
            self.assertEqual(crypto.GpgSpec(config=cfg).gnupghome_resolved, cfg_val)


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
        cls.cfg = cfg = trtc.Config()
        cfg.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.GpgSpec.keys_to_import = [test_pgp_key]
        cfg.GpgSpec.trust_to_import = test_pgp_trust
        gpg_spec = crypto.GpgSpec(config=cfg)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()   # @UndefinedVariable
        crypto.GitAuthSpec.clear_instance()       # @UndefinedVariable
        crypto.VaultSpec.clear_instance()         # @UndefinedVariable

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

        if isinstance(exp_msg, str):
            csig = crypto.pgp_split_clearsigned(clearsigned)
            self.assertIsInstance(csig, dict)
            self.assertEqual(len(csig), 4)
            self.assertEqual(csig['msg'], exp_msg)
            self.assertIsNotNone(csig['sigarmor'])
        else:
            with self.assertRaisesRegex(
                    ValueError,
                    "-line text is not a PGP-clear-sig!"):
                crypto.pgp_split_clearsigned(clearsigned)

        ## Check with \r\n at the end.
        #
        clearsigned = re.sub('$\n^', '\r\n', clearsigned, re.MULTILINE)
        if isinstance(exp_msg, str):
            csig = crypto.pgp_split_clearsigned(clearsigned)
            self.assertIsInstance(csig, dict)
            self.assertEqual(len(csig), 4)
            self.assertEqual(csig['msg'], re.sub('$\n^', '\r\n', exp_msg), re.MULTILINE)
            self.assertIsNotNone(csig['sigarmor'])
        else:
            with self.assertRaisesRegex(
                    ValueError,
                    "-line text is not a PGP-clear-sig!"):
                crypto.pgp_split_clearsigned(clearsigned)

    def test_parse_git_tag_unknown_pubkey(self):
        import os
        from unittest.mock import patch
        import git

        repo = git.Repo(my_repo)

        tagref = repo.tag('refs/tags/test_tag')
        tag = tagref.tag
        self.assertEqual(tag.hexsha, '0abf209dbf4c30370c1e2c7625f75a2aa0f0c9db')
        self.assertEqual(tagref.commit.hexsha, '76b8bf7312770a488eaeab4424d080dea3272435')

        bytes_sink = io.BytesIO()
        tag.stream_data(bytes_sink)
        tag_bytes = bytes_sink.getvalue()

        tag_csig = crypto.pgp_split_sig(tag_bytes)

        env = os.environ.copy()
        env.pop('GNUPGHOME', None)
        with patch.dict('os.environ',  # @UndefinedVariable
                        env, clear=True):  # Exclude any test-user's keys.
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

        repo = git.Repo(my_repo)

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
        cls.cfg = cfg = trtc.Config()
        cfg.VaultSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')

        ## Clean memories from past tests
        #
        crypto.VaultSpec.clear_instance()                   # @UndefinedVariable
        vault = crypto.VaultSpec.instance(config=cfg)       # @UndefinedVariable

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
        vault = crypto.VaultSpec.instance()       # @UndefinedVariable

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
        from unittest.mock import patch

        ## Check GNUPGKEY/master_key interaction.
        #
        env_val = 'some_value'
        cfg_val = 'some_value'
        cfg = trtc.Config()
        cfg.VaultSpec.master_key = cfg_val
        with patch.dict('os.environ',  # @UndefinedVariable
                        {'GNUPGKEY': env_val}):
            self.assertEqual(crypto.VaultSpec().master_key_resolved,
                             env_val)
            self.assertEqual(crypto.VaultSpec(config=cfg).master_key_resolved,
                             env_val)
        self.assertEqual(crypto.VaultSpec(config=cfg).master_key_resolved,
                         cfg_val)

        vault = crypto.VaultSpec.instance()  # @UndefinedVariable

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
        from unittest.mock import patch

        vault = crypto.VaultSpec.instance()  # @UndefinedVariable
        gpg_del_key(vault.GPG, vault.master_key)

        ## Check GNUPGKEY/master_key interaction.
        #
        env_val = 'some_value'
        cfg_val = 'some_value'
        cfg = trtc.Config()
        cfg.VaultSpec.master_key = cfg_val
        with patch.dict('os.environ',  # @UndefinedVariable
                        {'GNUPGKEY': env_val}):
            self.assertEqual(crypto.VaultSpec().master_key_resolved,
                             env_val)
            self.assertEqual(crypto.VaultSpec(config=cfg).master_key_resolved,
                             env_val)
        self.assertEqual(crypto.VaultSpec(config=cfg).master_key_resolved,
                         cfg_val)


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
        vault = crypto.VaultSpec.instance()  # @UndefinedVariable
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


class TestKey(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = trtc.Config()
        cfg.VaultSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.VaultSpec.keys_to_import = [test_pgp_key]
        cfg.GpgSpec.trust_to_import = test_pgp_trust

        ## Clean memories from past tests
        #
        crypto.VaultSpec.clear_instance()               # @UndefinedVariable
        vault = crypto.VaultSpec.instance(config=cfg)   # @UndefinedVariable

        cls.ok_key = gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.VaultSpec.gnupghome)

    def test_dencrypt(self):
        pswdid, obj = 'fooid', 'bar'
        vault = crypto.VaultSpec.instance()  # @UndefinedVariable

        vault.master_key = self.ok_key.fingerprint
        ciphertext = vault.encryptobj(pswdid, obj)
        msg = (obj, ciphertext)
        self.assertTrue(crypto.is_pgp_encrypted(ciphertext), msg)

        vault.master_key = test_pgp_key_id
        with self.assertRaisesRegex(CmdException, "After July 27 2017"):
            ciphertext = vault.encryptobj(pswdid, obj)

        vault.allow_test_key = True
        ciphertext = vault.encryptobj(pswdid, obj)
        msg = (obj, ciphertext)
        self.assertTrue(crypto.is_pgp_encrypted(ciphertext), msg)

        vault.allow_test_key = False
        with self.assertRaisesRegex(CmdException, "After July 27 2017"):
            vault.decryptobj(pswdid, ciphertext)

        vault.allow_test_key = True
        plainbytes2 = vault.decryptobj(pswdid, ciphertext)
        self.assertEqual(obj, plainbytes2, msg)

    def test_dencrypt_binary(self):
        vault = crypto.VaultSpec.instance()  # @UndefinedVariable

        pswdid = 'fooid'
        plain = b'bar'
        vault.master_key = self.ok_key.fingerprint
        ciphertext = vault.encryptobj(pswdid, plain, no_pickle=True, no_armor=True)
        plain2 = vault.decryptobj(pswdid, ciphertext, no_pickle=True)
        assert plain2 == plain


class TCipherTrait(unittest.TestCase):
    """See :class:`co2dice.test_baseapp`."""


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
