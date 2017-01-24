#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
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

test_pgp_key = tw.dedent(
    """
    -----BEGIN PGP PRIVATE KEY BLOCK-----
    Version: GnuPG v2

    lQHYBFbirf0BBADdjkrRSJ+ua9RvlzZsrpr4tAvj6+rK9eRwuue+jaIu5JucIvV5
    nq5AlsQGACTy88qsnYcbqqro15AURdZ3fS2uy4yGvcNzMNP0w6Jh+DPZM9ubZXXQ
    xsejMltjynHe3gJNeBy5D7xGOURv1sAaHoCxYw4zC/SjkbnNqpvMZNmvTwARAQAB
    AAP/UyULsdepyUjBK/GY5JdwJAZZcfr+nZVC2gViY9H+O8/iD+nUqoQgy69ouAHE
    3AIenMHvSrQ1OHVxJhKBZk0tX3lUZ60W8yBY5GydWu0ylHeTXzU2tgCqMDi36TKE
    w/p1jx5nkJ6qwfXtMTWLsaWOllRaGOpOv+pR3yEkRmTQe20CAOvbwxax33bFDXJM
    WSL0GEdTXDM/M5XdoFoV+M9wnx4eJj4yPnL0SkpWgkq38FtwrINdxg46OfOw7hml
    hlKXz80CAPB52jgRTImm2VAENdMa0O+yu5YDmRyyuyJWyoYdNQlbOrzraQjiJTfQ
    zsv9bfvpxeFS+PRXyAenwRV3qQryR4sCALAvfd068PZkg4K3+TP5F45fFBTfnHqR
    eCeUX/uU4CYzh+LcM5E89IQ45O+FIwUgyvVx8kyUe79vwlzbJXnf/1mjlrQxU2Ft
    cGxpbmcgVGVzdCA8c2FtcGxpbmdAY28ybXBhcy5qcmMuZWMuZXVyb3BhLmV1Poi5
    BBMBCAAjBQJW4q39AhsvBwsJCAcDAgEGFQgCCQoLBBYCAwECHgECF4AACgkQ/77E
    oYwAhAN16QQAp79/wkUQNqXZTWhY5ji/jazisK4ggaJcWIvpsrbiwL1wJ5ZbbY/9
    gHXslyDWhyKK/KbgAdAurAXgekRDLTXxR1c+Q0hm0phrmyhIENRbwF04T2S4JExQ
    PiKw+6RfuxfTV22f6sRSkwuK4MP9veZ/0u6Mid6kRHJFo+mVus8XuZA=
    =clrj
    -----END PGP PRIVATE KEY BLOCK-----
    """)
test_pgp_trust = tw.dedent("""\
    # List of assigned trustvalues, created 01/24/17 18:16:11 Eur
    # (Use "gpg --import-ownertrust" to restore them)
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
    ## Separate class for no classSetUp/ClassTearDown methods.

    def test_GPG_EXECUTABLE(self):
        from unittest.mock import patch

        with patch.dict('os.environ', {'GNUPGEXE': '/bad_path'}):  # @UndefinedVariable
            with self.assertRaisesRegex(OSError, 'Unable to run gpg - it may not be available.'):
                crypto.GnuPGSpec().GPG

            cfg = trtc.get_config()
            cfg.GnuPGSpec.gnupgexe = 'gpg'
            crypto.GnuPGSpec(config=cfg).GPG
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


_signed_tag = tw.dedent("""\
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


_splitted_signed_tag = [
    tw.dedent("""\
        object 76b8bf7312770a488eaeab4424d080dea3272435
        type commit
        tag test_tag
        tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1485272439 +0100

        - Is bytes (utf-8 encodable);
        - all lines end with LF, and any trailing whitespace truncated;
        - any line can start with dashes;
        - any empty lines at the bottom are truncated,
        - apart from the last LF, which IS part of the msg.
        """).encode('utf-8'),
    tw.dedent("""\
        -----BEGIN PGP SIGNATURE-----
        Version: GnuPG v2

        iJwEAAEIAAYFAliHdXwACgkQ/77EoYwAhAMxDgQAhlqOjb0bHGxLcyYIpFg9kEmp
        4poL5eA7cdmq3eU1jXTfb5UXJV6BnP+DUsJ4TG+7KoUimgli0djG7ZisRvNYBWGD
        PNO2X5LqNx7tzgj/fQT5CzWcWMXfjUd337pfoj3K3kDroCNl7oQl/bSIR46z9l/3
        JS/kbngOONtzIkPbQvU=
        =bEkN
        -----END PGP SIGNATURE-----
        """).encode('utf-8')
]


@ddt.ddt
class TGnuPGSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = trtc.get_config()
        cfg.GnuPGSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.GnuPGSpec.keys_to_import = test_pgp_key
        cfg.GnuPGSpec.trust_to_import = test_pgp_trust
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

    @ddt.data(*_clearsigned_msgs)
    def test_parse_clearsigned(self, case):
        exp_msg, clearsigned = case

        groups = crypto.split_clearsigned(clearsigned)
        if isinstance(exp_msg, str):
            self.assertIsInstance(groups, dict)
            self.assertEqual(groups['msg'], exp_msg)
            self.assertIsNotNone(groups['sig'])
        else:
            self.assertIsNone(groups)

        ## Check with \r\n at the end.
        #
        clearsigned = re.sub('$\n^', '\r\n', clearsigned, re.MULTILINE)
        groups = crypto.split_clearsigned(clearsigned)
        if isinstance(exp_msg, str):
            self.assertIsInstance(groups, dict)
            self.assertEqual(groups['msg'], re.sub('$\n^', '\r\n', exp_msg), re.MULTILINE)
            self.assertIsNotNone(groups['sig'])
        else:
            self.assertIsNone(groups)

    def test_parse_git_tags(self):
        import git

        repo = git.Repo(myproj)

        tagref = repo.tag('refs/tags/test_tag')
        tag = tagref.tag
        self.assertEqual(tag.hexsha, '0abf209dbf4c30370c1e2c7625f75a2aa0f0c9db')
        self.assertEqual(tagref.commit.hexsha, '76b8bf7312770a488eaeab4424d080dea3272435')

        bytes_sink = io.BytesIO()
        tag.stream_data(bytes_sink)
        tag_bytes = bytes_sink.getvalue()
        self.assertEqual(tag_bytes, _signed_tag)

        res = crypto.split_git_signed(tag_bytes)
        self.assertEqual(len(res), 2)
        msg, sig = res  # encode(sys.getdefaultencoding())
        print(msg,)
        print(_splitted_signed_tag[0])
        self.assertEqual(msg, _splitted_signed_tag[0])
        self.assertEqual(sig, _splitted_signed_tag[1])

        gpg_spec = crypto.GnuPGSpec(config=self.cfg)
        ver = gpg_spec.verify_detached(sig, msg)
        pp(vars(ver))
        self.assertTrue(ver)

    def test_clearsign_verify(self):
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

        ## Check parsing.
        #
        groups = crypto.split_clearsigned(signed2)
        self.assertIsInstance(groups, dict)
        self.assertEqual(groups['msg'], msg)
        self.assertIsNotNone(groups['sig'])


_ciphertexts = set()


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
