#!/usr/bin/env python
#
# Copyright 2014-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""
PGP-dencrypt passwords so as to be stored in text-files.

The general idea is to use a PGP key to securely store many passwords in configuration files.
The code using these passwords must never store them as is, but use and immediately discard them.
"""

from co2mpas.sampling import baseapp
from collections import namedtuple
import io
import os
import re
from typing import Text, Tuple, Union, Dict  # @UnusedImport

import os.path as osp
import traitlets as trt
import traitlets.config as trtc


_pgp_regex = re.compile(r'^\s*-----[A-Z ]*PGP[A-Z ]*-----.+-----[A-Z ]*PGP[A-Z ]*-----\s*$', re.DOTALL)


def is_pgp_encrypted(obj) -> bool:
    return bool(isinstance(obj, str) and _pgp_regex.match(obj))


_pgp_clearsig_regex = re.compile(
    r"""
    (?P<armor>
        ^-{5}BEGIN\ PGP\ SIGNED\ MESSAGE-{5}\r?\n
        (?:^Hash:\ [^\r\n]+\r?\n)*                   ## 'Hash:'-header(s)
        ^\r?\n                                       ## blank-line
        (?P<msg>^.*?)
        \r?\n                                        ## NOT part of plaintext!
        (?P<sig>
            ^-{5}BEGIN\ PGP\ SIGNATURE-{5}\r?\n
            (?:
              .*
              (?:^Comment:\ Stamper\ Reference\ Id:\ (?P<stamper>\d+)\r?\n)?
              .*
            )?
            ^\r?\n                                   ## blank-line
            [^-]+                                    ## sig-body
            ^-{5}END\ PGP\ SIGNATURE-{5}(?:\r?\n)?
        )
    )
    """,
    re.DOTALL | re.VERBOSE | re.MULTILINE)
_pgp_clearsig_dedashify_regex = re.compile(r'^- ', re.MULTILINE)
_pgp_clearsig_eol_canonical_regex = re.compile(r'[ \t\r]*\n')


Clearsigned = namedtuple('Clearsigned', 'armor msg sig stamper')


def split_clearsigned(text: str) -> Dict:
    """
    Parses text RFC 4880 PGP-signed message with ``gpg --clearsing`` command.

    Clear-signed messages are like that::

        -----BEGIN PGP SIGNED MESSAGE-----
        Hash: MD5                                ## Optional

        - --The\r\n
        Message\r\n
        body\r\n
        \r\n
        \r\n                                    ## This last newline chars ARE NOT part of `msg`.
        -----BEGIN PGP SIGNATURE-----
        Header: value

        InvalidSig/gdgdfgdggdf2dgdfg9g8g97gfggdfg6sdf3qw
        2dgdfg9g8g97gfggdfg6sdfipowqoerifkl&9
        pi23o5890tuao=
        -----END PGP SIGNATURE-----

    Specifically for the message note that:
      - All message lines, except the last one, end with CRLF;
      - any trailing whitespace from all lines are removed;
      - any line starting with `'-'` is prepended with `'- '`.

    :return:
        a dict with keys:
        - `armor`: the whole armored text, plaintext + sig included.
        - `msg`: the plaintext de-dashified, rfc4880 clear-sign-normalized
          CRLF everywhere apart without any eol at the very end.
        - `sig`: the armored signature
        - `stamper`: the id in case it was signed by stamper

    .. seealso::
        - https://tools.ietf.org/html/rfc4880#page-59
        - http://gnupg.10057.n7.nabble.com/splitting-up-an-inline-signed-OpenPGP-message-td48681.html#a48715
    """
    m = _pgp_clearsig_regex.search(text)
    if m:
        groups = m.groupdict()
        msg = groups['msg']
        msg = _pgp_clearsig_dedashify_regex.sub('', msg)
        msg = _pgp_clearsig_eol_canonical_regex.sub('\r\n', msg)
        groups['msg'] = msg

        return Clearsigned(**groups)


_pgp_signature_banner = b'-----BEGIN PGP SIGNATURE-----'


def split_git_signed(tag_content: bytes) -> (bytes, bytes):
    """
    Split PGP-signed Git objects (tags & commits) into plaintext & armor signature.

    Git-signed descriptions are bytes like that::

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

    Differences of Git-detach (vs Clearsign):
      - bytes vs string;
      - no msg-banner (only sig-banner), msg start from the very top;
      - LF instead of CRLF;
      - last msg newline preserved;
      - no de-dashification.

    In both formats, anything above or below armor is ignored (including newlines).


    :param tag_content:
            As fetched from ``git cat-file tag v1.2.1``, using LF and utf-8 encodable.
    :return:
            A 2-tuple(msg, sig), None if no sig found.

    See: https://lists.gnupg.org/pipermail/gnupg-users/2014-August/050780.html
    """
    nl = b'\n'
    lines = tag_content.split(nl)
    for i, l in enumerate(lines):
        if l.startswith(_pgp_signature_banner):
            return nl.join(lines[:i]) + nl, nl.join(lines[i:])


class GnuPGSpec(baseapp.Spec):
    """
    Configurable parameters for instantiating a GnuPG instance

    Class-parameters override values the following environment variables (if exist):
    - `GnuPGSpec.gnupgexe   --> GNUPGEXE`
    - `GnuPGSpec.gnupghome  --> GNUPGHOME`
    - `GnuPGSpec.master_key --> GPGKEY`

    """

    gnupgexe = trt.Unicode(
        None, allow_none=True,
        help="The path to GnuPG executable; if None, first `gpg` in PATH used, "
        "unless `GNUPGEXE`(%s) env-variable is set." % os.environ.get('GNUPGEXE')
    ).tag(config=True)

    gnupghome = trt.Unicode(
        None, allow_none=True,
        help="""
        The full pathname to the folder containing the public and private PGP-keyrings.

        If None, the executable decides:
          - POSIX:   %s/.gpg
          - Windows: %s\\GnuPG,
        unless the `GNUPGHOME`(%s) env-variable is set.
        """ % (os.environ.get('HOME', '~'),
               os.environ.get('APPDATA', '%APPDATA%'),
               os.environ.get('GNUPGHOME'))
    ).tag(config=True)

    keyring = trt.Unicode(
        None, allow_none=True,
        help="""
        The file-name of alternative keyring file to use, or TODO: list of such keyrings.
        If specified, the default keyring is not used.."""
    ).tag(config=True)

    secret_keyring = trt.Unicode(
        None, allow_none=True,
        help="""The file-name of alternative secret keyring file to use, or TODO: list of such keyrings.."""
    ).tag(config=True)

    gnupgoptions = trt.List(
        trt.Unicode(None, allow_none=False),
        default_value=['--allow-weak-digest-algos'],  # Timestamp's algos use MD5!
        allow_none=True,
        help="""A list of additional cmd-line options to pass to the GPG binary."""
    ).tag(config=True)

    master_key = trt.Unicode(
        None, allow_none=True,
        help="""
        The key-id (or recipient) of a secret PGP key to use for various crytpo operations.

        Usage in subclasses:
          - VaultSpec:         dencrypt 3rdp passwords
          - TstampSenderSpec:  sign email to timestamp service

        You MUST set either this configurable option or `GPGKEY`(%s) env-variable, if you have
        If you have more than one private keys in your PGP-keyring, or else
        the application will fail to start when any of the usages above is initiated.
        """ % os.environ.get('GPGKEY')
    ).tag(config=True)

    keys_to_import = trt.List(
        trt.Unicode(
            None, allow_none=True,
        ),
        help="""
        Armored text of keys (pub/sec) to import.

        Use and one of these commands:
            gpg --export-secret-keys <key-id-1> ..
            gpg --export-keys <key-id-1> ..
        """
    ).tag(config=True)

    #:
    trust_to_import = trt.Unicode(
        None, allow_none=True,
        help="The text of ``gpg --export-owner-trust`` to import."
    ).tag(config=True)

    @trt.observe('gnupgexe', 'gnupghome', 'keyring', 'secret_keyring', 'options',
                 'keys_to_import', 'trust_to_import')
    def _remove_cached_GPG(self, change):
        self._GPG = None

    def _guess_master_key(self) -> Text:
        master_key = self.master_key or os.environ.get('GPGKEY')
        if not master_key:
            GPG = self.GPG
            seckeys = GPG.list_keys(secret=True)
            nseckeys = len(seckeys)
            if nseckeys != 1:
                raise ValueError("Cannot guess master-key! Found %d keys in secret keyring."
                                 "\n  Please set the `VaultSpec.master_key` config-param or `GPGKEY` env-var." %
                                 nseckeys)

            master_key = seckeys[0]['fingerprint']

        return master_key

    @property
    def gnupgexe_resolved(self):
        """Used for printing configurations."""
        import shutil

        gpg_exepath = self.gnupgexe or os.environ.get('GNUPGEXE', 'gpg')
        gpg_exepath = shutil.which(gpg_exepath) or gpg_exepath
        return gpg_exepath

    @property
    def gnupghome_resolved(self):
        """Used for printing configurations."""
        gnupghome = self.gnupghome
        if not gnupghome:
            if os.name == 'nt':
                gnupghome = '%s\\GnuPG' % os.environ.get('APPDATA', '%APPDATA%')
            else:
                gnupghome = '%s/.gnupg' % os.environ.get('HOME', '~')
        return gnupghome

    def _import_keys_and_trust(self, GPG, key_blocks, trust_text):
        """
        Load in GPG-keyring from :attr:`GnuPGSpec.keys_to_import` and :attr:`GnuPGSpec.trust_to_import`.

        :param GPG:
            Given to avoid inf recursion.
        """
        import gnupg

        log = self.log

        def import_trust(trust_text):
            ## Remember to submit to *gnupg* project.
            class NoResult(object):
                def handle_status(self, key, value):
                    pass

            log.debug('--import-owner input: %r', trust_text[:256])
            data = gnupg._make_binary_stream(trust_text, GPG.encoding)
            result = NoResult()
            GPG._handle_io(['--import-ownertrust'], data, result, binary=True)
            data.close()

            result = result.stderr
            return result if ' error' in result else 'ok'

        ## TODO: Fail if not imported keys/trust!
        keys_res = []
        for armor_key in key_blocks:
            keys_res.append(GPG.import_keys(armor_key))
        if trust_text:
            trust_res = import_trust(trust_text)

        key_summaries = [k.summary() for k in keys_res]
        log.info('Import: Keys: %s, Trust: %s', key_summaries, trust_res)

    @property
    def GPG(self) -> 'gnupg.GPG':
        import gnupg
        GPG = getattr(self, '_GPG', None)
        if not GPG:
            gnupgexe = self.gnupgexe or os.environ.get('GNUPGEXE', 'gpg')
            GPG = self._GPG = gnupg.GPG(
                gpgbinary=gnupgexe,
                gnupghome=self.gnupghome,
                verbose=self.verbose,
                use_agent=True,
                keyring=self.keyring,
                options=self.gnupgoptions,
                secret_keyring=self.secret_keyring)
            GPG.encoding = 'utf-8'

            if self.keys_to_import:
                self._import_keys_and_trust(GPG, self.keys_to_import, self.trust_to_import)

        return GPG

    def encryptobj(self, pswdid: Text, plainobj) -> Text:
        """
        Encrypt `plainobj` in PGP-armor format, suitable to be stored in a file.

        :param pswdid:
            Used to identify the encrypted item when reporting problems.
        :return:
            PGP-armored text

        .. Tip::
            Discard `pswd` immediately after encryption,
            to reduce opportunity window of greping it from memory.
        """
        assert not is_pgp_encrypted(plainobj), "PswdId('%s'): already encrypted!" % pswdid

        import dill

        try:
            plainbytes = dill.dumps(plainobj)  # type: bytes
        except Exception as ex:
                raise ValueError("PswdId('%s'): encryption failed due to: %s" % (pswdid, ex))

        cipher = self.GPG.encrypt(plainbytes, self._guess_master_key(), armor=True)
        if not cipher.ok:
            self.log.debug("PswdId('%s'): encryption stderr: %s", pswdid, cipher.status, cipher.stderr)
            raise ValueError("PswdId('%s'): %s!" % (pswdid, cipher.status))

        return str(cipher)

    def decryptobj(self, pswdid: Text, armor_text: Text):
        """
        PGP-decrypt `armor_text` encrypted with :func:`pgp_encrypt()`.

        :param pswdid:
            Used to identify the encrypted item when reporting problems.
        :return:
            The original object unencrypted.
        :raise ValueError:
            when decryption fail

        .. Tip::
            Discard returned pswd immediately after usage,
            to reduce opportunity window of greping it from memory.
        """
        if not is_pgp_encrypted:
            raise ValueError("PswdId('%s'): cannot encrypt!  Not in pgp-armor format!" % pswdid)

        try:
            plain = self.GPG.decrypt(armor_text)
        except Exception as ex:
            raise ValueError("PswdId('%s'): decryption failed due to: %s" % (pswdid, ex))

        if not plain.ok:
            self.log.debug("PswdId('%s'): decryption stderr: %s", pswdid, plain.stderr)
            raise ValueError("PswdId('%s'): %s!" % (pswdid, plain.status))

        import dill
        plainobj = dill.loads(plain.data)

        return plainobj

    def clearsign_text(self, text: Text) -> Text:
        """Clear-signs a textual-message with :attr:`master_key`."""
        try:
            signed = self.GPG.sign(text, keyid=self._guess_master_key())
        except Exception as ex:
            raise ValueError("Signing failed due to: %s" % ex)

        if not signed.data:
            self.log.debug("Signing stderr: %s", signed.stderr)
            raise ValueError("No signed due to: %s!" % getattr(signed, 'status'))

        return str(signed)

#     ## NO, very simple, do it directly in client code.
#     def verify_clearsigned(self, text: Text) -> bool:
#         """Verifies a clear-signed textual-message."""
#        verified = self.GPG.verify(text)
#
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

    def verify_detached(self, sig: bytes, data: bytes):
        """Verify `sig` on the `data`."""
        import gnupg
        import tempfile

        assert isinstance(data, bytes), data
        assert isinstance(sig, bytes), sig

        with tempfile.TemporaryDirectory() as tdir:
            sig_fn = osp.join(tdir, 'sig')
            with io.open(sig_fn, 'wb+') as sig_fp:
                sig_fp.write(sig)

            GPG = self.GPG
            args = ['--verify', gnupg.no_quote(sig_fn), '-']
            result = GPG.result_map['verify'](GPG)
            data_stream = io.BytesIO(data)
            GPG._handle_io(args, data_stream, result, binary=True)

        return result

    def verify_git_signed(self, git_bytes: bytes):
        msg, sig = split_git_signed(git_bytes)
        return self.verify_detached(sig, msg)


class VaultSpec(trtc.SingletonConfigurable, GnuPGSpec):
    """A store of 3rdp passwords and othe secret objects in textual format."""


def get_vault(config) -> VaultSpec:
    """Use this to get hold of the *vault* singletton."""
    return VaultSpec.instance(config=config)


class Cipher(trt.TraitType):
    """A trait that auto-dencrypts its value using PGP (can be anything that is Dill-ed)."""
    ## See also :class:`baseapp.HasCiphersMixin`

    info_text = 'any value (will be PGP-encrypted)'
    allow_none = True

    def validate(self, obj, value):
        if value is None or is_pgp_encrypted(value):
            pass
        else:
            cls_name = type(obj).__name__
            pswdid = '%s.%s' % (cls_name, self.name)
            vault = get_vault(obj.config)
            vault.log.debug("Auto-encrypting cipher-trait(%r)...", pswdid)
            value = vault.encryptobj(pswdid, value)

        return value
