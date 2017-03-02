#!/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
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
from collections import OrderedDict
import io
import os
import re
from typing import Text, Tuple, Union, Dict  # @UnusedImport

import os.path as osp
import traitlets as trt
import traitlets.config as trtc


_pgp_regex = re.compile(r'^\s*-----[A-Z ]*PGP[A-Z ]*-----.+-----[A-Z ]*PGP[A-Z ]*-----\s*$', re.DOTALL)


def is_pgp_encrypted(obj) -> bool:
    """Note that it encrypts also `None`."""
    return bool(isinstance(obj, str) and _pgp_regex.match(obj))


_pgp_clearsig_regex = re.compile(
    r"""
    (?P<whole>
        ^-{5}BEGIN\ PGP\ SIGNED\ MESSAGE-{5}\r?\n
        (?:^Hash:\ [^\r\n]+\r?\n)*                   ## 'Hash:'-header(s)
        ^\r?\n                                       ## blank-line
        (?P<msg>^.*?)
        \r?\n                                        ## NOT part of plaintext!
        (?P<sigarmor>
            ^-{5}BEGIN\ PGP\ SIGNATURE-{5}\r?\n
            .*?
            ^\r?\n                                   ## blank-line
            (?P<sig>[^-]+)
            ^-{5}END\ PGP\ SIGNATURE-{5}(?:\r?\n)?
        )
    )    """,
    re.DOTALL | re.VERBOSE | re.MULTILINE)

_pgp_clearsig_dedashify_regex = re.compile(
    r'^- ', re.MULTILINE)

_pgp_clearsig_eol_canonical_regex = re.compile(
    r'[ \t\r]*\n')

#: The `'gpgsig '` optional prefix works for signed Git-commits.
_pgp_signature_banner_regex = re.compile(
    br'^(?:gpgsig )?-----BEGIN PGP SIGNATURE-----', re.MULTILINE)

_git_detachsig_strip_top_empty_lines_regexb = re.compile(
    br'^\s*\n?')

_git_detachsig_canonical_regexb = re.compile(
    br'\s*$|[ \t\r]*\n')


def pgp_split_clearsigned(text: str) -> Dict:
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

        return groups


def pgp_split_sig(git_content: bytes) -> (bytes, bytes):
    """
    Split any PGP-signed text in 2 parts: top-part (armored or not), armored-sig.

    Git objects (tags & commits) are structured like this, but in bytes,
    like that::

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


    :param git_content:
            Bytes as fetched from ``git cat-file tag/commit <HASHID>``.
    :return:
            A 2-tuple(msg, sig), None if no sig found.

    See: https://lists.gnupg.org/pipermail/gnupg-users/2014-August/050780.html
    """
    m = _pgp_signature_banner_regex.search(git_content)
    if m:
        split_pos = m.start()

        return OrderedDict([
            ('whole', git_content),
            ('msg', git_content[:split_pos]),
            ('sigarmor', git_content[split_pos:]),
        ])


class GpgSpec(baseapp.Spec):
    """
    Configurable parameters for instantiating a GnuPG instance

    Class-parameters override values the following environment variables (if exist):
    - :attr:`GpgSpec.gnupgexe`   --> `GNUPGEXE`
    - :attr:`GpgSpec.gnupghome`  --> `GNUPGHOME`
    - :attr:`GpgSpec.master_key` --> `GNUPGKEY`

    """

    gnupgexe = trt.Unicode(
        os.environ.get('GNUPGEXE', 'gpg2'), allow_none=True,
        help="The path to GnuPG-v2 executable; read from `GNUPGEXE`(%s) env-variable or 'gpg'."
        % os.environ.get('GNUPGEXE')
    ).tag(config=True)

    gnupghome = trt.Unicode(
        os.environ.get('GNUPGHOME'), allow_none=True,
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
        default_value=[
            '--allow-weak-digest-algos',  # Timestamp-service's key use MD5!
            '--armor',
            '--keyid-format', 'long',
        ],
        allow_none=True,
        help="""A list of additional cmd-line options to pass to the GPG binary."""
    ).tag(config=True)

    master_key = trt.Unicode(
        os.environ.get('GNUPGKEY'), allow_none=True,
        help="""
        The key-id (or recipient) of a *secret* PGP key to use for various crytpo operations.

        Usage in subclasses:
          - VaultSpec:         dencrypt 3rdp passwords
          - TstampSenderSpec:  sign email to timestamp service

        You MUST set either this configurable option or `GNUPGKEY`(%s) env-variable, if you have
        If you have more than one private keys in your PGP-keyring, or else
        the application will fail to start when any of the usages above is initiated.
        """ % os.environ.get('GNUPGKEY')
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

    #: Lazily created.
    _GPG = None

    @trt.observe('gnupgexe', 'gnupghome', 'keyring', 'secret_keyring', 'options')
    def _remove_cached_GPG(self, change):
        self._GPG = None

    @property
    def master_key_resolved(self) -> Text:
        master_key = self.master_key
        if not master_key:
            GPG = self.GPG
            seckeys = GPG.list_keys(secret=True)
            nseckeys = len(seckeys)
            if nseckeys != 1:
                seckeys = ['\n    %s: %s' % (k['keyid'], k['uids'][0])
                           for k
                           in seckeys]
                raise ValueError(
                    "Cannot guess master-key! Found %d keys in secret keyring: %s"
                    "\n  Please set the `GpgSpec.master_key` config-param or `GNUPGKEY` env-var."
                    % (nseckeys, ', '.join(seckeys)))

            master_key = seckeys[0]['fingerprint']

        return master_key

    @property
    def gnupgexe_resolved(self):
        """Used for printing configurations only."""
        import shutil

        gnupgexe = self.gnupgexe
        gnupgexe = shutil.which(gnupgexe) or gnupgexe

        if not re.search(r'gpg2(:?.exe)?$', gnupgexe, re.I) or osp.isdir(gnupgexe):
            self.log.warning(
                "The path `%s.gnupgexe = '%s'` may point to a FOLDER(!) or GPG-v1.x, "
                "\n  instead of pointing to a `gpg2` executable!",
                type(self).__name__, gnupgexe)

        return gnupgexe

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

    @trt.observe('keys_to_import', 'trust_to_import')
    def _reimport_keys_and_trust(self, change):
        if self._GPG:
            self._import_keys_and_trust(self._GPG)

    def _import_keys_and_trust(self, GPG):
        """
        Load in GPG-keyring from :attr:`keys_to_import` and :attr:`trust_to_import`.

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
        for armor_key in self.keys_to_import:
            keys_res.append(GPG.import_keys(armor_key))
        if self.trust_to_import:
            trust_res = import_trust(self.trust_to_import)
        else:
            trust_res = 'no trust found'

        key_summaries = [k.summary() for k in keys_res]
        log.info('Import: Keys: %s, Trust: %s', key_summaries, trust_res)

    @property
    def GPG(self) -> 'gnupg.GPG':
        import gnupg
        GPG = self._GPG
        if not GPG:
            self.gnupgexe_resolved  # Just to wearn user...
            gnupgexe = self.gnupgexe
            GPG = self._GPG = gnupg.GPG(
                gpgbinary=gnupgexe,
                gnupghome=self.gnupghome,
                verbose=self.verbose,
                use_agent=True,
                keyring=self.keyring,
                options=self.gnupgoptions,
                secret_keyring=self.secret_keyring)

            if self.keys_to_import:
                self._import_keys_and_trust(GPG)

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
        import pickle

        assert not is_pgp_encrypted(plainobj), "PswdId('%s'): already encrypted!" % pswdid

        try:
            plainbytes = pickle.dumps(plainobj)  # type: bytes
        except Exception as ex:
                raise ValueError("PswdId('%s'): encryption failed due to: %s" % (pswdid, ex))

        cipher = self.GPG.encrypt(plainbytes, self.master_key_resolved, armor=True)
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
        import pickle

        if not is_pgp_encrypted:
            raise ValueError("PswdId('%s'): cannot encrypt!  Not in pgp-armor format!" % pswdid)

        try:
            plain = self.GPG.decrypt(armor_text)
        except Exception as ex:
            raise ValueError("PswdId('%s'): decryption failed due to: %s" % (pswdid, ex))

        if not plain.ok:
            self.log.debug("PswdId('%s'): decryption stderr: %s", pswdid, getattr(plain, 'stderr', ''))
            raise ValueError("PswdId('%s'): %s!" % (pswdid, plain.status))

        plainobj = pickle.loads(plain.data)

        return plainobj

    def clearsign_text(self, text: Text) -> Text:
        """Clear-signs a textual-message with :attr:`master_key`."""
        try:
            signed = self.GPG.sign(text, keyid=self.master_key_resolved)
        except Exception as ex:
            raise ValueError("Signing failed due to: %s" % ex)

        if not signed.data:
            self.log.debug("Signing stderr: %s", signed.stderr)
            raise ValueError("No signed due to: %s!" % getattr(signed, 'status'))

        return str(signed)

    def _proc_verfication(self, ver, keep_stderr: bool=None):
        """Convert *gnupg* lib's results into dict, hidding `stderr` if OK."""
        keep_stderr = keep_stderr is None and not bool(ver)
        if not keep_stderr:
            ver.stderr = ''

        return ver

    def verify_clearsigned(self, text: Text, keep_stderr: bool=None) -> bool:
        """Verifies a clear-signed textual-message."""
        return self._proc_verfication(self.GPG.verify(text), keep_stderr)

    def verify_detached(self, sig: bytes, data: bytes, keep_stderr=None):
        """Verify binary `sig` on the `data`."""
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

        return self._proc_verfication(result, keep_stderr)

    def verify_git_signed(self, git_bytes: bytes, keep_stderr: bool=None):
        """
        Splits and verify the normalized top-part against the bottom armored-sig.

        :return:
            The object returned by *gnupg*, with the splitted parts in `parts` attribute.
        """
        csig = pgp_split_sig(git_bytes)
        if csig:
            msg = _git_detachsig_canonical_regexb.sub(b'\n', csig['msg'])
            msg = _git_detachsig_strip_top_empty_lines_regexb.sub(b'', msg)
            ver = self.verify_detached(csig['sigarmor'], msg)

            ver.parts = csig

            return ver


########################################
## Singletons for handling GPG-keys,
## separated so that they can be configured independently.
##
class VaultSpec(trtc.SingletonConfigurable, GpgSpec):
    """A store of 3rdp passwords and othe secret objects in PGP-encrypted armore format."""


def get_vault(config: trtc.Config) -> VaultSpec:
    return VaultSpec.instance(config=config)


class GitAuthSpec(trtc.SingletonConfigurable, GpgSpec):
    """The private key of the TA/TS importing filesnto git-repos is stored here."""


def get_git_auth(config: trtc.Config) -> GitAuthSpec:
    return GitAuthSpec.instance(config=config)


class StamperAuthSpec(trtc.SingletonConfigurable, GpgSpec):
    """Used to verify stamper's timestamp service."""


def get_stamper_auth(config: trtc.Config) -> StamperAuthSpec:
    return StamperAuthSpec.instance(config=config)
##
########################################


class Cipher(trt.TraitType):
    """A trait that auto-dencrypts its value using PGP (can be anything that is pickled-ed)."""
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
