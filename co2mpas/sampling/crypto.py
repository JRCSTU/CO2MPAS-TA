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
import os
import re
from typing import Text, Tuple, Union  # @UnusedImport

import traitlets as trt
import traitlets.config as trtc


_pgp_regex = re.compile(r'^\s*-----[A-Z ]*PGP[A-Z ]*-----.+-----[A-Z ]*PGP[A-Z ]*-----\s*$', re.DOTALL)


def is_pgp_encrypted(obj) -> bool:
    return bool(isinstance(obj, str) and _pgp_regex.match(obj))


class GnuPGSpec(baseapp.Spec):
    """
    Configurable parameters for instantiating a GnuPG instance

    Class-parameters override values of `GPG_EXECUTABLE` and `GNUPGHOME` environ-variables.
    """

    gpgbinary = trt.Unicode(
        'gpg', allow_none=True,
        help="""
        The path to GnuPG executable; if None, the first `gpg` command in PATH variable is used,
        unless the GPG_EXECUTABLE env-variable is set.
        """
    ).tag(config=True)

    gnupghome = trt.Unicode(
        None, allow_none=True,
        help="""
        The full pathname containing the keys to where we can find the public and private keyrings.
        If None, the executable decides (POSIX: `~/.gpg`, Windows: `%APPDATA%\Roaming\GnuPG`),
        unless the GNUPGHOME env-variable is set.
        """
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

    options = trt.List(
        trt.Unicode(None, allow_none=False),
        None, allow_none=True,
        help="""A list of additional cmd-line options to pass to the GPG binary."""
    ).tag(config=True)

    master_key = trt.Unicode(
        None, allow_none=True,
        help="""
        The key-id (or recipient) of a secret PGP key to use for various crytpo operations.

        Usage in subclasses:
            VaultSpec:         dencrypt 3rdp passwords
            TstampSenderSpec:  sign email to timestamp service

        You MUST set either this configurable option or `GPGKEY` env-var, if you have
        If you have more than one private keys in your PGP-keyring, or else
        the application will fail to start when any of the usages above is initiated.
        """
    ).tag(config=True)

    @trt.observe('gpgbinary', 'gnupghome', 'keyring', 'secret_keyring', 'options')
    def _gpg_args_changed(self, change):
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
    def GPG(self) -> 'gnupg.GPG':
        import gnupg
        GPG = getattr(self, '_GPG', None)
        if not GPG:
            GPG = self._GPG = gnupg.GPG(
                gpgbinary=self.gpgbinary,
                gnupghome=self.gnupghome,
                verbose=self.verbose,
                use_agent=True,
                keyring=self.keyring,
                options=self.options,
                secret_keyring=self.secret_keyring)
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
            self.log.debug("PswdId('%s'): decryption stderr: %s", pswdid, plain.status, plain.stderr)
            raise ValueError("PswdId('%s'): %s!" % (pswdid, plain.status))

        import dill
        plainobj = dill.loads(plain.data)

        return plainobj

#    def verify_detached_armor(self, sig: str, data: str):
#    #def verify_file(self, file, data_filename=None):
#        """Verify `sig` on the `data`."""
#        logger = gnupg.logger
#        #with tempfile.NamedTemporaryFile(mode='wt+',
#        #                encoding='latin-1') as sig_fp:
#        #sig_fp.write(sig)
#        #sig_fp.flush(); sig_fp.seek(0) ## paranoid seek(), Windows at least)
#        #sig_fn = sig_fp.name
#        sig_fn = osp.join(tempfile.gettempdir(), 'sig.sig')
#        logger.debug('Wrote sig to temp file: %r', sig_fn)
#
#        args = ['--verify', gnupg.no_quote(sig_fn), '-']
#        result = self.result_map['verify'](self)
#        data_stream = io.BytesIO(data.encode(self.encoding))
#        self._handle_io(args, data_stream, result, binary=True)
#        return result
#
#
#    def verify_detached(self, sig: bytes, msg: bytes):
#        with tempfile.NamedTemporaryFile('wb+', prefix='co2dice_') as sig_fp:
#            with tempfile.NamedTemporaryFile('wb+', prefix='co2dice_') as msg_fp:
#                sig_fp.write(sig)
#                sig_fp.flush()
#                sig_fp.seek(0) ## paranoid seek(), Windows at least)
#
#                msg_fp.write(msg)
#                msg_fp.flush();
#                msg_fp.seek(0)
#
#                sig_fn = gnupg.no_quote(sig_fp.name)
#                msg_fn = gnupg.no_quote(msg_fp.name)
#                args = ['--verify', sig_fn, msg_fn]
#                result = self.result_map['verify'](self)
#                p = self._open_subprocess(args)
#                self._collect_output(p, result, stdin=p.stdin)
#                return result


class VaultSpec(trtc.SingletonConfigurable, GnuPGSpec):
    """A store of 3rdp passwords and othe secret objects in textual format."""


def get_vault(config) -> VaultSpec:
    """Use this to get hold of the *vault* singletton."""
    return VaultSpec.instance(config=config)


class Cipher(trt.TraitType):
    """A trait that auto-dencrypts its value using PGP (can be anything that is Dill-ed)."""

    info_text = 'any value (will be PGP-encrypted)'

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

    def decrypted(self, obj: trt.HasTraits):
        """
        Decrypts a cipher trait of some instance.

        :param obj:
            The instance holding the trait-values.
        :return:
            The unencrypted object, or None if trait-value was None.

        .. Tip::
            Invoke it on the class, not on the trait: ``ObjClass.ctrait.decrypt(obj)``.
        """
        assert isinstance(obj, trt.HasTraits), "%r not a HasTraits!" % obj

        value = self.get(obj, type(obj))
        if value is not None:
            cls_name = type(obj).__name__
            pswdid = '%s.%s' % (cls_name, self.name)
            if not is_pgp_encrypted(value):
                self.log.warning("Found non-encrypted param %r!", pswdid)
            else:
                vault = get_vault(obj.config)
                vault.log.debug("Decrypting cipher-trait(%r)...", pswdid)
                value = vault.decryptobj(pswdid, value)
        return value
