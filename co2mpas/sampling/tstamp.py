#!/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""A *report* contains the co2mpas-run values to time-stamp and disseminate to TA authorities & oversight bodies."""
from collections import (
    defaultdict, OrderedDict, namedtuple, Mapping)  # @UnusedImport
import io
import os.path as osp
import random
import re
import sys
from typing import (
    List, Sequence, Iterable, Text, Tuple, Dict, Callable, Union)  # @UnusedImport

from pandalone import utils as pndlu

import functools as fnt

from . import CmdException, baseapp, dice, crypto
from .. import (__version__, __updated__, __uri__, __copyright__, __license__,  # @UnusedImport
                vehicle_family_id_pattern)
from .._vendor import traitlets as trt


_undefined = object()


class UnverifiedSig(CmdException):
    def __init__(self, msg, verdict):
        self.verdict = verdict
        super().__init__(msg)


def _to_bytes(s, encoding='ASCII', errors='surrogateescape'):
    if not isinstance(s, bytes):
        return s.encode(encoding, errors=errors)


def _to_str(b, encoding='ASCII', errors='surrogateescape'):
    if not isinstance(b, str):
        return b.decode(encoding, errors=errors)


def pgp_sig_to_sig_id_num(sig_id: Text) -> int:
    import base64
    import binascii

    sig_bytes = base64.b64decode(sig_id + '==')
    num = int(binascii.b2a_hex(sig_bytes), 16)

    return num


def num_to_dice100(num: int, is_randomize: bool) -> (int, int):
    """
    :return:
        ``(num, dice100)``
    """
    if is_randomize:
        dice100 = random.Random(num).randint(0, 99)
    else:
        ## Cancel the effect of trailing zeros, but is biased,
        #  see #422
        num = int(str(num).strip('0'))
        dice100 = num % 100

    return num, dice100


###################
##     Specs     ##
###################

STARTTLS_PORTS = (587, 143)


class TstampSpec(dice.DiceSpec):
    """Common parameters and methods for both SMTP(sending emails) & IMAP(receiving emails)."""

    user_account = trt.Unicode(
        None, allow_none=True,
        help="""
        The username for the account on the SMTP/IMAP server!
        If not set, `user_email` is used.
        """
    ).tag(config=True)

    user_pswd = crypto.Cipher(
        None, allow_none=True,
        help="""
        The SMTP/IMAP server's password matching `user_account`/`user_email` param.

        For *GMail* with 2-factor authentication, see:
            https://support.google.com/accounts/answer/185833
        """
    ).tag(config=True)

    host = trt.Unicode(
        help="""The SMTP/IMAP server, e.g. 'smtp.gmail.com'."""
    ).tag(config=True)

    port = trt.Int(
        None, allow_none=True,
        help="""
            The SMTP/IMAP server's port, usually 587/465 for SSL, 25 otherwise.
            If undefined, does its best according to the `ssl` config,
            which when `True`, uses SSL/TLS ports.
        """).tag(config=True)

    ssl = trt.Union(
        (trt.Bool(), trt.FuzzyEnum(['SSL/TLS', 'STARTTLS'])),
        default_value=True,
        help="""
        Bool/enumeration for what encryption to use when connecting to SMTP/IMAP servers:
        - 'SSL/TLS':  Connect only through TLS/SSL, fail if server supports it
                      (usual ports SMTP:465 IMAP:993).
        - 'STARTTLS': Connect plain & upgrade to TLS/SSL later, fail if server supports it
                      (usual ports SMTP:587 IMAP:143).
        - True:       enforce most secure encryption, based on server port above;
                      If port is `None`, identical to 'SSL/TLS'.
        - False:      Do not use any encryption;  better use `no_auth` param,
                      not to reveal credentials in plain-text.

        See also:
          - https://www.fastmail.com/help/technical/ssltlsstarttls.html
          - http://forums.mozillazine.org/viewtopic.php?t=2730845
        """
    ).tag(config=True)

    no_auth = trt.Bool(
        False,
        help="""Whether not to send any user/password credentials to the server"""
    ).tag(config=True)

    @property
    def _ssl_resolved(self):
        """:return: a tuple of bool: (is_SSL, is_STARTTLS) """
        ssl, port = self.ssl, self.port
        if ssl is False:
            return False, False
        if ssl == 'SSL/TLS':
            return True, False
        if ssl == 'STARTTLS':
            return False, True
        if ssl is True:
            is_starttls = port in STARTTLS_PORTS
            return not is_starttls, is_starttls

        assert False, ("Unexpected logic-branch:", ssl, port)

    mail_kwds = trt.Dict(
        help="""
            Any extra key-value pairs passed to the SMTP/IMAP mail-client libraries.
            For instance, :class:`smtlib.SMTP_SSL` and :class:`smtlib.IMAP4_SSL`
            support `keyfile`, `certfile`, `ssl_context`(unusable :-/) and `timeout`,
            while SMTP/SSL support additionally `local_hostname` and `source_address`.

            The keyfile and certfile parameters specify paths to optional files which contain
            a certificate to be used to identify the local side of the connection.
            see https://docs.python.org/3/library/ssl.html#ssl.wrap_socket
        """
    ).tag(config=True)

    socks_host = trt.Unicode(
        None, allow_none=True,
        help="""
        The hostname/ip of the SOCKS-proxy server for send/recv emails.
        If not set, SOCKS-proxying is disabled.

        Note:
          GMail, Yahoo (and possibly Outlook in the future) need you
          to make first a successful login with your browser THROUGH PROXY,
          before letting DICE to login.  Search: https://goo.gl/vb4wAi
        """
    ).tag(config=True)

    socks_skip_resolve = trt.Bool(
        False,
        help="""
        Should DNS queries be performed on the remote side of the SDOCKS tunnel?

          - This has no effect with SOCKS4 servers.
          - Prefer to set an IP in `socks_host` when setting this to True.
        """
    ).tag(config=True)

    socks_type = trt.FuzzyEnum(
        ['SOCKS4', 'SOCKS5', 'HTTP', 'disabled'], allow_none=True,
        help="""
        The SOCKS-proxy protocol to use for send/recv emails (case-insensitive).
        If not set, becomes 'SOCKS5' if `socks_user` is defined, 'SOCKS4' otherwise.
        """
    ).tag(config=True)

    socks_port = trt.Int(
        None, allow_none=True,
        help="""
        The port of the SOCKS-proxy server for send/recv emails.
        If not set, defaults to 1080 for SOCKS-v4/5 proxies, 8080 for HTTP-proxy.
        """
    ).tag(config=True)

    socks_user = trt.Unicode(
        None, allow_none=True,
        help="""The username of the SOCKS-v5-proxy server for send/recv emails."""
    ).tag(config=True)

    socks_pswd = crypto.Cipher(
        None, allow_none=True,
        help="""The password of the SOCKS-v5-proxy server for send/recv emails."""
    ).tag(config=True)

    subject_prefix = trt.Unicode(
        '[co2dice.test]: ',
        allow_none=True,
        help="""
        Prefixes project-ids when sending emails, and used as search term when receiving.

        - The *sender* uses this value as the 1st part of the subject-line for
          the dice-report email that is send to the timestamper.  None/empty is not
          allowed!
        - The *receiver* uses this value to filter emails containing this string in
          their subject line. If None, no extra filter on the subject line is used.

        Tip:
          set to sender's ``c.TstampSender.subject_prefix = None`` if dice cannot
          receive the emails from your account that you know are there
          (assuming the other search criteria, such as dates, are correct).
          Yahoo needs this!
        """
    ).tag(config=True)

    @property
    def user_account_resolved(self):
        return self.user_account is not None and self.user_account or self.user_email

    def choose_server_class(self):
        raise NotImplemented()

    def make_server(self):
        host = self.host
        port = self.port
        srv_kwds = self.mail_kwds.copy()
        if port is not None:
            srv_kwds['port'] = port
        srv_cls = self.choose_server_class()
        _, is_startssl = self._ssl_resolved

        self.log.info("Connecting to %s%s: %s(%s:%s)%s...", srv_cls.__name__,
                      '(STARTTLS)' if is_startssl else '',
                      self.user_account_resolved, host, port, srv_kwds)
        srv = srv_cls(host, **srv_kwds)

        return srv

    def check_login(self, dry_run):
        """Logs only and returns true/false; does not throw any exception!"""
        ok = False
        srv_name = srv_sock = ''
        try:
            with self.make_server() as srv:
                srv_name = type(srv).__name__
                srv_sock = srv.sock
                self.log.debug("Authenticating %s: %s@%s ...", srv_name,
                               self.user_account_resolved, srv.sock)
                self.login_srv(srv,  # If login denied, raises.
                               self.user_account_resolved,
                               self.decipher('user_pswd'))
            ok = True
            return True
        except Exception as ex:
            ok = ex
            self.log.error("Connection FAILED due to: %s", ex, exc_info=True)

            return False
        finally:
            self.log.info("Connected to %s: %s@%s, ok? %s", srv_name,
                          self.user_account_resolved, srv_sock, ok)

    _socks_patched_modules = set()

    def monkeypatch_socks_module(self, module):
        """
        If :attr:`socks_host` is defined, wrap module to use PySocks-proxy(:mod:`socks`).
        """
        if (self.socks_host and
                self.socks_type != 'disabled' and
                module not in self._socks_patched_modules):
            import socks

            if self.socks_type is None:
                socks_type = (socks.SOCKS5
                              if self.socks_type == 'socks5' or self.socks_user is not None
                              else socks.SOCKS4)
            else:
                socks_type = socks.PROXY_TYPES(socks_type.upper)
            self.log.debug("Using proxy(%s)-->%s:%s",
                           socks.PRINTABLE_PROXY_TYPES[socks_type],
                           self.socks_host, self.socks_port)
            socks.set_default_proxy(socks_type,
                                    self.socks_host,
                                    self.socks_port,
                                    rdns=not self.socks_skip_resolve,
                                    username=self.socks_user,
                                    password=self.decipher('socks_pswd'))
            socks.wrapmodule(module)
            self._socks_patched_modules.add(module)


#: The key of of the base64 blob of the report wihen it is too-wide.
SCRABLE_KEY = 'base64(tag)'


@fnt.lru_cache()
def _make_send_transfer_encoders_map():
    """Add 2 capital/lower keys for each Content-Transfer-Encoder in :mod:`email import encoders`."""
    from email import encoders as enc

    encoders = [enc.encode_base64, enc.encode_quopri, enc.encode_7or8bit]
    enc_kv = [(e.__name__[len('encode_'):].upper(), e)
              for e in encoders]

    ## Add the same encoders in lower for conditionall application.
    #
    def apply_conditionally(msg, encoder):
        mbytes = msg.get_payload(decode=True)
        max_line_length = msg.policy.max_line_length

        apply = not max_line_length or any(len(l.rstrip()) > max_line_length
                                           for l in mbytes.split(b'\n'))
        try:
            mbytes.decode('ascii')
        except:  # @IgnorePep8
            apply = True

        if apply:
            encoder(msg)

    enc_kv += [(k.lower(), fnt.partial(apply_conditionally, encoder=v))
               for k, v in enc_kv]

    enc_kv.append(('noenc', None))  # Old deleted and then nothing added.

    return dict(enc_kv)


class TstampSender(TstampSpec):
    """SMTP & timestamp parameters and methods for sending dice emails."""

    ## TODO: delete deprecated trait
    timestamping_addresses = trt.List(
        trt.Unicode(),
        help="Deprecated, but still functional.  Prefer `TstampSender.tstamper_address` instead"
        "  Note: it is not a list!"
    ).tag(config=True)

    tstamper_address = trt.Unicode(
        help="""The plain email-address of the timestamp-service. Ask JRC to provide that. """
    ).tag(config=True)

    tstamp_recipients = trt.List(
        trt.Unicode(),
        help="""
        The plain email-address of the receivers of the timestamped-response.
        Ask JRC to provide that. You don't have to provide your sender-account here.
    """).tag(config=True)

    cc_addresses = trt.List(
        trt.Unicode(),
        help="Any carbon-copy (CC) recipients. "
    ).tag(config=True)

    bcc_addresses = trt.List(
        trt.Unicode(),
        help="Any blind-carbon-copy (BCC) recipients. "
    ).tag(config=True)

    ## TODO: delete deprecated trait
    x_recipients = trt.List(
        trt.Unicode(),
        help="Deprecated, but still functional.  Prefer `tstamp_recipients` list  instead."
    ).tag(config=True)

    from_address = trt.Unicode(
        None, allow_none=True,
        help="""Your email-address to use as `From:` for timestamp email, or none to use `user_email`.
        Specify you correct address, or else you will never receive the tstamped-response!
        """
    ).tag(config=True)

    other_headers = trt.Dict(
        key_trait=trt.Unicode(), value_trait=trt.Unicode(),
        default_value=None, allow_none=True,
        help="""
        List of more (IMAP or EWS) email headers, given in this format:
            <header-name>=<string-value>

        Examples:
            (cmd_chain)s --TstampSender.other_headers Reply-To=mymail@foo.com \
                         --TstampSender.other_headers=Thread-Topic=Bar
        """
    ).tag(config=True)

    subject = trt.Unicode(
        allow_none=True,
        help="""Deprecated, and NON functional.

        Replaced either by the top-level `TstampDice.subject_prefix` or
        the `subject_prefix` options in `TstampSender` and `TstampReceiver` classes.
        The later is needed for when searching old tstamps.
        """
    ).tag(config=True)

    send_transfer_encoding = trt.FuzzyEnum(
        list(_make_send_transfer_encoders_map()),
        None, allow_none=True,
        case_sensitive=True,
        help="""
        Set the Content-Transfer-Encoding MIME Header and encodes appropriately outgoing mails.

        - CAPITAL encodings mean "always applied"; `lower` applied only if
          non-ASCII (or long-lines?).
        - Experiment with this to avoid strange `'=0A=0D=0E'` chars scattered in the email
          (MS Outlook Exchange servers have this problem but seem immune to this switch!)
        - Note that base64 encoding DOES NOT work with Tstamper, for sure.
        - Modifying `recv_transfer_encoding` is not necessary.
        - `noenc` removes the MIME header completely.
        - Setting None means "default set by python".
        """
    ).tag(config=True)

    scramble_tag = trt.Bool(
        help="""Base64-encode dice-tag to mask any non-ASCII and long-lines.

        That happend right before sending it it may resolve email-corruption
        problems, particularly when following the "manual" procedure."""
    ).tag(config=True)

    @property
    def _from_address_resolved(self):
        return self.from_address or self.user_email

    def __init__(self, *args, **kwds):
        self._register_validator(
            type(self)._is_not_empty,
            ['host', 'tstamper_address', 'subject_prefix'])
        self._register_validator(
            type(self)._is_pure_email_address,
            ['tstamper_address'])
        self._register_validator(
            ## Apply only one sender, IMAP may(?) search UTF-8.
            type(self)._is_all_latin,
            ['subject_prefix', 'tstamp_recipients', 'x_recipients'])
        self._register_validator(
            type(self)._warn_deprecated,
            ['x_recipients', 'timestamping_addresses', 'subject'])
        super().__init__(*args, **kwds)

    ## TODO: delete deprecated trait
    @property
    def _tstamper_address_resolved(self):
        adrs = [a
                for a in self.timestamping_addresses + [self.tstamper_address]
                if a]
        if not adrs:
            myname = type(self).__name__
            raise trt.TraitError('One of `%s.tstamper_address` and ``%s.timestamping_addresses` must not be empty!'
                                 % (myname, myname))
        return adrs

    def _append_tstamp_recipients(self, msg):
        x_recs = '\n'.join('X-Stamper-To: %s' % rec
                           for rec
                           in self.tstamp_recipients + self.x_recipients)
        if not x_recs:
            myname = type(self).__name__
            raise trt.TraitError('One of `%s.tstamp_recipients` and ``%s.x_recipients` must not be empty!'
                                 % (myname, myname))

        msg = "%s\n\n%s" % (x_recs, msg)

        return msg

    def _apply_send_transfer_encoding(self, mail, encoding):

        ## CAPITAL/lower names define conditional-application.
        #
        apply = True
        check_if_utf8 = encoding.isupper()
        if check_if_utf8:
            try:
                mail.get_payload(decode=True).decode('ascii')
                apply = False
            except UnicodeError:
                apply = True
        self.log.info("%s email Transfer-Encoding: %s",
                      'Setting' if apply else 'Skipped (because ASCII)',
                      encoding)
        if not apply:
            return

        ## Delete existing or else:
        #    ValueError: There may be at most 1 Content-Transfer-Encoding headers in a message
        del mail['Content-Transfer-Encoding']

        enc_map = _make_send_transfer_encoders_map()
        encoder = enc_map[encoding]
        if encoder:
            encoder(mail)

    def _prepare_mail(self, msg, subject_suffix):
        from email.mime.text import MIMEText
        from email import policy

        mail = MIMEText(msg, 'plain')
        mail['Subject'] = '%s %s' % (self.subject_prefix, subject_suffix)
        mail['From'] = self._from_address_resolved
        mail['To'] = ', '.join(self._tstamper_address_resolved)
        if self.cc_addresses:
            mail['Cc'] = ', '.join(self.cc_addresses)
        if self.bcc_addresses:
            mail['Bcc'] = ', '.join(self.bcc_addresses)

        if self.other_headers is not None:
            for k, v in self.other_headers.items():
                mail[k] = v

        ## Instruct serializers to dissregard line-length.
        mail.policy = policy.default.clone(max_line_length=0)

        encoding = self.send_transfer_encoding
        if encoding:
            self._apply_send_transfer_encoding(mail, encoding)

        return mail

    def choose_server_class(self):
        import smtplib

        self.monkeypatch_socks_module(smtplib)
        is_ssl, _ = self._ssl_resolved
        cls = smtplib.SMTP_SSL if is_ssl else smtplib.SMTP
        cls.debuglevel = int(self.verbose)

        return cls

    def login_srv(self, srv, user, pswd):
        _, is_starttls = self._ssl_resolved
        if is_starttls:
            self.log.debug('STARTTLS...')
            srv.ehlo()
            srv.starttls(keyfile=self.mail_kwds.get('keyfile'),
                         certfile=self.mail_kwds.get('certfile'),
                         context=None)
            srv.ehlo()

        srv.noop()

        if not self.no_auth:
            (code, resp) = srv.login(user, pswd)  # If login denied, raises.
            # 235: 'Authentication successful'
            if code == 503:
                self.log.info('Already authenticated: %s', resp)

    def _scramble_tag(self, tag_text: Union[Text, bytes],
                      header: Text):
        """
        If email too wide or non-latin, encode32 it, an set just the header.

        The result is this::

            tag: dices/IP-10-AAA-2017-0012/4
            base64(tag): |
              N5RGUZLDOQQDAMBTMY2DANTCME4GCOJSMU3TONLBHAYWKY3CGEZTSM3BHA3TOM
              DEMU2WKM3GMUZQU5DZOBSSAY3PNVWWS5AKORQWOIDENFRWK4ZPJFIC2MJQFVAU
              ...
              JNFUWQU==
            -----BEGIN PGP SIGNATURE-----
        """
        assert all(ord(c) < 128 for c in header), "Non-ASCII in: %s" % header
        sep = b'\n' if isinstance(tag_text, bytes) else '\n'
        max_line_len = max(len(l) for l in tag_text.split(sep))
        too_wide = max_line_len > 78
        all_latin = all(ord(c) < 128 for c in tag_text)
        ## According to RFC5322, 78 is the maximum width for textual emails;
        #  mails with width > 78 may be sent as HTML-encoded and/or mime-multipart.
        #  QuotedPrintable has 76 as limit, probably to account for CR+NL
        #  end-ofline chars
        self.log.info("Email content: non_ASCII: %s, line_len: %s, too_wide: %s",
                      not all_latin, max_line_len, too_wide)
        if (self.scramble_tag):
            import base64

            wrapped_rbody = base64.b64encode(_to_bytes(tag_text, 'utf-8'))

            ## Chnuk blob in lines:
            #      -2: ident, 64: margin copied from GPG.
            #
            new_width = 64 - 2
            wrapped_rbody = b'\n  '.join(
                wrapped_rbody[pos:pos + new_width]
                for pos
                in range(0, len(wrapped_rbody), new_width))

            tag_text = '%s: %s\n%s: |\n  %s' % ('tag', header,
                                                SCRABLE_KEY,
                                                wrapped_rbody.decode())

        return tag_text

    def send_timestamped_email(self, msg: Union[str, bytes],
                               subject_suffix='', dry_run=False):
        ## TODO: Schedula to the rescue!

        ## Allow to skip report syntax-errors/verification if --force,
        #  but still report the kind of syntax/sig failure
        #
        try:
            git_auth = crypto.get_git_auth(self.config)
            ver = git_auth.verify_git_signed(_to_bytes(msg, 'utf-8'))
            verdict = _mydump(sorted(vars(ver).items()))
        except Exception as ex:
            err = "Failed to extract signed dice-report from tstamp!\n%s" % ex
            if self.force:
                self.log.warning(err)
            else:
                raise CmdException(err) from ex
        else:
            if not ver:
                err = "Cannot verify dice-report's signature!\n%s" % verdict
                if self.force:
                    self.log.warning(err)
                else:
                    raise CmdException(err)
            else:
                err = "The dice-report in timestamp got verified OK: %s"
                self.log.debug(err, verdict)

        msg = self._scramble_tag(msg, subject_suffix)
        msg = self._append_tstamp_recipients(msg)
        mail = self._prepare_mail(msg, subject_suffix)

        if dry_run:
            self.log.warning("DRY-RUN:  No email has been sent!\n  "
                             "the printed %d-char TEXT-email from '%s' to %s-->%s",
                             len(msg), self._from_address_resolved,
                             self._tstamper_address_resolved,
                             self.tstamp_recipients + self.x_recipients)
        else:
            with self.make_server() as srv:
                self.login_srv(srv,  # If login denied, raises.
                               self.user_account_resolved,
                               self.decipher('user_pswd'))

                self.log.info("Timestamping %d-char email from %s-->%s through %s",
                              len(msg), self._from_address_resolved,
                              self.tstamp_recipients + self.x_recipients,
                              self._tstamper_address_resolved)
                srv.send_message(mail)

        return mail


_stamper_id_regex = re.compile(r"Comment:\s+Stamper\s+Reference\s+Id:\s+(\d+)")
_stamper_banner_regex = re.compile(r"^#{56}\r?\n(?:^#[^\n]*\n)+^#{56}\r?\n\r?\n(.*)",
                                   re.MULTILINE | re.DOTALL)  # @UndefinedVariable

#: If it exists in some tstamp, it signifies new "randomized" dice routine.
#: If it start with undescore(_), the full line must be removed prior verification.
_stamp_version_regex = re.compile(r"\b(_)?stamp_version: (\d+\.\d+\.\d+)[ \t\r]*\n")


def _should_extract_stamp_version_line(match: 're.Match') -> bool:
    return bool(match.group(1))

def extract_any_stamp_version_line(mail_text: Text) -> (Text, Text):
    """
    :return
        ``(stamp_ver, text_without_ver_line)`` where `stamp_ver`
        possibly None
    """
    stamp_ver = None
    m = _stamp_version_regex.search(mail_text)
    if m:
        stamp_ver = m.group(2)
        if _should_extract_stamp_version_line(m):
            mail_text = _stamp_version_regex.sub('', mail_text)

    return mail_text, stamp_ver


class TstampReceiver(TstampSpec):
    """IMAP & timestamp parameters and methods for receiving & parsing dice-report emails."""

    auth_mechanisms = trt.List(
        trt.FuzzyEnum(['CRAM-MD5', 'PLAIN']), default_value=['CRAM-MD5', 'PLAIN'],
        help="""The order for IMAP authentications to try; CRAM-MD5 and PLAIN supported only."""
    ).tag(config=True)

    vfid_extraction_regex = trt.CRegExp(
        r'vehicle_family_id[^\n]+(%s)' % vehicle_family_id_pattern,
        help="""An approximate way to get the project if timestamp parsing has failed. """
    ).tag(config=True)

    dicetag_regex = trt.CRegExp(
        r'dices/%s/\d+' % vehicle_family_id_pattern,
        help="""Tag-name to search on Subject and Body of tstamp response emails. """
    ).tag(config=True)

    mailbox = trt.Unicode(
        b'INBOX',
        help="""
        The mailbox folder name (case-sensitive except "INBOX") to search for tstamp responses in.

          - Use `mailbox` subcmd to list all mailboxes.
          - Tip: Although Gmail is not recommended, use "[Gmail]/All Mail"
            to search in all its folders.
        """
    ).tag(config=True)

    rfc_criteria = trt.List(
        trt.Unicode(allow_none=True), allow_none=True,
        default_value=[
            'FROM "mailer@stamper.itconsult.co.uk"',
            'SUBJECT "Proof of Posting Certificate"',
        ],
        help="""
        RFC3501 IMAP search terms ANDed together for fetching Stamper responses.

        - Note that list-elements are combinations of criteria like those below,
          grouped with NOT, AND & OR keywords, all parenthesized:
            SUBJECT | BODY | "foo bar"
            TEXT "foo bar"        # Search subject & body
            FROM  | TO | CC | BCC "foo@bar"

          or just those 'special" flags:
            [UN]ANSWERED | [UN]FLAGGED | [UN]SEEN | RECENT | NEW | OLD | DRAFT
            [UN]KEYWORD <flag>
            HEADER <field-name> <string>
            CHARSET UTF-8         # Applies for all strings given.

          where `NEW` means `AND(RECENT UNSEEN)`.
          Any date-related fields also given with --on, --before and --after
          disregard time & timezone:
            BEFORE | SINCE | SENTBEFORE | SENTSINCE | SENTON "15-May-2017"

        - A single space-delimiter and double-quoted strings are both compulsory;
        - When multiple terms are given, they are ANDed rtogether.
        - More criteria are appended on runtime, ie `TstampSpec.subject_prefix`,
          `wait_criterio` if --wait, and any args to `recv` command as ORed
          and searched as subject terms (i.e. the (projects-ids").
        - If you want to fetch tstamps sent to `tstamp_recipients`,
          either leave this empty, or set it to email-address of the sender
          (bash syntax):
            --rfc-criteria='From "tstamp-sender@foo.com"'
        - See https://tools.ietf.org/html/rfc3501#section-6.4.4
        """
    ).tag(config=True)

    wait_criterio = trt.Unicode(
        'NEW', allow_none=True,
        help="""
        The RFC3501 IMAP search criteria for when IDLE-waiting;

        See `co2dice config desc rfc_criteria` for examples.
        """
    ).tag(config=True)

    poll_delay = trt.Int(
        60,
        help="""How often(in sec) to poll IMAP server if it does not supported IDLE command. """
    ).tag(config=True)

    dates_locale = trt.Unicode(
        'en_US', allow_none=True,
        help="""
        Locale to use when parsing dates (see `before_date`), or default if undefined.

        locales: de_DE, en_AU, en_US, es, nl_NL, pt_BR, ru_RU, fr_FR
        """
    ).tag(config=True)

    before_date = trt.Unicode(
        None, allow_none=True,
        help="""
        Search messages sent before the specified day, in human readable form.

          - Read docs: https://dateparser.readthedocs.io
          - For available locales, see `date_locale` param
          - See also `co2dice config desc rfc_criteria`.

        Examples:
          - yesterday, last year, previous Wednesday
          - 18/3/53              ## order depending on your locale
          - 10 days ago
          - two days after eom   ## 2 days after end-of-moth
          - Jan                  ## NOTE: If January, refers to NEXT year's January!
        """
    ).tag(config=True)

    after_date = trt.Unicode(
        None, allow_none=True,
        help="""Search messages sent before the specified date, in human readable form (see `before_date`)"""
    ).tag(config=True)

    on_date = trt.Unicode(
        None, allow_none=True,
        help="""Search messages for this day, in human readable form (see `before_date`)"""
    ).tag(config=True)

    email_infos = trt.List(
        trt.Unicode(allow_none=True),
        default_value=['From', 'To', 'Subject', 'Date'], allow_none=True,
        help="""
        The email items to print for each matched email; all if None or contains a None.

        - Might be case-insensitive!
        - The Message-Id is always printed.
        - Other standard fields: Received, Delivered-To
        - Set it to `None` to see all available fields for a specific email-provider.
        - Use "special" item `Body` to include email-payload
          (not fetched if None).
        """
    ).tag(config=True)

    un_quote_printable = trt.FuzzyEnum(
        'TAG FULL tag full'.split(),
        'tag', allow_none=True,
        case_sensitive=True,
        help="""
        Whether to un-quote-printable Stamp or selectively the containing Tag only.

          - TAG: unquote just the enclosed tag of the dice-report.
          - FULL: unquote full dice-response email.
          - "lower" case values: try verbatim text first, fallback unoquoting later".
          - None: don't unquote anything.

        Tip:
          Trials may be needed to decide, e.g. Outlook servers need 'TAG', but
          a more resilient choice is 'tag'.
        """
    ).tag(config=True)

    def try_unquoting(self, unquot_choice, txt: Text, check_func, stage):
        """
        Tries processing text twice, original :attr:`un_quote_printable`, unless CAPTIALS

        :param check_func:
             callable(txt) to process unquoted text and raise if invalid (e.g. sig-check).
        :param stage:
            'full' or 'tag', caseless
        """
        if not unquot_choice or stage.lower() != unquot_choice.lower():
            return check_func(txt)

        def unquote():
            import quopri
            self.log.info('Unquoting printable %s...', stage)
            return quopri.decodestring(txt).decode('utf-8')

        if unquot_choice.isupper():
            return check_func(unquote())

        try:
            return check_func(txt)
        except Exception as ex:
            self.log.warning('1st validation of original %s failed due to: %s',
                             stage, ex)
            try:
                return check_func(unquote())
            except Exception as _:
                raise ex from None  # Raise error from original (unquoted) text.

    @trt.validate('subject_prefix', 'wait_criterio',
                  'before_date', 'after_date', 'on_date')
    def _strip_trait(self, p):
        v = p.value
        return v and v.strip()

    def _capture_stamper_msg_and_id(self, ts_msg: Text, ts_heads: Text) -> int:
        stamper_id = msg = None
        m = _stamper_id_regex.search(ts_heads)
        if m:
            stamper_id = int(m.group(1))
        m = _stamper_banner_regex.search(ts_msg)
        if m:
            msg = m.group(1)

        return stamper_id, msg

    def _is_stamp_version_says_randomize_sig_id(self, stamp_ver: Text) -> int:
        """
        :param stamp_ver:
            A possibly none string, formated as ``<major>.<minor><micro>``.
        :return:
            True if `stamp_ver` defined, meaning sig-id should be randomized,
            None otherwise.
        """
        if stamp_ver:
            return True

    def _pgp_sig_to_dice100(self, sig_id: Text, stamp_ver: Text) -> int:
        """
        :return:
            ``(sig-id-as-20-bytes-number, dice100)``
        """
        num = pgp_sig_to_sig_id_num(sig_id)
        is_randomize = self._is_stamp_version_says_randomize_sig_id(stamp_ver)
        num, dice100 = num_to_dice100(num, is_randomize)

        return num, dice100

    def scan_for_project_name(self, mail_text: Text) -> int:
        """
        Search in the text for any coarsely identifiable project-name (`vehicle_family_id`).

        Use this if :meth:`parse_tstamp_response()` has failed to provide the answer.
        """
        project = None
        all_vfids = self.vfid_extraction_regex.findall(mail_text)
        if all_vfids:
            project = all_vfids[0][0]
            if not all(i[0] == project for i in all_vfids):
                project = None

        return project

    def _descramble_tag(self, tag_text: Text) -> int:
        if SCRABLE_KEY in tag_text:
            #
            ## Either report were too wide, or instructed to base64-encode
            #  the dice-report before embedding it into the tag.
            import base64

            blob_start = tag_text.index(SCRABLE_KEY)
            blob_start += len(SCRABLE_KEY) + len(': |')
            tag_b64 = re.sub('\\s', '', tag_text[blob_start:])
            tag_bytes = base64.b64decode(tag_b64)
        else:
            tag_bytes = _to_bytes(tag_text, 'utf-8')

        return tag_bytes

    def _verify_tag(self, tag_text: Text) -> OrderedDict:
        """return verdict or raise if tag invalid"""
        git_auth = crypto.get_git_auth(self.config)

        stag_bytes = self._descramble_tag(tag_text)
        ver = git_auth.verify_git_signed(stag_bytes)
        verdict = OrderedDict(sorted(vars(ver).items()))
        if not ver:
            raise UnverifiedSig(
                "Cannot verify (foreign?) dice-report's signature!\n%s" %
                _mydump(verdict), verdict)

        return verdict

    def parse_signed_tag(self, tag_text: Text) -> int:
        """
        :param msg_text:
            The tag as extracted from tstamp response by :meth:`crypto.pgp_split_clearsigned`.
        """
        ## TODO: Schedula to the rescue!

        ## Allow parsing signed/unsigned reports when --force,
        #
        try:
            verdict = self.try_unquoting(
                self.un_quote_printable, tag_text,
                self._verify_tag, 'tag')
        except CmdException as ex:
            ## Do not fail, it might be from an unknown sender,
            #  but log as much as possible and crop verdict.
            verdict = ex.verdict
            tag = verdict['parts']['msg']
            self.log.warning(str(ex))
        except Exception as ex:
            msg = "Failed to extract signed dice-report from tstamp!\n%s" % ex
            if self.force:
                self.log.warning(msg)
                verdict = OrderedDict(sig=msg)

                ## Fall-back assuming report was not signed at all.
                tag = _to_bytes(tag_text, 'utf-8')
            else:
                raise CmdException(msg) from ex
        else:
            self.log.debug("The dice-report in timestamp got verified OK: %s",
                           _mydump(verdict))
            tag = verdict['parts']['msg']

        ## Parse dice-report
        #
        from . import project

        try:
            cmsg = project._CommitMsg.parse_commit_msg(_to_str(tag).strip())
            verdict['commit_msg'] = cmsg._asdict()
            verdict['project'] = cmsg.p
            verdict['project_source'] = 'report'
        except Exception as ex:
            msg = "Cannot parse dice-report due to: %s" % ex
            if self.force:
                self.log.error(msg)
            else:
                raise CmdException(msg) from ex

        if 'project' not in verdict:
            verdict['project'] = self.scan_for_project_name(tag_text)
            verdict['project_source'] = 'grep'

        return verdict

    def _validate_stamp_version(self, stamp_ver: Text) -> int:
        """
        :param stamp_ver:
            A possibly none string, formated as ``<major>.<minor><micro>``.
        :raise:
            CmdException if `stamp_ver` defined but unsupported number:
            either diff major OR greater minor.
        """
        if stamp_ver:
            from .. import __dice_stamp_version__

            prog_ver = __dice_stamp_version__.split('.')
            major, minor, _ = stamp_ver.split('.')
            if int(major) != int(prog_ver[0]) or int(minor) > int(prog_ver[1]):
                ## No force can overcome this.
                raise CmdException(
                    "incompatible stamp-version '%s', expected <= '%s.%s.x'" %
                    (stamp_ver, prog_ver[0], prog_ver[1]))

    def _verify_tstamp(self, mail_text: Text) -> OrderedDict:
        """return verdict or raise if tstamp invalid"""
        stamper_auth = crypto.get_stamper_auth(self.config)

        mail_text, stamp_ver = extract_any_stamp_version_line(mail_text)

        ver = stamper_auth.verify_clearsigned(mail_text)
        verdict = vars(ver)
        ## Note: not `stamp_version` not to kick mail-detection routine.
        verdict['stamp_ver'] = stamp_ver
        verdict['mail_text'] = mail_text
        if not ver:
            errmsg = "Cannot verify timestamp-response's signature due to: %s"
            gpg_msg = ver.status or crypto.filter_gpg_stderr(ver.stderr)
            raise UnverifiedSig(errmsg % (gpg_msg), verdict)

        return verdict

    def extract_dice_tag_name(self, subject: str, msg: str) -> str:
        """Extract `dices/IP-12-WMI-1234/0` strings either from Subject or Body. """
        for place, txt in [('Subject-line', subject), ('email-Body', msg)]:
            if not txt:
                continue
            try:
                tag_name = self.dicetag_regex.search(txt).group()
                self.log.debug("Extracted tag-name '%s' from %s.",
                               tag_name, place)
                return tag_name
            except Exception as ex:
                self.log.warning(
                    "Failed extracting tag-name from %s due to: %r",
                    place, ex, exc_info=self.verbose)

    def _make_dice_results(self, ts_verdict, tag_verdict, tag_name):
        num, dice100 = self._pgp_sig_to_dice100(ts_verdict['signature_id'],
                                                ts_verdict.get('stamp_ver'))
        decision = 'OK' if dice100 < 90 else 'SAMPLE'

        #self.log.info("Timestamp sig did not verify: %s", _mydump(tag_verdict))
        dice_results = []

        stamp_ver = ts_verdict.get('stamp_ver')
        if stamp_ver:
            dice_results.append(('stamp_ver', stamp_ver))

        if tag_name:
            dice_results.append(('tag', tag_name))
        else:
            dice_results.append(('project', tag_verdict.get('project')))

        issuer = crypto.uid_from_verdict(tag_verdict)
        if issuer:
            dice_results.append(('issuer', issuer))

        if 'timestamp' in tag_verdict:
            dice_results.append(('issue_date',
                                 crypto.gpg_timestamp(tag_verdict['timestamp'])))

        stamper = crypto.uid_from_verdict(ts_verdict)
        if stamper:
            dice_results.append(('stamper', stamper))

        if 'timestamp' in ts_verdict:
            dice_results.append(('dice_date',
                                 crypto.gpg_timestamp(ts_verdict['timestamp'])))

        dice_results.extend([
            ('hexnum', '%X' % num),
            ('percent', dice100),
            ('decision', decision),
        ])

        return OrderedDict(dice_results)

    def parse_tstamp_response(self, mail_text: Text, tag_name: str=None) -> int:
        ## TODO: Could use dispatcher to parse tstamp-response, if failback routes were working...
        import textwrap as tw

        force = self.force
        errlog = self.log.error if self.force else self.log.debug

        try:
            ts_verdict = self.try_unquoting(
                self.un_quote_printable, mail_text,
                self._verify_tstamp, 'full')
        ## Let serious exceptions bubble up - cannot work with them.
        except CmdException as ex:
            ## Do not fail, it might be from an unknown sender,
            #  but log as much as possible and crop verdict.
            ts_verdict = ex.verdict
            if not force or 'signature_id' not in ts_verdict:  # Need sig-id for decision.
                self.log.debug("%s\n but got: %s", ex,
                               _mydump(sorted(ts_verdict.items())))
                raise CmdException(str(ex))
            else:
                self.log.error(str(ex))

        if not ts_verdict.get('valid'):
            self.log.warning(
                tw.dedent("""
                Timestamp's signature is valid, but not *trusted*!
                  You may sign Timestamp-service's key(81959DB570B61F81) with a *fully* trusted secret-key,
                  or assign *full* trust on JRC's key(TODO:JRC-keyid-here) that already has done so.
                    %s
                """), _mydump(sorted(ts_verdict.items())))

        ## NOTE: Text may have changed if `stamp-ver` line has been deleted.
        #  but still return the original stamp.
        ts_parts = crypto.pgp_split_clearsigned(ts_verdict['mail_text'])
        ts_verdict['parts'] = ts_parts
        if not ts_parts:
            errlog("Cannot parse timestamp-response:"
                   "\n  mail-txt: %s\n\n  ts-verdict: %s",
                   mail_text, _mydump(sorted(ts_verdict.items())))
            if not force:
                raise CmdException(
                    "Cannot parse timestamp-response!")
            stamper_id = tag_verdict = None
        else:
            stamper_id, tag = self._capture_stamper_msg_and_id(ts_parts['msg'], ts_parts['sigarmor'])
            ts_verdict['stamper_id'] = stamper_id
            if not tag:
                errlog("Failed parsing response content and/or stamper-id: %s\n%s",
                       _mydump(ts_parts), _mydump(sorted(ts_verdict.items())))
                if not force:
                    raise CmdException(
                        "Failed parsing response content and/or stamper-id: %s\n%s" %
                        (_mydump(ts_parts), _mydump(sorted(ts_verdict.items()))))

                tag_verdict = {'content_parsing': "failed"}
                tag_verdict = {'project': None}
            else:
                tag_verdict = self.parse_signed_tag(tag)

        if not tag_name:
            tag_name = self.extract_dice_tag_name(None, mail_text)

        dice_results = self._make_dice_results(ts_verdict, tag_verdict, tag_name)

        return OrderedDict([
            ('tstamp', ts_verdict),
            ('report', tag_verdict),
            ('dice', dice_results),
        ])

    def choose_server_class(self):
        import imaplib

        self.monkeypatch_socks_module(imaplib)
        monkeypatch_imaplib_for_IDLE(imaplib)
        monkeypatch_imaplib_noop_debug_26543(imaplib)

        imaplib.Debug = 1 + 2 * int(self.verbose) if self.verbose else 0
        is_ssl, _ = self._ssl_resolved
        return imaplib.IMAP4_SSL if is_ssl else imaplib.IMAP4

    def login_srv(self, srv, user, pswd):
        from imaplib import IMAP4

        _, is_starttls = self._ssl_resolved
        if is_starttls:
            self.log.debug('STARTTLS...')
            resp = srv.starttls()
            reject_IMAP_no_response("starttls", resp)

        resp = srv.noop()
        reject_IMAP_no_response("noop", resp)

        if self.no_auth or not self.auth_mechanisms:
            return

        authlist = [(auth, method) for auth, method
                    in zip(['CRAM-MD5', 'PLAIN'],
                           [srv.login_cram_md5, srv.login])
                    if 'AUTH=%s' % auth in srv.capabilities]

        if not authlist:
            msg = "IMAP AUTH extensions %s not supported by server: %s"
            sup_auths = [c for c in srv.capabilities if c.startswith('AUTH')]
            raise IMAP4.error(msg % (self.auth_mechanisms, sup_auths))

        auth, authmethod = next(iter(authlist))
        resp = authmethod(user, pswd)
        reject_IMAP_no_response("login", resp)

    def list_mailbox(self, directory='""', pattern='*'):
        with self.make_server() as srv:
            self.login_srv(srv,  # If login denied, raises.
                           self.user_account_resolved,
                           self.decipher('user_pswd'))

            resp = srv.list(directory, pattern)
            data = reject_IMAP_no_response("list mailboxes", resp)
            res = [d.decode() for d in data if d]
            return ["Found %i mailboxes:" % len(res)] + res

    def _prepare_search_criteria(self, is_wait, projects):
        subj = self.subject_prefix
        dates = [self.before_date, self.after_date, self.on_date]
        waitcrt = self.wait_criterio

        if not self.rfc_criteria:
            criteria = []
        else:
            criteria = [c and c.strip() for c in self.rfc_criteria]
            criteria = [c for c in criteria if c]

        if subj:
            criteria.append('SUBJECT "%s"' % subj)

        if is_wait and waitcrt:
            criteria.append(waitcrt)

        if any(dates):
            import parsedatetime as pdt

            kw_date_pairs = [(kw, dt)
                             for kw, dt
                             in zip(['SENTBEFORE', 'SINCE', 'ON'], dates)
                             if dt]
            c = self.dates_locale and pdt.Constants(self.dates_locale)
            cal = pdt.Calendar(c)

            for kw, dt in kw_date_pairs:
                rfc_date = parse_as_RFC3501_date(cal, dt)
                criteria.append('%s "%s"' % (kw, rfc_date))

        projects = [c and c.strip() for c in projects]
        projects = list(set(c for c in projects if c))
        if projects:
            criteria.append(pairwise_ORed(projects,
                                          lambda i: '(SUBJECT "%s")' % i))

        criteria = [c.strip() for c in criteria]
        criteria = [c if c.startswith('(') else '(%s)' % c
                    for c in criteria]
        criteria = ' '.join(criteria)

        return criteria

    #: Server-capabillity captured after authed-connection established,
    #: to decide which IDLE or POLL loop to use when `is_wait`.
    _IDLE_supported = None

    # IMAP receive, see https://pymotw.com/2/imaplib/ for IMAP example.
    #      https://yuji.wordpress.com/2011/06/22/python-imaplib-imap-example-with-gmail/
    def receive_timestamped_emails(self, is_wait, projects, read_only):
        """
        Yields all matched :class:`email.message.Message` emails from IMAP.

        :param read_only:
            when true, doesn mark fetched emails as `Seen`.
        """
        criteria = self._prepare_search_criteria(is_wait, projects)
        self._IDLE_supported = None

        while True:
            yield from self._proc_emails1(
                is_wait, read_only, criteria)

            if not is_wait:
                break

            if not self._IDLE_supported:
                ## POLL within this external loop.
                #
                import time
                time.sleep(self.poll_delay)

            # IDLE loops again with new server (ie in case of failures).

    def _proc_emails1(self, is_wait, read_only, criteria):
        with self.make_server() as srv:
            self.login_srv(srv,  # If login denied, raises.
                           self.user_account_resolved,
                           self.decipher('user_pswd'))

            self._IDLE_supported = 'IDLE' in srv.capabilities

            resp = srv.select(self.mailbox, read_only)
            reject_IMAP_no_response("select mailbox", resp)

            while True:
                yield from self._proc_emails2(is_wait, criteria, srv)

                if is_wait and self._IDLE_supported:
                    ## IDLE within this internal loop
                    #
                    self.log.info("IDLE waiting for emails...")
                    event_line = wait_IDLE_IMAP_change(srv)
                    self.log.debug("Broke out of IDLE due to: %s", event_line)

                else:  # External loop will handle POLL.
                    break

    def _proc_emails2(self, is_wait, criteria, srv):
        import email

        ## SEARCH for tstamp emails.
        #
        self.log.info("Searching: %s", criteria)
        resp = srv.uid('SEARCH', criteria.encode())
        data = reject_IMAP_no_response("search emails", resp)

        uids = data[0].split()
        self.log.info("Found %s tstamp emails: %s",
                      len(uids), [u.decode() for u in uids])

        if not uids:
            return

        ## FETCH tstamp emails.
        #
        resp = srv.uid('FETCH', b','.join(uids), "(UID RFC822)")
        data = reject_IMAP_no_response("fetch emails", resp)

        for i, (uid, d) in enumerate(zip(uids, data)):
            if i % 2 == 1:
                assert d == b')', 'Unexpected FETCH data(%i): %s' % (i, d)
                continue
            m = email.message_from_bytes(d[1])

            yield uid.decode(), m

    def get_recved_email_infos(self, mail, verdict_or_ex=None,
                               verbose=None, email_infos=_undefined) -> OrderedDict:
        """
        Decide email-fields to include based on :attr:`email_infos`.

        :param verbose:
            Override :attr:`verbose` and if true, updates result with verdict.
        :param email_infos:
            override :attr:`email_infos`
        :return:
            Results contain `project` & `dice` if possible.
        :raise: never
        """
        verbose = self.verbose if verbose is None else verbose
        email_infos = self.email_infos if email_infos is _undefined else email_infos

        ## Any None signifies include all.
        is_all = email_infos is None or any(i is None for i in email_infos)
        if is_all:
            infos = OrderedDict((k.title(), '\n'.join(mail.get_all(k)))
                                for k in mail)
        else:
            infos = OrderedDict((k.title(), '\n'.join(mail.get_all(k)))
                                for k in email_infos
                                if k in mail)
        ## Body last one, for console.
        #  (mind that `email_infos` may be or contain None)
        #
        if email_infos and 'body' in [i.lower() for i in email_infos if i]:
            infos['Body'] = mail.get_payload()

        if verdict_or_ex is None:
            pass
        elif isinstance(verdict_or_ex, Exception):
            infos['dice'] = "Failed due to: %s" % verdict_or_ex
        else:
            verdict = verdict_or_ex
            if verbose:
                infos.update(verdict)
            else:
                try:
                    infos['dice'] = verdict['dice']
                except:  # @IgnorePep8
                    pass
            try:
                infos['project'] = verdict['report']['project']
            except:  # @IgnorePep8
                pass

        return infos


def _mydump(obj, indent=2, **kwds):
    import yaml

    return yaml.dump(obj, indent=indent, **kwds)


def reject_IMAP_no_response(cmd, resp):
    ok, data = resp
    if ok == 'OK':
        return data
    raise CmdException("Command %s: %s, %s" % (cmd, ok, data))


def parse_as_RFC3501_date(cal, date):
    """
    >>> import parsedatetime as pdt

    >>> cal = pdt.Calendar()
    >>> t = cal.parse('9 May 2017')[0]
    >>> [parse_as_RFC3501_date(cal, d, sourceTime=t)
    ...  for d in ['previous Jan', '-2months', '10 days after eom']]
    ['01-Jan-2017', '09-Mar-2017', '10-Jun-2017']
    """
    dt = cal.parseDT(date)[0]
    dts = dt.strftime('%d-XXX-%Y')
    months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ]
    dts = dts.replace('XXX', months[dt.month - 1])

    return dts


def pairwise_ORed(items, item_printer):
    """
    :param items:
        a non empty list/tuple

    >>> print(pairwise_ORed([1]))
    "1"
    >>> print(pairwise_ORed([1,2]))
    (OR "1" "2")
    >>> print(pairwise_ORed([1,2,3]))
    (OR (OR "1" "2") "3")
    >>> print(pairwise_ORed([1,2,3,4,5,6,7]))
    (OR (OR (OR "1" "2") (OR "3" "4")) (OR (OR "5" "6") "7"))
    """
    if isinstance(items, (tuple, list)):
        n = len(items)
        assert n > 0, items
        if n == 1:
            return item_printer(items[0])
        else:
            middle = (n + 1) // 2
            return '(OR %s %s)' % (
                pairwise_ORed(items[:middle], item_printer),
                pairwise_ORed(items[middle:], item_printer))


def monkeypatch_imaplib_for_IDLE(imaplib):
    imaplib.Commands['IDLE'] = ('SELECTED',)
    imaplib.Commands['DONE'] = ('SELECTED',)


def monkeypatch_imaplib_noop_debug_26543(imaplib):
    ## see https://bugs.python.org/issue26543
    import time

    def GOOD_untagged_response(self, typ, dat, name):
        if typ == 'NO':
            return typ, dat
        if name not in self.untagged_responses:
            return typ, [None]
        data = self.untagged_responses.pop(name)
        if __debug__:
            if self.debug >= 5:
                self._mesg('untagged_responses[%s] => %s' % (name, data))
        return typ, data

    if __debug__:

        def _mesg(self, s, secs=None):
            if secs is None:
                secs = time.time()
            tm = time.strftime('%M:%S', time.localtime(secs))
            sys.stderr.write('  %s.%02d %s\n' % (tm, (secs * 100) % 100, s))
            sys.stderr.flush()

        def _dump_ur(self, d):  # @ReservedAssignment
            # Dump untagged responses (in `d').
            if d:
                self._mesg('untagged responses dump:' +
                           ''.join('\n\t\t%s: %r' % x for x in d.items()))

        def _log(self, line):
            # Keep log of last `_cmd_log_len' interactions for debugging.
            self._cmd_log[self._cmd_log_idx] = (line, time.time())
            self._cmd_log_idx += 1
            if self._cmd_log_idx >= self._cmd_log_len:
                self._cmd_log_idx = 0

        def print_log(self):
            self._mesg('last %d IMAP4 interactions:' % len(self._cmd_log))
            i, n = self._cmd_log_idx, self._cmd_log_len
            while n:
                try:
                    self._mesg(*self._cmd_log[i])
                except:  # @IgnorePep8
                    pass
                i += 1
                if i >= self._cmd_log_len:
                    i = 0
                n -= 1

    imaplib.IMAP4._untagged_response = GOOD_untagged_response


def wait_IDLE_IMAP_change(srv):
    """
    Use RFC2177 `IDLE` IMAP command to wait for any status change.

    :param sock_timeout:
        After 29 min, IMAP-server closes socket.
    :return:
        the raw line that broke IDLE, just for logging it,
        because is not included in `imaplib` moduele's logs.
    """
    cmd = 'IDLE'
    tag = srv._command(cmd)

    ## Pump untill continuation response `'+ idling'`.
    #
    while srv._get_response():
        pass
    assert srv.continuation_response.lower() == b'idling', srv.continuation_response

    ev_reply = srv._get_response()  # Blocks waiting status-change.

    ## Finalize `IDLE` cmd.
    #
    srv.send(b'DONE\r\n')
    ok, data = srv._command_complete(cmd, tag)
    ok, data = srv._untagged_response(ok, data, cmd)
    assert ok == 'OK', "IDLE-Done failed due to: %s: %s" % (ok, data)

    srv.noop()  # To update stats

    return ev_reply


###################
##    Commands   ##
###################

class TstampCmd(baseapp.Cmd):
    """
    Commands to manage the communications with the Timestamp server.

    The time-stamp service is used to disseminate the *dice-report* to the TA authorities
    and to oversight bodies.
    From its response the *sampling decision* is be deduced.
    """

    examples = trt.Unicode("""
        - Pick an existing dice-report tag to send for timestamping::
              co2dice project report | %(cmd_chain)s send

        - Await for the response, and paste its content to this command::
              %(cmd_chain)s parse

        Server Configuration Samples:
        -----------------------------
        - GMAIL:
          - Instructions: https://support.google.com/mail/answer/7126229
          - Allow SMTP/IMAP access: https://support.google.com/accounts/answer/6010255
            and login with Browser through SOCKS and open link read from response!!
          - app-passwords (2-factor auth): https://myaccount.google.com/apppasswords
            ::
                c.DiceSpec.user_email = 'foo@gmail.com'
                c.TstampSender.host   = 'smtp.gmail.com'
                c.TstampReceiver.host = 'imap.gmail.com'

        - OUTLOOK/OFFICE-365:
          - Instructions: https://goo.gl/tJQvi7 & https://goo.gl/jiTVZt
          - Allow SMTP/IMAP access: https://outlook.live.com/owa/?path=/options/popandimap
            ::
                c.DiceSpec.user_email = 'foo@outlook.com'   # But not a username alone!
                c.TstampSender.host   = 'smtp.mail.yahoo.com'  OR  'outlook.office365.com'
                c.TstampReceiver.host = 'imap-mail.outlook.com'  OR  'smtp.office365.com'
                #c.TstampSender.port   = 587                # Try this, if not working without it.

        - YAHOO:
          - Host/Port: https://help.yahoo.com/kb/SLN4075.html
          - Allow SMTP/IMAP access: https://login.yahoo.com/account/security
            ::
                c.DiceSpec.user_email = 'foo@yahoo.com'     # A username alone is ok.
                c.TstampSender.host   = 'smtp.mail.yahoo.com'
                c.TstampReceiver.host = c.imap.mail.yahoo.com'
                c.TstampReceiver.subject_prefix = None      # Cannot parse [] chars
    """)

    def __init__(self, **kwds):
        kwds.setdefault('subcommands', baseapp.build_sub_cmds(*all_subcmds))
        super().__init__(**kwds)


class SendCmd(baseapp.Cmd):
    """
    Send emails to be timestamped.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<report-file-1> ...]

    - Do not use this command directly (unless experimenting) - prefer
      the `project tsend` sub-command.
    - If '-' is given or no files at all, it reads from STDIN.
    - Many options related to sending & receiving the email are expected to be stored in the config-file.
    - Use --verbose to print the timestamped email.
    """

    examples = trt.Unicode("""
        - To send a dice-report for a prepared project you have to know the `vehicle_family_id`::
              git  cat-file  tag  tstamps/RL-12-BM3-2017-0001/1 | %(cmd_chain)s
    """)

    dry_run = trt.Bool(
        help="Verify dice-report and login to SMTP-server but do not actually send email to timestamp-service."
    ).tag(config=True)

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [TstampSender, crypto.GitAuthSpec])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    type(self).__name__: {'dry_run': True},
                },
                pndlu.first_line(type(self).dry_run.help)
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        from boltons.setutils import IndexedSet as iset

        files = iset(args) or ['-']

        sender = TstampSender(config=self.config)
        for file in files:
            if file == '-':
                self.log.info("Timestamping STDIN; paste message verbatim!")
                mail_text = sys.stdin.read()
            else:
                if not osp.exists(file):
                    self.log.error("File to timestamp '%s' not found!", file)
                    continue

                try:
                    with io.open(file, 'rt') as fin:
                        mail_text = fin.read()
                except Exception as ex:
                    self.log.error(
                        "Reading file to timestamp '%s' failed due to: %r",
                        file, ex, exc_info=self.verbose)
                    continue

            self.log.info("Timestamping '%s'...", pndlu.convpath(file))

            try:
                mail = sender.send_timestamped_email(mail_text, dry_run=self.dry_run)
                if self.verbose or self.dry_run:
                    return str(mail)
            except CmdException as ex:
                self.log.error("Timestamping file '%s' stopped due to: %s", 
                               ex, file, exc_info=1)  # one-off event, must not loose ex.
            except Exception as ex:
                self.log.error("Timestamping file '%s' failed due to: %r", 
                               ex, file, exc_info=1)  # one-off event, must not loose ex.


class MailboxCmd(baseapp.Cmd):
    """Lists mailboxes in IMAP server. """

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [TstampReceiver])
        super().__init__(**kwds)

    def run(self, *args):
        ## If `verbose`, too many small details, need flow.
        rcver = TstampReceiver(config=self.config)
        return rcver.list_mailbox(*args)


#: Reused also by `project trecv` cmd
recv_cmd_aliases = {
    'before': 'TstampReceiver.before_date',
    'after': 'TstampReceiver.after_date',
    'on': 'TstampReceiver.on_date',
    'mailbox': 'TstampReceiver.mailbox',
    'rfc-criteria': 'TstampReceiver.rfc_criteria',
    'wait-criterio': 'TstampReceiver.wait_criterio',
    'subject': 'TstampReceiver.subject_prefix',
    'email-infos': 'TstampReceiver.email_infos',
}


class RecvCmd(baseapp.Cmd):
    """
    Fetch tstamps from IMAP server and derive *decisions* OK/SAMPLE flags.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<search-term-1> ...]

    - Use --view to just fetch and view emails, not validate.
    - Fetch of emails in one-shot search, or use --wait.
    - The terms are ORed and searched within the email's subject-line;
      tip: use the project name(s).
    """

    examples = trt.Unicode("""
        - Search today's emails::
              %(cmd_chain)s --after today "IP-10-AAA-2017-1003"

        - Just view (not validate) emails on some date::
              %(cmd_chain)s --on "28 Feb 2018"  --raw

        - Other search formats::
              %(cmd_chain)s --after "1 year ago" --before "18 March 2017"
              %(cmd_chain)s --after "yesterday" --search 'From "foo@bar.com"'

        - Wait for new mails to arrive (and not to block console),
          - on Linux::
                %(cmd_chain)s --wait &
                ## wait...
                kill %%1  ## Asumming this was the only job started.

          - on Windows::
                START \\B %(cmd_chain)s --wait

            and kill with one of:
              - `[Ctrl+Beak]` or `[Ctrl+Pause]` keystrokes,
              - `TASKLIST/TASKKILL` console commands, or
              - with the "Task Manager" GUI.
    """)

    wait = trt.Bool(
        False,
        help="""
            Whether to wait reading IMAP for any email(s) satisfying the criteria and report them.

            WARN:
              Process must be killed afterwards, so start it in the background (see examples).
              On Windows try:
                - `[Ctrl+Beak]` or `[Ctrl+Pause]` keystrokes,
                - `TASKLIST/TASKKILL` console commands, or
                - with the "Task Manager" GUI.

            Note:
              Development flag, use `co2dice project trecv` cmd for type-aprooval.
        """
    ).tag(config=True)

    form = trt.FuzzyEnum(
        ['list', 'raw'],
        allow_none=True,
        help="""If not none, skip tstamp verification and print raw email or `email_infos`."""
    ).tag(config=True)

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [
            TstampSender, TstampReceiver,
            crypto.GitAuthSpec, crypto.StamperAuthSpec])
        kwds.setdefault('cmd_flags', {
            'wait': (
                {type(self).__name__: {'wait': True}},
                type(self).wait.help
            ),
            'list': (
                {type(self).__name__: {'form': 'list'}},
                "Just list matched emails."
            ),
            'raw': (
                {type(self).__name__: {'form': 'raw'}},
                "Just print matched email content(s)."
            ),
        })
        kwds.setdefault('cmd_aliases', recv_cmd_aliases)
        super().__init__(**kwds)

    def run(self, *args):
        ## If `verbose`, too many small details, need flow.
        default_flow_style = None if self.verbose else False
        rcver = TstampReceiver(config=self.config)

        emails = rcver.receive_timestamped_emails(self.wait, args,
                                                  read_only=True)
        for uid, mail in emails:
            mid = mail.get('Message-Id')
            mail_text = mail.get_payload()

            if self.form == 'raw':
                yield "\n\n" + '=' * 40
                yield "Email_id: %s" % mid
                for k, v in rcver.get_recved_email_infos(mail).items():
                    yield "%s: %s" % (k, v)
                yield '=' * 40
                ## In PY3 stdout duplicates \n as \r\n, hence \r\n --> \r\r\n.
                #  and signed text always has \r\n EOL.
                yield mail_text.replace('\r\n', '\n')
            else:
                if self.form == 'list':
                    verdict = None
                else:
                    try:
                        tag_name = rcver.extract_dice_tag_name(mail['Subject'], mail_text)
                        verdict = rcver.parse_tstamp_response(mail_text, tag_name)
                    except CmdException as ex:
                        verdict = ex
                        self.log.warning("[%s]%s: parsing tstamp failed due to: %s",
                                         uid, mid, ex)
                    except Exception as ex:
                        verdict = ex
                        self.log.error("[%s]%s: parsing tstamp failed due to: %r",
                                       uid, mid, ex, exc_info=self.verbose)

                infos = rcver.get_recved_email_infos(mail, verdict)

                yield _mydump({'[%s]%s' % (uid, mid): infos},
                              default_flow_style=default_flow_style)


class ParseCmd(baseapp.Cmd):
    """
    Verifies and derives the *decision* OK/SAMPLE flag from tstamped-response email.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<tstamped-file-1> ...]

    - If '-' is given or no files at all, it reads from STDIN.
    """
    examples = trt.Unicode("""cat <mail> | %(cmd_chain)s""")

    parse_as_tag = trt.Bool(
        default_value=None, allow_none=True,
        help="""
        true: tag given, false: stamp given, None: guess based on '"' chars.
        """
    ).tag(config=True)

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [TstampReceiver,
                                         crypto.GitAuthSpec, crypto.StamperAuthSpec])
        kwds.setdefault('cmd_flags', {
            'tag': (
                {type(self).__name__: {'parse_as_tag': True}},
                "Parse input as tag."
            ),
            'stamp': (
                {type(self).__name__: {'parse_as_tag': False}},
                "Parse input as stamp."
            ),
        })
        super().__init__(**kwds)

    def _is_parse_tag(self, mail_text):
        if self.parse_as_tag is None:
            ## tstamper produces 57 x '*'.
            return ('#' * 50) not in mail_text
        else:
            return bool(self.parse_as_tag)

    def run(self, *args):
        from boltons.setutils import IndexedSet as iset

        default_flow_style = None if self.verbose else False
        files = iset(args) or ['-']

        rcver = TstampReceiver(config=self.config)
        for file in files:
            if file == '-':
                self.log.info("Parsing STDIN; paste message verbatim!")
                mail_text = sys.stdin.read()
            else:
                if not osp.exists(file):
                    self.log.error("File to parse '%s' not found!", file)
                    continue

                try:
                    with io.open(file, 'rt') as fin:
                        mail_text = fin.read()
                except Exception as ex:
                    self.log.error(
                        "Reading file to parse '%s' failed due to: %r",
                        file, ex, exc_info=self.verbose)
                    continue

            is_parse_tag = self._is_parse_tag(mail_text)
            self.log.info("Parsing file '%s' as %s...",
                          file, 'TAG' if is_parse_tag else 'STAMP')

            try:
                if is_parse_tag:
                    verdict = rcver.parse_signed_tag(mail_text)
                else:
                    verdict = rcver.parse_tstamp_response(mail_text)

                if not self.verbose:
                    from toolz import dicttoolz as dtz

                    verdict = dtz.keyfilter(lambda k: k == 'dice',
                                            verdict)

                if file != '-':
                    verdict['fpath'] = file
            except CmdException as ex:
                verdict = ex
                self.log.warning("Failed parsing tstamp '%s' due to: %s",
                                 file, ex)
            except Exception as ex:
                verdict = ex
                self.log.error("Failed parsing tstamp '%s' due to: %r",
                               file, ex, exc_info=self.verbose)
            else:
                yield _mydump(verdict, default_flow_style=default_flow_style)


class LoginCmd(baseapp.Cmd):
    """Attempts to login into SMTP server. """

    dry_run = trt.Bool(
        help="Verify dice-report and login to SMTP-server but do not actually send email to timestamp-service."
    ).tag(config=True)

    srv = trt.FuzzyEnum(
        ['SMTP', 'IMAP'], allow_none=True,
        help="""Which server to attempt to login; attempts to both if `None`."""
    ).tag(config=True)

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [TstampSender, TstampReceiver])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {type(self).__name__: {'dry_run': True}},
                pndlu.first_line(type(self).dry_run.help)
            ),
            'smtp': (
                {type(self).__name__: {'srv': 'SMTP'}},
                "Attempts to login only to SMTP."
            ),
            'imap': (
                {type(self).__name__: {'srv': 'IMAP'}},
                "Attempts to login only to IMAP."
            ),
        })
        super().__init__(**kwds)

    def run(self, *args):
        nargs = len(args)
        if nargs > 0:
            raise CmdException("Cmd '%s' takes no arguments, received %d: %r!"
                               % (self.name, len(args), args))

        srv = self.srv
        servers = []
        if not srv or self.srv == 'SMTP':
            servers.append(TstampSender(config=self.config))
        if not srv or self.srv == 'IMAP':
            servers.append(TstampReceiver(config=self.config))

        return (s.check_login(self.dry_run) for s in servers)


all_subcmds = (
    LoginCmd,
    SendCmd,
    RecvCmd,
    MailboxCmd,
    ParseCmd
)
