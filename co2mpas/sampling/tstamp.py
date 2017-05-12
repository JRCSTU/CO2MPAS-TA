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
import re
import sys
from typing import (
    List, Sequence, Iterable, Text, Tuple, Dict, Callable, Union)  # @UnusedImport

from .._vendor import traitlets as trt

from . import CmdException, baseapp, dice, crypto
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport


def _to_bytes(s):
    if not isinstance(s, bytes):
        return s.encode('ASCII', errors='surrogateescape')


def _to_str(b):
    if not isinstance(b, str):
        return b.decode('ASCII', errors='surrogateescape')


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

        Tip:
          Microsoft Outlook/Yahoo servers use STARTTLS.
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

        Tip:
          Prefer a real IP and set `socks_skip_resolve=True`, or else,
          hostnames may resolve to _unsupported_ IPv6.
        """
    ).tag(config=True)

    socks_skip_resolve = trt.Bool(
        help="""Whether to skip DNS resolve of `socks_host` value."""
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
        Prefixes project-ids when sending emails, used as search term when receiving.

        If none, Receiver will not add it to its criteria; Sender will scream.
        """
    ).tag(config=True)

    #@trt.validate('subject_prefix')  # Only @sender, IMAP may search UTF-8?
    def _is_all_latin(self, proposal):
        value = proposal.value
        if any(ord(c) >= 128 for c in value):
            myname = type(self).__name__
            raise trt.TraitError('%s.%s must not contain non-ASCII chars!'
                                 % (myname, proposal.trait.name))
        return value

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

        self.log.info("Connecting to %s%s: %s@%s(%s)...", srv_cls.__name__,
                      '(STARTTLS)' if is_startssl else '',
                      self.user_account_resolved, host, srv_kwds or '')
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

    def monkeypatch_socks_module(self, module):
        """
        If :attr:`socks_host` is defined, wrap module to use PySocks-proxy(:mod:`socks`).
        """
        if self.socks_host and self.socks_type != 'disabled':
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

    subject = trt.Unicode(
        allow_none=True,
        help="""Deprecated, and NON functional.

        Replaced either by the top-level `TstampDice.subject_prefix` or
        the `subject_prefix` options in `TstampSender` and `TstampReceiver` classes.
        The later is needed for when searching old tstamps.
        """
    ).tag(config=True)

    @property
    def _from_address_resolved(self):
        return self.from_address or self.user_email

    def __init__(self, *args, **kwds):
        self._register_validator(
            TstampSender._is_not_empty,
            ['host', 'subject_prefix'])
        self._register_validator(
            TstampSender._warn_deprecated,
            ['x_recipients', 'timestamping_addresses', 'subject'])
        self._register_validator(
            TstampSender._is_all_latin,
            ['subject_prefix'])
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

    def _prepare_mail(self, msg, subject_suffix):
        from email.mime.text import MIMEText

        mail = MIMEText(msg, 'plain')
        mail['Subject'] = '%s %s' % (self.subject_prefix, subject_suffix)
        mail['From'] = self._from_address_resolved
        mail['To'] = ', '.join(self._tstamper_address_resolved)
        if self.cc_addresses:
            mail['Cc'] = ', '.join(self.cc_addresses)
        if self.bcc_addresses:
            mail['Cc'] = ', '.join(self.bcc_addresses)

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
            srv.starttls(keyfile=self.mail_kwds.get('keyfile'),
                         certfile=self.mail_kwds.get('certfile'),
                         context=None)

        srv.noop()

        if not self.no_auth:
            (code, resp) = srv.login(user, pswd)  # If login denied, raises.
            # 235: 'Authentication successful'
            if code == 503:
                self.log.info('Already authenticated: %s', resp)

    def send_timestamped_email(self, msg: Union[str, bytes],
                               subject_suffix='', dry_run=False):
        ## TODO: Schedula to the rescue!

        msg_bytes = msg
        if isinstance(msg, str):
            msg_bytes = _to_bytes(msg)
        git_auth = crypto.get_git_auth(self.config)

        ## Allow to skip report syntxa-errors/verification if --force,
        #  but still report the kind of syntax/sig failure
        #
        try:
            ver = git_auth.verify_git_signed(msg_bytes)
            verdict = _mydump(sorted(vars(ver).items()))
        except Exception as ex:
            err = "Failed to extract signed dice-report from tstamp!\n%s" % ex
            if self.force:
                self.log.warning(err)
            else:
                raise CmdException(err)
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

                self.log.info("Timestamping %d-char email from '%s' to %s-->%s",
                              len(msg), self._from_address_resolved,
                              self._tstamper_address_resolved,
                              self.tstamp_recipients + self.x_recipients)
                srv.send_message(mail)

        return mail


_stamper_id_regex = re.compile(r"Comment: Stamper Reference Id: (\d+)")
_stamper_banner_regex = re.compile(r"^#{56}\r?\n(?:^#[^\n]*\n)+^#{56}\r?\n\r?\n\r?\n(.*)",
                                   re.MULTILINE | re.DOTALL)  # @UndefinedVariable


class TstampReceiver(TstampSpec):
    """IMAP & timestamp parameters and methods for receiving & parsing dice-report emails."""

    auth_mechanisms = trt.List(
        trt.FuzzyEnum(['CRAM-MD5', 'PLAIN']), default_value=['CRAM-MD5', 'PLAIN'],
        help="""The order for IMAP authentications to try; CRAM-MD5 and PLAIN supported only."""
    ).tag(config=True)

    vfid_extraction_regex = trt.CRegExp(
        r"vehicle_family_id[^\n]+((?:IP|RL|RM|PR)-\d{2}-\w{2,3}-\d{4}-\d{4})",  # See also co2mpas.io.schema!
        help="""An approximate way to get the project if timestamp parsing has failed. """
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

    email_criteria = trt.List(
        trt.Unicode(),
        default_value=[
            'From "mailer@stamper.itconsult.co.uk"',
            'Subject "Proof of Posting Certificate"',
        ],
        help="""
        RFC3501 IMAP search terms ANDed together for fetching Stamper responses.

        - Note that elements are not just string - most probably you want:

            TEXT "foo bar"

        - More criteria are appended on runtime, ie `TstampSpec.subject_prefix`,
          `wait_criteria` if --wait, and any args to `recv` command as ORed
          and searched as subject terms.
        - If you want to fetch tstamps sent to `tstamp_recipients`,
          either leave this empty, or set it to email-address of the sender:

            ['From "tstamp-sender@foo.com"']
        """
    ).tag(config=True)

    wait_criteria = trt.Unicode(
        'NEW', allow_none=True,
        help="""The RFC3501 IMAP search criteria for when IDLE-waiting, usually RECENT+UNSEEN messages."""
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
        Search messages sent before this point in time, in human readable form.

        - eg:
          - yesterday, last year, previous Wednesday
          - 18/3/53              ## order depending on your locale
          - 10 days ago
          - two days after eom   ## 2 days after end-of-moth
          - Jan                  ## NOTE: after Feb, refers to NEXT January!
        - For available locales, see `date_locale` param
        - see https://github.com/bear/parsedatetime/blob/master/parsedatetime/pdt_locales/base.py)
        """
    ).tag(config=True)

    after_date = trt.Unicode(
        None, allow_none=True,
        help="""Search messages sent before this point in time, in human readable form (see `before_date`)"""
    ).tag(config=True)

    email_infos = trt.List(
        trt.Unicode(),
        default_value=['To', 'Subject', 'Date'],
        help="""
        The email items to fetch for each matched email.

        Usually one of:
            Delivered-To, Received, From, To, Subject, Date,
            Message-Id (always printed)
        """
    ).tag(config=True)

    def _capture_stamper_msg_and_id(self, ts_msg: Text, ts_heads: Text) -> int:
        stamper_id = msg = None
        m = _stamper_id_regex.search(ts_heads)
        if m:
            stamper_id = int(m.group(1))
        m = _stamper_banner_regex.search(ts_msg)
        if m:
            msg = m.group(1)

        return stamper_id, msg

    def _pgp_sig2int(self, sig_id: str) -> int:
        import base64
        import binascii

        sig_bytes = base64.b64decode(sig_id + '==')
        num = int(binascii.b2a_hex(sig_bytes), 16)

        ## Cancel the effect of trailing zeros.

        num = int(str(num).strip('0'))

        return num

    def scan_for_project_name(self, mail_text: Text) -> int:
        """
        Search in the text for any coarsely identifiable project-name (`vehicle_family_id`).

        Use this if :meth:`parse_tstamp_response()` has failed to provide the answer.
        """
        project = None
        all_vfids = self.vfid_extraction_regex.findall(mail_text)
        if all_vfids:
            project = all_vfids[0]
            if not all(i == project for i in all_vfids):
                project = None

        return project

    def parse_tstamped_tag(self, tag_text: Text) -> int:
        """
        :param msg_text:
            The tag as extracted from tstamp response by :meth:`crypto.pgp_split_clearsigned`.
        """
        ## TODO: Schedula to the rescue!

        stag_bytes = _to_bytes(tag_text)
        git_auth = crypto.get_git_auth(self.config)

        ## Allow parsing signed/unsigned reports when --force,
        #
        ver = None
        try:
            ver = git_auth.verify_git_signed(stag_bytes)
            verdict = OrderedDict(sorted(vars(ver).items()))
        except Exception as ex:
            msg = "Failed to extract signed dice-report from tstamp!\n%s" % ex
            if self.force:
                self.log.warning(msg)
                verdict = OrderedDict(sig=msg)

                ## Fall-back assuming report was not signed at all.
                tag = stag_bytes
            else:
                raise CmdException(msg)

        else:
            if not ver:
                #
                ## Do not fail, it might be from an unknown sender.

                msg = "Cannot verify dice-report's signature!\n%s"
                self.log.warning(msg, _mydump(verdict))
            else:
                msg = "The dice-report in timestamp got verified OK: %s"
                self.log.debug(msg, _mydump(verdict))

            tag = verdict['parts']['msg']

        ## Parse dice-report
        #
        from . import project

        try:
            cmsg = project._CommitMsg.parse_commit_msg(_to_str(tag))
            verdict['commit_msg'] = cmsg._asdict()
            verdict['project'] = cmsg.p
            verdict['project_source'] = 'report'
        except Exception as ex:
            if not self.force:
                raise
            else:
                self.log.error("Cannot parse dice-report due to: %s", ex)

        if 'project' not in verdict:
            verdict['project'] = self.scan_for_project_name(tag_text)
            verdict['project_source'] = 'grep'

        return verdict

    def parse_tstamp_response(self, mail_text: Text) -> int:
        ## TODO: Could use dispatcher to parse tstamp-response, if failback routes were working...
        import textwrap as tw

        force = self.force
        stamper_auth = crypto.get_stamper_auth(self.config)
        errlog = self.log.error if self.force else self.log.debug

        ts_ver = stamper_auth.verify_clearsigned(mail_text)
        ts_verdict = vars(ts_ver)
        if not ts_ver:
            errmsg = "Cannot verify timestamp-response's signature due to: %s"
            if not force or not ts_ver.signature_id:  # Need sig-id for decision.
                self.log.debug(errmsg, _mydump(sorted(ts_verdict.items())))
                raise CmdException(errmsg % ts_ver.status)
            else:
                self.log.error(errmsg, _mydump(sorted(ts_verdict.items())))

        if not ts_ver.valid:
            self.log.warning(
                tw.dedent("""
                Timestamp's signature is valid, but not *trusted*!
                  You may sign Timestamp-service's key(81959DB570B61F81) with a *fully* trusted secret-key,
                  or assign *full* trust on JRC's key(TODO:JRC-keyid-here) that already has done so.
                    %s
                """), _mydump(sorted(ts_verdict.items())))

        ts_parts = crypto.pgp_split_clearsigned(mail_text)
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
                    raise CmdException("Timestamp-response had no *stamper-id*: %s" % ts_parts['sigarmor'])

                tag_verdict = {'content_parsing': "failed"}
                tag_verdict = {'project': None}
            else:
                tag_verdict = self.parse_tstamped_tag(tag)

        num = self._pgp_sig2int(ts_ver.signature_id)
        dice100 = num % 100
        decision = 'OK' if dice100 < 90 else 'SAMPLE'

        #self.log.info("Timestamp sig did not verify: %s", _mydump(tag_verdict))
        return OrderedDict([
            ('tstamp', ts_verdict),
            ('report', tag_verdict),
            ('dice', {
                'hexnum': '%X' % num,
                'percent': dice100,
                'decision': decision,
            }),
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
        criteria = list(self.email_criteria)
        if self.subject_prefix:
            criteria.append('Subject "%s"' % self.subject_prefix)
        if is_wait:
            criteria.append(self.wait_criteria)

        before, after = [self.before_date, self.after_date]
        if before or after:
            import parsedatetime as pdt

            c = self.dates_locale and pdt.Constants(self.dates_locale)
            cal = pdt.Calendar(c)

            if before:
                criteria.append('SENTBEFORE "%s"' %
                                parse_as_RFC3501_date(cal, before))
            if after:
                criteria.append('SINCE "%s"' %
                                parse_as_RFC3501_date(cal, after))

        if projects:
            criteria.append(pairwise_ORed(projects,
                                          lambda i: '(SUBJECT "%s")' % i))

        criteria = [c.strip() for c in criteria]
        criteria = [c if c.startswith('(') else '(%s)' % c for c in criteria]
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

        for i, d in enumerate(data):
            if i % 2 == 1:
                assert d == b')', 'Unexpected FETCH data(%i): %s' % (i, d)
                continue
            m = email.message_from_bytes(d[1])

            yield m

    def _get_recved_email_infos(self, mail, verdict_or_ex, verbose=None):
        """Does not raise anything."""
        verbose = verbose is None and self.verbose or verbose

        infos = OrderedDict((i, mail.get(i)) for i in self.email_infos)

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
                    infos['project'] = verdict['report']['project']
                except:
                    pass
                try:
                    infos['dice'] = verdict['dice']
                except:
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
                except:
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

    examples = trt.Unicode(
        """
        Pick an existing dice-report tag to send for timestamping:
            git cat-object tag dices/RL-12-BM3-2016-000/1 | %(cmd_chain)s send

        Await for the response, and paste its content to this command:
            %(cmd_chain)s parse
    """)

    def __init__(self, **kwds):
        kwds.setdefault('subcommands', baseapp.build_sub_cmds(*all_subcmds))
        super().__init__(**kwds)


class SendCmd(baseapp.Cmd):
    """
    Send emails to be timestamped.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<report-file-1> ...]

    - Do not use this command directly (unless experimenting) - prefer the `project tstamp` sub-command.
    - If '-' is given or no files at all, it reads from STDIN.
    - Many options related to sending & receiving the email are expected to be stored in the config-file.
    - Use --verbose to print the timestamped email.
    """

    examples = trt.Unicode("""
        To send a dice-report for a prepared project you have to know the `vehicle_family_id`:

            git  cat-file  tag  tstamps/RL-12-BM3-2017-0001/1 | %(cmd_chain)s
        """)

    dry_run = trt.Bool(
        help="Verify dice-report and login to SMTP-server but do not actually send email to timestamp-service."
    ).tag(config=True)

    def __init__(self, **kwds):
        from pandalone import utils as pndlu

        kwds.setdefault('conf_classes', [TstampSender, crypto.GitAuthSpec])
        kwds.setdefault('cmd_flags', {
            ('n', 'dry-run'): (
                {
                    type(self).__name__: {'dry_run': True},
                },
                pndlu.first_line(type(self).dry_run.help)
            )
        })
        super().__init__(**kwds)

    def run(self, *args):
        from boltons.setutils import IndexedSet as iset
        from pandalone import utils as pndlu

        files = iset(args) or ['-']
        self.log.info("Timestamping '%s'...", tuple(files))

        sender = TstampSender(config=self.config)
        for file in files:
            if file == '-':
                self.log.info("Reading STDIN; paste message verbatim!")
                mail_text = sys.stdin.read()
            else:
                self.log.debug("Reading '%s'...", pndlu.convpath(file))
                with io.open(file, 'rt') as fin:
                    mail_text = fin.read()

            mail = sender.send_timestamped_email(mail_text, dry_run=self.dry_run)
            if self.verbose or self.dry_run:
                return str(mail)


class MailboxCmd(baseapp.Cmd):
    """Lists mailboxes in IMAP server. """

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [TstampReceiver])
        super().__init__(**kwds)

    def run(self, *args):
        ## If `verbose`, too many small details, need flow.
        rcver = TstampReceiver(config=self.config)
        return rcver.list_mailbox(*args)


class RecvCmd(baseapp.Cmd):
    """
    Fetch tstamps from IMAP server and derive *decisions* OK/SAMPLE flags.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<search-term-1> ...]

    - Fetch of emails in one-shot search, or use --wait.
    - The terms are ORed and searched within the email's subject-line;
      tip: use the project name(s).
    """

    examples = trt.Unicode("""
        To search emails in one-shot (bash):
            %(cmd_chain)s --after today "IP-10-AAA-2017-1003"
            %(cmd_chain)s --after "1 year ago" --before "18 March 2017"
            %(cmd_chain)s --after "yesterday" --search 'From "foo@bar.com"'

        To wait for new mails arriving (and not to block console),
        on Linux:
            %(cmd_chain)s --wait &
            ## wait...
            kill %%1  ## Asumming this was the only job started.

        On Windows:
            START \\B %(cmd_chain)s --wait

        and kill with `TASKLIST/TASKKILL or with "Task Manager" GUI.
    """)

    wait = trt.Bool(
        False,
        help="""
        Whether to wait reading IMAP for any email(s) satisfying the criteria and report them.

        WARN:
          Process must be killed afterwards, so start it in the background (see examples).
        NOTE:
          Development flag, use `co2dice project trecv` cmd for type-aprooval.
        """
    ).tag(config=True)

    form = trt.FuzzyEnum(
        ['list', 'raw'],
        allow_none=True,
        help="""If not none, skip tstamp verification and print raw email or `email_infos`."""
    ).tag(config=True)

    def __init__(self, **kwds):
        from pandalone import utils as pndlu

        kwds.setdefault('conf_classes', [
            TstampSender, TstampReceiver,
            crypto.GitAuthSpec, crypto.StamperAuthSpec])
        kwds.setdefault('cmd_flags', {
            'wait': (
                {type(self).__name__: {'wait': True}},
                pndlu.first_line(type(self).wait.help)
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
        kwds.setdefault('cmd_aliases', {
            'before': 'TstampReceiver.before_date',
            'after': 'TstampReceiver.after_date',
            'mailbox': 'TstampReceiver.mailbox',
            'search': 'TstampReceiver.email_criteria',
        })
        super().__init__(**kwds)

    def run(self, *args):
        ## If `verbose`, too many small details, need flow.
        default_flow_style = None if self.verbose else False
        rcver = TstampReceiver(config=self.config)

        emails = rcver.receive_timestamped_emails(self.wait, args,
                                                  read_only=True)
        for mail in emails:
            mid = mail.get('Message-Id')
            if self.form == 'raw':
                yield mail.get_payload()
            else:
                if self.form == 'list':
                    verdict = None
                else:
                    try:
                        verdict = rcver.parse_tstamp_response(mail.get_payload())
                    except Exception as ex:
                        verdict = ex
                        self.log.warning("Failed parsing %s tstamp due to: %s",
                                         mid, ex)

                infos = rcver._get_recved_email_infos(mail, verdict)

                yield _mydump({mid: infos}, default_flow_style=default_flow_style)


class ParseCmd(baseapp.Cmd):
    """
    Verifies and derives the *decision* OK/SAMPLE flag from tstamped-response email.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<tstamped-file-1> ...]

    - If '-' is given or no files at all, it reads from STDIN.
    """
    examples = trt.Unicode("""cat <mail> | %(cmd_chain)s""")

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [TstampReceiver,
                                         crypto.GitAuthSpec, crypto.StamperAuthSpec])
        super().__init__(**kwds)

    def run(self, *args):
        from boltons.setutils import IndexedSet as iset
        from pandalone import utils as pndlu

        default_flow_style = None if self.verbose else False
        files = iset(args) or ['-']
        self.log.info("Parsing '%s'...", tuple(files))

        rcver = TstampReceiver(config=self.config)
        for file in files:
            if file == '-':
                self.log.info("Reading STDIN; paste message verbatim!")
                mail_text = sys.stdin.read()
            else:
                self.log.debug("Reading '%s'...", pndlu.convpath(file))
                with io.open(file, 'rt') as fin:
                    mail_text = fin.read()

            res = rcver.parse_tstamp_response(mail_text)

            yield _mydump(res, default_flow_style=default_flow_style)


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
        from pandalone import utils as pndlu

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
