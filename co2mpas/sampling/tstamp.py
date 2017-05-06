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


#
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
        - False:      Do not use any encryption;  better use `skip_auth` param,
                      not to reveal credentials in plain-text.

        Tip: Microsoft Outlook/Yahoo servers use STARTTLS.
        See also:
         - https://www.fastmail.com/help/technical/ssltlsstarttls.html
         - http://forums.mozillazine.org/viewtopic.php?t=2730845
        """
    ).tag(config=True)

    skip_auth = trt.Bool(
        False,
        help="""Whether not to send any user/password credentials to the server"""
    ).tag(config=True)

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
          prefer a real IP and set `socks_skip_resolve=True`, or else,
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

    @property
    def user_account_resolved(self):
        return self.user_account is not None and self.user_account or self.user_email

    def choose_server_class(self):
        raise NotImplemented()

    def make_server(self, dry_run):
        from unittest.mock import MagicMock

        host = self.host
        port = self.port
        srv_kwds = self.mail_kwds.copy()
        if port is not None:
            srv_kwds['port'] = port
        srv_cls = self.choose_server_class()
        _, is_startssl = self._ssl_resolved()

        self.log.info("Connecting to %s%s: %s@%s(%s)...", srv_cls.__name__,
                      '(STARTTLS)' if is_startssl else '',
                      self.user_account_resolved, host, srv_kwds or '')
        srv = MagicMock() if dry_run else srv_cls(host, **srv_kwds)

        return srv

    def check_login(self, dry_run):
        """Logs only and returns true/false; does not throw any exception!"""
        ok = False
        srv_name = srv_sock = ''
        try:
            with self.make_server(dry_run) as srv:
                srv_name = type(srv).__name__
                srv_sock = srv.sock
                self.log.debug("Authenticating %s: %s@%s ...", srv_name,
                               self.user_account_resolved, srv.sock)
                ok = self.login_srv(srv, self.user_account_resolved, self.decipher('user_pswd'))

            return True
        except Exception as ex:
            ok = ex
            self.log.error("Connection FAILED due to: %s", ex, exc_info=ex)

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
        help="Deprecated, but still functional.  Prefer `TstampSender.tstamp_recipients` list  instead."
    ).tag(config=True)

    subject = trt.Unicode(
        help="""The subject-line to use for email sent to timestamp service. """
    ).tag(config=True)

    from_address = trt.Unicode(
        None, allow_none=True,
        help="""Your email-address to use as `From:` for timestamp email, or none to use `user_email`.
        Specify you correct address, or else you will never receive the tstamped-response!
        """
    ).tag(config=True)

    def __init__(self, *args, **kwds):
        from .dice import DiceSpec
        self._register_validator(
            DiceSpec._is_not_empty,
            ['host', 'subject'])
        self._register_validator(
            DiceSpec._warn_deprecated,
            ['x_recipients', 'timestamping_addresses'])
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
        mail['Subject'] = '%s %s' % (self.subject, subject_suffix)
        mail['From'] = self.from_address or self.user_email
        mail['To'] = ', '.join(self._tstamper_address_resolved)
        if self.cc_addresses:
            mail['Cc'] = ', '.join(self.cc_addresses)
        if self.bcc_addresses:
            mail['Cc'] = ', '.join(self.bcc_addresses)

        return mail

    def choose_server_class(self):
        import smtplib

        self.monkeypatch_socks_module(smtplib)
        is_ssl, _ = self._ssl_resolved()
        return smtplib.SMTP_SSL if is_ssl else smtplib.SMTP

    def login_srv(self, srv, user, pswd):
        srv.set_debuglevel(self.verbose)
        _, is_starttls = self._ssl_resolved()
        if is_starttls:
            self.log.debug('STARTTLS...')
            srv.starttls(keyfile=self.mail_kwds.get('keyfile'),
                         certfile=self.mail_kwds.get('certfile'),
                         context=None)

        srv.noop()

        if not self.skip_auth:
            return srv.login(user, pswd)

    def send_timestamped_email(self, msg: Union[str, bytes], subject_suffix='', dry_run=False):
        from pprint import pformat

        msg_bytes = msg if isinstance(msg, bytes) else msg.encode('utf-8')
        git_auth = crypto.get_git_auth(self.config)
        ver = git_auth.verify_git_signed(msg_bytes)
        verdict = None if ver is None else pformat(vars(ver))
        if not ver:
            if self.force:
                self.log.warning("Content to timestamp failed signature verification!  %s",
                                 verdict)
            else:
                raise CmdException("Content to timestamp failed signature verification!\n  %s"
                                   % verdict)
        else:
            self.log.debug("Content to timestamp gets verified OK: %s" % verdict)

        msg = self._append_tstamp_recipients(msg)
        mail = self._prepare_mail(msg, subject_suffix)

        with self.make_server(dry_run) as srv:
            self.login_srv(srv, self.user_account_resolved, self.decipher('user_pswd'))

            from logging import WARNING, INFO
            level = WARNING if dry_run else INFO
            prefix = "DRY-RUN:  No email has been sent!\n  " if dry_run else ''
            self.log.log(level, "%sTimestamping %d-char email from '%s' to %s-->%s",
                         prefix, len(msg), self.from_address,
                         self._tstamper_address_resolved,
                         self.tstamp_recipients + self.x_recipients)
            srv.send_message(mail)

        return mail


_stamper_id_regex = re.compile(r"Comment: Stamper Reference Id: (\d+)")
_stamper_banner_regex = re.compile(r"^#{56}\r?\n(?:^#[^\n]*\n)+^#{56}\r?\n\r?\n\r?\n(.*)",
                                   re.MULTILINE | re.DOTALL)  # @UndefinedVariable


class TstampReceiver(TstampSpec):
    """IMAP & timestamp parameters and methods for receiving & parsing dice-report emails."""

    cram_md5_login = trt.Bool(
        False,
        help="""Whether to authenticate using CRAM-MD5 method; leave it False, not oftenly used."""
    ).tag(config=True)

    vfid_extraction_regex = trt.CRegExp(
        r"vehicle_family_id[^\n]+((?:IP|RL|RM|PR)-\d{2}-\w{2,3}-\d{4}-\d{4})",  # See also co2mpas.io.schema!
        help=""""An approximate way to get the project if timestamp parsing has failed. """
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
        Search in the text for any coarsly identifiable project-name (`vehicle_family_id`).

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
            The tag as extracted from tstamp response email by :meth:`crypto.pgp_split_clearsigned`.
        """
        from pprint import pformat

        git_auth = crypto.get_git_auth(self.config)
        tag_ver = git_auth.verify_git_signed(tag_text.encode('utf-8'))
        tag_verdict = OrderedDict({} if tag_ver is None else sorted(vars(tag_ver).items()))
        if not tag_ver:
            ## Do not fail, it might be from an unknown sender.
            #
            self.log.warning(
                "Cannot verify dice-report's signature due to: %s", pformat(tag_verdict))

        ## Parse dice-report
        #
        from . import project

        tag_csig = tag_verdict['parts']
        tag = tag_csig['msg']
        try:
            cmsg = project._CommitMsg.parse_commit_msg(tag.decode('utf-8'))
            tag_verdict['commit_msg'] = cmsg._asdict()
            tag_verdict['project'] = cmsg.p
            tag_verdict['project_source'] = 'report'
        except Exception as ex:
            if not self.force:
                raise
            else:
                self.log.error("Cannot parse dice-report due to: %s", ex)

        if 'project' not in tag_verdict:
            tag_verdict['project'] = self.scan_for_project_name(tag_text)
            tag_verdict['project_source'] = 'grep'

        return tag_verdict

    def parse_tstamp_response(self, mail_text: Text) -> int:
        ## TODO: Could use dispatcher to parse tstamp-response, if failback routes were working...
        import textwrap as tw
        from pprint import pformat

        force = self.force
        stamper_auth = crypto.get_stamper_auth(self.config)

        ts_ver = stamper_auth.verify_clearsigned(mail_text)
        ts_verdict = vars(ts_ver)
        if not ts_ver:
            self.log.error("Cannot verify timestamp-response's signature due to: %s", pformat(ts_verdict))
            if not force or not ts_ver.signature_id:  # Need sig-id for decision.
                raise CmdException(
                    "Cannot verify timestamp-reponse signature due to: %s" % ts_ver.status)
        if not ts_ver.valid:
            self.log.warning(
                tw.dedent("""\
                Timestamp's signature is valid, but not *trusted*!
                  You may sign Timestamp-service's key(81959DB570B61F81) with a *fully* trusted secret-key,
                  or assign *full* trust on JRC's key(TODO:JRC-keyid-here) that already has done so.
                    %s
                """), pformat(ts_verdict))

        ts_parts = crypto.pgp_split_clearsigned(mail_text)
        ts_verdict['parts'] = ts_parts
        if not ts_parts:
            self.log.error("Cannot parse timestamp-response:"
                           "\n  mail-txt: %s\n\n  ts-verdict: %s",
                           mail_text, pformat(ts_verdict))
            if not force:
                raise CmdException(
                    "Cannot parse timestamp-response!")
            stamper_id = tag_verdict = None
        else:
            stamper_id, tag = self._capture_stamper_msg_and_id(ts_parts['msg'], ts_parts['sigarmor'])
            ts_verdict['stamper_id'] = stamper_id
            if not tag:
                self.log.error("Failed parsing response content and/or stamper-id: %s\n%s",
                               pformat(ts_parts), pformat(ts_verdict))
                if not force:
                    raise CmdException("Timestamp-response had no *stamper-id*: %s" % ts_parts['sigarmor'])

                tag_verdict = {'content_parsing': "failed"}
                tag_verdict = {'project': None}
            else:
                tag_verdict = self.parse_tstamped_tag(tag)

        num = self._pgp_sig2int(ts_ver.signature_id)
        dice100 = num % 100
        decision = 'OK' if dice100 < 90 else 'SAMPLE'

        #self.log.info("Timestamp sig did not verify: %s", pformat(tag_verdict))
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
        is_ssl, _ = self._ssl_resolved()
        return imaplib.IMAP4_SSL if is_ssl else imaplib.IMAP4

    def login_srv(self, srv, user, pswd):
        srv.debug = int(self.verbose)
        _, is_starttls = self._ssl_resolved()
        if is_starttls:
            self.log.debug('STARTTLS...')
            srv.starttls()

        srv.noop()

        if not self.skip_auth:
            if self.cram_md5_login:
                return srv.login_cram_md5(user, pswd)
            else:
                return srv.login(user, pswd)

    # TODO: IMAP receive, see https://pymotw.com/2/imaplib/ for IMAP example.
    def receive_timestamped_email(self, dry_run):
        with self.make_server(dry_run) as srv:
            repl = self.login_srv(srv, self.user_account_resolved, self.decipher('user_pswd'))
            """GMAIL-2FAuth: imaplib.error: b'[ALERT] Application-specific password required:
            https://support.google.com/accounts/answer/185833 (Failure)'"""
            self.log.debug("Sent IMAP user/pswd, server replied: %s", repl)

            resp = srv.list()
            print(resp[0])
            return [srv.retr(i + 1) for i, msg_id in zip(range(10), resp[1])]  # @UnusedVariable


###################
##    Commands   ##
###################


class _Subcmd(baseapp.Cmd):
    pass


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

    class SendCmd(_Subcmd):
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
                        'SendCmd': {'dry_run': True},
                    },
                    pndlu.first_line(TstampCmd.SendCmd.dry_run.help)
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

    class ParseCmd(_Subcmd):
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
            from pprint import pformat
            from pandalone import utils as pndlu

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

                resp = rcver.parse_tstamp_response(mail_text)

                yield pformat(resp)

    class RecvCmd(_Subcmd):
        """
        TODO: tstamp receive command

        The receiving command waits for the response.and returns: 1: SAMPLE | 0: NO-SAMPLE
        Any other code is an error-code - communicate it to JRC.

        To wait for the response after you have sent the dice-report, use this bash commands:

            %(cmd_chain)s
            if [ $? -eq 0 ]; then
                echo "NO-SAMPLE"
            elif [ $? -eq 1 ]; then
                echo "SAMPLE!"
            else
                echo "ERROR CODE: $?"
        """
        def run(self, *args):
            raise CmdException("Not implemented yet!")

    class LoginCmd(_Subcmd):
        """Attempts to login into SMTP server. """

        dry_run = trt.Bool(
            help="Verify dice-report and login to SMTP-server but do not actually send email to timestamp-service."
        ).tag(config=True)

        def __init__(self, **kwds):
            from pandalone import utils as pndlu

            kwds.setdefault('conf_classes', [TstampSender, TstampReceiver])
            kwds.setdefault('cmd_flags', {
                ('n', 'dry-run'): (
                    {
                        'SendCmd': {'dry_run': True},
                    },
                    pndlu.first_line(TstampCmd.SendCmd.dry_run.help)
                )
            })
            super().__init__(**kwds)

        def run(self, *args):
            nargs = len(args)
            if nargs > 0:
                raise CmdException("Cmd '%s' takes no arguments, received %d: %r!"
                                   % (self.name, len(args), args))

            return (s.check_login(self.dry_run)
                    for s in [TstampSender(config=self.config),
                              TstampReceiver(config=self.config)])

    def __init__(self, **kwds):
        kwds.setdefault('subcommands', baseapp.build_sub_cmds(*all_subcmds))
        super().__init__(**kwds)


all_subcmds = (
    TstampCmd.LoginCmd,
    TstampCmd.SendCmd,
    TstampCmd.RecvCmd,
    TstampCmd.ParseCmd
)
