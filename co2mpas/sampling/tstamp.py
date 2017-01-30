#!/usr/bin/env python
#
# Copyright 2014-2016 European Commission (JRC);
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
    List, Sequence, Iterable, Text, Tuple, Dict, Callable)  # @UnusedImport

import traitlets as trt

from . import CmdException, baseapp, dice, crypto, project
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport


#
###################
##     Specs     ##
###################


class TstampSpec(dice.DiceSpec):
    """Common parameters and methods for both SMTP(sending emails) & IMAP(receiving emails)."""

    user_account = trt.Unicode(
        None, allow_none=False,
        help="""The username for the account on the SMTP/IMAP server!"""
    ).tag(config=True)

    user_pswd = crypto.Cipher(
        help="""
        The SMTP/IMAP server's password matching `user_account` param.

        For *GMail* with 2-factor authentication, see:
            https://support.google.com/accounts/answer/185833
        """
    ).tag(config=True)

    host = trt.Unicode(
        allow_none=False,
        help="""The SMTP/IMAP server, e.g. 'smtp.gmail.com'."""
    ).tag(config=True)

    port = trt.Int(
        None,
        allow_none=True,
        help="""
            The SMTP/IMAP server's port, usually 587/465 for SSL, 25 otherwise.
            If undefined, does its best.
        """).tag(config=True)

    ssl = trt.Bool(
        True,
        help="""Whether to talk TLS/SSL to the SMTP/IMAP server; configure `port` separately!"""
    ).tag(config=True)

    mail_kwds = trt.Dict(
        help="""
            Any extra key-value pairs passed to the SMTP/IMAP mail-client libraries.
            For instance, :class:`smtlib.SMTP_SSL` and :class:`smtlib.IMAP4_SSL`
            support `keyfile` and `timeout`, while SMTP/SSL support additionally
            `local_hostname` and `source_address`.
        """
    ).tag(config=True)

    @trt.validate('host')
    def _valid_host(self, proposal):
        value = proposal['value']
        if not value:
            raise trt.TraitError('%s.%s must not be empty!'
                                 % (proposal['owner'].name, proposal['trait'].name))
        return value

    def choose_server_class(self):
        raise NotImplemented()

    def make_server(self):
        host = self.host
        port = self.port
        srv_kwds = self.mail_kwds.copy()
        if port is not None:
            srv_kwds['port'] = port
        srv_cls = self.choose_server_class()

        self.log.info("Login %s: %s@%s(%s)...", srv_cls.__name__,
                      self.user_account, host, srv_kwds or '')
        return srv_cls(host, **srv_kwds)

    def check_login(self):
        ok = False
        with self.make_server() as srv:
            try:
                srv.login(self.user_account, self.decipher('user_pswd'))
                ok = True
            finally:
                self.log.info("Login %s: %s@%s ok? %s", type(srv).__name__,
                              self.user_account, srv.sock, ok)


class TstampSender(TstampSpec):
    """SMTP & timestamp parameters and methods for sending dice emails."""

    login = trt.CaselessStrEnum(
        'login simple'.split(), default_value=None, allow_none=True,
        help="""Which SMTP mechanism to use to authenticate: [ login | simple | <None> ]. """
    ).tag(config=True)

    timestamping_addresses = trt.List(
        type=trt.Unicode(), allow_none=False,
        help="""The plain email-address(s) of the timestamp service must be here. Ask JRC to provide that. """
    ).tag(config=True)

    x_recipients = trt.List(
        type=trt.Unicode(), allow_none=False,
        help="""The plain email-address of the receivers of the timestamped response. Ask JRC to provide that."""
    ).tag(config=True)

    subject = trt.Unicode(
        '[dice test]', allow_none=False,
        help="""The subject-line to use for email sent to timestamp service. """
    ).tag(config=True)

    from_address = trt.Unicode(
        None,
        allow_none=True,
        help="""Your email-address to use as `From:` for timestamp email, or none to use `user_email`.
        Specify you correct address, or else you will never receive the sampling flag!
        """
    ).tag(config=True)

    def _append_x_recipients(self, msg):
        x_recs = '\n'.join('X-Stamper-To: %s' % rec for rec in self.x_recipients)
        msg = "%s\n\n%s" % (x_recs, msg)

        return msg

    def _prepare_mail(self, msg):
        from email.mime.text import MIMEText

        mail = MIMEText(msg, 'plain')
        mail['Subject'] = self.subject
        mail['From'] = self.from_address or self.user_email
        mail['To'] = ', '.join(self.timestamping_addresses)

        return mail

    def choose_server_class(self):
        import smtplib

        return smtplib.SMTP_SSL if self.ssl else smtplib.SMTP

    def send_timestamped_email(self, msg, dry_run=False):
        from pprint import pformat

        git_auth = crypto.get_git_auth(self.config)
        ver = git_auth.verify_git_signed(msg.encode('utf-8'))
        verdict = pformat(vars(ver))
        if not ver:
            if self.force:
                self.log.warning("Content to timestamp failed signature verification!  %s",
                                 verdict)
            else:
                raise CmdException("Content to timestamp failed signature verification!\n  %s"
                                   % verdict)
        else:
            self.log.info("Content to timestamp signed OK: %s" % verdict)

        msg = self._append_x_recipients(msg)
        mail = self._prepare_mail(msg)

        with self.make_server() as srv:
            srv.login(self.user_account, self.decipher('user_pswd'))

            from logging import WARNING, INFO
            level = WARNING if dry_run else INFO
            prefix = "DRY-RUN:  No email has been sent!\n  " if dry_run else ''
            self.log.log(level, "%sTimestamping %d-char email from '%s' to %s-->%s",
                         prefix, len(msg), self.from_address,
                         self.timestamping_addresses, self.x_recipients)
            if not dry_run:
                srv.send_message(mail)

        return mail


_stamper_id_regex = re.compile(r"Comment: Stamper Reference Id: (\d+)")
_stamper_banner_regex = re.compile(r"^#{56}\r?\n(?:^#[^\n]*\n)+^#{56}\r?\n\r?\n\r?\n(.*)",
                                   re.MULTILINE | re.DOTALL)


class TstampReceiver(TstampSpec):
    """IMAP & timestamp parameters and methods for receiving & parsing dice-report emails."""

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

    def parse_tsamp_response(self, mail_text: Text) -> int:
        from pprint import pformat
        import textwrap as tw

        stamper_auth = crypto.get_stamper_auth(self.config)
        ts_ver = stamper_auth.verify_clearsigned(mail_text)
        ts_verdict = vars(ts_ver)
        if not ts_ver:
            self.log.error("Cannot verify timestamp-response's signature due to: %s", pformat(ts_verdict))
            if not self.force or not ts_ver.signature_id:  # Need sig-id for decision.
                raise ValueError(
                    "Cannot verify timestamp-reponse signature due to: %s" % ts_ver.status)
        if not ts_ver.valid:
            self.log.warning(
                tw.dedent("""\
                Timestamp's signature is valid, but not *trusted*!
                  You may sign Timestamp-service's key(81959DB570B61F81) with a *fully* trusted secret-key,
                  or assign *full* trust on JRC's key(TODO:JRC-keyid-here) that already has done so.
                    %s
                """), pformat(ts_verdict))

        csig = crypto.pgp_split_clearsigned(mail_text)
        stamper_id, tag = self._capture_stamper_msg_and_id(csig.msg, csig.sigheads)
        if not stamper_id:
            self.log.error("Timestamp-response had no *stamper-id*: %s\n%s",
                           pformat(csig), pformat(ts_verdict))
            if not self.force:
                raise ValueError("Timestamp-response had no *stamper-id*: %s" % csig.sig)

        ## Verify inner tag.
        #
        if tag:
            git_auth = crypto.get_git_auth(self.config)
            tag_ver = git_auth.verify_git_signed(tag.encode('utf-8'))
            tag_verdict = vars(tag_ver)
            if not tag_ver:
                self.log.warning(
                    "Cannot verify dice-report's signature due to: %s", pformat(tag_verdict))

        num = self._pgp_sig2int(ts_ver.signature_id)
        dice100 = num % 100
        decision = 'OK' if dice100 < 90 else 'SAMPLE'

        #self.log.info("Timestamp sig did not verify: %s", pformat(tag_verdict))
        return {
            'tstamp': {
                'sig': ts_verdict,
                'sig_armor': csig.sig,
                'stamper_id': stamper_id,
            },
            'tag': {
                'sig': tag_verdict,
            },
            'dice_hex': '%X' % num,
            'dice_%100': dice100,
            'dice_decision': decision,
        }

    def choose_server_class(self):
        import imaplib

        return imaplib.IMAP4_SSL if self.ssl else imaplib.IMAP4

    # TODO: IMAP receive, see https://pymotw.com/2/imaplib/ for IMAP example.
    def receive_timestamped_email(self):
        with self.make_server() as srv:
            repl = srv.login(self.user_account, self.decipher('user_pswd'))
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
    @property
    def projects_db(self):
        p = project.ProjectsDB.instance(config=self.config)
        p.config = self.config
        return p


class TstampCmd(baseapp.Cmd):
    """
    Commands to manage the communications with the Timestamp server.

    The time-stamp service is used to disseminate the *dice-report* to the TA authorities
    and to oversight bodies.
    From its response the *sampling decision* is be deduced.
    """

    class SendCmd(_Subcmd):
        """
        Send emails to be timestamped.

        SYNTAX
            co2dice tstamp send [OPTIONS] [<file-1> ...]

        - Do not use this command directly (unless experimenting) - preffer the `co2dice project tstamp` command.
        - If no files are given, it reads from STDIN.
        - Many options related to sending & receiving the email are expected to be stored in the config-file.
        """

        examples = trt.Unicode("""
            To send a dice-report for a prepared project you have to know the `vehicle_family_id`:
                git  cat-file  tag  tstamps/RL-12-BM3-2017-0001/1 | co2dice tstamp send
            """)

        dry_run = trt.Bool(
            help="Verify dice-report and login to SMTP-server but do not actually send email to timestamp-service."
        ).tag(config=True)

        def __init__(self, **kwds):
            from pandalone import utils as pndlu

            kwds.setdefault('conf_classes', [TstampSender])
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
            self.log.info('Timestamping %r...', args)

            files = self.extra_args
            if not files:
                files = '-'

            sender = TstampSender(config=self.config)
            for file in files:
                if file == '-':
                    self.log.info("TimeReading STDIN; paste message verbatim!")
                    mail_text = sys.stdin.read()
                else:
                    with io.open(file, 'rt') as fin:
                        mail_text = fin.read()

                sender.send_timestamped_email(mail_text, dry_run=self.dry_run)

    class ParseCmd(_Subcmd):
        """
        Derives the *decision* OK/SAMPLE flag from time-stamped email.

        SYNTAX
            co2dice tstamp send [OPTIONS] [<tstamp-response-file-1> ...]

        """
        examples = trt.Unicode("""cat <mail> | co2dice tstamp parse""")

        def __init__(self, **kwds):
            kwds.setdefault('conf_classes', [TstampReceiver])
            super().__init__(**kwds)

        def run(self, *args):
            from pprint import pformat

            self.log.info('Timestamping %r...', args)

            files = self.extra_args
            if not files:
                files = '-'

            rcver = TstampReceiver(config=self.config)
            for file in files:
                if file == '-':
                    self.log.info("TimeReading STDIN; paste message verbatim!")
                    mail_text = sys.stdin.read()
                else:
                    with io.open(file, 'rt') as fin:
                        mail_text = fin.read()

                resp = rcver.parse_tsamp_response(mail_text)

                yield pformat(resp)

    class RecvCmd(_Subcmd):
        """
        TODO: tstamp receive command

        The receiving command waits for the response.and returns: 1: SAMPLE | 0: NO-SAMPLE
        Any other code is an error-code - communicate it to JRC.

        To wait for the response after you have sent the dice-report, use this bash commands:

            co2dice tstamp recv
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

        def __init__(self, **kwds):
            with self.hold_trait_notifications():
                kwds.setdefault('conf_classes', [TstampSender, TstampReceiver])
                super().__init__(**kwds)

        def run(self, *args):
            nargs = len(args)
            if nargs > 0:
                raise CmdException("Cmd '%s' takes no arguments, received %d: %r!"
                                   % (self.name, len(args), args))

            sender = TstampSender(config=self.config)
            sender.check_login()

            rcver = TstampReceiver(config=self.config)
            rcver.check_login()

    def __init__(self, **kwds):
        kwds.setdefault('subcommands', baseapp.build_sub_cmds(*all_subcmds))
        super().__init__(**kwds)


all_subcmds = (
    TstampCmd.LoginCmd,
    TstampCmd.SendCmd,
    TstampCmd.RecvCmd,
    TstampCmd.ParseCmd
)
