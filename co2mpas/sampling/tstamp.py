#!/usr/bin/env python
#
# Copyright 2014-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""A *report* contains the co2mpas-run values to time-stamp and disseminate to TA authorities & oversight bodies."""
from collections import (
    defaultdict, OrderedDict, namedtuple, Mapping)  # @UnusedImport
from collections import namedtuple
import imaplib
import io
import re
import smtplib
import sys
from typing import (
    List, Sequence, Iterable, Text, Tuple, Dict, Callable)  # @UnusedImport

import traitlets as trt
import traitlets.config as trtc

from . import CmdException, baseapp, dice, crypto, project
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport


#
###################
##     Specs     ##
###################


class TstampSpec(dice.DiceSpec, crypto.GnuPGSpec):
    """Common parameters and methods for both SMTP(sending emails) & IMAP(receiving emails)."""

    user_pswd = crypto.Cipher(
        help="""
        The SMTP/IMAP server's password matching `user_name` param.

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
                      self.user_name, host, srv_kwds or '')
        return srv_cls(host, **srv_kwds)

    def check_login(self):
        ok = False
        with self.make_server() as srv:
            try:
                srv.login(self.user_name, self.decipher('user_pswd'))
                ok = True
            finally:
                self.log.info("Login %s: %s@%s ok? %s", type(srv).__name__,
                              self.user_name, srv.sock, ok)


class TstampSender(TstampSpec):
    """SMTP & timestamp parameters and methods for sending dice emails."""

    login = trt.CaselessStrEnum(
        'login simple'.split(), default_value=None, allow_none=True,
        help="""Which SMTP mechanism to use to authenticate: [ login | simple | <None> ]. """
    ).tag(config=True)

    timestamping_addresses = trt.List(
        type=trtc.Unicode(), allow_none=False,
        help="""The plain email-address(s) of the timestamp service must be here. Ask JRC to provide that. """
    ).tag(config=True)

    x_recipients = trt.List(
        type=trtc.Unicode(), allow_none=False,
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
        return smtplib.SMTP_SSL if self.ssl else smtplib.SMTP

    def send_timestamped_email(self, msg):
        msg = self._append_x_recipients(msg)

        self.log.info("Timestamping %d-char email from '%s' to %s-->%s",
                      len(msg),
                      self.from_address,
                      self.timestamping_addresses,
                      self.x_recipients)
        mail = self._prepare_mail(msg)

        with self.make_server() as srv:
            srv.login(self.user_name, self.decipher('user_pswd'))

            srv.send_message(mail)

        return mail


_stamper_id_regex = re.compile(r"Comment: Stamper Reference Id: (\d+)")
_stamper_banner_regex = re.compile(r"^#{56}\r?\n(?:^#[^\n]*\n)+^#{56}\r?\n\r?\n\r?\n(.*)",
                                   re.MULTILINE | re.DOTALL)


#DiceResponse = namedtuple('DiceResponse',
#                          '')
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
        ts_ver = self.GPG.verify(mail_text)
        if not ts_ver:
            self.log.error("Cannot verify timestamp-response's signature due to: %s", pformat(vars(ts_ver)))
            raise ValueError("Cannot verify timestamp-reponse signature due to: %s" % ts_ver.status)

        csig = crypto.split_clearsigned(mail_text)
        stamper_id, tag = self._capture_stamper_msg_and_id(csig.msg, csig.sigheads)
        if not stamper_id:
            self.log.error("Timestamp-response had no *stamper-id*: %s\n%s",
                           pformat(csig), pformat(vars(ts_ver)))
            raise ValueError("Timestamp-response had no *stamper-id*: %s" % csig.sig)

        # Verify inner tag.
        if tag:
            tag_ver = self.verify_git_signed(tag.encode('utf-8'))

        num = self._pgp_sig2int(ts_ver.signature_id)
        dice100 = num % 100
        decision = 'OK' if dice100 < 90 else 'SAMPLE'

        #self.log.info("Timestamp sig did not verify: %s", pformat(vars(ts_ver)))
        return {
            'tstamp': pformat(vars(ts_ver)),
            'tstamp_sig': csig.sig,
            'tstamp_sig_date': ts_ver.creation_date,
            'stamper_id': stamper_id,
            'tag': pformat(vars(tag_ver)),
            'tag_keyid': tag_ver.key_id,
            'tag_sig_date': ts_ver.creation_date,
            'dice': num,
            'dice_100': dice100,
            'dice_flag': decision,
        }

    def choose_server_class(self):
        return imaplib.IMAP4_SSL if self.ssl else imaplib.IMAP4

    # TODO: IMAP receive, see https://pymotw.com/2/imaplib/ for IMAP example.
    def receive_timestamped_email(self):
        with self.make_server() as srv:
            repl = srv.login(self.user_name, self.decipher('user_pswd'))
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
    """Commands to manage the communications with the Timestamp server."""

    class SendCmd(_Subcmd):
        """
        Send emails to be timestamped and parse back the response.

        The time-stamp service is used to disseminate the dice-report to the TA authorities & oversight bodies.
        From its response the sampling decision will be deduced.

        Many options related to sending & receiving the email are expected to be stored in the config-file.

        - The sending command is NOT to be used directly (just for experimenting).
          If neither `--file` nor `--project` given, reads dice-report from stdin.
        - The receiving command waits for the response.and returns: 1: SAMPLE | 0: NO-SAMPLE
          Any other code is an error-code - communicate it to JRC.

        SYNTAX
            co2dice tstamp send [ file=<dice-report-file> ]
            co2dice tstamp send [ file=<dice-report-file> ]
            co2dice tstamp recv
        """

        examples = trt.Unicode("""
            To wait for the response after you have sent the dice-report, use this bash commands:

                co2dice tstamp recv
                if [ $? -eq 0 ]; then
                    echo "NO-SAMPLE"
                elif [ $? -eq 1 ]; then
                    echo "SAMPLE!"
                else
                    echo "ERROR CODE: $?"
            """)

        file = trt.Unicode(
            None, allow_none=True,
            help="""If not null, read mail body from the specified file."""
        ).tag(config=True)

        def __init__(self, **kwds):
            with self.hold_trait_notifications():
                dkwds = {
                    'conf_classes': [project.ProjectsDB, TstampSender],
                    'cmd_aliases': {
                        'file': 'SendCmd.file',
                    },
                }
                dkwds.update(kwds)
                super().__init__(**dkwds)

        def run(self, *args):
            nargs = len(args)
            if nargs > 0:
                raise CmdException("Cmd '%s' takes no arguments, received %d: %r!"
                                   % (self.name, len(args), args))

            file = self.file

            sender = TstampSender(config=self.config)
            if not file:
                self.log.warning("Time-stamping STDIN; paste message verbatim!")
                mail_text = sys.stdin.read()
            else:
                self.log.info('Time-stamping files %r...', file)
                with io.open(file, 'rt') as fin:
                    mail_text = fin.read()

            sender.send_timestamped_email(mail_text)

    class ParseCmd(_Subcmd):
        """
        Derives the *decision* OK/SAMPLE flag from time-stamped email.

        SYNTAX
            cat <mail> | co2dice tstamp parse
        """

        def run(self, *args):
            nargs = len(args)
            if nargs > 0:
                raise CmdException("Cmd '%s' takes no arguments, received %d: %r!"
                                   % (self.name, len(args), args))

            from pprint import pformat

            rcver = TstampReceiver(config=self.config)
            mail_text = sys.stdin.read()
            res = rcver.parse_tsamp_response(mail_text)

            return pformat(res)

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


all_subcmds = (TstampCmd.SendCmd, TstampCmd.ParseCmd, TstampCmd.LoginCmd,)
