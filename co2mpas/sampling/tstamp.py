#!/usr/bin/env python
#
# Copyright 2014-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""A *report* contains the co2mpas-run values to time-stamp and disseminate to TA authorities & oversight bodies."""
#
###################
##     Specs     ##
###################

from collections import (
    defaultdict, OrderedDict, namedtuple, Mapping)  # @UnusedImport
import imaplib
import io
import re
import smtplib
import sys
import tempfile
from typing import (
    List, Sequence, Iterable, Text, Tuple, Dict, Callable)  # @UnusedImport

import traitlets as trt
import traitlets.config as trtc

from . import CmdException, baseapp, dice, crypto, project
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport


class TStampSpec(dice.DiceSpec):
    """Common parameters and methods for both SMTP(sending emails) & IMAP(receiving emails)."""

    user_pswd = crypto.Cipher(
        allow_none=True,
        help="""The SMTP/IMAP server's password matching `user_name` param."""
    ).tag(config=True)

    host = trt.Unicode(
        None, _none=False,
        help="""The SMTP/IMAP server, e.g. 'smtp.gmail.com'."""
    ).tag(config=True)

    port = trt.Int(
        allow_none=True,
        help="""
            The SMTP/IMAP server's port, usually 587/465 for SSL, 25 otherwise.
            If undefined, defaults to 0 and does its best.
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


class TstampSender(TStampSpec):
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
        '[dice test]',
        help="""The subject-line to use for email sent to timestamp service. """
    ).tag(config=True)

    from_address = trt.Unicode(
        None,
        allow_none=False,
        help="""Your email-address to use as `From:` for email sent to timestamp service.
        Specify you correct address, or else you will never receive the sampling flag!
        """
    ).tag(config=True)

    def _sign_msg_body(self, text):
        return text

    def _append_x_recipients(self, msg):
        x_recs = '\n'.join('X-Stamper-To: %s' % rec for rec in self.x_recipients)
        msg = "%s\n\n%s" % (x_recs, msg)

        return msg

    def _prepare_mail(self, msg):
        from email.mime.text import MIMEText

        mail = MIMEText(msg, 'plain')
        mail['Subject'] = self.subject
        mail['From'] = self.from_address
        mail['To'] = ', '.join(self.timestamping_addresses)

        return mail

    def send_timestamped_email(self, msg):
        msg = self._sign_msg_body(msg)

        msg = self._append_x_recipients(msg)

        host = self.host
        port = self.port
        srv_kwds = self.mail_kwds.copy()
        if port is not None:
            srv_kwds['port'] = port

        self.log.info("Timestamping %d-char email from %r through %r%s to %s-->%s",
                      len(msg),
                      self.from_address,
                      host, srv_kwds or '',
                      self.timestamping_addresses,
                      self.x_recipients)
        mail = self._prepare_mail(msg)

        with (smtplib.SMTP_SSL(host, **srv_kwds)
              if self.ssl else smtplib.SMTP(host, **srv_kwds)) as srv:
            srv.login(self.user_name, TstampSender.user_pswd.decrypted(self))

            srv.send_message(mail)
        return mail


_PGP_SIG_REGEX = re.compile(
    br"-----BEGIN PGP SIGNED MESSAGE-----"
    br"\s*(.+)"
    br"-----BEGIN PGP SIGNATURE-----"
    br".+Comment: Stamper Reference Id: (\d+)"
    br"\n\n(.+?)\n"
    br"-----END PGP SIGNATURE-----",
    re.DOTALL)


class TstampReceiver(TStampSpec):
    """IMAP & timestamp parameters and methods for receiving & parsing dice-report emails."""

    def _pgp_split(self, sig_msg_bytes: bytes) -> Tuple[bytes, bytes, bytes]:
        m = _PGP_SIG_REGEX.search(sig_msg_bytes)
        if not m:
            raise CmdException("Invalid signed message: %r" % sig_msg_bytes)

        msg, ts_id, sig = m.groups()

        return msg, ts_id, sig

    def _pgp_sig2int(self, sig: bytes) -> int:
        import base64
        import binascii

        sig_bytes = base64.decodebytes(sig)
        num = int(binascii.b2a_hex(sig_bytes), 16)

        return num

    def _verify_detached_armor(self, sig: str, data: str):
        """Verify `sig` on the `data`."""
    #def verify_file(self, file, data_filename=None):
        #with tempfile.NamedTemporaryFile(mode='wt+',
        #                encoding='latin-1') as sig_fp:
        #sig_fp.write(sig)
        #sig_fp.flush(); sig_fp.seek(0) ## paranoid seek(), Windows at least)
        #sig_fn = sig_fp.name
        with tempfile.TemporaryFile('wb+', prefix='dicesig_') as dicesig_file:
            sig_fn = dicesig_file.name
            self.log.debug('Wrote sig to temp file: %r', sig_fn)

            args = ['--verify', gnupg.no_quote(sig_fn), '-']
            result = self.result_map['verify'](self)
            data_stream = io.BytesIO(data.encode(self.encoding))
            self._handle_io(args, data_stream, result, binary=True)

            return result

    def parse_tsamp_response(self, mail_text: Text) -> int:
        mbytes = mail_text.encode('utf-8')

        # TODO: validate sig!

        msg, ts_id, sig = self._pgp_split(mbytes)
        num = self._pgp_sig2int(sig)
        mod100 = num % 100
        decision = 'OK' if mod100 < 90 else 'SAMPLE'

        return sig.decode(), num, mod100, decision

    # see https://pymotw.com/2/imaplib/ for IMAP example.
    def receive_timestamped_email(self, host, login_cb, ssl=False, **srv_kwds):
        prompt = 'IMAP(%r)' % host

        with (imaplib.IMAP4_SSL(host, **srv_kwds)
              if ssl else imaplib.IMAP4(host, **srv_kwds)) as srv:
            repl = srv.login(self.user_name, self.user_pswd)
            """GMAIL-2FAuth: imaplib.error: b'[ALERT] Application-specific password required:
            https://support.google.com/accounts/answer/185833 (Failure)'"""
            self.log.debug("Sent %s-user/pswd, server replied: %s", prompt, repl)

            resp = srv.list()
            print(resp[0])
            return [srv.retr(i + 1) for i, msg_id in zip(range(10), resp[1])]  # @UnusedVariable


###################
##    Commands   ##
###################


class _Subcmd(baseapp.Cmd):
    @property
    def projects_db(self):
        p = project.ProjectsDB.instance()
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

        __sender = None

        @property
        def sender(self):
            if not self.__sender:
                self.__sender = TstampSender(config=self.config)
            return self.__sender

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
                raise CmdException(
                    "Cmd '%s' takes no arguments, received %d: %r!"
                    % (self.name, len(args), args))

            file = self.file

            if not file:
                self.log.warning("Time-stamping STDIN; paste message verbatim!")
                mail_text = sys.stdin.read()
            else:
                self.log.info('Time-stamping files %r...', file)
                with io.open(file, 'rt') as fin:
                    mail_text = fin.read()
            self.sender.send_timestamped_email(mail_text)

    class ParseCmd(_Subcmd):
        """
        Derives the *decision* OK/SAMPLE flag from time-stamped email.

        SYNTAX
            cat <mail> | co2dice tstamp parse
        """

        __recver = None

        @property
        def recver(self) -> TstampReceiver:
            if not self.__recver:
                self.__recver = TstampReceiver(config=self.config)
            return self.__recver

        def run(self, *args):
            nargs = len(args)
            if nargs > 0:
                raise CmdException(
                    "Cmd '%s' takes no arguments, received %d: %r!"
                    % (self.name, len(args), args))

            rcver = self.recver
            mail_text = sys.stdin.read()
            decision_tuple = rcver.parse_tsamp_response(mail_text)

            return ('SIG: %s\nNUM: %s\nMOD100: %s\nDECISION: %s' %
                    decision_tuple)

    def __init__(self, **kwds):
        with self.hold_trait_notifications():
            dkwds = {
                'conf_classes': [project.ProjectsDB, TstampReceiver],
                'subcommands': baseapp.build_sub_cmds(*all_subcmds),
            }
            dkwds.update(kwds)
            super().__init__(**dkwds)

all_subcmds = (TstampCmd.SendCmd, TstampCmd.ParseCmd)


if __name__ == '__main__':
    from traitlets.config import get_config
    # Invoked from IDEs, so enable debug-logging.
    c = get_config()
    c.Application.log_level = 0
    #c.Spec.log_level='ERROR'

    argv = None
    ## DEBUG AID ARGS, remember to delete them once developed.
    #argv = ''.split()
    #argv = '--debug'.split()

    #TstampCmd(config=c).run('--text ')
    from . import dice
    baseapp.run_cmd(baseapp.chain_cmds(
        [dice.MainCmd, TstampCmd],
        config=c))
