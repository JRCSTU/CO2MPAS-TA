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

from pandalone import utils as pndlu

import os.path as osp

from . import CmdException, baseapp, crypto, tstamp
from .._vendor import traitlets as trt


class TstampSigner(tstamp.TstampReceiver):
    """
    To run securely on a server see: https://wiki.gnupg.org/AgentForwarding
    """

    stamper_name = trt.Unicode('JRC-stamper')
    sender = trt.Unicode()
    stamps_folder = trt.Unicode(
        help="""The folder to store all signed stamps and derive cert-chain.""",
        config=True
    )

    @trt.default('stamps_folder')
    def default_stamps_folder(self):
        return osp.join(baseapp.default_config_dir(), 'MyStamper')

    signed_opening = trt.CRegExp(
        R"""
        ^-{5}BEGIN\ PGP\ SIGNED\ MESSAGE-{5}\r?\n
        (?:^Hash:\ [^\r\n]+\r?\n)*                   ## 'Hash:'-header(s)
        ^\r?\n                                       ## blank-line
        """,
        re.DOTALL | re.VERBOSE | re.MULTILINE,
        config=True)

    validate_decision = trt.Bool(
        True,
        help="""Validate dice and append-report at the bottom of dreport.  """,
        config=True)

    trim_dreport = trt.Bool(
        help="""Remove any garbage after dreport's signature? """,
        config=True)

    recipients = trt.List(
        trt.Unicode(),
        default_value=["JRC-CO2MPAS@ec.europa.eu",
                       "CLIMA-LDV-CO2-CORRELATION@ec.europa.eu"],
        allow_none=True)

    sender = trt.Unicode(
        allow_none=True)


    def sign_text_as_tstamper(self, text: Text,
                              sender: Text=None,
                              full_output: bool=False):
        """
        :param full_output:
            if true, return `gnupg` output object, otherwise, signed text.
        """
        from datetime import datetime

        stamper_name = self.stamper_name
        stamper_auth = crypto.get_stamper_auth(self.config)
        recipients = '\n#     '.join(self.recipients)
        issue_date = datetime.now().isoformat()
        stamp_id = 2
        parent_stamp = 'wettryioulmngvf'
        tstamp_text = f"""\
########################################################
#
# Proof of posting certificate from {stamper_name}
# certifying that:-
#   {sender}
# requested to email this message to:-
#   {recipients}
#
# certificate_date: {issue_date}
# reference: {stamp_id:07}
# parent_stamp: {parent_stamp}
#
########################################################


{text}
"""
        stamper_comment = f"Stamper Reference Id: {stamp_id:07}"
        sign = stamper_auth.clearsign_text(
            tstamp_text, extra_args=['--comment', stamper_comment],
            full_output=full_output)

        if full_output:
            #
            ## GnuPG does not return signature-id when signing :-(
            from toolz import dicttoolz as dtz

            ## Merge gnupg-verify results
            #
            ts_ver = stamper_auth.verify_clearsigned(str(sign))
            ts_ver = dtz.valfilter(bool, vars(ts_ver))
            sign.__dict__.update(ts_ver)

        return sign

    def sign_dreport_as_tstamper(self, dreport: Text):
        import pprint as pp
        import textwrap as tw

        tag_verdict = self.parse_signed_tag(dreport)
        tag_signer = crypto.uid_from_verdict(tag_verdict)
        if tag_verdict['valid']:
            if self.trim_dreport:
                dreport = tag_verdict['parts']['msg'].decode('utf-8')
        else:
            err = "Invalid dice-report due to: %s \n%s" % (
                tag_verdict['status'], tw.indent(pp.pformat(tag_verdict), '  '))
            if self.force:
                self.log.warning(err)
            else:
                raise CmdException(err)

        sender = self.sender or tag_signer or '<unknown>'

        sign = self.sign_text_as_tstamper(
            dreport, sender, full_output=True)

        stamp, ts_verdict = str(sign), vars(sign)
        tag_name = self.extract_dice_tag_name(None, dreport)
        dice_decision = self.make_dice_results(ts_verdict,
                                               tag_verdict,
                                               tag_name)
        if self.validate_decision:
            signed_text = self.append_decision(stamp, dice_decision)
        else:
            signed_text = stamp

        return signed_text, dice_decision


class SignCmd(baseapp.Cmd):
    """Private stamper service."""

    def __init__(self, **kwds):
        kwds.setdefault('conf_classes', [TstampSigner,
                                         crypto.GitAuthSpec, crypto.StamperAuthSpec])
        super().__init__(**kwds)

    def run(self, *args):
        from boltons.setutils import IndexedSet as iset

        files = iset(args) or ['-']
        self.log.info("Signining '%s'...", tuple(files))

        signer = TstampSigner(config=self.config)
        for file in files:
            if file == '-':
                self.log.info("Reading STDIN; paste message verbatim!")
                mail_text = sys.stdin.read()
            else:
                self.log.debug("Reading '%s'...", pndlu.convpath(file))
                with io.open(file, 'rt') as fin:
                    mail_text = fin.read()

            try:
                tstamp, _decision = signer.sign_dreport_as_tstamper(mail_text)

                ## In PY3 stdout duplicates \n as \r\n, hence \r\n --> \r\r\n.
                #  and signed text always has \r\n EOL.
                yield tstamp.replace('\r\n', '\n')
            except Exception as ex:
                self.log.error("%s: signig %i-char message failed due to: %s",
                               file, len(mail_text), ex, exc_info=1)
