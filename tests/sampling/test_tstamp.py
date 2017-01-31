#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import tstamp, crypto
import logging
from pprint import pprint as pp
import shutil
import tempfile
import unittest

import ddt

import os.path as osp
import traitlets.config as trtc

from . import test_pgp_fingerprint, test_pgp_key, test_pgp_trust


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


_tstamp_responses = [(941136, '346C4B1FDF5343D6F4B0BF10D660FEDC25B66', 74, 'OK',
                      {'trust_text': 'TRUST_FULLY'}, {'trust_text': None},
""" #@IgnorePep8
 by mail-zone.jrc.it
 (Sun Java(tm) System Messaging Server 7.3-11.01 64bit (built Sep  1 2009))
 with SMTP id <0OKA00BLJZCS1BF0@mail-zone.jrc.it> for
 post@stamper.itconsult.co.uk; Tue, 24 Jan 2017 22:22:04 +0100 (CET)
Received: from STUW025.nube.local ([unknown] [139.191.229.250])
 by email-02-ext.jrc.it
 (Sun Java(tm) System Messaging Server 7.3-11.01 64bit (built Sep  1 2009))
 with ESMTPA id <0OKA005MBZCREH00@email-02-ext.jrc.it> for
 post@stamper.itconsult.co.uk; Tue, 24 Jan 2017 22:22:03 +0100 (CET)
Date: Tue, 24 Jan 2017 22:22:03 +0100 (CET)
Message-id: <0OKA005MCZCREH00@email-02-ext.jrc.it>
Subject: [dice test]
From: ankostis@gmail.com
X-DCC-INFN-TO-Metrics: et05 1233; Body=1 Fuz1=1 Fuz2=1
X-DNSBL: 0

-----BEGIN PGP SIGNED MESSAGE-----

########################################################
#
# This is a proof of posting certificate from
# stamper.itconsult.co.uk certifying that a user
# claiming to be:-
#     ankostis@gmail.com
# requested that this message be sent to:-
#     post@stamper.itconsult.co.uk
#     ankostis@gmail.com
#     konstantinos.anagnostopoulos@ext.jrc.ec.europa.eu
#
# This certificate was issued at 21:25 (GMT)
# on Tuesday 24 January 2017 with reference 0941136
#
# CAUTION: while the message may well be from the sender
#          indicated in the "From:" header, the sender
#          has NOT been authenticated by this service
#
# For information about the Stamper service see
#        http://www.itconsult.co.uk/stamper.htm
#
########################################################


object 76b8bf7312770a488eaeab4424d080dea3272435
type commit
tag test_tag
tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1485272439 +0100

- - Is bytes (utf-8 encodable);
- - all lines end with LF, and any trailing whitespace truncated;
- - any line can start with dashes;
- - any empty lines at the bottom are truncated,
- - apart from the last LF, which IS part of the msg.
- -----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iJwEAAEIAAYFAliHdXwACgkQ/77EoYwAhAMxDgQAhlqOjb0bHGxLcyYIpFg9kEmp
4poL5eA7cdmq3eU1jXTfb5UXJV6BnP+DUsJ4TG+7KoUimgli0djG7ZisRvNYBWGD
PNO2X5LqNx7tzgj/fQT5CzWcWMXfjUd337pfoj3K3kDroCNl7oQl/bSIR46z9l/3
JS/kbngOONtzIkPbQvU=
=bEkN
- -----END PGP SIGNATURE-----

-----BEGIN PGP SIGNATURE-----
Version: 2.6.3i
Charset: noconv
Comment: Stamper Reference Id: 0941136

iQEVAgUBWIfGLIGVnbVwth+BAQHg2gf8DKahMniitB2df76Sa4d0TJJD/wpR5vfc
O8T0TQ7VblcOVniAh4VEuHKN5Kqd3Q9sVs3K/yzsusHRTPiB06AmepVrt69PhfpF
6PUBk09wqoheSnOeQm4/nzBM1qluXyIYvvWk85eEqaWfunWAKrdMbNUz+0BPuIah
+X+GBwQZ8+tEdBPnJcF2Gu5LpBDtfL2C/jo9TPBQd+wdMt92pMryPsYuHFzbbj0T
SjzL9Fp7gP5OsJZ1uRMtP9MzMTjvjMS1IKmNwPvVsvSe+77S3+urMgklH7ciypsT
8vg3VUDkJesTXOU4oDaWNwclNOAfumM4m+pZFKxKkq7FgoYcSjj4Ug==
=vFH9
-----END PGP SIGNATURE-----

extra stuff

"""), (941144, 'A6C6E3771EA412EE56B4E61A10CDE90776D53419', 41, 'OK',
       {'trust_text': 'TRUST_FULLY'}, {'trust_text': None},
""" #@IgnorePep8
-----BEGIN PGP SIGNED MESSAGE-----

########################################################
#
# This is a proof of posting certificate from
# stamper.itconsult.co.uk certifying that a user
# claiming to be:-
#     ankostis@gmail.com
# requested that this message be sent to:-
#     post@stamper.itconsult.co.uk
#     ankostis@gmail.com
#     konstantinos.anagnostopoulos@ext.jrc.ec.europa.eu
#
# This certificate was issued at 23:00 (GMT)
# on Tuesday 24 January 2017 with reference 0941144
#
# CAUTION: while the message may well be from the sender
#          indicated in the "From:" header, the sender
#          has NOT been authenticated by this service
#
# For information about the Stamper service see
#        http://www.itconsult.co.uk/stamper.htm
#
########################################################


object 76b8bf7312770a488eaeab4424d080dea3272435
type commit
tag test_tag
tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1485272439 +0100

- - Is bytes (utf-8 encodable);
- - all lines end with LF, and any trailing whitespace truncated;
- - any line can start with dashes;
- - any empty lines at the bottom are truncated,
- - apart from the last LF, which IS part of the msg.
- -----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iJwEAAEIAAYFAliHdXwACgkQ/77EoYwAhAMxDgQAhlqOjb0bHGxLcyYIpFg9kEmp
4poL5eA7cdmq3eU1jXTfb5UXJV6BnP+DUsJ4TG+7KoUimgli0djG7ZisRvNYBWGD
PNO2X5LqNx7tzgj/fQT5CzWcWMXfjUd337pfoj3K3kDroCNl7oQl/bSIR46z9l/3
JS/kbngOONtzIkPbQvU=
=bEkN
- -----END PGP SIGNATURE-----

-----BEGIN PGP SIGNATURE-----
Version: 2.6.3i
Charset: noconv
Comment: Stamper Reference Id: 0941144

iQEVAgUBWIfccIGVnbVwth+BAQEuzgf7B7SxVeN+410yNZE8UHSfHfPBuSsUSMkr
oyyyp1VtLnFHk4vPKgaPQ23f6o6poRlWI18qDWV7q2MbElA+VvxyH3pIpCMCndik
KUmmZIas27iC6jI6EHMdYLPN3fydsPLdxPzKItuBSlFdkoVU65D925C2Pmq3ErGE
5w7ouRQXAfF7lOd1V0JF8NhGZF/MtkFv6ZhkrC+JHpJXk0/J6i7mvLOGJRlclE7v
K1nTqtFTPxDfDEnBZgzDNT6jD4rsPyDhNIydJsESc9ypVPB7ExwVKQT4wfSH+FLE
aVryU+Z1cn1UO+59VsUeoaUcJqr7wNmwR5Zzyzp7Obm7ZlEvE5Gqfg==
=y4Fb
-----END PGP SIGNATURE-----
"""), (941518, 'A100EBD962AEA3349AFC6396D48015131BCA866F', 19, 'OK',
       {'trust_text': 'TRUST_FULLY'}, {'trust_text': 'TRUST_ULTIMATE'},
""" #@IgnorePep8
-----BEGIN PGP SIGNED MESSAGE-----

########################################################
#
# This is a proof of posting certificate from
# stamper.itconsult.co.uk certifying that a user
# claiming to be:-
#     ankostis@gmail.com
# requested that this message be sent to:-
#     post@stamper.itconsult.co.uk
#     ankostis@gmail.com
#     konstantinos.anagnostopoulos@ext.jrc.ec.europa.eu
#
# This certificate was issued at 23:00 (GMT)
# on Thursday 26 January 2017 with reference 0941518
#
# CAUTION: while the message may well be from the sender
#          indicated in the "From:" header, the sender
#          has NOT been authenticated by this service
#
# For information about the Stamper service see
#        http://www.itconsult.co.uk/stamper.htm
#
########################################################


object 3334bcde283480883f2fb209efcf84ae24da8335
type commit
tag tests/signed_by_CBBB52FF
tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1485442847 +0100

Test-tag for crypto TCs, signed by:
  CO2MPAS Test <sampling@co2mpas.jrc.ec.europa.eu>  1024R/B124C999CBBB52FF 2017-01-26 [expires: 2017-07-25]
- -----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iJwEAAEIAAYFAliKDx8ACgkQsSTJmcu7Uv9HsAP+KmK4+cSXvScwg5UHDq7VVj1B
XjtEHZp6VwKndmMCQNIOsyR3F7o5qsleU2NympSVxQyOTL0WlFaJqdNMSLwqV/px
oWZdPlYCw6lc1BFjRkYF5YVCb6E7dJG6WbUJTVys5lt3AIIN3l1WuO2JlhmXvubN
021zAo8TJIn1aFQEkVw=
=nxOG
- -----END PGP SIGNATURE-----

-----BEGIN PGP SIGNATURE-----
Version: 2.6.3i
Charset: noconv
Comment: Stamper Reference Id: 0941518

iQEVAgUBWIp/cIGVnbVwth+BAQFHYgf9FlwGDnTXG8n6z9mxY3D/0iyQPCBcl7GA
F82u0+R/QjCxEpy/CpdAKPH0r3wbFvyvBHgmxwkHFY/dpk1g9NyRVp7Fj/ANGuNx
QF3ORr0JG55ZjxpHZM+OZye0PWWIrvMpqK4Rv+EEWFgYoo4/RJmX0uyjTW3eRBxy
gzZN0TCHF5YFHWJcsaqiAFogszuFcaHaq2v9m+8X252knKM9NZ/0mIjjQhRynNss
0d7bzQbtRJFJOHgYC7WRBgaRsokNbMVNUaWvygiC+Q7ccV7mVVvNQZ3fqQMiSKxH
oCxi53i/Agi1pCvJ/WCC9HJ7papOA9+Gd2R7x3F2XVRSP1+/9g7wRA==
=OVvz
-----END PGP SIGNATURE-----
""")
]


@ddt.ddt
class TRX(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = trtc.get_config()

        cfg.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        cfg.GpgSpec.keys_to_import = test_pgp_key
        cfg.GpgSpec.trust_to_import = test_pgp_trust
        cfg.GpgSpec.master_key = test_pgp_fingerprint
        crypto.GpgSpec(config=cfg)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()
        crypto.GitAuthSpec.clear_instance()
        crypto.VaultSpec.clear_instance()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)

    def test_send_timestamp(self):
        snd = tstamp.TstampSender(config=self.cfg)
        ex_msg = r"Content to timestamp failed signature verification!\s+None"
        with self.assertRaisesRegex(tstamp.CmdException, ex_msg):
            snd.send_timestamped_email("", dry_run=True)

    def test_parse_timestamp_bad(self):
        rcv = tstamp.TstampReceiver(config=self.cfg)
        ex_msg = r"Cannot verify timestamp-reponse signature due to: incorrect passphrase"
        with self.assertRaisesRegex(tstamp.CmdException, ex_msg):
            rcv.parse_tsamp_response("")

    @ddt.data(*_tstamp_responses)
    def test_parse_timestamps(self, case):
        (stamper_id, dice, dice100, dice_decision,
         ts_verdict, tag_verdict, tstamp_response) = case
        rcv = tstamp.TstampReceiver(config=self.cfg)
        resp = rcv.parse_tsamp_response(tstamp_response)
        pp(resp)
        self.assertEqual(resp['tstamp']['stamper_id'], stamper_id)
        self.assertDictContainsSubset({
            'dice_hex': dice,
            'dice_%100': dice100,
            'dice_decision': dice_decision,
        }, resp)
        ts_verdict.update({
            'key_id': '81959DB570B61F81',
            'pubkey_fingerprint': '4B12BCD5788511063B543190E09DF306',
        })
        self.assertDictContainsSubset(ts_verdict, resp['tstamp']['sig'])
        self.assertDictContainsSubset(tag_verdict, resp['tag']['sig'])
