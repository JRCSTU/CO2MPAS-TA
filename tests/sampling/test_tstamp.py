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
import textwrap as tw
import traitlets.config as trtc


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


test_pgp_key = [
""" #@IgnorePep8
## Key for running TCs::
#
#    pub   1024R/B124C999CBBB52FF 2017-01-26 [expires: 2017-07-25]
#    uid               [ unknown] CO2MPAS Test <sampling@co2mpas.jrc.ec.europa.eu>
#    sig 3        B124C999CBBB52FF 2017-01-26  CO2MPAS Test <sampling@co2mpas.jrc.ec.europa.eu>
#
-----BEGIN PGP PRIVATE KEY BLOCK-----

lQHYBFiJ7q0BBAC7SXZw+XbnbT9QuB7mQUlMaw9NPtqj8iRRvJZOejV0PSY0F1Ub
jNhLlmrBX+m4zoPtreEmjeGOa5uPDoqqiD1ft9kWf9Byr1Uq3L++NtDwIcetZzl1
hHiG/wtY7kaWDZRgHXKMbf5TPjsFKyXS8lnyIRlD6nuU4xvMTzmiCdp4FQARAQAB
AAP9Erl8SDzEvMwRG7igzDwEQEnm4H3zfZcotuxQKb3xqLKxZl1b0rKQ8HO0Liuw
8hthmMp8224teh/7kECvr++JlSN8+EiXZ+DTffFdMRZKfAkB6uktfRNIuY98qH0h
AUgRS0StXQEPgm3SzguzA+1TYTa2Khay8wVjIXCBU0M6EbkCAMO+T94VICGNlUXm
mC/R1VdzO8o9XVFWhfGVUvVR0U4tzb5+izLSf4aau74OnICGqHnIQaEmU1DVit22
ALQlgGkCAPTwvDKRkSfDAKhvOu5Flb7k0AsC5wdQMQrfs0m5lwXDhuojB2XB77zG
ODwkhWIT46qGZZlvmcKPSQcOXnkpBM0B/ii7PsEWw7SNgVnRjOGeKpu/drpluUwa
uT0B9x6sy+Fyx/IVZuNRsbG4Xetay7MeC+m7MaLAwe+ZezmgxVT4kn2gq7QwQ08y
TVBBUyBUZXN0IDxzYW1wbGluZ0BjbzJtcGFzLmpyYy5lYy5ldXJvcGEuZXU+iL8E
EwEIACkFAliJ7q0CGy8FCQDtTgAHCwkIBwMCAQYVCAIJCgsEFgIDAQIeAQIXgAAK
CRCxJMmZy7tS/53UA/9G+m7bmn/HCKSRsH4fIkveq4jZRVmq1NEPmXm4pXCwROGK
fRAw0pl/l+eGW3adMctTOxaX3lI/nz+g2QTEURgDHxLGaghDEGuy1VyjFFt0WXef
2l5xTONpi4gs/G4M1+TY/MantEDRUJPh3EMgoEuT0H6gffhxsejI/YD1BH0RGQ==
=AFgK
-----END PGP PRIVATE KEY BLOCK-----
""",
""" #@IgnorePep8
## Timestamp-service's keys, as imported from:
#  from: http://www.itconsult.co.uk/stamper/stampinf.htm
#
#    pub   1024R/02B02F070712FEBD 1994-09-25
#    uid               [ unknown] Matthew Richardson <matthew@itconsult.co.uk>
#    sig          02B02F070712FEBD 1995-10-12  Matthew Richardson <matthew@itconsult.co.uk>
#    uid               [ unknown] Matthew Richardson <Jersey, Channel Islands>
#    sig          0A617C533F98D37D 1997-04-30  [User ID not found]
#    sig          39B795BF6DD50C41 1996-02-06  [User ID not found]
#    sig          9330B7B0558E09A5 1995-05-14  [User ID not found]
#    sig          02B02F070712FEBD 1994-09-25  Matthew Richardson <matthew@itconsult.co.uk>
#
#    pub   1024R/E0F5F876083A08A9 1995-08-27
#    uid               [ unknown] Scheduler Service (MER_Schedule on \\MER_DECXL)
#    sig          E0F5F876083A08A9 1995-08-27  Scheduler Service (MER_Schedule on \\MER_DECXL)
#    sig          02B02F070712FEBD 1995-08-27  Matthew Richardson <matthew@itconsult.co.uk>
#
#    pub   2046R/81959DB570B61F81 1995-10-11
#    uid               [ unknown] Timestamp Service <stamper@itconsult.co.uk>
#    sig          81959DB570B61F81 2014-05-25  Timestamp Service <stamper@itconsult.co.uk>
#    sig          B124C999CBBB52FF 2017-01-26  CO2MPAS Test <sampling@co2mpas.jrc.ec.europa.eu>
#
#  Note that only the "timestamp-key" is signed by test-key, above.
#
-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: GnuPG v2

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
El4C5af/8COxQcDn608=
=TF9x
-----END PGP PUBLIC KEY BLOCK-----
""",
]

test_pgp_trust = tw.dedent("""\
    ## CO2MPAS test-key
    5464E04EE547D1FEDCAC4342B124C999CBBB52FF:6:
    """)

_tstamp_responses = [(941136, '346C4B1FDF5343D6F4B0BF10D660FEDC25B66', 74, 'OK',
                      {'trust_text': None},
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
       {'trust_text': None},
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
       {'trust_text': 'TRUST_FULLY'},
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
        cls.gnupghome = tempfile.mkdtemp(prefix='gpghome-')

        cfg.GpgSpec.gnupghome = cls.gnupghome
        cfg.GpgSpec.keys_to_import = test_pgp_key
        cfg.GpgSpec.trust_to_import = test_pgp_trust
        cfg.GpgSpec.master_key = '8C008403'  # Dice's TestKey
        crypto.GpgSpec(config=cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.gnupghome)

    @ddt.data(*_tstamp_responses)
    def test_timestamp_verify(self, case):
        (stamper_id, dice, dice100, dice_decision,
         tag_verdict, tstamp_response) = case
        rcv = tstamp.TstampReceiver(config=self.cfg)
        resp = rcv.parse_tsamp_response(tstamp_response)
        pp(resp)
        self.assertEqual(resp['tstamp']['stamper_id'], stamper_id)
        self.assertDictContainsSubset({
            'dice_hex': dice,
            'dice_%100': dice100,
            'dice_decision': dice_decision,
        }, resp)
        self.assertDictContainsSubset({
            'key_id': '81959DB570B61F81',
            'pubkey_fingerprint': '4B12BCD5788511063B543190E09DF306',
        }, resp['tstamp']['sig'])
        self.assertDictContainsSubset(tag_verdict, resp['tag']['sig'])
