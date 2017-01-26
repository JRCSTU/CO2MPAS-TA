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
    tw.dedent(
        """\
        ## CO2MPAS-test sec-key.
        #
        -----BEGIN PGP PRIVATE KEY BLOCK-----
        Version: GnuPG v2

        lQHYBFbirf0BBADdjkrRSJ+ua9RvlzZsrpr4tAvj6+rK9eRwuue+jaIu5JucIvV5
        nq5AlsQGACTy88qsnYcbqqro15AURdZ3fS2uy4yGvcNzMNP0w6Jh+DPZM9ubZXXQ
        xsejMltjynHe3gJNeBy5D7xGOURv1sAaHoCxYw4zC/SjkbnNqpvMZNmvTwARAQAB
        AAP/UyULsdepyUjBK/GY5JdwJAZZcfr+nZVC2gViY9H+O8/iD+nUqoQgy69ouAHE
        3AIenMHvSrQ1OHVxJhKBZk0tX3lUZ60W8yBY5GydWu0ylHeTXzU2tgCqMDi36TKE
        w/p1jx5nkJ6qwfXtMTWLsaWOllRaGOpOv+pR3yEkRmTQe20CAOvbwxax33bFDXJM
        WSL0GEdTXDM/M5XdoFoV+M9wnx4eJj4yPnL0SkpWgkq38FtwrINdxg46OfOw7hml
        hlKXz80CAPB52jgRTImm2VAENdMa0O+yu5YDmRyyuyJWyoYdNQlbOrzraQjiJTfQ
        zsv9bfvpxeFS+PRXyAenwRV3qQryR4sCALAvfd068PZkg4K3+TP5F45fFBTfnHqR
        eCeUX/uU4CYzh+LcM5E89IQ45O+FIwUgyvVx8kyUe79vwlzbJXnf/1mjlrQxU2Ft
        cGxpbmcgVGVzdCA8c2FtcGxpbmdAY28ybXBhcy5qcmMuZWMuZXVyb3BhLmV1Poi5
        BBMBCAAjBQJW4q39AhsvBwsJCAcDAgEGFQgCCQoLBBYCAwECHgECF4AACgkQ/77E
        oYwAhAN16QQAp79/wkUQNqXZTWhY5ji/jazisK4ggaJcWIvpsrbiwL1wJ5ZbbY/9
        gHXslyDWhyKK/KbgAdAurAXgekRDLTXxR1c+Q0hm0phrmyhIENRbwF04T2S4JExQ
        PiKw+6RfuxfTV22f6sRSkwuK4MP9veZ/0u6Mid6kRHJFo+mVus8XuZA=
        =clrj
        -----END PGP PRIVATE KEY BLOCK-----
    """),
    tw.dedent(
        """\
        ## Timestamp-service pub-key.
        #
        -----BEGIN PGP PUBLIC KEY BLOCK-----
        Version: GnuPG v2

        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
        q6kfyfg=
        =gL5r
        -----END PGP PUBLIC KEY BLOCK-----
    """),
]

test_pgp_trust = tw.dedent("""\
    ## CO2MPAS test-key
    8922372A2983334307D7DA90FFBEC4A18C008403:4:
    ## Timestamp-service
    4B12BCD5788511063B543190E09DF30600000000:5:
    """)

_tstamp_responses = [
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

""",
"""
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
"""
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
    def test_timestamp_verify(self, tstamp_response):
        rcv = tstamp.TstampReceiver(config=self.cfg)
        resp = rcv.parse_tsamp_response(tstamp_response)
        pp(resp)
