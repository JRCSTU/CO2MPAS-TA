#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2dice.utils.logconfutils import init_logging
from co2dice._vendor.traitlets import config as trtc
from co2dice import crypto, tstamp
from co2dice.cmdlets import collect_cmd, Cmd
from collections import Counter
from pprint import pformat as pf
import logging
import os
import shutil
import tempfile
import unittest

import ddt
import yaml

import co2dice._vendor.traitlets as trt
import os.path as osp
import subprocess as sbp

from . import TEST_CONF_ENVAR, CONF_ENVAR, test_pgp_fingerprint, test_pgp_keys, test_pgp_trust


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


tstamp_responses = [(
    """Failed parsing commit message due to: ValueError\("incompatible message,""",
    941136, {
        'hexnum': '346C4B1FDF5343D6F4B0BF10D660FEDC25B66',
        'percent': 74,
        'decision': 'OK',
    }, {
        'trust_text': 'TRUST_FULLY',
    }, {
        'trust_text': None,
        'project': None,
    },
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

"""), (
    """Failed parsing commit message due to: ValueError\("incompatible message,""",
    941144, {
        'hexnum': 'A6C6E3771EA412EE56B4E61A10CDE90776D53419',
        'percent': 41,
        'decision': 'OK',
    }, {
        'trust_text': 'TRUST_FULLY',
    }, {
        'trust_text': None,
        'project': None,
    },
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
"""), (
    r"""Failed parsing commit message due to: ValueError\("incompatible message,""",
    941518, {
        'hexnum': 'A100EBD962AEA3349AFC6396D48015131BCA866F',
        'percent': 19,
        'decision': 'OK',
    }, {
        'trust_text': 'TRUST_FULLY',
    }, {
        'trust_text': 'TRUST_ULTIMATE',
        'project': None,
    },
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
"""), (
    None,
    942920, {
        'hexnum': 'C87BB5C47CDF17D98978A4F7756713428A7F333',
        'percent': 19,
        'decision': 'OK',
    }, {
        'trust_text': 'TRUST_FULLY',
    }, {
        'trust_text': 'TRUST_ULTIMATE',
        'project': 'FT-12-ABC-2016-0001',
        'project_source': 'report',
        'vehicle_family_id': 'RL-99-BM3-2017-0001',
    },
""" #@IgnorePep8
-----BEGIN PGP SIGNED MESSAGE-----

########################################################
#
# This is a proof of posting certificate from
# stamper.itconsult.co.uk certifying that a user
# claiming to be:-
#     ankostis@gmail.com
# requested that this message be sent to:-
#     kostis.anagnostopoulos@ext.ec.europa.eu
#     ankostis@gmail.com
#
# This certificate was issued at 18:00 (GMT)
# on Saturday 04 February 2017 with reference 0942920
#
# CAUTION: while the message may well be from the sender
#          indicated in the "From:" header, the sender
#          has NOT been authenticated by this service
#
# For information about the Stamper service see
#        http://www.itconsult.co.uk/stamper.htm
#
########################################################


object df6cfcbb3b8840cc843f7cbf922c1451dbd7cc08
type commit
tag dices/FT-12-ABC-2016-0001/6
tagger Kostis <ankostis@gmail.com> 1486226540 +0100

- - {v: 1.0.0, a: drep 2 files, p: FT-12-ABC-2016-0001, s: tagged}
- - {file: input.xlsx, iokind: inp, vehicle_family_id: RL-99-BM3-2017-0001}
- - file: output-longer_than-usual-file-012325336475546776.xlsx
  iokind: out
  vehicle_family_id: RL-99-BM3-2017-0001
  content_type: dice_report
  content:
    CO2MPAS_deviation: [-4.14, null]
    CO2MPAS_version: [1.5.0.dev1, null]
    Model_scores: [vehicle-H, vehicle-L]
    TA_mode: [true, null]
    Vehicle: [vehicle-H, vehicle-L]
    alternator_model: [4.56, null]
    at_model: [-0.95, null]
    clutch_torque_converter_model: [4.71, null]
    co2_params: [0, null]
    datetime: ['2017/01/29-23:42:41', null]
    engine_capacity: [997, 997]
    engine_cold_start_speed_model: [18.74, null]
    engine_coolant_temperature_model: [0.59, null]
    engine_is_turbo: [true, true]
    engine_speed_model: [0.02, 91.36]
    fuel_type: [diesel, diesel]
    gear_box_type: [automatic, automatic]
    start_stop_model: [-0.99, null]
    vehicle_family_id: [RL-99-BM3-2017-0001, null]
- -----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iJwEAAEIAAYFAliWBGwACgkQsSTJmcu7Uv+fggP+IJPS3QqkBa7BF4YbPYkKjD0O
F/FJFySXCh7iX+LRwNGAudHgGmVw3puaaY10ksIeKAyjiSYEck8TF8hB+4GDG2fZ
FQ+/P9eF3EpFaIXQ8YsKsPmMuW/5iSTdU9PfbM3YSTvUYq94Jz+Xjt1ZCnOTUE/p
+YVxyIe3UwYBJWuciUY=
=AYp4
- -----END PGP SIGNATURE-----

-----BEGIN PGP SIGNATURE-----
Version: 2.6.3i
Charset: noconv
Comment: Stamper Reference Id: 0942920

iQEVAgUBWJYWoIGVnbVwth+BAQGvogf9G79Xt/bDNqTcom5yD8xpp6uZ42E0MBCI
joee4b6cwsHucRdW6T8BkOHvjwbtkIXLDdsH9z1LHg/MNg7EBA/sABerTnKJdPBB
wc7IrKoeHQsnrQR7eb7b4jDrMKd/Cg5Hgv5o+Wv/eJVbWkWAEQ0iWWPwJWLDr9Hh
lfbwrniAAlkCnJGLP2urcGgB9ikGvhyFniGfl5EbRrVyXjmEhaOLBJ13wV1CbJp3
gaNYL4yCuCJ+zP4BV5rMbTXNHSxhZw6xmgSnD4x731Bz8DrZd5DCwuKUkXkH8vhI
SHI4X+XpZSHFeBFVucZySOwr57AhDYCZpgFI0uXV+k+C94wRBBA1yA==
=GgO8
-----END PGP SIGNATURE-----
"""), (
    None,
    942919, {
        'hexnum': '8171C03C97F199C6011AF7ED2825A7712DEA0135',
        'percent': 93,
        'decision': 'SAMPLE',
    }, {
        'trust_text': 'TRUST_FULLY',
    }, {
        'trust_text': 'TRUST_ULTIMATE',
        'project': 'FT-12-ABC-2016-0001',
        'project_source': 'report',
        'vehicle_family_id': 'RL-99-BM3-2017-0001',
    },
""" #@IgnorePep8
-----BEGIN PGP SIGNED MESSAGE-----

########################################################
#
# This is a proof of posting certificate from
# stamper.itconsult.co.uk certifying that a user
# claiming to be:-
#     ankostis@gmail.com
# requested that this message be sent to:-
#     kostis.anagnostopoulos@ext.ec.europa.eu
#     ankostis@gmail.com
#
# This certificate was issued at 17:55 (GMT)
# on Saturday 04 February 2017 with reference 0942919
#
# CAUTION: while the message may well be from the sender
#          indicated in the "From:" header, the sender
#          has NOT been authenticated by this service
#
# For information about the Stamper service see
#        http://www.itconsult.co.uk/stamper.htm
#
########################################################


object df6cfcbb3b8840cc843f7cbf922c1451dbd7cc08
type commit
tag dices/FT-12-ABC-2016-0001/6
tagger Kostis <ankostis@gmail.com> 1486226540 +0100

- - {v: 1.0.0, a: drep 2 files, p: FT-12-ABC-2016-0001, s: tagged}
- - {file: input.xlsx, iokind: inp, vehicle_family_id: RL-99-BM3-2017-0001}
- - file: output-longer_than-usual-file-012325336475546776.xlsx
  iokind: out
  vehicle_family_id: RL-99-BM3-2017-0001
  content_type: dice_report
  content:
    CO2MPAS_deviation: [-4.14, null]
    CO2MPAS_version: [1.5.0.dev1, null]
    Model_scores: [vehicle-H, vehicle-L]
    TA_mode: [true, null]
    Vehicle: [vehicle-H, vehicle-L]
    alternator_model: [4.56, null]
    at_model: [-0.95, null]
    clutch_torque_converter_model: [4.71, null]
    co2_params: [0, null]
    datetime: ['2017/01/29-23:42:41', null]
    engine_capacity: [997, 997]
    engine_cold_start_speed_model: [18.74, null]
    engine_coolant_temperature_model: [0.59, null]
    engine_is_turbo: [true, true]
    engine_speed_model: [0.02, 91.36]
    fuel_type: [diesel, diesel]
    gear_box_type: [automatic, automatic]
    start_stop_model: [-0.99, null]
    vehicle_family_id: [RL-99-BM3-2017-0001, null]
- -----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iJwEAAEIAAYFAliWBGwACgkQsSTJmcu7Uv+fggP+IJPS3QqkBa7BF4YbPYkKjD0O
F/FJFySXCh7iX+LRwNGAudHgGmVw3puaaY10ksIeKAyjiSYEck8TF8hB+4GDG2fZ
FQ+/P9eF3EpFaIXQ8YsKsPmMuW/5iSTdU9PfbM3YSTvUYq94Jz+Xjt1ZCnOTUE/p
+YVxyIe3UwYBJWuciUY=
=AYp4
- -----END PGP SIGNATURE-----

-----BEGIN PGP SIGNATURE-----
Version: 2.6.3i
Charset: noconv
Comment: Stamper Reference Id: 0942919

iQEVAgUBWJYVdIGVnbVwth+BAQHz+Qf+KEUrHLdcg0d2dhM7yuaGm+ymLZ7oIY5A
0lO2UtoO81akGizetpWv+oHd4PvYRDBONjiW1shl+KsYlTSmz3mG0XAij9lrT7uf
4+F+/6x5QZJG5g6ktgB46rw2/mdW3ZvF6JHf+FWQP2ibDCz0mrGhuMUlHX0UJxBe
1w8Rku3iS5n2yKK4fHmQdoCHQNk/1m2QcKrajHKV3oAdCXrkiSThlMwJky6vrubt
1FtVIaVXn6c+1qwfL57r11rNY2BcHbTswa99LnRxJVpIyHe/2UXPKmcPnf375viF
JywXRfQktpKlMZeyYQPF+cGsXL+TJO2xQzCTLjR9fWoK3HuoO26/eQ==
=uUII
-----END PGP SIGNATURE-----
"""), (
    None,
    942924, {
        'hexnum': 'B006192C9D64D265F59A58C29A7C95E71BC80324',
        'percent': 84,
        'decision': 'OK',
    }, {
        'trust_text': 'TRUST_FULLY',
    }, {
        'trust_text': 'TRUST_ULTIMATE',
        'project': 'FT-12-ABC-2016-0001',
        'project_source': 'report',
        'vehicle_family_id': 'RL-99-BM3-2017-0001',
    },
""" #@IgnorePep8
-----BEGIN PGP SIGNED MESSAGE-----

########################################################
#
# This is a proof of posting certificate from
# stamper.itconsult.co.uk certifying that a user
# claiming to be:-
#     ankostis@gmail.com
# requested that this message be sent to:-
#     kostis.anagnostopoulos@ext.ec.europa.eu
#     ankostis@gmail.com
#
# This certificate was issued at 18:55 (GMT)
# on Saturday 04 February 2017 with reference 0942924
#
# CAUTION: while the message may well be from the sender
#          indicated in the "From:" header, the sender
#          has NOT been authenticated by this service
#
# For information about the Stamper service see
#        http://www.itconsult.co.uk/stamper.htm
#
########################################################


object a1875e2a105c594da67e3e899ca153a1a1d38924
type commit
tag dices/FT-12-ABC-2016-0001/8
tagger Kostis <ankostis@gmail.com> 1486234422 +0100

- - {v: 1.0.0, a: drep 2 files, p: FT-12-ABC-2016-0001, s: tagged}
- - {file: input.xlsx, iokind: inp, project: RL-99-BM3-2017-0001}
- - file: output-longer_than-usual-file-012325336475546776.xlsx
  iokind: out
  project: RL-99-BM3-2017-0001
  content_type: dice_report
  content:
    vehicle_family_id: [RL-99-BM3-2017-0001, null]
    CO2MPAS_version: [1.5.0.dev1, null]
    datetime: ['2017/01/29-23:42:41', null]
    TA_mode: [true, null]
    CO2MPAS_deviation: [-4.14, null]
    Vehicle: [vehicle-H, vehicle-L]
    fuel_type: [diesel, diesel]
    engine_capacity: [997, 997]
    gear_box_type: [automatic, automatic]
    engine_is_turbo: [true, true]
    Model_scores: [vehicle-H, vehicle-L]
    alternator_model: [4.56, null]
    at_model: [-0.95, null]
    clutch_torque_converter_model: [4.71, null]
    co2_params: [0, null]
    engine_cold_start_speed_model: [18.74, null]
    engine_coolant_temperature_model: [0.59, null]
    engine_speed_model: [0.02, 91.36]
    start_stop_model: [-0.99, null]
- -----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iJwEAAEIAAYFAliWIzYACgkQsSTJmcu7Uv+apgQAlos9dvBlBrBxPaAfyoc5W0E3
I+Xnk2pyJRtlWLdxM91JP8FuhAh6jaIE9BQUznMv5u+GSJ7zZ+6x9XnI56jxSi1R
DIfPzigZsstD4v2X7ltu7DXkXHlwbfOKH+CjzmyZm+WKVh/jlX9jN6c444iqKlhe
/yYUReezAO9BlbOw/xU=
=nI2k
- -----END PGP SIGNATURE-----

-----BEGIN PGP SIGNATURE-----
Version: 2.6.3i
Charset: noconv
Comment: Stamper Reference Id: 0942924

iQEVAgUBWJYjhYGVnbVwth+BAQH8uAf9FEVu6OxDsmlfGtI1H3CdWpn90Z61hFjD
4wwhXu4ULCEZ8ZU+13vhEKSa3YvsmMzrVFMB0dE2JieFllraOs4P+0aenD76XFPg
zY30ZtlzW62nGaJYXMuBWI/yz+WGGwGEF0qY+wJpV88rStQHaTej/XFGufHRnOoF
VuyNXVCEQ7Ml719QBKjuYmCdD/kH2fPp7xwouuMmevuEv/zFzp7nPdt2mxOXT/VL
ac22O9+e8ELvVv/XfVhHcU6ginw6oBWnkzHs59+pPBXK+mOVodo2H6TzVRNFF84Y
ct0p7ZWQqb7xn2Q3IFuU/vOiUTc5XZTnrpUr5QkHV00IMOnPnvSnag==
=QXAM
-----END PGP SIGNATURE-----
"""),
]

signed_tag = """\
object a1875e2a105c594da67e3e899ca153a1a1d38924
type commit
tag dices/FT-12-ABC-2016-0001/8
tagger Kostis <ankostis@gmail.com> 1486234422 +0100

- {v: 1.0.0, a: drep 2 files, p: FT-12-ABC-2016-0001, s: tagged}
- {file: input.xlsx, iokind: inp, project: RL-99-BM3-2017-0001}
- file: output-longer_than-usual-file-012325336475546776.xlsx
  iokind: out
  project: RL-99-BM3-2017-0001
  content_type: dice_report
  content:
    vehicle_family_id: [RL-99-BM3-2017-0001, null]
    CO2MPAS_version: [1.5.0.dev1, null]
    datetime: ['2017/01/29-23:42:41', null]
    TA_mode: [true, null]
    CO2MPAS_deviation: [-4.14, null]
    Vehicle: [vehicle-H, vehicle-L]
    fuel_type: [diesel, diesel]
    engine_capacity: [997, 997]
    gear_box_type: [automatic, automatic]
    engine_is_turbo: [true, true]
    Model_scores: [vehicle-H, vehicle-L]
    alternator_model: [4.56, null]
    at_model: [-0.95, null]
    clutch_torque_converter_model: [4.71, null]
    co2_params: [0, null]
    engine_cold_start_speed_model: [18.74, null]
    engine_coolant_temperature_model: [0.59, null]
    engine_speed_model: [0.02, 91.36]
    start_stop_model: [-0.99, null]
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iJwEAAEIAAYFAliWIzYACgkQsSTJmcu7Uv+apgQAlos9dvBlBrBxPaAfyoc5W0E3
I+Xnk2pyJRtlWLdxM91JP8FuhAh6jaIE9BQUznMv5u+GSJ7zZ+6x9XnI56jxSi1R
DIfPzigZsstD4v2X7ltu7DXkXHlwbfOKH+CjzmyZm+WKVh/jlX9jN6c444iqKlhe
/yYUReezAO9BlbOw/xU=
=nI2k
-----END PGP SIGNATURE-----
"""

missing_pub_key_id = '5124D06F'
missing_pub_key_tag = """
object 6beb4f527ee7ed0a1b77dc62a27b12e2fb0da074
type commit
tag dices/RL-99-BM3-2017-0001/1
tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1507037486 +0200

- {v: 1.0.2, a: drep 3 files, p: RL-99-BM3-2017-0001, s: tagged}

- file: input.xlsx
  iokind: inp
  report: {vehicle_family_id: RL-99-BM3-2017-0001}
- file: output.xlsx
  iokind: out
  report:
    0.vehicle_family_id: [RL-99-BM3-2017-0001, null]
    1.CO2MPAS_version: [1.5.0.dev1, null]
    2.report_type: [dice_report, null]
    3.datetime: ['2017/01/29-23:42:41', null]
    4.TA_mode: [1.0, .nan]
    5.CO2MPAS_deviation: [-4.14, .nan]
    6.Vehicle: [vehicle-H, vehicle-L]
    7.fuel_type: [diesel, diesel]
    8.engine_capacity: [997.0, 997.0]
    9.gear_box_type: [automatic, automatic]
    10.engine_is_turbo: [1.0, 1.0]
    11.Model_scores: [vehicle-H, vehicle-L]
    12.alternator_model: [4.56, .nan]
    13.at_model: [-0.95, .nan]
    14.clutch_torque_converter_model: [4.71, .nan]
    15.co2_params: [0.0, .nan]
    16.engine_cold_start_speed_model: [18.74, .nan]
    17.engine_coolant_temperature_model: [0.59, .nan]
    18.engine_speed_model: [0.02, 91.36]
    19.start_stop_model: [-0.99, .nan]
- {file: LICENSE.txt, iokind: other, report: null}
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iQEcBAABCAAGBQJZ05EvAAoJED4LzmdRJNBv4v4H/ionDWrPZ9Ipi3XGjKSK6TRA
c70zv7Th02NQLYnOOhr8W4L+OUXXw9j3raR+tw1ogrCxJy4s3LWUgayd4mJN8O/t
c2bu/FXiaNspEVzfKtBOcawoDYb9hu1HF/dS1Qhw2n009FV43dnORasMYxSpmQkW
yG8KIYCHCUbHoznvGoflJKyAAbHmoeL7vYIQeAI/hcq8Fa8IvRibg57ForG108hK
O0NLUdBGBHkH3wPn7zwdlaB9m6a4ZderNnbweprLkHVRVAUxs3DcjxWKkQmproRG
DaU+3OO7CzzFmUrGIargbUWsuHok4WRCiCZKUq1l/ooDbtz3Tv+va3NdD1+/hVc=
=1gRO
-----END PGP SIGNATURE-----
"""
missing_pub_key_decision = """
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

########################################################
#
# Proof of posting certificate from JRC-stamper
# certifying that:-
#   3E0BCE675124D06F: Orphan Public <orphan@public.key>
# requested to email this message to:-
#   dd@ee
#
# certificate_date: 2017-10-03T14:31:58.900537
# reference: 0000009
# parent_stamp: 7fdc30757850bd65d0dcc95fac70eec4887985fe
#
########################################################


object 6beb4f527ee7ed0a1b77dc62a27b12e2fb0da074
type commit
tag dices/RL-99-BM3-2017-0001/1
tagger Kostis Anagnostopoulos <ankostis@gmail.com> 1507037486 +0200

- - {v: 1.0.2, a: drep 3 files, p: RL-99-BM3-2017-0001, s: tagged}

- - file: input.xlsx
  iokind: inp
  report: {vehicle_family_id: RL-99-BM3-2017-0001}
- - file: output.xlsx
  iokind: out
  report:
    0.vehicle_family_id: [RL-99-BM3-2017-0001, null]
    1.CO2MPAS_version: [1.5.0.dev1, null]
    2.report_type: [dice_report, null]
    3.datetime: ['2017/01/29-23:42:41', null]
    4.TA_mode: [1.0, .nan]
    5.CO2MPAS_deviation: [-4.14, .nan]
    6.Vehicle: [vehicle-H, vehicle-L]
    7.fuel_type: [diesel, diesel]
    8.engine_capacity: [997.0, 997.0]
    9.gear_box_type: [automatic, automatic]
    10.engine_is_turbo: [1.0, 1.0]
    11.Model_scores: [vehicle-H, vehicle-L]
    12.alternator_model: [4.56, .nan]
    13.at_model: [-0.95, .nan]
    14.clutch_torque_converter_model: [4.71, .nan]
    15.co2_params: [0.0, .nan]
    16.engine_cold_start_speed_model: [18.74, .nan]
    17.engine_coolant_temperature_model: [0.59, .nan]
    18.engine_speed_model: [0.02, 91.36]
    19.start_stop_model: [-0.99, .nan]
- - {file: LICENSE.txt, iokind: other, report: null}
- -----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iQEcBAABCAAGBQJZ05EvAAoJED4LzmdRJNBv4v4H/ionDWrPZ9Ipi3XGjKSK6TRA
c70zv7Th02NQLYnOOhr8W4L+OUXXw9j3raR+tw1ogrCxJy4s3LWUgayd4mJN8O/t
c2bu/FXiaNspEVzfKtBOcawoDYb9hu1HF/dS1Qhw2n009FV43dnORasMYxSpmQkW
yG8KIYCHCUbHoznvGoflJKyAAbHmoeL7vYIQeAI/hcq8Fa8IvRibg57ForG108hK
O0NLUdBGBHkH3wPn7zwdlaB9m6a4ZderNnbweprLkHVRVAUxs3DcjxWKkQmproRG
DaU+3OO7CzzFmUrGIargbUWsuHok4WRCiCZKUq1l/ooDbtz3Tv+va3NdD1+/hVc=
=1gRO
- -----END PGP SIGNATURE-----
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2
Comment: Stamper Reference Id: 0000009

iJwEAQEIAAYFAlnTkU8ACgkQsSTJmcu7Uv+ItgP/Y8vUx1rtMwiYY9Dq2U3/PiS6
RBIiCmwcJDkm5YDkaxa+ZF5UtYFdLapKXgELpeHeaEw2nkcsCVCgls6CkvpLUN92
ND7glj08urOjKWwR4TRl7KmrIEnyVp1zCGNWe3ILWjLswRFzvcZLTesELn3ge3FL
etr1GmB6vyJQiIuBPwI=
=XN7g
-----END PGP SIGNATURE-----


dice:
  tag: 'dices/RL-99-BM3-2017-0001/1: 6beb4f527ee7ed0a1b77dc62a27b12e2fb0da074'
  issuer: '3E0BCE675124D06F: Orphan Public <orphan@public.key>'
  issue_date: '2017-10-03T14:31:27'
  stamper: 'B124C999CBBB52FF: CO2MPAS Test <JRC-CO2MPAS@ec.europa.eu>'
  dice_date: '2017-10-03T14:31:59'
  hexnum: 778E8240750DEF59828E2364135D2404A72F679A
  percent: 58
  decision: OK

"""


@ddt.ddt
class TRX(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = c = trtc.Config()

        c.GpgSpec.gnupghome = tempfile.mkdtemp(prefix='gpghome-')
        c.GpgSpec.keys_to_import = test_pgp_keys
        c.GpgSpec.trust_to_import = test_pgp_trust
        c.GpgSpec.master_key = test_pgp_fingerprint
        c.GpgSpec.allow_test_key = True
        crypto.GpgSpec(config=c)

        ## Clean memories from past tests
        #
        crypto.StamperAuthSpec.clear_instance()     # @UndefinedVariable
        crypto.GitAuthSpec.clear_instance()         # @UndefinedVariable
        crypto.VaultSpec.clear_instance()           # @UndefinedVariable

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.GpgSpec.gnupghome)

    def check_timestamp_fails(self, rcv, mail_text, msg_regex):
        with self.assertRaisesRegex(tstamp.CmdException, msg_regex):
            rcv.parse_tstamp_response(mail_text)

    def check_timestamp(self, rcv, stamper_id, dice, ts_verdict, tag_verdict, tstamp_response):
        resp = rcv.parse_tstamp_response(tstamp_response)
        print(pf(resp))
        self.assertEqual(resp['tstamp']['stamper_id'], stamper_id, pf(resp))
        self.assertDictContainsSubset(dice, resp['dice'], pf(resp))
        ts_verdict.update({'key_id': '81959DB570B61F81',
                           'pubkey_fingerprint': '4B12BCD5788511063B543190E09DF306'})
        self.assertDictContainsSubset(ts_verdict, resp['tstamp'], pf(resp))
        tv = tag_verdict.copy()
        tv.pop('vehicle_family_id', None)  # Exist for test_scan_vfid_regex(), below.
        self.assertDictContainsSubset(tv, resp['report'], pf(resp))

    def test_send_timestamp(self):
        snd = tstamp.TstampSender(config=self.cfg)
        ex_msg = ("Failed to extract signed dice-report from tstamp!\\s+"
                  "0-line text is not a PGP-sig!")
        with self.assertRaisesRegex(tstamp.CmdException, ex_msg):
            snd.send_timestamped_email("", dry_run=True)

    def test_extract_base64_blob(self):
        tag_text = """Τιριρερεμ"""
        snd = tstamp.TstampSender(config=self.cfg)
        snd.scramble_tag = True
        b64_text = snd._scramble_tag(tag_text, 'Hi there')
        self.assertTrue(all(ord(c) < 128 for c in b64_text), b64_text)
        self.assertTrue(all(len(l) < 78 for l in b64_text.split('\n')), b64_text)

        rcv = tstamp.TstampReceiver(config=self.cfg)
        tstamp_text = 'Some\nFobar\r\n%s\r\n\n' % b64_text
        new_tag_text = rcv._descramble_tag(tstamp_text)
        self.assertEqual(tag_text, new_tag_text.decode('utf-8'))

    @ddt.data(*tstamp_responses)
    def test_scan_vfid_regex(self, case):
        rcv = tstamp.TstampReceiver(config=self.cfg)
        vfid = rcv.scan_for_project_name(case[-1])
        exp_vfid = (case[4]['project'], case[4].get('vehicle_family_id'))
        self.assertIn(vfid, exp_vfid)

    @ddt.data('tstamp_recipients', 'cc_addresses', 'bcc_addresses')
    def test_email_lists_validations(self, trait_name):
        ex_msg = "needs a proper email"
        with self.assertRaisesRegex(trt.TraitError, "%s.+bad_email" % ex_msg):
            ts = tstamp.TstampSender(**{trait_name: ['bad_email']})
            print(ts.tstamp_recipients)

    @ddt.data(
        ('project:  dices/IP-10-AAA-2017-1003/1\n', '', 'dices/IP-10-AAA-2017-1003/1'),
        (None,
         "object 76b8bf7312770a488eaeab4424d080dea3272435\n"
         "type commit\r\r\n"
         "tag dices/IP-10-AAA-2017-0012/12\r\n",
         'dices/IP-10-AAA-2017-0012/12: 76b8bf7312770a488eaeab4424d080dea3272435'),
        ('sdfd dsfsd', '0trpe Γρεεκ', None),
    )
    def test_extract_tag_name(self, case):
        subject, msg, tag = case
        rcv = tstamp.TstampReceiver(config=self.cfg)
        tag_name = rcv.extract_dice_tag_name(subject, msg)
        self.assertEqual(tag_name, tag)

    def test_parse_timestamp_bad(self):
        rcv = tstamp.TstampReceiver(config=self.cfg)
        ex_msg = (r"Cannot verify timestamp-response's signature due to: "
                  "error - verify 4294967295")
        with self.assertRaisesRegex(tstamp.CmdException, ex_msg):
            rcv.parse_tstamp_response("")

    @ddt.data(*tstamp_responses)
    def test_parse_timestamps(self, case):
        fail_regex, *verdicts = case
        rcv = tstamp.TstampReceiver(config=self.cfg)
        if fail_regex:
            self.check_timestamp_fails(rcv, verdicts[-1], fail_regex)
            rcv.force = True
        self.check_timestamp(rcv, *verdicts)

    def test_parse_signed_tag(self):
        rcv = tstamp.TstampReceiver(config=self.cfg)
        verdict = rcv.parse_signed_tag(signed_tag)
        sverdict = yaml.dump(verdict, indent=2)
        self.assertTrue(verdict['valid'], sverdict)
        self.assertIn(crypto._TEST_KEY_ID, verdict['fingerprint'], sverdict)
        self.assertEqual(verdict['commit_msg']['p'],
                         'FT-12-ABC-2016-0001',
                         sverdict)

    def test_parse_missing_pub_key_tag(self):
        rcv = tstamp.TstampReceiver(config=self.cfg)
        verdict = rcv.parse_signed_tag(missing_pub_key_tag)
        print(yaml.dump(verdict, indent=2))
        self.assertFalse(verdict['valid'], verdict)
        self.assertEqual(verdict['status'], 'no public key')
        self.assertIn(missing_pub_key_id, verdict['key_id'], verdict)

    def test_parse_missing_pub_key_decision(self):
        cfg = self.cfg.copy()
        cfg.Spec.force = True
        rcv = tstamp.TstampReceiver(config=cfg)
        verdict = rcv.parse_tstamp_response(missing_pub_key_decision)
        print(yaml.dump(verdict, indent=2))
        self.assertTrue(verdict['tstamp']['valid'])
        self.assertFalse(verdict['report']['valid'])
        self.assertEqual(verdict['report']['status'], 'no public key')
        self.assertIn(missing_pub_key_id, verdict['report']['key_id'])

    @ddt.data(
        (None, None, []),
        (None, '', []),
        (None, '  ', []),

        ([], None, []),
        ([], '', []),
        ([], '  ', []),

        (['', '  '], None, ['', '  ']),
        (['', '  '], '', ['', '  ']),
        (['', '  '], '  ', ['', '  ']),
    )
    def test_criteria_stripping(self, case):
        ecrts, one, projects = case
        rcv = tstamp.TstampReceiver(rfc_criteria=ecrts,
                                    wait_criterio=one,
                                    subject_prefix=one)
        crt = rcv._prepare_search_criteria(True, projects)
        self.assertEqual(crt, '')

    def test_criteria_dupe_projects(self):
        rcv = tstamp.TstampReceiver(rfc_criteria=[],
                                    subject_prefix='')
        crt = rcv._prepare_search_criteria(False, [])  # sanity
        self.assertEqual(crt, '')

        crt = rcv._prepare_search_criteria(False, ['ab', 'ab'])
        self.assertNotIn('OR', crt)

        crt = rcv._prepare_search_criteria(False, ['ab', 'foo', 'ab'])
        self.assertEqual(crt.count('OR'), 1)

    def test_capture_tstamper_parts(self):
        s= """  # @IgnorePep8
########################################################\r
#\r
# This is a proof of posting certificate from\r
# stamper.itconsult.co.uk certifying that a user\r
# claiming to be:-\r
#     ankostis@outlook.com\r
# requested that this message be sent to:-\r
#     anagnko@gmail.com=0A=\r
#     kostis.anagnostopoulos@ext.ec.europa.eu=0A=\r
#\r
# This certificate was issued at 22:25 (GMT)\r
# on Wednesday 28 June 2017 with reference 0967126\r
#\r
# CAUTION: while the message may well be from the sender\r
#          indicated in the "From:" header, the sender\r
#          has NOT been authenticated by this service\r
#\r
# For information about the Stamper service see\r
#        http://www.itconsult.co.uk/stamper.htm\r
#\r
########################################################\r
\r
=0A=\r
tag: dices/IP-10-AAA-2017-0012/4=0A=\r
base32(tag): |=0A=\r
  N5RGUZLDOQQDAMBTMY2DANTCME4GCOJSMU3TONLBHAYWKY3CGEZTSM3BHA3TOM=0A=\r
  JNFUWQU=3D=3D=3D=\r
"""
        m = tstamp._stamper_banner_regex.search(s)
        self.assertIsNotNone(m)

    def test_num_to_decision(self):
        import random

        rnd = random.Random(0)
        pgp_sig_id_nbytes = 20
        max_sig_id = 2 ** (8 * pgp_sig_id_nbytes) - 1

        sig_ids = [rnd.randint(0, max_sig_id)
                   for _ in range(10_000)]
        stats = Counter(tstamp.num_to_dice100(sig_id, True)[1]
                        for sig_id in sig_ids)

        print('\n    %10s' % 'NEW_DICE100')
        print('\n'.join('%2s: %10s' % p for p in stats.items()))

        limit = 90

        def make_prcnt(counter):
            ge_limit = sum(v
                           for k, v in counter.items()
                           if k >= limit)
            return ge_limit / sum(counter.values())

        pcrnt = make_prcnt(stats)
        print('%s-prcnt: %.6f' % (limit, pcrnt))
        self.assertAlmostEqual(0.1, pcrnt, 2)

    def test_tranfer_encoders_map(self):
        enc_map = tstamp._make_send_transfer_encoders_map()
        self.assertEqual(len(set(enc_map)), 7, enc_map)
        for k, v in enc_map.items():
            if v is not None:
                self.assertTrue(callable(v), (k, v))

    @ddt.data(
        ('tag.txt', 0),
        ('tstamp.txt', 1),
    )
    def test_parse_out(self, case):
        fname, keys_indx = case
        cmd = tstamp.ParseCmd(config=self.cfg)
        res = collect_cmd(cmd.run(osp.join(mydir, fname)))

        substrings = [
            ['model', 'project', 'project_source'],
            ['tag', 'issuer', 'issue_date', 'dice_date',
             'hexnum', 'percent', 'decision'],
        ]
        for k in substrings[keys_indx]:
            tail = res[-500:]
            self.assertIn('%s:' % k, tail, (k, tail))

@ddt.ddt
class TstampShell(unittest.TestCase):
    """Set ``TEST_CO2DICE_CONFIG_PATHS`` and optionally HOME env appropriately! to run"""
    @classmethod
    def setUpClass(cls):
        cls._old_envvar = os.environ.get(CONF_ENVAR)
        if TEST_CONF_ENVAR in os.environ:
            os.environ[CONF_ENVAR] = os.environ[TEST_CONF_ENVAR]

    @classmethod
    def tearDownClass(cls):
        if cls._old_envvar is not None:
            os.environ[CONF_ENVAR] = cls._old_envvar

    def _get_vault_gnupghome(self):
        """Allow cipher-traits encrypted with the GPG of the user running tests."""
        cmd = Cmd()
        cmd.initialize([])
        crypto.VaultSpec.clear_instance()  # @UndefinedVariable
        vault = crypto.VaultSpec(config=cmd.config)
        return ('--VaultSpec.gnupghome=%s' % vault.gnupghome
                if vault.gnupghome
                else '')

    def test_config_paths(self):
        ret = sbp.check_call('co2dice config paths')
        self.assertEqual(ret, 0)

    def test_parse_tstamp_file(self):
        ret = sbp.check_call('co2dice tstamp parse tstamp.txt',
                             cwd=mydir)
        self.assertEqual(ret, 0)

    def test_parse_tag_file(self):
        ret = sbp.check_call('co2dice tstamp parse tag.txt',
                             cwd=mydir)
        self.assertEqual(ret, 0)

    def test_login_smoketest(self):
        vault_cli = self._get_vault_gnupghome()
        ret = sbp.check_call('co2dice tstamp login %s' % vault_cli)
        self.assertEqual(ret, 0)

    def test_mailbox_smoketest(self):
        vault_cli = self._get_vault_gnupghome()
        ret = sbp.check_call('co2dice tstamp mailbox %s' % vault_cli)
        self.assertEqual(ret, 0)

    def test_recv_smoketest(self):
        vault_cli = self._get_vault_gnupghome()
        ret = sbp.check_call('co2dice tstamp recv %s' % vault_cli)
        self.assertEqual(ret, 0)

        ret = sbp.check_call('co2dice tstamp recv --page=10 %s' % vault_cli)
        self.assertEqual(ret, 0)

        ret = sbp.check_call('co2dice tstamp recv --page=-2: %s' % vault_cli)
        self.assertEqual(ret, 0)

        ret = sbp.check_call('co2dice tstamp recv --raw %s' % vault_cli)
        self.assertEqual(ret, 0)

        ret = sbp.check_call('co2dice tstamp recv --list %s' % vault_cli)
        self.assertEqual(ret, 0)

    def test_send_smoketest(self):
        vault_cli = self._get_vault_gnupghome()
        fpath = osp.join(mydir, '..', '..', 'setup.py')
        ret = sbp.check_call('co2dice tstamp send %s --dry-run -f %s'
                             % (fpath, vault_cli))
        self.assertEqual(ret, 0)
