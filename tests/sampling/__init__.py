#! python
# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os.path as osp
import textwrap as tw


mydir = osp.dirname(__file__)
test_inp_fpath = osp.join(mydir, 'input.xlsx')
test_out_fpath = osp.join(mydir, 'output.xlsx')
test_vfid = 'RL-99-BM3-2017-0001'

test_pgp_fingerprint = 'B124C999CBBB52FF'
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
#  and with just the "Timestamp Service" key signed by the test-key, above.
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
#  NOTE: to import it, need --allow-weak-digest-algos
#
-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: GnuPG v2

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
El4C5af/8COxQcDn608=
=TF9x
-----END PGP PUBLIC KEY BLOCK-----
""",
]

test_pgp_trust = """\
## CO2MPAS test-key
5464E04EE547D1FEDCAC4342B124C999CBBB52FF:6:
"""
