#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import tstamp
import logging
import unittest

import os.path as osp


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


dice_txt = """
-----BEGIN PGP SIGNED MESSAGE-----

########################################################
#
# This is a proof of posting certificate from
# stamper.itconsult.co.uk certifying that a user
# claiming to be:-
#     ankostis@gmail.com
# requested that this message be sent to:-
#     konstantinos.anagnostopoulos@ext.jrc.ec.europa.eu
#     sundancedawngold@gmail.com
#
# This certificate was issued at 01:25 (GMT)
# on Wednesday 23 November 2016 with reference 0931201
#
# CAUTION: while the message may well be from the sender
#          indicated in the "From:" header, the sender
#          has NOT been authenticated by this service
#
# For information about the Stamper service see
#        http://www.itconsult.co.uk/stamper.htm
#
########################################################



- -----BEGIN PGP PUBLIC KEY BLOCK-----

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
=R0r8
- -----END PGP PUBLIC KEY BLOCK-----

-----BEGIN PGP SIGNATURE-----
Version: 2.6.3i
Charset: noconv
Comment: Stamper Reference Id: 0931201

iQEVAgUBWDTv7IGVnbVwth+BAQF+Wwf+JfBf3/dhvOJPnxVWTHG0LcdS8Jmm7Zpc
5HvJdlPUI2cltorkEKXPw+pFnXYnAX5iZLO7CjraSvliW9fzTt+KW176Sf8ZaNmy
pfeU513Aj/yDkkUPpHCK1Ib4M6WAy7o5RDgQJBRdqrjXY/tWQRrhiS7poJ/9yrD6
WN4D9P9vOtuwINFONAb971zSXQI0+IH3ft8Waqp+7AnIk8Yq/rcQznoWOVNOetlw
8J+nFld2zJMFBJT0ITDaedTk/HLk43PmLj+iXHmhfEgrS3LAr8LXfSUIJMxegSDs
8x/OVP1oYY0ueYTZvk8r4x6t3IOI+bW39b6TAl+dyRjzyB/1oVNG7w==
=i2wn
-----END PGP SIGNATURE-----

"""


class T(unittest.TestCase):

    def test_parse(self):
        c = tstamp.TstampReceiver()
        sig, num, mod100, decision = c.parse_tsamp_response(dice_txt)
        print(sig, num, mod100, decision)
