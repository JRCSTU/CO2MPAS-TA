#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp
import io
import json
import logging
import unittest

import os.path as osp
import pandalone.utils as pndlu
import traitlets as trt


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


def prepare_persistent_config_file(pfile):
    j = {}
    if osp.isfile(pfile):
        with io.open(pfile, 'rt') as finp:
            j = json.load(finp)

    ## Add an arbitrary parameters in pfile to see if it is preserved.
    j['ANY'] = {'a': 1}

    ## Add an arbitrary parameters in pfile to see if it is preserved.
    j['MyCmd'] = {'ptrait': False}

    with io.open(pfile, 'wt') as fout:
        json.dump(j, fout)


class TBaseApp(unittest.TestCase):

    def check_persistent_config_file(self, pfile, flag):
        with io.open(pfile, 'rt') as finp:
            j = json.load(finp)

        self.assertEqual(j['ANY']['a'], 1, j)
        self.assertEqual(j['MyCmd']['ptrait'], flag, j)

    def test_ptraits(self,):
        pfile = pndlu.ensure_file_ext(baseapp.default_config_fpath(), '.json')
        prepare_persistent_config_file(pfile)

        class MyCmd(baseapp.Cmd):
            "No desc"
            ptrait = trt.Bool().tag(config=True, persistTrue)

        c = MyCmd()
        self.check_persistent_config_file(pfile, False)
        c.initialize([])
        self.check_persistent_config_file(pfile, False)

        c.ptrait = True
        self.check_persistent_config_file(pfile, False)

        c.store_pconfig_file(pfile)
        self.check_persistent_config_file(pfile, True)

    def test_unconfig_ptraits(self):
        class MyCmd(baseapp.Cmd):
            "No desc"
            bad_ptrait = trt.Bool().tag(persist=True)

        c = MyCmd()
        with self.assertLogs(c.log, logging.FATAL) as cm:
            try:
                c.initialize([])
            except SystemExit:
                pass
        exp_msg = "Persistent trait 'bad_ptrait' not tagged as 'config'!"
        self.assertTrue(any(exp_msg in m for m in cm.output), cm.output)
