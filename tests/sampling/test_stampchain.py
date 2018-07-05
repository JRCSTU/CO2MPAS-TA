#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas._vendor.traitlets import config as trtc
from co2mpas.sampling import tsigner, tstamp
import logging
import os
import unittest

import ddt

import os.path as osp


init_logging(level=logging.WARNING)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)

@ddt.ddt
class TcfgcmdShell(unittest.TestCase):
    def _stamp_dir(self):
        return os.environ['STAMP_CHAIN_DIR']

    def test_DB_ENVAR(self):
        assert 'STAMP_CHAIN_DIR' in os.environ

    def _SigChain(self, cfg=None):
        cfg = cfg or trtc.Config()
        cfg.SigChain.stamp_chain_dir = self._stamp_dir()
        return tsigner.SigChain(config=cfg)

    def test_build_parent_chain(self):
        signer = self._SigChain()
        chain = signer.load_stamp_chain()
        assert len(chain) > 0
        nfiles = 0
        for _dirpath, _dirs, files in os.walk(self._stamp_dir()):
            nfiles += len(files)

        assert len(chain) + 1 == nfiles  # +1 for HEAD

    def test_parse_stamps(self):
        cfg = trtc.Config()
        cfg.TstampReceiver.force = True
        signer = self._SigChain(cfg=cfg)
        trecv = tstamp.TstampReceiver(config=cfg)
        chain = signer.load_stamp_chain()

        errors = []
        for sig_hex in chain:
            try:
                stamp = signer.load_sig_file(sig_hex)
                _verdict = trecv.parse_tstamp_response(stamp)
                #print(_verdict)
            except Exception as ex:
                errors.append(ex)

        assert len(errors) == 0