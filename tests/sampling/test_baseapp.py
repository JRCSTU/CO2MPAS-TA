#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp, crypto
import io
import json
import logging
import os
import shutil
import tempfile
from unittest import mock
import unittest

import ddt

import itertools as itt
import os.path as osp
import pandalone.utils as pndlu
import traitlets as trt
import traitlets.config as trtc

from . import test_crypto as cryptotc


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


def mix_dics(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


class TCfgFilesRegistry(unittest.TestCase):

    def test_consolidate_1(self):
        visited = [
            ('D:\\Apps\\cygwin64\\home\\anastkn\\.co2dice', None),
            ('D:\\Apps\\cygwin64\\home\\anastkn\\.co2dice', 'co2dice_config.py'),
            ('D:\\Apps\\cygwin64\\home\\anastkn\\.co2dice', 'co2dice_config.json'),
            ('d:\\apps\\cygwin64\\home\\anastkn\\work\\compas.vinz\\co2mpas\\sampling', None),
            ('d:\\apps\\cygwin64\\home\\anastkn\\work\\compas.vinz\\co2mpas\\sampling', None),
            ('d:\\apps\\cygwin64\\home\\anastkn\\work\\compas.vinz\\co2mpas\\sampling', None),
            ('d:\\apps\\cygwin64\\home\\anastkn\\work\\compas.vinz\\co2mpas\\sampling', None),
        ]
        c = baseapp.CfgFilesRegistry()
        cons = c._consolidate(visited)

        exp = [
            ('D:\\Apps\\cygwin64\\home\\anastkn\\.co2dice', ['co2dice_config.py', 'co2dice_config.json']),
            ('d:\\apps\\cygwin64\\home\\anastkn\\work\\compas.vinz\\co2mpas\\sampling', [])
        ]
        print('FF\n', cons)
        self.assertListEqual(cons, exp, visited)

    def test_consolidate_2(self):
        visited =  [
            ('C:\\Users\\anastkn\\.co2dice', 'co2dice_persist.json'),
            ('C:\\Users\\anastkn\\.co2dice', 'co2dice_config.py'),
            ('C:\\Users\\anastkn\\.co2dice', None),
            ('D:\\Work\\compas.vinz\\co2mpas\\sampling', None),
            ('D:\\Work\\compas.vinz\\co2mpas\\sampling', None),
            ('D:\\Work\\compas.vinz\\co2mpas\\sampling', None),
            ('D:\\Work\\compas.vinz\\co2mpas\\sampling', None),
        ]
        c = baseapp.CfgFilesRegistry()
        cons = c._consolidate(visited)

        exp =   [
            ('C:\\Users\\anastkn\\.co2dice', ['co2dice_persist.json', 'co2dice_config.py']),
            ('D:\\Work\\compas.vinz\\co2mpas\\sampling', [])
        ]
        #print('FF\n', cons)
        self.assertListEqual(cons, exp, visited)

    def test_default_loaded_paths(self):
        f = '%s.py' % baseapp.default_config_fpaths()[0]

        ## Ensure at least one default home config-file.
        if not osp.isfile(f):
            io.open(f, 'w').close()
        cmd = baseapp.Cmd()
        cmd.initialize([])
        self.assertGreaterEqual(len(cmd.loaded_config_files), 1)
        print(cmd._cfgfiles_registry.visited_files)


@ddt.ddt
class TPConfFiles(unittest.TestCase):
    def check_cmd_params(self, cmd, values):
        self.assertSequenceEqual([cmd.a, cmd.b, cmd.c], values)

    @mock.patch('co2mpas.sampling.baseapp.default_config_fname', lambda: 'c')
    @ddt.data(
        (None, None, []),
        (['cc', 'cc.json'], None, []),


        ## Because of ext-stripping.
        (['b.py', 'a.json'], None, ['b.json', 'a.py']),
        (['c.json'], None, ['c.json']),

        ## Because 'c' monekypatched default-name.
        ([''], None, ['c.py', 'c.json']),

        (['a'], None, ['a.py']),
        (['b'], None, ['b.json']),
        (['c'], None, ['c.py', 'c.json']),

        (['c.json', 'c.py'], None, ['c.json', 'c.py']),
        (['c.json;c.py'], None, ['c.json', 'c.py']),

        (['c', 'c.json;c.py'], None, ['c.py', 'c.json']),
        (['c;c.json', 'c.py'], None, ['c.py', 'c.json']),

        (['a', 'b'], None, ['a.py', 'b.json']),
        (['b', 'a'], None, ['b.json', 'a.py']),
        (['c'], None, ['c.py', 'c.json']),
        (['a', 'c'], None, ['a.py', 'c.py', 'c.json']),
        (['a', 'c'], None, ['a.py', 'c.py', 'c.json']),
        (['a;c'], None, ['a.py', 'c.py', 'c.json']),
        (['a;b', 'c'], None, ['a.py', 'b.json', 'c.py', 'c.json']),

        (None, 'a', ['a.py']),
        (None, 'b', ['b.json']),
        (None, 'c', ['c.py', 'c.json']),
        (None, 'b;c', ['b.json', 'c.py', 'c.json']),

        ('b', 'a', ['b.json']),
    )
    def test_collect_static_fpaths(self, case):
        with tempfile.TemporaryDirectory(prefix='co2conf-') as tdir:
            for f in ('a.py', 'b.json', 'c.py', 'c.json'):
                io.open(osp.join(tdir, f), 'w').close()

            try:
                param, var, exp = case
                exp = [osp.join(tdir, f) for f in exp]

                cmd = baseapp.Cmd()
                if param is not None:
                    cmd.config_paths = [osp.join(tdir, ff)
                                        for f in param
                                        for ff in f.split(os.pathsep)]
                if var is not None:
                    os.environ['CO2DICE_CONFIG_PATH'] = os.pathsep.join(
                        osp.join(tdir, ff)
                        for f in var
                        for ff in f.split(os.pathsep))

                paths = cmd._collect_static_fpaths()
                self.assertListEqual(paths, exp)
            finally:
                try:
                    del os.environ['CO2DICE_CONFIG_PATH']
                except:
                    pass

    def test_move_both_config_persist_files(self):
        class MyCmd(baseapp.Cmd):
            "Ok Cmd"
            a = trt.Int().tag(config=True)
            b = trt.Int().tag(config=True)
            c = trt.Int().tag(config=True, persist=True)

        texts = ("c.MyCmd.a=3\nc.MyCmd.b=3;c.MyCmd.c=3",
                 '{"MyCmd": {"a": 1, "b": 1, "c": 1}}',
                 '{"MyCmd": {"a": 2,  "c": 2}}')

        cmd = MyCmd()
        self.check_cmd_params(cmd, (0, 0, 0))
        cmd.initialize([])
        self.check_cmd_params(cmd, (0, 0, 0))

        with tempfile.TemporaryDirectory(prefix='co2conf-') as tdir:
            fnames = [osp.join(tdir, f)
                      for f in
                      ('stat.py', 'stat.json', 'dyna.json')]
            for f, txt in zip(fnames, texts):
                with io.open(f, 'wt') as fp:
                    fp.write(txt)

            cmd = MyCmd()
            cmd.config_paths = fnames[:2]
            cmd.persist_path = fnames[2]
            cmd.initialize([])
            self.check_cmd_params(cmd, (2, 3, 2))

            cmd = MyCmd()
            cmd.config_paths = [fnames[1], fnames[0]]
            cmd.persist_path = fnames[2]
            cmd.initialize([])
            self.check_cmd_params(cmd, (2, 1, 2))

            cmd = MyCmd()
            cmd.config_paths = fnames[:2]
            cmd.persist_path = ''
            cmd.initialize([])
            self.check_cmd_params(cmd, (3, 3, 3))

            cmd = MyCmd()
            cmd.config_paths = fnames[:1]
            cmd.persist_path = ''
            cmd.initialize([])
            self.check_cmd_params(cmd, (3, 3, 3))

            cmd = MyCmd()
            cmd.config_paths = [fnames[1] + osp.pathsep + fnames[0]]
            cmd.persist_path = ''
            cmd.initialize([])
            self.check_cmd_params(cmd, (1, 1, 1))

    def test_check_non_encrypted_in_config_files(self):
        class MyCmd(baseapp.Cmd):
            "Ok Cmd"
            enc = crypto.Cipher().tag(config=True, persist=True)

        with tempfile.TemporaryDirectory(prefix='co2conf-') as tdir:
            js = '{"MyCmd": {"enc": "BAD_ENC"}}'

            persist_path = osp.join(tdir, 'a.json')
            with io.open(persist_path, 'w') as fp:
                fp.write(js)

            ## Setup vault not to scream.
            #
            vault = crypto.VaultSpec.instance()
            vault.gnupghome = tdir
            fingerprint = cryptotc.gpg_gen_key(
                vault.GPG,
                key_length=1024,
                name_real='test user',
                name_email='test@test.com')
            vault.master_key = fingerprint

            cmd = MyCmd()
            cmd.config_paths = [persist_path]

            with self.assertLogs(cmd.log, 'ERROR') as cm:
                cmd.initialize([])
            self.assertNotEqual(cmd.enc, "BAD_ENC")
            logmsg = "Found 1 non-encrypted params in static-configs: ['MyCmd.enc']"
            self.assertIn(logmsg, [r.message for r in cm.records], cm.records)

            ## But if persist-config, autoencrypted
            cmd = MyCmd()
            cmd.persist_path = persist_path
            cmd.initialize([])
            self.assertTrue(crypto.is_pgp_encrypted(cmd.enc), cmd.enc)


@ddt.ddt
class TBase(unittest.TestCase):

    def check_persistent_config_file(self, pfile, clsname=None, trait=None, value=None):
        with io.open(pfile, 'rt') as finp:
            j = json.load(finp)

        ## Check arbitrary params preserved.
        #
        self.assertIn('MySpec', j, j)
        self.assertIn('MyCmd', j, j)
        self.assertEqual(j['ANY']['a'], 1, j)

        if clsname is not None:
            self.assertEqual(j[clsname][trait], value, j)

        return j


def prepare_persistent_config_file(pfile, extra_configs=None):
    j = {}
    if osp.isfile(pfile):
        with io.open(pfile, 'rt') as finp:
            j = json.load(finp)

    ## Add an arbitrary parameters in pfile to see if it is preserved.
    j['ANY'] = {'a': 1}

    ## Add an arbitrary parameters in pfile to see if it is preserved.
    j['MyCmd'] = {'ptrait': False}
    j['MySpec'] = {'ptrait': False}

    if extra_configs:
        j.update(extra_configs)

    with io.open(pfile, 'wt') as fout:
        json.dump(j, fout)


@ddt.ddt
class TPTraits(TBase):
    @classmethod
    def setUpClass(cls):
        cls._tdir = tempfile.mkdtemp(prefix='co2persist-')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tdir)


    def test_ptraits_spec(self):
        pfile = osp.join(self._tdir, 'foo.json')
        prepare_persistent_config_file(pfile)

        class MySpec(baseapp.Spec):
            "Ok Spec"
            ptrait = trt.Bool().tag(config=True, persist=True)

        c = MySpec()
        ## Needed bc only final Cmds load ptraits.
        #
        c.load_pconfig(pfile)
        c.update_config(c._pconfig)

        self.check_persistent_config_file(pfile, 'MySpec', 'ptrait', False)

        c.ptrait = True
        self.check_persistent_config_file(pfile, 'MySpec', 'ptrait', False)

        c.store_pconfig(pfile)
        self.check_persistent_config_file(pfile, 'MySpec', 'ptrait', True)

    def test_ptraits_cmd(self):
        pfile = osp.join(self._tdir, 'foo.json')
        prepare_persistent_config_file(pfile)

        class MyCmd(baseapp.Cmd):
            "Ok Cmd"
            ptrait = trt.Bool().tag(config=True, persist=True)

        c = MyCmd()
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', False)
        c.persist_path = pfile
        c.initialize([])
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', False)

        c.ptrait = True
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', False)

        c.store_pconfig(pfile)
        self.check_persistent_config_file(pfile, 'MyCmd', 'ptrait', True)

    @ddt.data(
        {}, {'config': False}, {'config': None}, {'config': 0}, {'config': 1}, {'config': -2})
    def test_invalid_ptraits_on_spec(self, tags):
        class MySpec(baseapp.Spec):
            "Spec with invalid ptrait"
            bad_ptrait = trt.Bool().tag(persist=True, **tags)

        with self.assertRaisesRegex(trt.TraitError,
                                    "Persistent trait 'bad_ptrait' not tagged as 'config'!"):
            MySpec()

    @ddt.data(
        {}, {'config': False}, {'config': None}, {'config': 0}, {'config': 1}, {'config': -2})
    def test_invalid_ptraits_on_cmd(self, tags):
        class MyCmd(baseapp.Cmd):
            "Cmd with invalid ptrait"
            bad_ptrait = trt.Bool().tag(persist=True, **tags)

        c = MyCmd()
        with self.assertLogs(c.log, logging.FATAL) as cm:
            try:
                c.initialize([])
            except SystemExit:
                pass
        exp_msg = "Persistent trait 'bad_ptrait' not tagged as 'config'!"
        self.assertTrue(any(exp_msg in m for m in cm.output), cm.output)


@ddt.ddt
class TCipherTraits(TBase):
    @classmethod
    def setUpClass(cls):
        cls._tdir = tdir = tempfile.mkdtemp(prefix='co2cipher-')
        cfg = trtc.get_config()
        cfg.VaultSpec.gnupghome = tdir
        vault = crypto.VaultSpec.instance(config=cfg)

        fingerprint = cryptotc.gpg_gen_key(
            vault.GPG,
            key_length=1024,
            name_real='test user',
            name_email='test@test.com')
        vault.master_key = fingerprint

    @classmethod
    def tearDownClass(cls):
        vault = crypto.VaultSpec.instance()
        assert vault.gnupghome
        cryptotc.gpg_del_key(vault.GPG, vault.master_key)
        shutil.rmtree(cls._tdir)

    def test_chiphertraits_spec(self):
        plainval = 'foo'
        pfile = osp.join(self._tdir, 'foo.json')
        prepare_persistent_config_file(pfile, {'MySpec': {'ctrait': plainval}})

        class MySpec(baseapp.Spec):
            "OK Spec"

            ctrait = crypto.Cipher(None, allow_none=True).tag(config=True, persist=True)

        c = MySpec()
        ## Needed bc only final Cmds load ptraits.
        #
        c.load_pconfig(pfile)
        self.assertIsNone(c.ctrait)
        c.update_config(c._pconfig)

        cipher0 = c.ctrait      # from pconfig-file
        self.assertTrue(crypto.is_pgp_encrypted(cipher0))

        c.ctrait = plainval
        cipher1 = c.ctrait      # 1st runtime encryption
        self.assertTrue(crypto.is_pgp_encrypted(cipher1))
        self.assertEqual(c.ctrait, cipher1)  # Preserved among gets.
        self.assertNotEqual(cipher0, cipher1)
        self.assertEqual(cipher1, c._trait_values['ctrait'])

        c.ctrait = plainval
        cipher2 = c.ctrait      # 2nd runtime encryption
        self.assertTrue(crypto.is_pgp_encrypted(cipher2))
        self.assertNotEqual(cipher1, cipher2)  # Due to encryption nonse.

        j = self.check_persistent_config_file(pfile)
        c.store_pconfig(pfile)
        self.check_persistent_config_file(pfile, 'MySpec', 'ctrait', cipher2)

    def test_chiphertraits_cmd(self):
        plainval = 'foo'
        pfile = osp.join(self._tdir, 'foo.json')
        prepare_persistent_config_file(pfile, {'MyCmd': {'ctrait': plainval}})

        class MyCmd(baseapp.Cmd):
            "OK Cmd"

            ctrait = crypto.Cipher(None, allow_none=True).tag(config=True, persist=True)

        c = MyCmd()
        self.assertIsNone(c.ctrait)

        c.config_paths = [self._tdir]
        c.persist_path = pfile
        c.initialize([])

        cipher0 = c.ctrait      # from pconfig-file
        self.assertIsNotNone(c.ctrait)
        self.assertTrue(crypto.is_pgp_encrypted(cipher0))

        c.ctrait = plainval
        cipher1 = c.ctrait      # 1st runtime encryption
        self.assertTrue(crypto.is_pgp_encrypted(cipher1))
        self.assertEqual(c.ctrait, cipher1)  # Preserved among gets.
        self.assertNotEqual(cipher0, cipher1)
        self.assertEqual(cipher1, c._trait_values['ctrait'])

        c.ctrait = plainval
        cipher2 = c.ctrait      # 2nd runtime encryption
        self.assertTrue(crypto.is_pgp_encrypted(cipher2))
        self.assertNotEqual(cipher1, cipher2)  # Due to encryption nonse.

        j = self.check_persistent_config_file(pfile)
        c.store_pconfig(pfile)
        self.check_persistent_config_file(pfile, 'MyCmd', 'ctrait', cipher2)

