#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import io
import os
import re
import unittest
from unittest.mock import patch

import compas
from compas import __main__ as compas_main


mydir = os.path.dirname(__file__)
readme_path = os.path.join(mydir, '..', 'README.rst')
tutorial_path = os.path.join(mydir, '..', 'doc', 'tutorial.rst')


class Doctest(unittest.TestCase):

    def test_README_version_opening(self):
        ver = compas.__version__
        header_len = 20
        mydir = os.path.dirname(__file__)
        with open(readme_path) as fd:
            for i, l in enumerate(fd):
                if ver in l:
                    break
                elif i >= header_len:
                    msg = "Version(%s) not found in README %s header-lines!"
                    raise AssertionError(msg % (ver, header_len))

    def test_README_version_from_cmdline(self):
        ver = compas.__version__
        mydir = os.path.dirname(__file__)
        with open(readme_path) as fd:
            ftext = fd.read()
            with patch('sys.stdout', new=io.StringIO()) as stdout:
                try:
                    compas_main.main('--version')
                except SystemExit as ex:
                    pass
            ver_str = stdout.getvalue().strip()
            assert ver_str
            m = re.match('(co2mpas-[^ ]+)', ver_str)
            self.assertIsNotNone(m, 'Version(%s) not found!' % ver_str)
            proj_ver = m.group(1)
            self.assertIn('%s ' % proj_ver, ftext,
                          "Version(%s) not found in README cmd-line version-check!" %
                          ver)

    def test_README_contains_main_help_msg(self):
        help_msg = compas_main.__doc__  # @UndefinedVariable
        mydir = os.path.dirname(__file__)
        with open(readme_path) as fd:
            ftext = fd.read()
            for i, l in enumerate(help_msg.split('\n')):
                l = l.strip()
                self.assertIn(l, ftext,
                              "main's help-msg line[%i] not found in README: %s" %
                              (i, l))
