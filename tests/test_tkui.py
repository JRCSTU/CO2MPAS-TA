#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import logging
import unittest

import ddt

from co2mpas import co2gui, __main__ as cmain
import os.path as osp
import tkinter as tk


cmain.init_logging(level=logging.WARNING)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)


@ddt.ddt
class Test(unittest.TestCase):

    @ddt.data(
        ("", ()),
        ("""Some clean text.""", ()),
        ("""A [broken] (link) cause of spaces.""", ()),
        ("""A [img:broken] (image) cause of spaces.""", ()),

        ("""[  img:  test  ](  s p a c e s  ).""",
         (
             ('img:', 'test', 's p a c e s'),
         )
         ),

        ("""[ some &^& sdragbge ](  a &(& DDJ&*( bad url ).""",
         (
             (None, 'some &^& sdragbge', 'a &(& DDJ&*( bad url'),
         )
         ),

        ("""[Einstein](https://en.wikipedia.org/wiki/Einstein) on the [img:beach](images/keratokampos.png).""",
         (
             (None, 'Einstein', 'https://en.wikipedia.org/wiki/Einstein'),
             ('img:', 'beach', 'images/keratokampos.png'),
         )
         ),

        ("""Hi [wdg:foo](bar) there [img:abc](def) and [mplah](blah).""",
         (
             ('wdg:', 'foo', 'bar'),
             ('img:', 'abc', 'def'),
             (None, 'mplah', 'blah'),
         )
         ),

        ("""] (Hi [ wdg: foo] there [img: abc ] and [ mplah ].""",
         (
             ('wdg:', 'foo', None),
             ('img:', 'abc', None),
             (None, 'mplah', None),
         )
         ),

        ("[foo] (bar)", ((None, 'foo', None), )),
        ("[wdg:foo] (bar)", (('wdg:', 'foo', None), )),
        ("[foo](bar\ntender)", ((None, 'foo', None), )),
        ("[img:foo](bar\ntender)", (('img:', 'foo', None), )),

)
    def test_makdown_parsing_regex(self, case):
        txt, exp_groups_seq = case
        exp_nmatches = len(exp_groups_seq)
        regex = co2gui._img_in_txt_regex

        nmatches = 0
        for nmatches, (m, exp_groups) in enumerate(zip(regex.finditer(txt), exp_groups_seq), 1):
            self.assertEqual(m.groups(), exp_groups)

        self.assertEqual(nmatches, exp_nmatches)

    @ddt.data(
        'asd [ foo f',
        'asd ( foo f',
        'asd ( foo )f',
        'asd  foo )f',
        'asd  foo ]f',
        ' [bar)sf',
        ' (bar)sf',
        ' [img:bar)sf',
        ' [wdg:bar)sf',

        'NL? [wdg:foo \n bar] 1',

        '\\[wdg:escaped] 1',
        '\\[wdg:escaped](url) 2',
    )
    def test_makdown_parsing_regex_bad(self, txt):
        regex = co2gui._img_in_txt_regex

        self.assertIsNone(regex.search(txt))

#    def test_smoketest(self):
#        root = tk.Tk()
#        try:
#            app = co2gui.TkUI(root)
#            root.after_idle(app._do_about)
#            root.after(3000, root.quit)
#            app.mainloop()
#        finally:
#            try:
#                root.destroy()
#            except tk.TclError:
#                pass

    def test_about(self):
        root = tk.Tk()
        try:
            co2gui.show_about(root, verbose=True)
            root.after(700, root.quit)
            root.mainloop()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass
