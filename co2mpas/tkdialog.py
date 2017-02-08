#!/usr/bin/env/python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
The base widget for dialogs (i.e. password boxes).

Adapted from http://effbot.org/tkinterbook/tkinter-dialog-windows.htm
"""
import tkinter as tk
import tkinter.ttk as ttk


class TkDialog(tk.Toplevel):
    """
    Override :meth:`body(), :meth:`validate()`, , :meth:`apply()` and
    optionally :meth:`cancel()` & :meth:`buttonbox()`.
    """
    def __init__(self, parent, title, *args, **kwds):
        super().__init__(parent, *args, **kwds)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = ttk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
                                  parent.winfo_rooty() + 50))

        self.initial_focus.focus_set()

        self.wait_window(self)

    #
    # construction hooks

    def body(self, master):
        """
        Override to create the dialog-body.

        :return: the widget that should have initial focus.
        """

    def buttonbox(self):
        """"Add a standard button box; override if you don't want them."""

        box = ttk.Frame(self)

        w = ttk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = ttk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def close(self):
        self.parent.focus_set()  # Put focus back to the parent window.
        self.destroy()

    #
    def ok(self, event=None):
        """Standard button semantics."""

        if not self.validate():
            self.initial_focus.focus_set()  # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.close()

    def cancel(self, event=None):
        self.close()

    #
    # command hooks

    def validate(self):
        return 1

    def apply(self):
        pass
