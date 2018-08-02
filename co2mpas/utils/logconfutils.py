#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
import sys


def _colorama_init(autoreset=False, convert=None, strip=None, wrap=True):
    "Patch-func for `colorama` to stop wrapping STDOUT and convert ANSI seqs."
    import atexit
    from colorama import initialise

    if not wrap and any([autoreset, convert, strip]):
        raise ValueError('wrap=False conflicts with any other arg=True')

    #global wrapped_stdout, wrapped_stderr
    #global orig_stdout, orig_stderr

    #orig_stdout = sys.stdout
    initialise.orig_stderr = sys.stderr

    ## Fix https://github.com/JRCSTU/co2mpas/issues/475
    #
    #if sys.stdout is None:
    #    wrapped_stdout = None
    #else:
    #    sys.stdout = wrapped_stdout = \
    #        wrap_stream(orig_stdout, convert, strip, autoreset, wrap)
    if sys.stderr is None:
        initialise.wrapped_stderr = None
    else:
        sys.stderr = initialise.wrapped_stderr = \
            initialise.wrap_stream(initialise.orig_stderr,
                                   convert, strip, autoreset, wrap)

    #global atexit_done
    if not initialise.atexit_done:
        atexit.register(initialise.reset_all)
        initialise.atexit_done = True


def patch_colorama_not_to_wrap_stdout():
    """
    Monkey patch `colorama` lib to fix invalid-char crashes when piping STDOUT.


    As explained in 2nd problem of https://github.com/JRCSTU/co2mpas/issues/475
    `colorama` breaks when STDOUT is piped with invalid chars crashing while
    flushing the stream.
    But in reality, co2mpas/dice don't use color in STDOUT, so this fixes
    a common annoyance in command-line.
    """
    import colorama

    colorama.init = _colorama_init
