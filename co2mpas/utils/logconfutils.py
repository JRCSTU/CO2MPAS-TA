#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
import logging
import sys


def set_numpy_errors_enabled(enabled):
    import numpy as np

    action = 'warn' if enabled else 'ignore'
    np.seterr(divide=action, invalid=action)


def set_warnings_enabled(enabled):
    import warnings

    logging.captureWarnings(True)

    if enabled:
        action = 'default'
    else:
        action = 'ignore'

    warnings.filterwarnings(action=action, category=DeprecationWarning)
    warnings.filterwarnings(action=action, module="scipy",
                            message="^internal gelsd")
    warnings.filterwarnings(action=action, module="dill",
                            message="^unclosed file")
    warnings.filterwarnings(action=action, module="importlib",
                            message="^can't resolve")


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


def _count_multiflag_in_argv(args, short, long, eliminate=False):
    """
    Match flags in `argvs` list, in short/long form, and optionally remove them.

    :param eliminate:
        If true, returned flags will have those matching, removed.
    :return:
        the 2-tuple (num-of-matches, new-args) where `new-args` possibly
        have flags missing.
    """
    import re

    long = '--%s' % long
    nmatches = 0
    new_args = []
    for flag in args:
        if flag == long:
            nmatches += 1
            if eliminate:
                continue

        elif re.match('^-[a-z]+', flag, re.I):
            nmatches += flag.count(short)
            if eliminate:
                flag = flag.replace(short, '')
                if flag == '-':
                    continue

        new_args.append(flag)

    return nmatches, new_args


def log_level_from_argv(args,
                        start_level: int,
                        eliminate_verbose=False,
                        eliminate_quiet=False,
                        verbosity_step=10):
    """
    :param start_level_index:
        some existing level
    :return:
        a 2-tuple (level, new_args), where `new_args` is
        the updated list of args
    """
    if not isinstance(start_level, int):
        raise ValueError(
            "Expecting an *integer* for logging level, got '%s'!" % start_level)
    if not args:
        return start_level, args

    levels = list(sorted(logging._levelToName))

    nverbose, new_args = _count_multiflag_in_argv(args, 'v', 'verbose',
                                                  eliminate_verbose)
    nquiet, new_args = _count_multiflag_in_argv(new_args, 'q', 'quiet',
                                                eliminate_quiet)

    level = start_level + verbosity_step * (nquiet - nverbose)
    level = max(0, min(levels[-1], level))

    return level, new_args
