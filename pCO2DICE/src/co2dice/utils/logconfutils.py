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
import io
import os
import os.path as osp


def _set_numpy_logging():
    rlog = logging.getLogger()
    if not rlog.isEnabledFor(logging.DEBUG):
        import numpy as np
        np.seterr(divide='ignore', invalid='ignore')


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


def init_logging(level=None, frmt=None, logconf_file=None,
                 color=False, default_logconf_file=None,
                 not_using_numpy=False, **kwds):
    """
    :param level:
        tip: use :func:`is_any_log_option()` to decide if should be None
        (only if None default HOME ``logconf.yaml`` file is NOT read).
    :param default_logconf_file:
        Read from HOME only if ``(level, frmt, logconf_file)`` are none.
    :param kwds:
        Passed directly to :func:`logging.basicConfig()` (e.g. `filename`);
        used only id default HOME ``logconf.yaml`` file is NOT read.
    """
    ## Only read default logconf file in HOME
    #  if no explicit arguments given.
    #
    no_args = all(i is None for i in [level, frmt, logconf_file])
    if no_args and default_logconf_file and osp.exists(default_logconf_file):
        logconf_file = default_logconf_file

    ## Monkeypatch `colorama` to fix invalid chars when STDOUT piped,
    #  see  https://github.com/JRCSTU/co2mpas/issues/475.
    patch_colorama_not_to_wrap_stdout()

    if logconf_file:
        from logging import config as lcfg

        logconf_file = osp.expandvars(osp.expanduser(logconf_file))
        if osp.splitext(logconf_file)[1] in '.yaml' or '.yml':
            import yaml

            with io.open(logconf_file) as fd:
                log_dict = yaml.safe_load(fd)
                lcfg.dictConfig(log_dict)
        else:
            lcfg.fileConfig(logconf_file)

        logconf_src = logconf_file
    else:
        if level is None:
            level = logging.INFO
        if not frmt:
            frmt = "%(asctime)-15s:%(levelname)5.5s:%(name)s:%(message)s"
        logging.basicConfig(level=level, format=frmt, **kwds)
        rlog = logging.getLogger()
        rlog.level = level  # because `basicConfig()` does not reconfig root-logger when re-invoked.

        logging.getLogger('pandalone.xleash.io').setLevel(logging.WARNING)

        if color and sys.stderr.isatty():
            from .._vendor.rainbow_logging_handler import RainbowLoggingHandler

            color_handler = RainbowLoggingHandler(
                sys.stderr,
                color_message_debug=('grey', None, False),
                color_message_info=('blue', None, False),
                color_message_warning=('yellow', None, True),
                color_message_error=('red', None, True),
                color_message_critical=('white', 'red', True),
            )
            formatter = logging.Formatter(frmt)
            color_handler.setFormatter(formatter)

            ## Be conservative and apply color only when
            #  log-config looks like the "basic".
            #
            if rlog.handlers and isinstance(rlog.handlers[0], logging.StreamHandler):
                rlog.removeHandler(rlog.handlers[0])
                rlog.addHandler(color_handler)
        logconf_src = 'explicit(level=%s)' % level

    if not not_using_numpy:
        _set_numpy_logging()

    logging.captureWarnings(True)

    ## Disable warnings on AIO but not when developing.
    #
    if os.environ.get('AIODIR'):
        import warnings

        warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        warnings.filterwarnings(action="ignore", module="importlib",
                                message="^can't resolve")

    logging.getLogger(__name__).debug('Logging-configurations source: %s',
                                      logconf_src)


def is_any_log_option(argv):
    """
    Return true if any -v/--verbose/--debug etc options are in `argv`

    :param argv:
        If `None`, use :data:`sys.argv`; use ``[]`` to explicitly use no-args.
    """
    log_opts = '-v --verbose -d --debug --vlevel'.split()
    if argv is None:
        argv = sys.argv
    return argv and set(log_opts) & set(argv)
