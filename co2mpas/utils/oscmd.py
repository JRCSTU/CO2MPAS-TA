#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""Utility to call OS commands through :func:`subprocess.run()` with logging.

The *polyvers* version-configuration tool is generating tags like::

    proj-foo-v0.1.0

On purpose python code here kept with as few dependencies as possible."""

from typing import Dict
import logging

import subprocess as sbp


#: Monkeypatch :class:`subprocess.CalledProcessError`
#: to always print STDERR on errors.
def err_includes_stderr(self):
    import signal

    tail = ('\n  STDERR: ' + self.stderr) if self.stderr else ''
    tail += ('\n  STDOUT: ' + self.stdout) if self.stdout else ''

    if self.returncode and self.returncode < 0:
        try:
            return "Command '%s' died with %r.%s" % (
                self.cmd, signal.Signals(-self.returncode), tail)
        except ValueError:
            return "Command '%s' died with unknown signal %d.%s" % (
                self.cmd, -self.returncode, tail)
    else:
        return "Command '%s' returned non-zero exit status %d.%s" % (
            self.cmd, self.returncode, tail)


sbp.CalledProcessError.__str__ = err_includes_stderr


def format_syscmd(cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join('"%s"' % s if ' ' in s else s
                       for s in cmd)
    else:
        assert isinstance(cmd, str), cmd

    return cmd


class _CmdName:
    "To `workaround `:classLFileNotFoundError` not explaing it's a command missing."
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return "'%s' (command)" % self.path


def exec_cmd(cmd,
             dry_run=False,
             check_stdout=True,
             check_stderr=True,
             check_returncode=True,
             encoding='utf-8', encoding_errors='surrogateescape',
             **popen_kws):
    """
    :param check_stdout:
        None: Popen(stdout=None), printed
        False: Popen(stdout=sbp.DEVNULL), ignored
        True: Popen(stdout=sbp.PIPE), collected & returned
    """
    log = logging.getLogger(__name__)
    call_types = {
        None: {'label': 'EXEC', 'stream': None},
        False: {'label': 'EXEC(no-stdout)', 'stream': sbp.DEVNULL},
        True: {'label': 'CALL', 'stream': sbp.PIPE},
    }
    stdout_ctype = call_types[check_stdout]
    cmd_label = stdout_ctype['label']
    cmd_str = format_syscmd(cmd)

    log.debug('%s%s %r', 'DRY_' if dry_run else '', cmd_label, cmd_str)

    if dry_run:
        return

    try:
        ##WARN: python 3.6 `encoding` & `errors` kwds in `Popen`.
        res: sbp.CompletedProcess = sbp.run(
            cmd,
            stdout=stdout_ctype['stream'],
            stderr=call_types[check_stderr]['stream'],
            encoding=encoding,
            errors=encoding_errors,
            **popen_kws
        )
    except FileNotFoundError as ex:
        ## On windows, no path provided!
        #
        if not ex.filename:
            ex.filename = _CmdName(cmd[0])

        raise

    if res.returncode:
        log.log(
            logging.DEBUG if check_returncode else logging.WARNING,
            '%s %r failed with %s!\n  stdout: %s\n  stderr: %s',
            cmd_label, cmd_str, res.returncode, res.stdout, res.stderr)
    elif check_stdout or check_stderr:
        log.debug('%s %r ok: \n  stdout: %s\n  stderr: %s',
                  cmd_label, cmd_str, res.stdout, res.stderr)

    if check_returncode:
        try:
            res.check_returncode()
        except sbp.CalledProcessError as ex:
            #
            #  A hackish wedge to report to the user
            ## the CWD when not in Git-repo.

            if "ot a git repository" in (ex.stderr.decode()
                                         if isinstance(ex.stderr, bytes) else
                                         ex.stderr):
                import os
                from .. import pvtags

                raise pvtags.NoGitRepoError(
                    "Current-dir '%s' is not within a git repository!" %
                    os.curdir) from ex
            raise

    return res


def _as_flag(k):
    return k.replace('_', '-')


class _Cli:
    def __init__(self, popen_kw: Dict, cmd: str):
        self._popen_kw = popen_kw
        self._cmdlist = [cmd]

    def _extend_cmdlist(self, args, kw):
        def kv2args(k, v):
            nk = len(k)

            if nk > 1:
                k = _as_flag(k)

            if v is None:
                return ()
            if isinstance(v, bool):
                if v:
                    flag = '-' + k if len(k) == 1 else '--' + k
                else:
                    if nk == 1:
                        raise ValueError(
                            "Cannot negate single-letter flag '-%s'!"
                            "\n  cmd: %s!" % (k, ' '.join(self._cmdlist)))
                    flag = '--no-' + k

                return (flag, )

            return ('-%s' % k, str(v)) if nk == 1 else ('--%s=%s' % (k, v), )

        arglist = self._cmdlist
        arglist.extend(str(a) for a in args)
        arglist.extend(arg
                       for kv in kw.items()
                       for arg in kv2args(*kv))

    def __getattr__(self, attr):
        if attr == '__wrapped__':  # PYTEST MAGIC!
            return None
        if attr:
            attr = _as_flag(attr)
        self._cmdlist.append(attr)
        return self

    def __call__(self, *args, **kw) -> str:
        self._extend_cmdlist(args, kw)
        res = exec_cmd(self._cmdlist, **self._popen_kw)
        self.rc = res.returncode
        self.stderr = res.stderr
        self.stdout = res.stdout  # keep unstripped stdout

        if self._popen_kw['check_stdout']:
            return res.stdout and res.stdout.strip()

    def _(self, *args, **kw):
        "Avoid immediate execution and continue building cmd-line."
        self._extend_cmdlist(args, kw)
        return self

    def __str__(self):
        return 'Cli(%s)' % ' '.join("''" if a == '' else a
                                    for a in self._cmdlist)


class PopenCmd:
    """
    A function --> cmd-line builder for executing (mostly) git commands.

    To run ``git log -n1``::

        out = cmd.git.log(n=1)

    To launch a short python program with ``python -c "print('a')"``::

        out = cmd.python._(c=True)('print(\'a\')')

    :raise sbp.CalledProcessError:
        if check_returncode=true and exit code is non-zero.

    if self.returncode:    .. Note::
       It's mostly for Git bc flags are produced like that:
           -f <value> --flag=<value>
    """
    def __init__(self,
                 dry_run=False,
                 check_stdout=True,
                 check_stderr=True,
                 check_returncode=True,
                 **popen_kw):
        """
        Set the Popen kw-args to use when the cmd will be executed.

        :param dry_run:
            log but don't actually exec cmd.
        :param check_stdout:
            None: Popen(stdout=None), printed
            False: Popen(stdout=sbp.DEVNULL), ignored
            True: Popen(stdout=sbp.PIPE), collected & returned
        :param check_stderr:
            same as `check_stdout`
        :param check_returncode:
            if true, raise `sbp.CalledProcessError` if return-code not 0.
        """
        popen_kw.update(locals())
        del popen_kw['self'], popen_kw['popen_kw']
        self._popen_kw = popen_kw

    def __getattr__(self, attr):
        return _Cli(self._popen_kw, attr)


cmd = PopenCmd()
