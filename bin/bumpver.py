#!/usr/bin/env python
#
"""
Script to bump, commit and tag new versions.

USAGE:
  bumpver [-n] [-f] [-c] [-a] [-t <message>]  [<new-ver>]

OPTIONS:
  -a, --amend       Amend current commit for setting the "chore(ver): ..." msg.
  -f, --force       Bump (and optionally) commit/tag if version exists/is same.
  -n, --dry-run     Do not write files - just pretend.
  -c, --commit      Commit afterwardswith a commit-message describing version bump.
  -t, --tag=<msg>   Adds a signed tag with the given message (commit implied).

Without <new-ver> prints version extracted from current file.

EXAMPLE:

  bumpver -t 'Mostly model changes' 1.6.2b0
"""

import os.path as osp
import sys
import re
import functools as fnt

import docopt


my_dir = osp.dirname(__file__)

VFILE = osp.join(my_dir, '..', 'co2mpas', '_version.py')
VFILE_regex_v = re.compile(r'__version__ = version = "([^"]+)"')
VFILE_regex_d = re.compile(r'__updated__ = "([^"]+)"')

RFILE = osp.join(my_dir, '..', 'README.rst')


class CmdException(Exception):
    pass


@fnt.lru_cache()
def read_txtfile(fpath):
    with open(fpath, 'rt', encoding='utf-8') as fp:
        return fp.read()


def extract_file_regexes(fpath, regexes):
    """
    :param regexes:
        A sequence of regexes to "search", having a single capturing-group.
    :return:
        One groups per regex, or raise if any regex did not match.
    """
    txt = read_txtfile(fpath)
    matches = [regex.search(txt) for regex in regexes]

    if not all(matches):
        raise CmdException("Failed extracting current version: "
                           "\n  ver: %s\n  date: %s" % matches)

    return [m.group(1) for m in matches]


def replace_substrings(files, subst_pairs):
    for fpath in files:
        txt = read_txtfile(fpath)

        replacements = []
        for old, new in subst_pairs:
            replacements.append((old, new, txt.count(old)))
            txt = txt.replace(old, new)

        yield (txt, fpath, replacements)


def format_syscmd(cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join('"%s"' % s if ' ' in s else s
                       for s in cmd)
    else:
        assert isinstance(cmd, str), cmd

    return cmd


def strip_ver2_commonprefix(ver1, ver2):
    cprefix = osp.commonprefix([ver1, ver2])
    if cprefix:
        striplen = cprefix.rfind('.')
        if striplen > 0:
            striplen += 1
        else:
            striplen = len(cprefix)
        ver2 = ver2[striplen:]

    return ver2


def run_testcases():
    ## From loading doc-TCs.
    sys.path.insert(0, osp.abspath(osp.normpath(osp.join(my_dir, '..'))))

    import unittest
    import tests.test_docs

    suite = unittest.TestLoader().loadTestsFromModule(tests.test_docs)
    res = unittest.TextTestRunner(failfast=True).run(suite)

    if not res.wasSuccessful():
        raise CmdException("Doc TCs failed, probably version-bumping has failed!")


def exec_cmd(cmd):
    import subprocess as sbp

    err = sbp.call(cmd, stderr=sbp.STDOUT)
    if err:
        raise CmdException("Failed(%i) on: %s" % (err, format_syscmd(cmd)))


def do_commit(new_ver, old_ver, dry_run, amend, ver_files):
    import pathlib

    new_ver = strip_ver2_commonprefix(old_ver, new_ver)
    cmt_msg = 'chore(ver): bump %s-->%s' % (old_ver, new_ver)

    ver_files = [pathlib.Path(f).as_posix() for f in ver_files]
    git_add = ['git', 'add'] + ver_files
    git_cmt = ['git', 'commit', '-m', cmt_msg]
    if amend:
        git_cmt.append('--amend')
    commands = [git_add, git_cmt]

    for cmd in commands:
        cmd_str = format_syscmd(cmd)
        if dry_run:
            yield "DRYRUN: %s" % cmd_str
        else:
            yield "EXEC: %s" % cmd_str
            exec_cmd(cmd)


def do_tag(tag, tag_msg, dry_run, force):
    cmd = ['git', 'tag', tag, '-s', '-m', tag_msg]
    if force:
        cmd.append('--force')
    cmd_str = format_syscmd(cmd)
    if dry_run:
        yield "DRYRUN: %s" % cmd_str
    else:
        yield "EXEC: %s" % cmd_str
        exec_cmd(cmd)


def bumpver(new_ver, dry_run=False, force=False, amend=False,
            tag_after_commit=None):
    """
    :param tag_after_commit:
        if true, do `git commit`, if string, also `git tag` with that as msg.
    """
    if amend:
        ## Restore previous version before extracting it.
        exec_cmd('git checkout HEAD~  -- README.rst co2mpas/_version.py'.split())

    regexes = [VFILE_regex_v, VFILE_regex_d]
    old_ver, old_date = extract_file_regexes(VFILE, regexes)

    if not new_ver:
        yield old_ver
        yield old_date
    else:
        if new_ver == old_ver:
            msg = "Version '%s'already bumped"
            if force:
                msg += ", but --force  effected."
                yield msg % new_ver
            else:
                msg += "!\n Use of --force recommended."
                raise CmdException(msg % new_ver)

        from datetime import datetime

        new_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')

        ver_files = [osp.normpath(f) for f in [VFILE, RFILE]]
        subst_pairs = [(old_ver, new_ver), (old_date, new_date)]

        for repl in replace_substrings(ver_files, subst_pairs):
            new_txt, fpath, replacements = repl

            if not dry_run:
                with open(fpath, 'wt', encoding='utf-8') as fp:
                    fp.write(new_txt)

            yield '%s: ' % fpath
            for old, new, nrepl in replacements:
                yield '  %i x (%24s --> %s)' % (nrepl, old, new)

        yield "...now launching DocTCs..."
        run_testcases()

        if tag_after_commit is not None:
            yield from do_commit(new_ver, old_ver, dry_run, amend, ver_files)

            if isinstance(tag_after_commit, str):
                tag = 'v%s' % new_ver
                yield from do_tag(tag, tag_after_commit, dry_run, force)


def main(*args):
    opts = docopt.docopt(__doc__, argv=args)

    commit = opts['--commit']
    tag = opts['--tag']
    if tag:
        tag_after_commit = tag
    elif commit:
        tag_after_commit = True
    else:
        tag_after_commit = None

    try:
        for i in bumpver(opts['<new-ver>'],
                         opts['--dry-run'],
                         opts['--force'],
                         opts['--amend'],
                         tag_after_commit):
            print(i)
    except CmdException as ex:
        sys.exit(str(ex))
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    main(*sys.argv[1:])
