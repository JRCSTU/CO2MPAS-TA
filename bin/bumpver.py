#!python
#
"""
Bump & commit a new version

USAGE:
    bumpver [-n] [-f] [-c | -t]  [<new-ver>]

OPTIONS:
  -f, --force       Bump (and optionally) commit even if version is the same.
  -n, --dry-run     Do not write files - just pretend.
  -c, --commit      Commit afterwards.
  -t, --tag         TODO: Tag afterwards (commit implied).

Without <new-ver> prints version extracted from current file.

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
    sys.path.append(osp.normpath(osp.join(my_dir, '..')))
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


def do_commit(new_ver, old_ver, dry_run, ver_files):
    import pathlib

    new_ver = strip_ver2_commonprefix(old_ver, new_ver)
    cmt_msg = 'chore(ver): bump %s-->%s' % (old_ver, new_ver)

    ver_files = [pathlib.Path(f).as_posix() for f in ver_files]
    commands = [
        ['git', 'add'] + ver_files,
        ['git', 'commit', '-m', cmt_msg]
    ]

    for cmd in commands:
        cmd_str = format_syscmd(cmd)
        if dry_run:
            yield "DRYRUN: %s" % cmd_str
        else:
            yield "EXEC: %s" % cmd_str
            exec_cmd(cmd)


def bumpver(new_ver, dry_run=False, force=False, tag_after_commit: bool=None):
    """
    :param tag_after_commit:
        if None, neither commit or tag applied.
    """
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
            yield from do_commit(new_ver, old_ver, dry_run, ver_files)

            if tag_after_commit is True:
                yield "TAGGED!"


def main(*args):
    opts = docopt.docopt(__doc__, argv=args)

    new_ver = opts['<new-ver>']
    dry_run = opts['--dry-run']
    force = opts['--force']

    commit = opts['--commit']
    tag = opts['--tag']
    if tag:
        tag_after_commit = True
    elif commit:
        tag_after_commit = False
    else:
        tag_after_commit = None

    try:
        for i in bumpver(new_ver, dry_run, force, tag_after_commit):
            print(i)
    except CmdException as ex:
        sys.exit(str(ex))
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    main(*sys.argv[1:])
