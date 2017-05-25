#!python
#
"""
Bump & commit a new version

USAGE:
    bumpver [-n] [-c | -t] [<new-ver>]

OPTIONS:
  -n, --dry-run     do not write files - just pretend
  -c                TODO: commit afterwards
  -t                TODO: tag afterwards (commit implied)

Without <new-ver> prints version extracted from current file.

"""

import os.path as osp
import sys
import re

import docopt


my_dir = osp.dirname(__file__)

VFILE = osp.join(my_dir, '..', 'co2mpas', '_version.py')
VFILE_regex_v = re.compile(r'__version__ = version = "([^"]+)"')
VFILE_regex_d = re.compile(r'__updated__ = "([^"]+)"')

RFILE = osp.join(my_dir, '..', 'README.rst')


class CmdException(Exception):
    pass


def read_txtfile(fpath):
    with open(fpath, 'rt', encoding='utf-8') as fp:
        return fp.read()


def replace_substrings(file_pairs, subst_pairs):
    for fpath, txt in file_pairs:
        if not txt:
            txt = read_txtfile(fpath)

        for old, new in subst_pairs:
            nrepl = txt.count(old)
            new_txt = txt.replace(old, new)

            yield (new_txt, fpath, nrepl, old, new)


def bumpver(new_ver, dry_run=False):
    vfile_txt = read_txtfile(VFILE)
    matches = [regex.search(vfile_txt)
               for regex
               in [VFILE_regex_v, VFILE_regex_d]]

    if not all(matches):
        raise CmdException("Failed extracting current version: "
                           "\n  ver: %s\n  date: %s" % matches)
    oldv, oldd = (m.group(1) for m in matches)

    if not new_ver:
        yield oldv
        yield oldd
    else:
        from datetime import datetime

        new_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')

        file_pairs = [(VFILE, vfile_txt), (RFILE, None)]
        subst_pairs = [(oldv, new_ver), (oldd, new_date)]

        for repl in replace_substrings(file_pairs, subst_pairs):
            new_txt, fpath, nrepl, old, new = repl

            if not dry_run:
                with open(fpath, 'wt', encoding='utf-8') as fp:
                    fp.write(new_txt)

            fpath = osp.normpath(fpath)
            yield '%s: %i x (%s --> %s)' % (fpath, nrepl, old, new)


def main(*args):
    opts = docopt.docopt(__doc__, argv=args)

    new_ver = opts['<new-ver>']
    dry_run = opts['--dry-run']

    try:
        for i in bumpver(new_ver, dry_run):
            print(i)
    except CmdException as ex:
        sys.exit(str(ex))
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    main(*sys.argv[1:])
