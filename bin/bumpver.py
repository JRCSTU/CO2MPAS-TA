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

        for old, new in subst_pairs:
            nrepl = txt.count(old)
            new_txt = txt.replace(old, new)

            yield (new_txt, fpath, nrepl, old, new)


def bumpver(new_ver, dry_run=False):
    regexes = [VFILE_regex_v, VFILE_regex_d]
    old_ver, old_date = extract_file_regexes(VFILE, regexes)

    if not new_ver:
        yield old_ver
        yield old_date
    else:
        from datetime import datetime

        new_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')

        ver_files = [osp.normpath(f) for f in [VFILE, RFILE]]
        subst_pairs = [(old_ver, new_ver), (old_date, new_date)]

        for repl in replace_substrings(ver_files, subst_pairs):
            new_txt, fpath, nrepl, old, new = repl

            if not dry_run:
                with open(fpath, 'wt', encoding='utf-8') as fp:
                    fp.write(new_txt)

            yield '%20s: %i x (%s --> %s)' % (fpath, nrepl, old, new)


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
