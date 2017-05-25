#!python
#
"""
Bump & commit a new version

USAGE:
    bumpver [-c | -t] [<new-ver>]

TODO OPTIONS:
  -c    commit afterwards
  -t    tag afterwards (commit implied)

Without any arg, just prints version from file.

"""

import os
import os.path as osp
import sys
import re

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

            with open(fpath, 'wt', encoding='utf-8') as fp:
                fp.write(new_txt)

            yield '%s: %i x (%s --> %s)' % (fpath, nrepl, old, new)


def bumpver(new_ver):
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

        yield from replace_substrings(file_pairs, subst_pairs)


def main(*argv):
    if len(argv) == 1:
        new_ver = None
    elif len(argv) == 2:
        new_ver = argv[1]
    else:
        sys.exit('Specify just <new-ver>, not: %s\n    bumpver <new-ver>' %
                 str(argv))

    try:
        for i in bumpver(new_ver):
            print(i)
    except CmdException as ex:
        sys.exit(str(ex))
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    main(*sys.argv)
