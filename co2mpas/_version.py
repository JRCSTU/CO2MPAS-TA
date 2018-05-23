# -*- coding: utf-8 -*-
#
# !/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl


#: Authoritative project's PEP 440 version.
from polyversion import polyversion

__version__ = version = "1.8.0a0.dev0"  # Also update README.rst, CHANGES.rst,

#: Input/Output file's version.
__file_version__ = "2.2.7"

#: Compatible Input file's version.
__input_file_version__ = "2"

__dice_report_version__ = '1.0.2'

# Please UPDATE TIMESTAMP WHEN BUMPING VERSIONS AND BEFORE RELEASE.
#: Release date.
__updated__ = "2018-05-23 05:49:21"


if __name__ == '__main__':
    import sys
    out = ';'.join(
        eval('__%s__' % a[2:].replace('-', '_'))
        for a in sys.argv[1:] if a[:2] == '--'
    )
    sys.stdout.write(out)