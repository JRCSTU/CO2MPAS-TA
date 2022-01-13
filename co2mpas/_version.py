# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Defines package metadata.
"""

__all__ = ['__version__', '__updated__', '__title__', '__author__',
           '__license__', '__copyright__', '__file_version__', '__uri__']

#: Authoritative project's PEP 440 version.
__version__ = version = "4.3.0"  # Also update README.rst

#: The :term:`Semantic Version` for Input/Output files.
__file_version__ = "4.4.1"

# Please UPDATE TIMESTAMP WHEN BUMPING VERSIONS AND BEFORE RELEASE.
#: Release date.
__updated__ = "2022-01-13 02:45:00"

__title__ = 'co2mpas'

__author__ = 'Vincenzo Arcidiacono <vinci1it2000@gmail.com>'

__license__ = 'EUPL, see LICENSE.txt'

__uri__ = "https://co2mpas.io"

__copyright__ = "Copyright (C) 2015-2022 European Commission (JRC)"

if __name__ == '__main__':
    import sys

    out = ';'.join(
        eval(a[2:].replace('-', '_')) for a in sys.argv[1:] if a[:2] == '--'
    )
    sys.stdout.write(out)
