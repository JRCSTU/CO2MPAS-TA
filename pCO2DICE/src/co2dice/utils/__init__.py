#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Utilities depending on STANDARD python libs only - non imports.

to avoid import-spamming when going for other package-submodules.
"""


def first_line(doc):
    for l in doc.split('\n', maxsplit=3):
        if l.strip():
            return l.strip()


def joinstuff(items, delimeter=', ', frmt='%s'):
    """
    Prefixes ALL items (not in-metween only) and then join them.

    :param frmt:
        must have one and only ``%s``

    For example, to create separate lines::

        joinstuff(items, '', '\n  %s')

    """
    return delimeter.join(frmt % (i, ) for i in items)
