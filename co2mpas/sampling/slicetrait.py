# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Enable Unicode-trait to pep3101-interpolate `{key}` patterns from "context" dicts.
"""

from typing import Text, Union

from .._vendor.traitlets.traitlets import TraitError, Instance


def _parse_slice(v: Union[Text, int]):
    """
    Parses text like python "slice" expression (ie ``-10::2``).

    :param v:
        the slice expression or a lone integer
    :return:
        - None if input is None/empty
        - a ``slice()`` instance (even if input a lone numbrt)
    :raise ValueError:
        input non-empty but invalid syntax
    """
    if isinstance(v, int):
        return slice(v, v + 1)

    orig_v = v
    v = v and v.strip()
    if not v:
        return

    try:
        if ':' not in v:
            ## A lone number given.
            v = int(v)
            return slice(v, v + 1)

        ## noqa: E501 From: https://stackoverflow.com/questions/680826/python-create-slice-object-from-string#comment3188450_681949
        return slice(*map(lambda x: int(x.strip()) if x.strip() else None,
                          v.split(':')))
    except Exception:
        pass

    raise TraitError("Syntax-error in '%s' slice!" % orig_v)


class Slice(Instance):
    """A trait parsing text like python "slice" expression (ie ``-10::2``)."""
    klass = slice
    _cast_types = str, int

    def validate(self, _obj, value):
        if isinstance(value, slice):
            return value

        try:
            return _parse_slice(value)
        except Exception:
            raise TraitError("Cannot parse '%r' as slice!" % value)
