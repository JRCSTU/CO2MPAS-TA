# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Enable Unicode-trait to pep3101-interpolate `{key}` patterns from "context" dicts.
"""
from typing import Iterable, List, Optional, Text, Union
import sys

from toolz import itertoolz as itz

from .._vendor.traitlets import config as trc
from .._vendor.traitlets import traitlets as trt


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

    raise trt.TraitError("Syntax-error in '%s' slice!" % orig_v)


class Slice(trt.Instance):
    """A trait parsing text like python "slice" expression (ie ``-10::2``)."""
    klass = slice
    _cast_types = str, int

    def validate(self, _obj, value):
        if isinstance(value, slice):
            return value

        try:
            return _parse_slice(value)
        except Exception:
            raise trt.TraitError("Cannot parse '%r' as slice!" % value)


def _slice_text_lines(txt_lines: List[str],
                      slices: Union[slice, List[slice]]) -> List[str]:
    "Extract lines based ob the slices given"
    if isinstance(slices, Iterable):
        line_groups = [txt_lines[sl] for sl in slices]
        txt_lines = sum(itz.interpose(['', '...', ''], line_groups), [])
    else:
        txt_lines = txt_lines[slices]

    return txt_lines


class ShrinkingOutputMixin(trc.Configurable):
    shrink = trt.Bool(
        None,
        allow_none=True,
        help="""
        A 3-state bool, deciding whether to shrink output according to `shrink_slices` param.

        - If none, shrinks if STDOUT is interactive (console).
        - Does not affect results written in `write-fpath` param.
        """
    ).tag(config=True)

    shrink_nlines_threshold = trt.Int(
        128,
        help="The maximum number of lines allowed to print without shrinking."
    ).tag(config=True)

    shrink_slices = trt.Union(
        (Slice(), trt.List(Slice())),
        default_value=[':64', '-32:'],
        help="""
        A slice or a list-of-slices applied when shrinking results printed.

        Examples:
            ':100'
            [':100', '-64:']
        """
    ).tag(config=True)

    def should_shrink_text(self, txt_lines):
        return (len(txt_lines) > self.shrink_nlines_threshold and
                (self.shrink or
                (self.shrink is None and sys.stdout.isatty())))

    def shrink_text(self, txt: Optional[str], shrink_slices=None) -> Optional[str]:
        shrink_slices = shrink_slices or self.shrink_slices
        if not (shrink_slices and txt):
            return txt

        txt_lines = txt.splitlines()

        if self.should_shrink_text(txt_lines):
            shrinked_txt_lines = _slice_text_lines(txt_lines,
                                                   shrink_slices)
            self.log.warning("Shrinked result text-lines from %i --> %i."
                             "\n  ATTENTION: result is not valid for stamping/validation!"
                             "\n  Write it to a file with `--write-fpath`(`-W`).",
                             len(txt_lines), len(shrinked_txt_lines))
            txt = '\n'.join(shrinked_txt_lines)

        return txt


shrink_flags_kwd = {
    'shrink': (
        {'ShrinkingOutputMixin': {'shrink': True}},
        "Omit lines of the report to facilitate console reading."
    ),
    'no-shrink': (
        {'ShrinkingOutputMixin': {'shrink': False}},
        "Print full report - don't omit any lines."
    ),
}
