# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions to read/write inputs/outputs from/on a .dill file.
"""

import logging

log = logging.getLogger(__name__)

import dill

__all__ = ['load_from_dill', 'save_dill']


def load_from_dill(fpath):
    """
    Load inputs from .dill file.

    :param fpath:
        File path.
    :type fpath: str

    :return:
        Input data.
    :rtype: dict
    """
    log.debug('Reading dill-file: %s', fpath)
    with open(fpath, 'rb') as f:
        return dill.load(f)


# noinspection PyUnusedLocal
def save_dill(data, fpath, *args, **kwargs):
    log.debug('Writing dill-file: %s', fpath)
    with open(fpath, 'wb') as f:
        return dill.dump(data, f, recurse=False)
