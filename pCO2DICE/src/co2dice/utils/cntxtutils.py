#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""Utilities with contextlib decorators etc"""
import contextlib


@contextlib.contextmanager
def chdir(path):
    import os

    opath = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(opath)
