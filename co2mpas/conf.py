#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides constants for the CO2MPAS.
"""

import co2mpas.utils as co2_utl
from .core.model.physical.defaults import dfl as _physical
from .core.load.validate.eng_mode import dfl as _eng_mode


# noinspection PyMissingOrEmptyDocstring
class Defaults(co2_utl.Constants):
    physical = _physical
    eng_mode = _eng_mode


dfl = Defaults()
