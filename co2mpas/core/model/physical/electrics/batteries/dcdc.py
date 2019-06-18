# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the DC/DC Converter.
"""

import numpy as np
import schedula as sh
from ...defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='DC/DC Converter',
    description='Models the DC/DC Converter.'
)


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True,
    outputs=['dcdc_converter_electric_powers_demand']
)
def calculate_dcdc_converter_electric_powers_demand(
        dcdc_converter_electric_powers, dcdc_converter_efficiency=.95):
    """
    Calculate DC/DC converter electric power demand [kW].

    :param dcdc_converter_electric_powers:
        DC/DC converter electric power [kW].
    :type dcdc_converter_electric_powers: numpy.array

    :param dcdc_converter_efficiency:
        DC/DC converter efficiency [-].
    :type dcdc_converter_efficiency: float

    :return:
        DC/DC converter electric power demand [kW].
    :rtype: numpy.array
    """
    return dcdc_converter_electric_powers / dcdc_converter_efficiency
