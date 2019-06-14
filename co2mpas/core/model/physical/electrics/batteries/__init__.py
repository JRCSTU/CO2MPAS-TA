# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the vehicle batteries.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics.batteries

.. autosummary::
    :nosignatures:
    :toctree: batteries/

    drive
    general
"""

import numpy as np
import schedula as sh
from .service import dsp as _general

dsp = sh.BlueDispatcher(
    name='Batteries', description='Models the vehicle batteries.'
)


@sh.add_function(dsp, outputs=['delta_state_of_charge'])
def calculate_delta_state_of_charge(lv_state_of_charges):
    """
    Calculates the overall delta state of charge of the battery [%].

    :param lv_state_of_charges:
        State of charge of the low voltage battery [%].

        .. note::

            `lv_state_of_charges` = 99 is equivalent to 99%.
    :type lv_state_of_charges: numpy.array

    :return:
        Overall delta state of charge of the low voltage battery [%].
    :rtype: float
    """
    return lv_state_of_charges[-1] - lv_state_of_charges[0]


dsp.add_dispatcher(
    dsp_id='general_porpoise_battery',
    dsp=_general,
    inputs=(
        'cycle_type', 'general_battery_currents', 'general_battery_capacity',
        'times', 'alternator_status_model',
    ),
    outputs=(
        'general_battery_state_of_charges',
        'general_battery_state_of_charge_balance',
        'general_battery_state_of_charge_balance_window'
    ),
    include_defaults=True
)
