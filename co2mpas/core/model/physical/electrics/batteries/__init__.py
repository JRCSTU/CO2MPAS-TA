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
    service
"""

import schedula as sh
from .service import dsp as _service

dsp = sh.BlueDispatcher(
    name='Batteries', description='Models the vehicle batteries.'
)

dsp.add_dispatcher(
    dsp_id='service_battery',
    dsp=_service,
    inputs=(
        'alternator_electric_powers', 'cycle_type',
        'dcdc_converter_electric_powers', 'engine_powers_out',
        'initial_service_battery_state_of_charge', 'on_engine',
        'service_battery_capacity', 'service_battery_currents',
        'service_battery_electric_powers', 'service_battery_load',
        'service_battery_loads', 'service_battery_nominal_voltage',
        'service_battery_state_of_charges', 'times'),
    outputs=(
        'initial_service_battery_state_of_charge',
        'service_battery_currents', 'service_battery_electric_powers',
        'service_battery_load', 'service_battery_loads',
        'service_battery_state_of_charges', 'service_battery_capacity',
        'service_battery_delta_state_of_charge'
    ),
    include_defaults=True
)
