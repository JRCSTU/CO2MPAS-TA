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
from .drive import dsp as _drive

dsp = sh.BlueDispatcher(
    name='Batteries', description='Models the vehicle batteries.'
)

dsp.add_dispatcher(
    dsp_id='service_battery',
    dsp=_service,
    inputs=(
        'alternator_electric_powers', 'cycle_type', 'on_engine', 'times',
        'dcdc_converter_electric_powers', 'engine_powers_out',
        'initial_service_battery_state_of_charge', 'service_battery_currents',
        'service_battery_capacity', 'service_battery_state_of_charges',
        'service_battery_electric_powers', 'service_battery_load',
        'service_battery_loads', 'service_battery_nominal_voltage',
    ),
    outputs=(
        'initial_service_battery_state_of_charge', 'service_battery_currents',
        'service_battery_electric_powers', 'service_battery_load',
        'service_battery_loads', 'service_battery_state_of_charges',
        'service_battery_capacity', 'service_battery_delta_state_of_charge'
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='drive_battery',
    dsp=_drive,
    inputs=(
        'drive_battery_electric_powers', 'drive_battery_voltages',
        'drive_battery_currents', 'drive_battery_capacity', 'times',
        'drive_battery_state_of_charges', 'drive_battery_loads',
        'motors_electric_powers', 'dcdc_converter_electric_powers_demand',
        'initial_drive_battery_state_of_charge', 'drive_battery_ocv',
        'electrical_hybridization_degree', 'dcdc_converter_electric_powers',
        'dcdc_converter_efficiency', 'drive_battery_load', 'drive_battery_r0',
        'drive_battery_n_parallel_cells', 'drive_battery_n_series_cells'
    ),
    outputs=(
        'maximum_driver_battery_electric_power', 'drive_battery_voltages',
        'drive_battery_currents', 'drive_battery_loads', 'drive_battery_load',
        'drive_battery_n_series_cells', 'drive_battery_ocv', 'drive_battery_r0',
        'initial_drive_battery_state_of_charge', 'drive_battery_capacity',
        'drive_battery_delta_state_of_charge', 'drive_battery_n_parallel_cells',
        'drive_battery_electric_powers', 'drive_battery_state_of_charges'
    ),
    include_defaults=True
)
