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

    dcdc
    drive
    service
"""

import schedula as sh
from .service import dsp as _service
from .drive import dsp as _drive
from .dcdc import dsp as _dcdc

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
        'drive_battery_electric_powers', 'drive_battery_voltages', 'times',
        'drive_battery_currents', 'drive_battery_capacity', 'drive_battery_r0',
        'drive_battery_state_of_charges', 'drive_battery_loads',
        'dcdc_converter_electric_powers_demand', 'motor_p0_electric_powers',
        'motor_p1_electric_powers', 'motor_p2_electric_powers',
        'motor_p3_electric_powers', 'motor_p4_electric_powers',
        'initial_drive_battery_state_of_charge', 'drive_battery_ocv',
        'electrical_hybridization_degree', 'drive_battery_load',
        'drive_battery_n_parallel_cells', 'drive_battery_n_series_cells'
    ),
    outputs=(
        'maximum_drive_battery_electric_power', 'drive_battery_voltages',
        'drive_battery_currents', 'drive_battery_loads', 'drive_battery_load',
        'drive_battery_n_series_cells', 'drive_battery_ocv', 'drive_battery_r0',
        'initial_drive_battery_state_of_charge', 'drive_battery_capacity',
        'drive_battery_delta_state_of_charge', 'drive_battery_n_parallel_cells',
        'drive_battery_electric_powers', 'drive_battery_state_of_charges'
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='dcdc_converter',
    dsp=_dcdc,
    inputs=('dcdc_converter_efficiency', 'dcdc_converter_electric_powers'),
    outputs=('dcdc_converter_electric_powers_demand',),
    include_defaults=True
)
