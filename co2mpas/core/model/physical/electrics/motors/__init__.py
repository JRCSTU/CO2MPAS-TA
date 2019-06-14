# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the vehicle electric motors.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics.motors

.. autosummary::
    :nosignatures:
    :toctree: motors/

    p0
    p1
    p2
    p3
    p4
    starter
"""

import schedula as sh
from .p4 import dsp as _p4

dsp = sh.BlueDispatcher(name='Motors', description='Models the vehicle motors.')

dsp.add_dispatcher(
    dsp_id='motor_p4',
    dsp=_p4,
    inputs=(
        'wheel_speeds', 'motor_p4_speed_ratio', 'motor_p4_speeds',
        'motor_p4_powers', 'motor_p4_torques', 'motor_p4_efficiency',
        'motor_p4_electric_power_loss_function', 'motor_p4_loss_param_a',
        'motor_p4_loss_param_b', 'motor_p4_electric_powers',
    ),
    outputs=(
        'motor_p4_speed_ratio', 'motor_p4_speeds', 'motor_p4_powers',
        'motor_p4_torques', 'motor_p4_efficiency', 'motor_p4_electric_powers',
        'motor_p4_efficiency_ratios'
    ),
    include_defaults=True
)
