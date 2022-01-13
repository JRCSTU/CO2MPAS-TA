# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the electric motors of the vehicle.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics.motors

.. autosummary::
    :nosignatures:
    :toctree: motors/

    alternator
    p0
    p1
    p2
    p3
    p4
    planet
    starter
"""

import schedula as sh
from .alternator import dsp as _alternator
from .p0 import dsp as _p0
from .p1 import dsp as _p1
from .p2 import dsp as _p2
from .p3 import dsp as _p3
from .p4 import dsp as _p4
from .planet import dsp as _planet
from .starter import dsp as _starter

dsp = sh.BlueDispatcher(name='Motors', description='Models the vehicle motors.')

dsp.add_dispatcher(
    dsp_id='alternator',
    dsp=_alternator,
    inputs=(
        'alternator_currents', 'alternator_nominal_voltage', 'motive_powers',
        'alternator_electric_powers', 'alternator_efficiency', 'accelerations',
        'service_battery_state_of_charges', 'service_battery_charging_statuses',
        'alternator_current_model', 'on_engine', 'alternator_charging_currents',
        'service_battery_initialization_time', 'times', 'is_hybrid',
    ),
    outputs=(
        'alternator_current_model', 'alternator_currents', 'alternator_powers',
        'alternator_electric_powers'
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='motor_p0',
    dsp=_p0,
    inputs=(
        'motor_p0_electric_powers', 'motor_p0_maximum_power', 'motor_p0_powers',
        'motor_p0_maximum_torque', 'motor_p0_speed_ratio', 'motor_p0_torques',
        'motor_p0_efficiency', 'motor_p0_rated_speed', 'motor_p0_speeds',
        'engine_speeds_out', 'has_motor_p0', 'times'
    ),
    outputs=(
        'motor_p0_electric_powers', 'motor_p0_maximum_power', 'motor_p0_powers',
        'motor_p0_maximum_torque', 'motor_p0_speed_ratio', 'motor_p0_torques',
        'motor_p0_maximum_powers', 'motor_p0_rated_speed', 'motor_p0_speeds',
        'engine_speeds_out', 'motor_p0_maximum_power_function', 'has_motor_p0',
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='motor_p1',
    dsp=_p1,
    inputs=(
        'motor_p1_electric_powers', 'motor_p1_maximum_power', 'motor_p1_powers',
        'motor_p1_maximum_torque', 'motor_p1_speed_ratio', 'motor_p1_torques',
        'motor_p1_efficiency', 'motor_p1_rated_speed', 'motor_p1_speeds',
        'engine_speeds_out', 'has_motor_p1', 'times'
    ),
    outputs=(
        'motor_p1_electric_powers', 'motor_p1_maximum_power', 'motor_p1_powers',
        'motor_p1_maximum_torque', 'motor_p1_speed_ratio', 'motor_p1_torques',
        'motor_p1_maximum_powers', 'motor_p1_rated_speed', 'motor_p1_speeds',
        'engine_speeds_out', 'motor_p1_maximum_power_function', 'has_motor_p1',
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='motor_p2_planetary',
    dsp=_planet,
    inputs=(
        'motor_p2_planetary_powers', 'final_drive_speeds_in', 'planetary_ratio',
        'motor_p2_planetary_maximum_power', 'motor_p2_planetary_maximum_torque',
        'planetary_mean_efficiency', 'planetary_speeds_in', 'engine_speeds_out',
        'motor_p2_planetary_rated_speed', 'motor_p2_planetary_torques', 'times',
        'motor_p2_planetary_electric_powers', 'motor_p2_planetary_speed_ratio',
        'motor_p2_planetary_efficiency', 'motor_p2_planetary_speeds',
        'has_motor_p2_planetary',
    ),
    outputs=(
        'motor_p2_planetary_powers', 'planetary_speeds_in', 'engine_speeds_out',
        'motor_p2_planetary_maximum_power', 'motor_p2_planetary_maximum_torque',
        'motor_p2_planetary_speeds', 'final_drive_speeds_in', 'planetary_ratio',
        'motor_p2_planetary_electric_powers', 'motor_p2_planetary_speed_ratio',
        'motor_p2_planetary_maximum_powers', 'motor_p2_planetary_rated_speed',
        'motor_p2_planetary_maximum_power_function', 'has_motor_p2_planetary',
        'motor_p2_planetary_torques', 'planetary_mean_efficiency',
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='motor_p2',
    dsp=_p2,
    inputs=(
        'motor_p2_electric_powers', 'motor_p2_maximum_power', 'motor_p2_powers',
        'motor_p2_maximum_torque', 'motor_p2_speed_ratio', 'motor_p2_torques',
        'motor_p2_efficiency', 'motor_p2_rated_speed', 'motor_p2_speeds',
        'gear_box_speeds_in', 'has_motor_p2', 'times'
    ),
    outputs=(
        'motor_p2_electric_powers', 'motor_p2_maximum_power', 'motor_p2_powers',
        'motor_p2_maximum_torque', 'motor_p2_speed_ratio', 'motor_p2_torques',
        'motor_p2_maximum_powers', 'motor_p2_rated_speed', 'motor_p2_speeds',
        'gear_box_speeds_in', 'has_motor_p2',
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='motor_p3',
    dsp=_p3,
    inputs=(
        'motor_p3_front_electric_powers', 'motor_p3_front_maximum_power',
        'motor_p3_front_powers', 'motor_p3_front_maximum_torque',
        'motor_p3_front_speed_ratio', 'motor_p3_front_torques',
        'motor_p3_front_efficiency', 'motor_p3_front_rated_speed',
        'motor_p3_front_speeds', 'has_motor_p3_front',
        'motor_p3_rear_electric_powers', 'motor_p3_rear_maximum_power',
        'motor_p3_rear_powers', 'motor_p3_rear_maximum_torque',
        'motor_p3_rear_speed_ratio', 'motor_p3_rear_torques',
        'motor_p3_rear_efficiency', 'motor_p3_rear_rated_speed',
        'motor_p3_rear_speeds', 'has_motor_p3_rear', 'final_drive_speeds_in',
        'times',
    ),
    outputs=(
        'motor_p3_front_electric_powers', 'motor_p3_front_maximum_power',
        'motor_p3_front_powers', 'motor_p3_front_maximum_torque',
        'motor_p3_front_speed_ratio', 'motor_p3_front_torques',
        'motor_p3_front_maximum_powers', 'motor_p3_front_rated_speed',
        'motor_p3_front_speeds', 'has_motor_p3_front',
        'motor_p3_rear_electric_powers', 'motor_p3_rear_maximum_power',
        'motor_p3_rear_powers', 'motor_p3_rear_maximum_torque',
        'motor_p3_rear_speed_ratio', 'motor_p3_rear_torques',
        'motor_p3_rear_maximum_powers', 'motor_p3_rear_rated_speed',
        'motor_p3_rear_speeds', 'has_motor_p3_rear', 'final_drive_speeds_in'
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='motor_p4',
    dsp=_p4,
    inputs=(
        'motor_p4_front_electric_powers', 'motor_p4_front_maximum_power',
        'motor_p4_front_powers', 'motor_p4_front_maximum_torque',
        'motor_p4_front_speed_ratio', 'motor_p4_front_torques',
        'motor_p4_front_efficiency', 'motor_p4_front_rated_speed',
        'motor_p4_front_speeds', 'has_motor_p4_front',
        'motor_p4_rear_electric_powers', 'motor_p4_rear_maximum_power',
        'motor_p4_rear_powers', 'motor_p4_rear_maximum_torque',
        'motor_p4_rear_speed_ratio', 'motor_p4_rear_torques',
        'motor_p4_rear_efficiency', 'motor_p4_rear_rated_speed',
        'motor_p4_rear_speeds', 'has_motor_p4_rear', 'wheel_speeds', 'times',
    ),
    outputs=(
        'motor_p4_front_electric_powers', 'motor_p4_front_maximum_power',
        'motor_p4_front_powers', 'motor_p4_front_maximum_torque',
        'motor_p4_front_speed_ratio', 'motor_p4_front_torques',
        'motor_p4_front_maximum_powers', 'motor_p4_front_rated_speed',
        'motor_p4_front_speeds', 'has_motor_p4_front',
        'motor_p4_rear_electric_powers', 'motor_p4_rear_maximum_power',
        'motor_p4_rear_powers', 'motor_p4_rear_maximum_torque',
        'motor_p4_rear_speed_ratio', 'motor_p4_rear_torques',
        'motor_p4_rear_maximum_powers', 'motor_p4_rear_rated_speed',
        'motor_p4_rear_speeds', 'has_motor_p3_rear', 'wheel_speeds'
    ),
    include_defaults=True
)

dsp.add_dispatcher(
    dsp_id='starter',
    dsp=_starter,
    inputs=(
        'delta_time_engine_starter', 'times', 'on_engine', 'starter_efficiency',
        'engine_moment_inertia', 'starter_nominal_voltage', 'engine_speeds_out',
    ),
    outputs=(
        'starter_electric_powers', 'starter_powers', 'starter_currents',
        'starter_model', 'delta_time_engine_starter'
    ),
    include_defaults=True
)


@sh.add_function(dsp, outputs=['is_hybrid'])
def identify_is_hybrid(
        has_motor_p0, has_motor_p1, has_motor_p2, has_motor_p2_planetary,
        has_motor_p3_front, has_motor_p3_rear, has_motor_p4_front,
        has_motor_p4_rear):
    """
    Identifies if the the vehicle is hybrid.

    :param has_motor_p0:
        Has the vehicle a motor in P0?
    :type has_motor_p0: bool

    :param has_motor_p1:
        Has the vehicle a motor in P1?
    :type has_motor_p1: bool

    :param has_motor_p2:
        Has the vehicle a motor in P2?
    :type has_motor_p2: bool

    :param has_motor_p2_planetary:
        Has the vehicle a motor in planetary P2?
    :type has_motor_p2_planetary: bool

    :param has_motor_p3_front:
        Has the vehicle a motor in P3 front?
    :type has_motor_p3_front: bool

    :param has_motor_p3_rear:
        Has the vehicle a motor in P3 rear?
    :type has_motor_p3_rear: bool

    :param has_motor_p4_front:
        Has the vehicle a motor in P4 front?
    :type has_motor_p4_front: bool

    :param has_motor_p4_rear:
        Has the vehicle a motor in P4 rear?
    :type has_motor_p4_rear: bool

    :return:
        Is the vehicle hybrid?
    :rtype: bool
    """
    b = has_motor_p0, has_motor_p1, has_motor_p2, has_motor_p2_planetary,
    b += has_motor_p3_front, has_motor_p3_rear, has_motor_p4_front
    b += has_motor_p4_rear,
    return any(b)
