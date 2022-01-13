#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to define theoretical profiles of WLTP cycle.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.cycle.WLTP

.. autosummary::
    :nosignatures:
    :toctree: WLTP/

    vel
"""
import logging
import copy
import schedula as sh
from ..NEDC import is_manual
from co2mpas.defaults import dfl
from .vel import dsp as _vel

logging.getLogger('wltp.experiment').setLevel(logging.WARNING)
log = logging.getLogger(__name__)

dsp = sh.BlueDispatcher(
    name='WLTP cycle model',
    description='Returns the theoretical times, velocities, and gears of WLTP.'
)

dsp.add_data(
    'initial_temperature', dfl.values.initial_temperature_WLTP,
    description='Initial temperature of the test cell [°C].'
)
dsp.add_data(
    'max_time', dfl.values.max_time_WLTP, 5, description='Maximum time [s].'
)
dsp.add_data('wltp_base_model', copy.deepcopy(dfl.values.wltp_base_model))


@sh.add_function(dsp, outputs=['base_model'])
def define_wltp_base_model(wltp_base_model):
    """
    Defines WLTP base model.

    :param wltp_base_model:
        WLTP base model params.
    :type wltp_base_model: dict

    :return:
        WLTP base model.
    :rtype: dict
    """
    import wltp.model as wltp_mdl
    # noinspection PyProtectedMember
    return sh.combine_dicts(wltp_mdl._get_model_base(), wltp_base_model)


dsp.add_dispatcher(
    dsp=_vel,
    inputs=(
        'times', 'base_model', 'velocities', 'speed_velocity_ratios',
        'inertial_factor', 'downscale_phases', 'max_velocity',
        'downscale_factor', 'downscale_factor_threshold', 'vehicle_mass',
        'unladen_mass', 'road_loads', 'engine_max_power', 'wltp_class',
        'engine_speed_at_max_power', 'max_speed_velocity_ratio'
    ),
    outputs=(
        'theoretical_motive_powers',
        {'theoretical_velocities': ('theoretical_velocities', 'velocities')}
    )
)


def wltp_gears(
        full_load_curve, velocities, accelerations, motive_powers,
        speed_velocity_ratios, idle_engine_speed, engine_speed_at_max_power,
        engine_max_power, engine_max_speed, base_model, initial_gears=None):
    """
    Returns the gear shifting profile according to WLTP [-].

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param speed_velocity_ratios:
        Speed velocity ratios of the gear box [h*RPM/km].
    :type speed_velocity_ratios: dict[int | float]

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_speed_at_max_power:
        Rated engine speed [RPM].
    :type engine_speed_at_max_power: float

    :param engine_max_power:
        Maximum power [kW].
    :type engine_max_power: float

    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :param base_model:
        WLTP base model params.
    :type base_model: dict

    :param initial_gears:
        Initial gear vector [-].
    :type initial_gears: numpy.array

    :return:
        Gear vector [-].
    :rtype: numpy.array
    """
    import numpy as np
    import wltp.experiment as wltp_exp
    n_min_drive = None
    svr = [v for k, v in sorted(speed_velocity_ratios.items()) if k]
    idle = idle_engine_speed[0]

    n_norm = np.arange(
        0.0, (engine_max_speed - idle) / (engine_speed_at_max_power - idle), .01
    )

    load_curve = {
        'n_norm': n_norm,
        'p_norm': full_load_curve(
            idle + n_norm * (engine_speed_at_max_power - idle)
        ) / engine_max_power
    }

    b = velocities < 0
    if b.any():
        v = velocities.copy()
        v[b] = 0
    else:
        v = velocities

    res = wltp_exp.run_cycle(
        v, accelerations, motive_powers, svr, idle_engine_speed[0], n_min_drive,
        engine_speed_at_max_power, engine_max_power, load_curve, base_model
    )

    if initial_gears:
        gears = initial_gears.copy()
    else:
        # noinspection PyUnresolvedReferences
        gears = res[0]

    # Apply Driveability-rules.
    # noinspection PyUnresolvedReferences
    wltp_exp.applyDriveabilityRules(v, accelerations, gears, res[1], res[-1])

    gears[gears < 0] = 0
    log.warning('The WLTP gear-shift profile generation is for engineering '
                'purposes and the results are by no means valid according to '
                'the legislation.\nActually they are calculated based on a pre '
                'phase-1a version of the GTR spec.\n '
                'Please provide the gear-shifting profile '
                'within `prediction.WLTP` sheet.')
    return gears


dsp.add_function(
    function=sh.add_args(wltp_gears),
    inputs=(
        'gear_box_type', 'full_load_curve', 'velocities', 'accelerations',
        'motive_powers', 'speed_velocity_ratios', 'idle_engine_speed',
        'engine_speed_at_max_power', 'engine_max_power', 'engine_max_speed',
        'base_model'
    ),
    outputs=['gears'],
    input_domain=is_manual,
    weight=sh.inf(2, 10)
)
