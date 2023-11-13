#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2023 European Commission (JRC);
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
    gears
"""
import logging
import schedula as sh
from ..NEDC import is_manual
from co2mpas.defaults import dfl
from .vel import dsp as _vel
from .gears import dsp as _gears

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

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_vel,
    inputs=(
        'times', 'velocities', 'speed_velocity_ratios', 'inertial_factor',
        'max_velocity', 'vehicle_mass', 'wltp_class', 'unladen_mass',
        'road_loads', 'engine_max_power', 'engine_speed_at_max_power',
        'max_speed_velocity_ratio', 'maximum_velocity_range',
        'has_capped_velocity'
    ),
    outputs=(
        'theoretical_motive_powers', 'max_time', 'wltp_class',
        {'theoretical_velocities': ('theoretical_velocities', 'velocities'),
         'theoretical_times': ('theoretical_times', 'times')}
    )
)


def wltp_gears(
        full_load_curve, velocities, accelerations, motive_powers,
        speed_velocity_ratios, idle_engine_speed, engine_speed_at_max_power,
        engine_max_power, engine_max_speed, initial_gears=None):
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

    :param initial_gears:
        Initial gear vector [-].
    :type initial_gears: numpy.array

    :return:
        Gear vector [-].
    :rtype: numpy.array
    """
    import numpy as np
    import wltp.experiment as wltp_exp
    import wltp.model as wltp_mdl
    # noinspection PyProtectedMember
    base_model = wltp_mdl._get_model_base()
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


def domain_gearshift(kwargs):
    b = kwargs.get('gear_box_type') == 'manual'
    return b and kwargs.get('enable_manual_wltp_gearshift')


dsp.add_data(
    'enable_manual_wltp_gearshift', dfl.values.enable_manual_wltp_gearshift
)
dsp.add_data('n_gears', filters=[int])
dsp.add_dispatcher(
    include_defaults=True,
    dsp=_gears,
    weight=sh.inf(1, 0),
    input_domain=domain_gearshift,
    inputs={
        'enable_manual_wltp_gearshift': sh.SINK,
        'gear_box_type': sh.SINK,
        "full_load_speeds": "full_load_speeds",
        "full_load_powers": "full_load_powers",
        "asm_margin": "asm_margin",
        "n_gears": "NoOfGears",
        "engine_max_speed_95": "Max95EngineSpeed",
        'engine_max_power': 'RatedEnginePower',
        'engine_speed_at_max_power': 'RatedEngineSpeed',
        'speed_velocity_ratios': 'speed_velocity_ratios',
        'idle_engine_speed_median': 'IdlingEngineSpeed',
        'vehicle_mass': 'VehicleTestMass',
        'road_loads': 'road_loads',
        "times": "TraceTimesInput",
        "velocities": "RequiredVehicleSpeedsInput",
        "hs_n_min1": "MinDriveEngineSpeed1st",
        "hs_n_min12": "MinDriveEngineSpeed1stTo2nd",
        "hs_n_min2d": "MinDriveEngineSpeed2ndDecel",
        "hs_n_min2": "MinDriveEngineSpeed2nd",
        "hs_n_min3": "MinDriveEngineSpeedGreater2nd",
        "hs_n_min3a": "MinDriveEngineSpeedGreater2ndAccel",
        "hs_n_min3d": "MinDriveEngineSpeedGreater2ndDecel",
        "hs_n_min3as": "MinDriveEngineSpeedGreater2ndAccelStartPhase",
        "hs_n_min3ds": "MinDriveEngineSpeedGreater2ndDecelStartPhase",
        "hs_t_start": "TimeEndOfStartPhase",
        "hs_supp0": "SuppressGear0DuringDownshifts",
        "hs_excl1": "ExcludeCrawlerGear",
        "hs_autom": "AutomaticClutchOperation",
        "hs_n_lim": "EngineSpeedLimitVMax",
        'gears': 'InitialGearsFinalAfterClutch',
        'theoretical_gears': 'InitialGearsFinalAfterClutch'
    },
    outputs={'InitialGearsFinalAfterClutch': ('theoretical_gears', 'gears')}
)

dsp.add_function(
    function=sh.add_args(wltp_gears),
    inputs=(
        'gear_box_type', 'full_load_curve', 'velocities', 'accelerations',
        'motive_powers', 'speed_velocity_ratios', 'idle_engine_speed',
        'engine_speed_at_max_power', 'engine_max_power', 'engine_max_speed'
    ),
    outputs=['gears'],
    input_domain=is_manual,
    weight=sh.inf(2, 10)
)
