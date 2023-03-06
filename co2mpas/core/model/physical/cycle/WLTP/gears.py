# -*- coding: UTF-8 -*-
#
# Copyright 2015-2023 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to calculate the WLTP theoretical velocities.
"""
import numpy as np
import schedula as sh
from gearshift.core.model.calculateShiftpointsNdvFullPC import dsp as _gears

dsp = _gears.register()
dsp.add_data('SafetyMargin', 0.1)
for k in [
    "AdditionalSafetyMargin0", "MinDriveEngineSpeed1st",
    "MinDriveEngineSpeed1stTo2nd", "MinDriveEngineSpeed2ndDecel",
    "MinDriveEngineSpeed2nd", "MinDriveEngineSpeedGreater2nd",
    "MinDriveEngineSpeedGreater2ndAccel", "MinDriveEngineSpeedGreater2ndDecel",
    "MinDriveEngineSpeedGreater2ndAccelStartPhase",
    "MinDriveEngineSpeedGreater2ndDecelStartPhase", "TimeEndOfStartPhase",
    "SuppressGear0DuringDownshifts", "ExcludeCrawlerGear",
    "AutomaticClutchOperation", "EngineSpeedLimitVMax"]:
    dsp.add_data(k, 0)


@sh.add_function(dsp, True, True, outputs=[
    'FullPowerCurve', 'StartEngineSpeed', 'EndEngineSpeed'
])
def define_FullPowerCurve(full_load_speeds, full_load_powers, asm_margin=None):
    speeds = full_load_speeds
    columns = [full_load_speeds, full_load_powers]
    if asm_margin is None:
        columns.append(np.zeros_like(full_load_speeds))  # TODO modify calculateShiftpointsNdvFullPC bug.
    else:
        columns.append(asm_margin)
    return np.column_stack(tuple(columns)), np.min(speeds), np.max(speeds)


@sh.add_function(dsp, outputs=['gear_nbrs', 'Ndv'])
def define_FullPowerCurve(speed_velocity_ratios):
    return zip(*sorted((k, v) for k, v in speed_velocity_ratios.items() if k))


dsp.add_function(
    'splitting', sh.bypass, ['road_loads'], ['f0', 'f1', 'f2']
)
