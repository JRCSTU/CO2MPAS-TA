# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the manual gear shifting.
"""
import collections
import numpy as np
import schedula as sh
from .at_gear import CMV

dsp = sh.BlueDispatcher(
    name='manual GS model', description='Models the manual gear shifting.'
)


@sh.add_function(dsp, outputs=['engine_max_speed_95'])
def calculate_engine_max_speed_95(
        full_load_speeds, idle_engine_speed, engine_max_speed, full_load_curve,
        engine_max_power):
    """
    Calculates the maximum engine speed [RPM] at 95% of the nominal power.

    :param full_load_speeds:
        T1 map speed vector [RPM].
    :type full_load_speeds: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param engine_max_power:
        Engine nominal power [kW].
    :type engine_max_power: float

    :return:
        Maximum engine speed [RPM] at 95% of the nominal power.
    :rtype: float
    """
    from scipy.interpolate import InterpolatedUnivariateSpline as Spl
    speeds = [idle_engine_speed[0] - idle_engine_speed[1], engine_max_speed]
    speeds.extend(full_load_speeds)
    speeds = np.unique(speeds)
    n = [engine_max_speed]
    n.extend(
        Spl(speeds, full_load_curve(speeds) / engine_max_power - 0.95).roots()
    )

    return max(n)


# noinspection PyMissingOrEmptyDocstring
class MGS(CMV):
    def __init__(self, *args, **kwargs):
        super(MGS, self).__init__(*args, **kwargs)

    # noinspection PyMethodOverriding,PyMethodOverriding
    def fit(self, full_load_curve, engine_speed_at_max_power, road_loads,
            engine_max_speed_95, velocity_speed_ratios, idle_engine_speed):
        self.velocity_speed_ratios = velocity_speed_ratios
        # noinspection PyProtectedMember
        from co2mpas.defaults import dfl
        from .mechanical import _calculate_req_power as calc_power_req

        d = dfl.functions.MGS
        vel = np.arange(d.MIN_VEL, d.MAX_VEL, d.DELTA_VEL, float)[:, None]

        g_id, vsr = zip(*[(k, v) for k, v in sorted(
            velocity_speed_ratios.items(), reverse=True
        ) if k])
        s = np.round(np.divide(vel, vsr), 1)

        p = full_load_curve(s)

        b = np.zeros_like(p, bool)
        g_id, idle = np.array(g_id), idle_engine_speed[0]
        j = g_id == 1
        b[:, j] = s[:, j] < idle
        j = g_id == 2
        b[:, j] = s[:, j] < idle * 0.9
        j = np.in1d(g_id, (1, 2), True, True)
        b[:, j] = s[:, j] < idle + 0.125 * (engine_speed_at_max_power - idle)

        b |= s > engine_max_speed_95
        b[:, j] |= p[:, j] < (calc_power_req(road_loads, vel) / d.PREC_FLC)

        v = np.repeat(vel, b.shape[1], 1)
        v[b] = np.nan
        d, u = np.nanmin(v, 0), np.nanmax(v, 0)
        d[-1], u[0] = 0, dfl.INF

        self.clear()
        self.update(collections.OrderedDict(zip(g_id, zip(d, u))))
        self[0] = (0, 0)
        return self


@sh.add_function(dsp, outputs=['MGS', 'MVL'])
def define_msg_and_mvl(
        full_load_curve, road_loads, velocity_speed_ratios,
        engine_max_speed_95, idle_engine_speed, engine_speed_at_max_power):
    """
    Calculates the maximum velocity from full load curve.

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]
    
    :param engine_max_speed_95:
        Maximum engine speed [RPM] at 95% of the nominal power.
    :type engine_max_speed_95: float
    
    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)
    
    :param engine_speed_at_max_power:
        Engine speed at engine nominal power [RPM].
    :type engine_speed_at_max_power: float

    :return:
        Maximum velocity and gear at maximum velocity [km/h, -].
    :return: float, int
    """
    from .at_gear import MVL
    mgs = MGS().fit(
        full_load_curve, engine_speed_at_max_power, road_loads,
        engine_max_speed_95, velocity_speed_ratios, idle_engine_speed
    )
    mvl = MVL()
    mvl.update(mgs)
    mvl.velocity_speed_ratios = velocity_speed_ratios

    return mgs, mvl


@sh.add_function(
    dsp,
    inputs=[
        'cycle_type', 'velocity_speed_ratios', 'MVL', 'idle_engine_speed',
        'full_load_curve', 'engine_speed_at_max_power'
    ],
    outputs=['correct_gear']
)
def correct_gear_v4(
        cycle_type, velocity_speed_ratios, mvl, idle_engine_speed,
        full_load_curve, engine_speed_at_max_power):
    """
    Returns a function to correct the gear predicted according to
    :func:`correct_gear_mvl`, :func:`correct_gear_full_load` and driveability
    rules.

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param mvl:
        Matrix velocity limits (upper and lower bound) [km/h].
    :type mvl: MVL

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param engine_speed_at_max_power:
        Rated engine speed [RPM].
    :type engine_speed_at_max_power: float

    :return:
        A function to correct the predicted gear.
    :rtype: callable
    """
    from .at_gear import correct_gear_v0
    cg = correct_gear_v0(
        cycle_type, velocity_speed_ratios, mvl, idle_engine_speed,
        full_load_curve
    )
    cg.fit_correct_driveability_rules(engine_speed_at_max_power)
    return cg


@sh.add_function(dsp, outputs=['gears'], weight=sh.inf(1, 0), inputs=[
    'correct_gear', 'MGS', 'times', 'velocities', 'accelerations',
    'motive_powers'
])
def prediction_gears_gsm_v1(
        correct_gear, gsm, times, velocities, accelerations, motive_powers):
    """
    Predicts gears with a gear shifting model (mgs) [-].

    :param correct_gear:
        A function to correct the gear predicted.
    :type correct_gear: callable

    :param gsm:
        A gear shifting model (cmv or gspv or dtgs).
    :type gsm: MSG

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :return:
        Predicted gears.
    :rtype: numpy.array
    """
    gears = gsm.predict(
        times, velocities, accelerations, motive_powers,
        correct_gear=correct_gear
    )
    return gears
