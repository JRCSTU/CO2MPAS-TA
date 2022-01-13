# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the basic mechanics of the gear box.
"""
import math
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='mechanical model',
    description='Models the gear box mechanical.'
)


def _correct_gear_shifts(
        times, ratios, gears, velocity_speed_ratios, shift_window=4.0):
    import scipy.optimize as sci_opt
    from . import calculate_gear_shifts
    shifts = calculate_gear_shifts(gears)
    vsr = np.vectorize(lambda v: velocity_speed_ratios.get(v, 0))
    s = len(gears)

    def _err(v, r):
        v = int(v) or 1
        return np.float32(co2_utl.mae(ratios[slice(v - 1, v + 1, 1)], r))

    k = 0
    new_gears = np.zeros_like(gears)
    dt = shift_window / 2
    for i in np.arange(s)[shifts]:
        g = gears[slice(i - 1, i + 1, 1)]
        j = int(i)
        if g[0] != 0 and g[-1] != 0:
            t = times[i]
            n = max(i - (((t - dt) <= times) & (times <= t)).sum(), min(i, k))
            m = min(i + ((t <= times) & (times <= (t + dt))).sum(), s)
            if n + 1 > m:
                # noinspection PyTypeChecker
                j = int(sci_opt.brute(
                    _err, (slice(n, m, 1),), args=(vsr(g),), finish=None)
                )
        x = slice(j - 1, j + 1, 1)
        new_gears[x] = g
        new_gears[k:x.start] = g[0]
        k = x.stop

    new_gears[k:] = new_gears[k - 1]

    return new_gears


dsp.add_data('stop_velocity', dfl.values.stop_velocity)
dsp.add_data('plateau_acceleration', dfl.values.plateau_acceleration)
dsp.add_data('change_gear_window_width', dfl.values.change_gear_window_width)


@sh.add_function(dsp, outputs=['gears'])
def identify_gears(
        times, velocities, accelerations, gear_box_speeds_in,
        velocity_speed_ratios, stop_velocity, plateau_acceleration,
        change_gear_window_width, idle_engine_speed=(0, 0)):
    """
    Identifies gear time series [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param plateau_acceleration:
        Maximum acceleration to be at constant velocity [m/s2].
    :type plateau_acceleration: float

    :param change_gear_window_width:
        Time window used to apply gear change filters [s].
    :type change_gear_window_width: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Gear vector identified [-].
    :rtype: numpy.array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        r = velocities / gear_box_speeds_in

    idle = (idle_engine_speed[0] - idle_engine_speed[1],
            idle_engine_speed[0] + idle_engine_speed[1])
    r[gear_box_speeds_in <= idle[0]] = 0

    vsr = velocity_speed_ratios
    g, vsr = np.array([(k, v) for k, v in sorted(vsr.items()) if k != 0]).T
    dr = np.abs(vsr[:, None] - r)
    i, j = np.argmin(dr, 0), np.arange(times.shape[0])
    b = velocities <= vsr[i] * idle[0]
    if idle[1]:
        b |= np.abs(velocities / idle[1] - r) < dr[i, j]
    b = (velocities <= stop_velocity) | (b & (accelerations < 0))
    gear = np.where(b, 0, g[i])
    b = (velocities > stop_velocity) & (accelerations > 0)
    b |= accelerations > plateau_acceleration
    gear[(gear == 0) & b] = 1

    gear = co2_utl.median_filter(times, gear, change_gear_window_width)

    gear = _correct_gear_shifts(times, r, gear, velocity_speed_ratios)

    gear = co2_utl.clear_fluctuations(times, gear, change_gear_window_width)

    return gear.astype(int)


def _shift(a):
    return np.where(np.ediff1d(a.astype(float), [1], [1]) != 0)[0].tolist()


def _calibrate_gsm(
        velocity_speed_ratios, on_engine, anomalies, gear, velocities,
        stop_velocity, idle_engine_speed):
    # noinspection PyProtectedMember
    from .at_gear.cmv import CMV, _filter_gear_shifting_velocity as filter_gs
    idle = idle_engine_speed[0] - idle_engine_speed[1]
    _vsr = sh.combine_dicts(velocity_speed_ratios, base={0: 0})

    limits = {
        0: {False: [0]},
        1: {True: [stop_velocity]},
        max(_vsr): {True: [dfl.INF]}
    }
    shifts = np.unique(sum(map(_shift, (on_engine, anomalies)), []))
    for i, j in co2_utl.pairwise(shifts):
        if on_engine[i:j].all() and not anomalies[i:j].any():
            for v in np.array(list(co2_utl.pairwise(_shift(gear[i:j])))) + i:
                if j != v[1]:
                    v, (g, ng) = velocities[slice(*v)], gear[[v[1] - 1, v[1]]]
                    up = g < ng
                    sh.get_nested_dicts(limits, g, up, default=list).append(
                        v.max() if up else v.min()
                    )

    for k, v in list(limits.items()):
        limits[k] = v.get(False, [_vsr[k] * idle] * 2), v.get(True, [])
    d = {j: i for i, j in enumerate(sorted(limits))}
    gsm = CMV(filter_gs(sh.map_dict(d, limits), stop_velocity))
    gsm.velocity_speed_ratios = sh.selector(gsm, sh.map_dict(d, _vsr))
    gsm.convert(_vsr)
    return gsm


@sh.add_function(dsp, inputs_kwargs=True, outputs=['gears'], weight=10)
@sh.add_function(
    dsp, function_id='identify_gears_v2', outputs=['gears'], weight=20
)
def identify_gears_v1(
        times, velocities, accelerations, on_engine,
        engine_speeds_out, velocity_speed_ratios, stop_velocity,
        plateau_acceleration, change_gear_window_width, idle_engine_speed,
        correct_gear, motive_powers=None):
    """
    Identifies gear time series [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param plateau_acceleration:
        Maximum acceleration to be at constant velocity [m/s2].
    :type plateau_acceleration: float

    :param change_gear_window_width:
        Time window used to apply gear change filters [s].
    :type change_gear_window_width: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param correct_gear:
        A function to correct the gear predicted.
    :type correct_gear: callable

    :return:
        Gear vector identified [-].
    :rtype: numpy.array
    """
    n = [k for k in velocity_speed_ratios if k != 0]
    if len(n) == 1:
        gears = np.ones_like(times, int) * n[0]
        gears[velocities <= stop_velocity] = 0
        return gears
    with np.errstate(divide='ignore', invalid='ignore'):
        r = velocities / engine_speeds_out

    idle = (idle_engine_speed[0] - idle_engine_speed[1],
            idle_engine_speed[0] + idle_engine_speed[1])
    r[engine_speeds_out <= idle[0]] = 0

    _vsr = sh.combine_dicts(velocity_speed_ratios, base={0: 0})
    g, vsr = np.array([(k, v) for k, v in sorted(_vsr.items())]).T
    dr = np.abs(vsr[:, None] - r)
    i, j = np.argmin(dr, 0), np.arange(times.shape[0])
    b = velocities <= vsr[i] * idle[0]
    if idle[1]:
        b |= np.abs(velocities / idle[1] - r) < dr[i, j]
    b = (velocities <= stop_velocity) | (b & (accelerations < 0))
    gears = np.where(b, 0, g[i]).astype(int)
    gears = co2_utl.median_filter(times, gears, change_gear_window_width)
    gears = _correct_gear_shifts(times, r, gears, velocity_speed_ratios)
    gears = co2_utl.clear_fluctuations(times, gears, change_gear_window_width)
    anomalies = velocities > stop_velocity
    anomalies &= (accelerations > 0) | ~on_engine
    anomalies |= accelerations > plateau_acceleration
    anomalies &= gears == 0
    from ..control.conventional import calculate_engine_speeds_out_hot
    ds = calculate_engine_speeds_out_hot(calculate_gear_box_speeds_in(
        gears, velocities, velocity_speed_ratios, stop_velocity
    ), on_engine, idle_engine_speed) - engine_speeds_out
    i = np.where(on_engine & ~anomalies)[0]
    med = np.nanmedian(ds[i])
    std = 3 * max(50, co2_utl.mad(ds[i], med=med))
    anomalies[i] |= np.abs(ds[i] - med) >= std
    b = (gears == 0) & (velocities <= stop_velocity)
    anomalies[b] = False
    anomalies = co2_utl.clear_fluctuations(
        times, anomalies.astype(int), change_gear_window_width
    )
    for i, j in co2_utl.pairwise(_shift(np.where(b, -1, gears))):
        anomalies[i:j] = anomalies[i:j].mean() > .3

    gsm = _calibrate_gsm(
        velocity_speed_ratios, on_engine, anomalies, gears, velocities,
        stop_velocity, idle_engine_speed
    ).init_gear(
        gears, times, velocities, accelerations, motive_powers,
        correct_gear=correct_gear
    )
    for i, v in enumerate(anomalies):
        if v:
            gears[i] = gsm(i)
    gears = co2_utl.median_filter(times, gears, change_gear_window_width)
    gears = co2_utl.clear_fluctuations(times, gears, change_gear_window_width)
    return gears.astype(int)


@sh.add_function(dsp, outputs=['gear_box_speeds_in'])
def calculate_gear_box_speeds_in_v1(
        gears, gear_box_speeds_out, gear_box_ratios):
    """
    Calculates Gear box speed vector [RPM].

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int]

    :return:
        Gear box speed vector [RPM].
    :rtype: numpy.array
    """

    d = {0: 0.0}

    d.update(gear_box_ratios)

    ratios = np.vectorize(lambda k: d[k])(gears)

    return gear_box_speeds_out * ratios


@sh.add_function(dsp, outputs=['gear_box_speeds_in'], weight=25)
def calculate_gear_box_speeds_in(
        gears, velocities, velocity_speed_ratios, stop_velocity):
    """
    Calculates Gear box speed vector [RPM].

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Gear box speed vector [RPM].
    :rtype: numpy.array
    """

    speeds = np.array(velocities, dtype=float)
    n = velocities >= stop_velocity
    b = ~n
    for k, r in velocity_speed_ratios.items():
        if r:
            speeds[n & (gears == k)] /= r
        else:
            b |= gears == k

    speeds[b] = 0.0
    return speeds


@sh.add_function(dsp, outputs=['speed_velocity_ratios'])
def calculate_speed_velocity_ratios(
        gear_box_ratios, final_drive_ratios, r_dynamic):
    """
    Calculates speed velocity ratios of the gear box [h*RPM/km].

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int]

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Speed velocity ratios of the gear box [h*RPM/km].
    :rtype: dict
    """

    c = 30 / (3.6 * math.pi * r_dynamic)

    svr = {k: c * v * final_drive_ratios[k] for k, v in gear_box_ratios.items()}

    svr[0] = dfl.INF

    return svr


@sh.add_function(dsp, outputs=['speed_velocity_ratios'], weight=5)
@sh.add_function(
    dsp, inputs=['gears', 'velocities', 'engine_speeds_out', 'stop_velocity'],
    outputs=['speed_velocity_ratios'], weight=10
)
def identify_speed_velocity_ratios(
        gears, velocities, gear_box_speeds_in, stop_velocity):
    """
    Identifies speed velocity ratios from gear vector [h*RPM/km].

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param gear_box_speeds_in:
        Gear box speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Speed velocity ratios of the gear box [h*RPM/km].
    :rtype: dict
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = gear_box_speeds_in / velocities

    ratios[velocities < stop_velocity] = 0

    svr = {k: co2_utl.reject_outliers(ratios[gears == k])[0]
           for k in range(1, int(max(gears)) + 1)
           if k in gears}
    svr[0] = dfl.INF

    return svr


@sh.add_function(dsp, outputs=['velocity_speed_ratios'])
def calculate_velocity_speed_ratios(speed_velocity_ratios):
    """
    Calculates velocity speed (or speed velocity) ratios of the gear box
    [km/(h*RPM) or h*RPM/km].

    :param speed_velocity_ratios:
        Constant speed velocity (or velocity speed) ratios of the gear box
        [h*RPM/km or km/(h*RPM)].
    :type speed_velocity_ratios: dict[int | float]

    :return:
        Constant velocity speed (or speed velocity) ratios of the gear box
        [km/(h*RPM) or h*RPM/km].
    :rtype: dict
    """

    def _inverse(v):
        if v <= 0:
            return dfl.INF
        elif v >= dfl.INF:
            return 0.0
        else:
            return 1 / v

    return {k: _inverse(v) for k, v in speed_velocity_ratios.items()}


@sh.add_function(dsp, outputs=['n_gears'])
@sh.add_function(dsp, inputs=['velocity_speed_ratios'], outputs=['n_gears'])
def identify_n_gears(gear_box_ratios):
    """
    Identify the number of gears [-].

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int]

    :return:
        Number of gears [-].
    :rtype: int
    """
    return max(gear_box_ratios)


@sh.add_function(dsp, outputs=['velocity_speed_ratios'], weight=48)
def identify_velocity_speed_ratios(
        n_gears, gear_box_speeds_in, velocities, stop_velocity):
    """
    Identifies velocity speed ratios from gear box speed vector [km/(h*RPM)].

    :param n_gears:
        Number of gears [-].
    :type n_gears: int

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :rtype: dict
    """
    return identify_velocity_speed_ratios_v1(
        n_gears, gear_box_speeds_in, velocities, (0, 0), stop_velocity
    )


@sh.add_function(dsp, outputs=['velocity_speed_ratios'], weight=49)
def identify_velocity_speed_ratios_v1(
        n_gears, engine_speeds_out, velocities, idle_engine_speed,
        stop_velocity):
    """
    Identifies velocity speed ratios from gear box speed vector [km/(h*RPM)].

    :param n_gears:
        Number of gears [-].
    :type n_gears: int

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :rtype: dict
    """
    import sklearn.cluster as sk_clu

    idle_speed = idle_engine_speed[0] + idle_engine_speed[1]

    b = (engine_speeds_out > idle_speed) & (velocities > stop_velocity)
    x = (velocities[b] / engine_speeds_out[b])[:, None]

    ms = sk_clu.KMeans(n_clusters=int(n_gears), copy_x=False, random_state=0)
    ms.fit(x)

    vsr = {k + 1: v for k, v in enumerate(sorted(ms.cluster_centers_[:, 0]))}

    vsr[0] = 0.0

    return vsr


@sh.add_function(dsp, outputs=['velocity_speed_ratios'], weight=50)
def identify_velocity_speed_ratios_v2(
        gear_box_speeds_in, velocities, stop_velocity):
    """
    Identifies velocity speed ratios from gear box speed vector [km/(h*RPM)].

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :rtype: dict
    """
    return identify_velocity_speed_ratios_v3(
        gear_box_speeds_in, velocities, (0, 0), stop_velocity
    )


@sh.add_function(dsp, outputs=['velocity_speed_ratios'], weight=51)
def identify_velocity_speed_ratios_v3(
        engine_speeds_out, velocities, idle_engine_speed, stop_velocity):
    """
    Identifies velocity speed ratios from gear box speed vector [km/(h*RPM)].

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :rtype: dict
    """
    import sklearn.cluster as sk_clu

    idle_speed = idle_engine_speed[0] + idle_engine_speed[1]

    b = (engine_speeds_out > idle_speed) & (velocities > stop_velocity)
    x = (velocities[b] / engine_speeds_out[b])[:, None]

    bandwidth = sk_clu.estimate_bandwidth(x, quantile=0.2)
    ms = sk_clu.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)

    vsr = {k + 1: v for k, v in enumerate(sorted(ms.cluster_centers_[:, 0]))}

    vsr[0] = 0.0

    return vsr


@sh.add_function(dsp, outputs=['gear_box_ratios'])
def calculate_gear_box_ratios(
        velocity_speed_ratios, final_drive_ratios, r_dynamic):
    """
    Calculates gear box ratios [-].

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Gear box ratios [-].
    :rtype: dict
    """

    c = 30 / (3.6 * math.pi * r_dynamic)

    r = calculate_velocity_speed_ratios(velocity_speed_ratios)

    return {k: v / (c * final_drive_ratios[k]) for k, v in r.items() if k != 0}


@sh.add_function(dsp, outputs=['max_gear'])
def identify_max_gear(speed_velocity_ratios):
    """
    Identifies the maximum gear of the gear box [-].

    :param speed_velocity_ratios:
        Speed velocity ratios of the gear box [h*RPM/km].
    :type speed_velocity_ratios: dict[int | float]

    :return:
        Maximum gear of the gear box [-].
    :rtype: int
    """

    return int(max(speed_velocity_ratios))


@sh.add_function(dsp, outputs=['first_gear_box_ratio'])
def identify_first_gear_box_ratio(gear_box_ratios):
    """
    Identify the gear box ratio of first gear.

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int]

    :return:
        Gear box ratio of first gear [-].
    :return: float
    """
    return gear_box_ratios.get(1, sh.NONE)


@sh.add_function(dsp, outputs=['last_gear_box_ratio'])
def identify_last_gear_box_ratio(gear_box_ratios):
    """
    Identify the gear box ratio of last gear.

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int]

    :return:
        Gear box ratio of last gear [-].
    :return: float
    """
    # noinspection PyUnresolvedReferences
    return max(gear_box_ratios.items())[1]


@sh.add_function(dsp, outputs=['engine_speed_at_max_velocity'])
def calculate_engine_speed_at_max_velocity(
        r_dynamic, final_drive_ratios, last_gear_box_ratio, maximum_velocity):
    """
    Calculates the maximum velocity from full load curve.

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float | numpy.array

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param last_gear_box_ratio:
        Gear box ratio of the last gear [-].
    :type last_gear_box_ratio: float | numpy.array

    :param maximum_velocity:
        Maximum velocity [km/h].
    :type maximum_velocity: float | numpy.array

    :return:
        Engine speed at maximum velocity [RPM].
    :return: float
    """
    speed = last_gear_box_ratio / calculate_last_gear_box_ratio(
        r_dynamic, final_drive_ratios, 1, maximum_velocity
    )
    return speed


@sh.add_function(dsp, outputs=['last_gear_box_ratio'])
def calculate_last_gear_box_ratio(
        r_dynamic, final_drive_ratios, engine_speed_at_max_velocity,
        maximum_velocity):
    """
    Calculates the gear box ratio of the last gear.

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param engine_speed_at_max_velocity:
        Engine speed at maximum velocity [RPM].
    :type engine_speed_at_max_velocity: float

    :param maximum_velocity:
        Maximum velocity [km/h].
    :type maximum_velocity: float

    :return:
        Gear box ratio of the last gear [-].
    :return: float
    """
    ratio = 3.6 * 2 * np.pi * r_dynamic * engine_speed_at_max_velocity
    # noinspection PyUnresolvedReferences
    ratio /= 60 * max(final_drive_ratios.items())[1] * maximum_velocity
    return ratio


def _calculate_req_power(road_loads, velocity):
    return np.polyval(road_loads[::-1], velocity) * velocity / 3600


@sh.add_function(dsp, outputs=['last_gear_box_ratio'], weight=5)
def calculate_last_gear_box_ratio_v1(
        full_load_curve, final_drive_ratios, road_loads, r_dynamic,
        maximum_velocity):
    """
    Calculates the gear box ratio of the last gear from full load curve.

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :param maximum_velocity:
        Maximum velocity [km/h].
    :type maximum_velocity: float

    :return:
        Gear box ratio of the last gear [-].
    :return: float
    """
    d = dfl.functions.calculate_last_gear_box_ratio_v1
    ratio = np.arange(d.MAX_RATIO, d.MIN_RATIO, -d.DELTA_RATIO)

    p = full_load_curve(calculate_engine_speed_at_max_velocity(
        r_dynamic, final_drive_ratios, ratio, maximum_velocity
    ))

    return ratio[p > _calculate_req_power(road_loads, maximum_velocity)][0]


@sh.add_function(dsp, outputs=['maximum_velocity', 'gear_at_maximum_velocity'])
def calculate_maximum_velocity(
        full_load_curve, road_loads, speed_velocity_ratios):
    """
    Calculates the maximum velocity from full load curve.

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param speed_velocity_ratios:
        Speed velocity ratios of the gear box [h*RPM/km].
    :type speed_velocity_ratios: dict[int | float]

    :return:
        Maximum velocity and gear at maximum velocity [km/h, -].
    :return: float, int
    """

    d = dfl.functions.calculate_maximum_velocity
    velocity = np.arange(d.MIN_VEL, d.MAX_VEL, d.DELTA_VEL, float)

    g_id, svr = zip(*[(k, v) for k, v in sorted(
        speed_velocity_ratios.items(), reverse=True
    ) if k])

    p = full_load_curve(np.round(np.multiply(velocity[:, None], svr), 1))

    b = (p * d.PREC_FLC) < _calculate_req_power(road_loads, velocity)[:, None]
    # noinspection PyUnresolvedReferences
    velocity = np.repeat(velocity[:, None], b.shape[1], 1)
    velocity[b] = -1

    i, j = np.unravel_index(np.argmax(velocity), velocity.shape)
    return velocity[i, j], g_id[j]


@sh.add_function(dsp, outputs=['maximum_velocity'], weight=10)
def calculate_maximum_velocity_v1(
        r_dynamic, final_drive_ratios, engine_speed_at_max_velocity,
        last_gear_box_ratio):
    """
    Calculates the maximum velocity [km/h].

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param engine_speed_at_max_velocity:
        Engine speed at maximum velocity [RPM].
    :type engine_speed_at_max_velocity: float

    :param last_gear_box_ratio:
        Gear box ratio of the last gear [-].
    :type last_gear_box_ratio: float

    :return:
        Maximum velocity [km/h].
    :return: float
    """
    vel = calculate_last_gear_box_ratio(
        r_dynamic, final_drive_ratios, engine_speed_at_max_velocity, 1
    ) / last_gear_box_ratio
    return vel


@sh.add_function(dsp, outputs=['maximum_velocity'], weight=15)
def calculate_maximum_velocity_v2(
        full_load_curve, final_drive_ratios, road_loads, r_dynamic,
        last_gear_box_ratio):
    """
    Calculates the maximum velocity from full load curve.

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :param last_gear_box_ratio:
        Gear box ratio of the last gear [-].
    :type last_gear_box_ratio: float

    :return:
        Maximum velocity [km/h].
    :return: float
    """
    d = dfl.functions.calculate_maximum_velocity_v2
    velocity = np.arange(d.MAX_VEL, d.MIN_VEL, -d.DELTA_VEL)

    p = full_load_curve(calculate_engine_speed_at_max_velocity(
        r_dynamic, final_drive_ratios, last_gear_box_ratio, velocity
    ))

    return velocity[p > _calculate_req_power(road_loads, velocity)][0]


@sh.add_function(dsp, outputs=['first_gear_box_ratio'], weight=100)
def calculate_first_gear_box_ratio(
        f0, r_dynamic, engine_max_torque, maximum_vehicle_laden_mass,
        final_drive_ratios):
    """
    Calculates the gear box ratio of the first gear.

    :param f0:
        Rolling resistance force [N] when angle_slope == 0.
    :type f0: float

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :param engine_max_torque:
        Engine Max Torque [N*m].
    :type engine_max_torque: float

    :param maximum_vehicle_laden_mass:
        Technically permissible maximum laden mass [kg].
    :type maximum_vehicle_laden_mass: float

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :return:
        Gear box ratio of the first gear [-].
    :return: float
    """
    d = dfl.functions.calculate_first_gear_box_ratio
    max_torque = engine_max_torque * d.MAX_TORQUE_PERCENTAGE
    slope = np.arctan(d.STARTING_SLOPE)
    ratio = f0 * np.cos(slope)
    ratio += maximum_vehicle_laden_mass * 9.81 * np.sin(slope)
    ratio /= (max_torque * final_drive_ratios[1]) / r_dynamic
    return ratio


@sh.add_function(dsp, outputs=['gear_box_ratios'])
def design_gear_box_ratios(n_gears, first_gear_box_ratio, last_gear_box_ratio):
    """
    Designs the gear box ratios [-].

    :param n_gears:
        Number of gears [-].
    :type n_gears: int

    :param first_gear_box_ratio:
        Gear box ratio of the first gear [-].
    :type first_gear_box_ratio: float

    :param last_gear_box_ratio:
        Gear box ratio of the last gear [-].
    :type last_gear_box_ratio: float

    :return:
        Gear box ratios [-].
    :rtype: dict
    """
    n_gears = int(n_gears)
    d = dfl.functions.design_gear_box_ratios
    f_two, f_tuning = np.asarray(d.f_two), np.asarray(d.f_tuning)

    ix = np.indices((len(f_two), len(f_tuning))).reshape(2, -1).T
    f_two, f_tuning = f_two[ix[:, 0]][:, None], f_tuning[ix[:, 1]][:, None]

    ratios = np.zeros((ix.shape[0], n_gears), float)
    ratios[:, 0], ratios[:, -1] = first_gear_box_ratio, last_gear_box_ratio

    n, fgbr, lgbr = n_gears - 1, first_gear_box_ratio, last_gear_box_ratio
    f_one = f_tuning * (fgbr / (f_two ** (n * (n - 1) / 2))) ** (1 / n)

    n = n_gears - np.arange(2, n_gears)
    ratios[:, 1:-1] = lgbr * (f_one ** n) * (f_two ** (n * (n - 1) / 2))
    dr = np.diff(ratios, axis=1)

    b = np.all(dr < 0, axis=1)
    if b.any():
        ratios, dr = ratios[b], dr[b]

    b = dr[:, 0] < 2
    if b.any():
        ratios, dr = ratios[b], dr[b]

    b = np.all(np.diff(dr, axis=1) > 0, axis=1)
    if b.any():
        ratios, dr = ratios[b], dr[b]

    res = np.linalg.lstsq(
        np.vander(np.arange(n_gears - 1), 3), dr.T, rcond=-1
    )[0]
    b = res[0] < 0

    if b.any():
        ratios, dr, res = ratios[b], dr[b], res[:, b]

    return dict(enumerate(ratios[np.argmin(res[0])], 1))
