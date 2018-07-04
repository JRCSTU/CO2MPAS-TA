# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions to predict the A/T gear shifting.
"""

import collections
import copy
import itertools
import pprint
import scipy.interpolate as sci_itp
import scipy.optimize as sci_opt
import sklearn.metrics as sk_met
import sklearn.pipeline as sk_pip
import sklearn.tree as sk_tree
import schedula as sh
import co2mpas.model.physical.defaults as defaults
import co2mpas.utils as co2_utl
import numpy as np


# noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit,PyUnusedLocal
class CorrectGear(object):
    def __init__(self, velocity_speed_ratios=None, idle_engine_speed=None):
        velocity_speed_ratios = velocity_speed_ratios or {}
        self.gears = np.array(sorted(k for k in velocity_speed_ratios if k > 0))
        self.vsr = velocity_speed_ratios
        self.min_gear = velocity_speed_ratios and self.gears[0] or None
        self.idle_engine_speed = idle_engine_speed
        self.pipe = []
        self.prepare_pipe = []

    def fit_basic_correct_gear(self):
        idle = self.idle_engine_speed[0] - self.idle_engine_speed[1]
        self.idle_vel = np.array([self.vsr[k] * idle for k in self.gears])
        self.prepare_pipe.append(self.prepare_basic)

    def prepare_basic(
            self, matrix, times, velocities, accelerations, motive_powers,
            engine_coolant_temperatures):

        max_gear = np.repeat(
            np.array([self.gears], int), velocities.shape[0], 0
        )
        max_gear[velocities[:, None] < self.idle_vel] = 0
        max_gear = max_gear.max(1)
        b = velocities > 0
        max_gear[~b] = 0
        b[-1] &= accelerations[-1] > 0
        b[:-1] &= (accelerations[:-1] > 0) | (np.diff(velocities) > 0)
        min_gear = np.minimum(np.where(b, self.min_gear, 0), max_gear)
        del b

        for gears in matrix.values():
            np.clip(gears, min_gear, max_gear, gears)
        del max_gear, min_gear
        return matrix

    def basic_correct_gear(
            self, gear, i, gears, times, velocities, accelerations,
            motive_powers, engine_coolant_temperatures, matrix):
        vel, mg = velocities[i], self.min_gear
        if not vel:
            gear = 0
        elif gear > mg:
            j = np.searchsorted(self.gears, gear)
            valid = (vel >= self.idle_vel)[::-1]
            k = valid.argmax()
            gear = valid[k] and self.gears[j - k] or mg
        if not gear:
            b = accelerations[i] > 0 or velocities.take(i + 1, mode='clip')
            gear = b and mg or 0
        return gear

    def fit_correct_gear_mvl(self, mvl):
        self.mvl = mvl
        self.mvl_acc = mvl.plateau_acceleration
        self.vl_dn, self.vl_up = np.array(list(zip(*map(mvl.get, self.gears))))
        self.pipe.append(self.correct_gear_mvl)

    def prepare_mvl(
            self, matrix, times, velocities, accelerations, motive_powers,
            engine_coolant_temperatures):

        vel = velocities[:, None]
        max_gear = np.repeat(np.array([self.gears], float), vel.shape[0], 0)
        max_gear[(vel > self.vl_up) | (vel < self.vl_dn)] = np.nan
        del vel

        b = velocities.astype(bool) & (accelerations < self.mvl_acc)
        max_gear = np.nanmax(max_gear, 1)
        c = ~np.isnan(max_gear)
        b, c = c & b, c & ~b
        max_gear = max_gear[b].astype(int), max_gear[c].astype(int)
        for gears in matrix.values():
            gears[b] = max_gear[0]
            gears[c] = np.minimum(max_gear[1], gears[c])

        del max_gear, b, c

        return matrix

    # noinspection PyUnusedLocal
    def correct_gear_mvl(
            self, gear, i, gears, times, velocities, accelerations,
            motive_powers, engine_coolant_temperatures, matrix):
        vel = velocities[i]
        if abs(accelerations[i]) < self.mvl_acc:
            j = np.where(self.vl_dn < vel)[0]
            if j.shape[0]:
                g = self.gears[j.max()]
                if g > gear:
                    return g
        if gear:
            while vel > self.mvl[gear][1]:
                gear += 1
        return gear

    def fit_correct_gear_full_load(
            self, full_load_curve, max_velocity_full_load_correction):
        self.max_velocity_full_load_corr = max_velocity_full_load_correction
        self.flc = full_load_curve
        self.np_vsr = np.array(list(map(self.vsr.get, self.gears)))
        self.pipe.append(self.correct_gear_full_load)

    def prepare_full_load(
            self, matrix, times, velocities, accelerations, motive_powers,
            engine_coolant_temperatures):

        max_gear = np.repeat(
            np.array([self.gears], float), velocities.shape[0], 0
        )
        speeds = velocities[:, None] / self.np_vsr
        speeds[:, 0] = np.maximum(speeds[:, 0], self.idle_engine_speed[0])
        max_gear[self.flc(speeds) <= motive_powers[:, None]] = np.nan
        del speeds
        max_gear = np.nanmax(max_gear, 1)
        max_gear[velocities > self.max_velocity_full_load_corr] = self.gears[-1]
        b = ~np.isnan(max_gear)
        max_gear = max_gear[b].astype(int)
        for gears in matrix.values():
            gears[b] = np.clip(gears[b], 0, max_gear)
        del max_gear, b

        return matrix

    def correct_gear_full_load(
            self, gear, i, gears, times, velocities, accelerations,
            motive_powers, engine_coolant_temperatures, matrix):
        vel = velocities[i]
        if vel > self.max_velocity_full_load_corr or gear <= self.min_gear:
            return gear

        j = np.searchsorted(self.gears, gear)
        delta = (self.flc(vel / self.np_vsr) - motive_powers[i])
        valid = delta[:j + 1][::-1] >= 0
        k = valid.argmax()
        if not valid[k]:
            return self.gears[np.argmax(delta)]
        return self.gears[j - k]

    def fit_correct_driveability_rules(self, engine_speed_at_max_power):
        idle = self.idle_engine_speed[0]
        n_min_drive = idle + 0.125 * (engine_speed_at_max_power - idle)
        idle = {1: idle, 2: idle * 0.9}
        self.min_gear_vel = {
            g: idle.get(g, n_min_drive) * self.vsr[g] for g in self.gears
        }
        self.pipe.append(self.correct_driveability_rules)
        self.next_gears = []

    # noinspection PyUnresolvedReferences
    def correct_driveability_rules(
            self, gear, i, gears, times, velocities, accelerations,
            motive_powers, engine_coolant_temperatures, matrix):
        pg = gears.take(i - 1, mode='clip')  # Previous gear.
        power = motive_powers[i]  # Current power.
        t0 = times[i]
        for j, (g, t) in enumerate(self.next_gears):
            if t0 <= t:
                gear = g
                self.next_gears = self.next_gears[j:]
                break
        else:
            self.next_gears = []

        # noinspection PyShadowingNames
        def get_next(k, g):
            t0 = times[k]
            for j, v in enumerate(self.next_gears):
                if t0 <= v[0]:
                    g = v[1]
                    break
            else:
                g = matrix[g][k]
            # 4.3
            if g and motive_powers[k] < 0 and \
                    self.min_gear_vel[g] > velocities[k]:
                return 0
            return g

        # 4.a During accelerations a gear have to last at least 2 seconds.
        if power > 0 or velocities.take(i + 1, mode='clip') > velocities[i]:
            j = np.searchsorted(times, times[i] - 2)
            gear = min(gear, pg + int((gears[j:i] == pg).all()))

        # 4.b
        if gear > 1 and power > 0:
            if pg < gear or motive_powers.take(i - 1, mode='clip') <= 0:
                g = gear
                t0 = times[i] + 10
                j = i + 1
                gen = enumerate(zip(times[j:], motive_powers[j:]), j)
                while True:
                    try:
                        k, (t, p) = next(gen)
                    except StopIteration:
                        break
                    if p <= 0:
                        break
                    g = get_next(k, g)
                    if g < gear:
                        if t <= t0:
                            gear = g
                        else:
                            t0 += 10
                            while True:
                                try:
                                    k, (t, p) = next(gen)
                                except StopIteration:
                                    break
                                if t > t0 or p <= 0:
                                    break
                                g = get_next(k, g)
                                if g < gear:
                                    gear = g
                                    break
                        break

            if pg > gear:
                g = gear
                t0 = times[i] + 10
                j = i + 1
                gen = enumerate(zip(times[j:], motive_powers[j:]), j)
                while True:
                    try:
                        k, (t, p) = next(gen)
                    except StopIteration:
                        break
                    g = get_next(k, g)
                    if g < pg:
                        break
                    if t > t0 or p <= 0:
                        gear = pg
                        break

        # 4.c, 4.d
        if pg and pg < gear:
            if power < 0 or motive_powers.take(i + 1, mode='clip') < 0:  # 4.d
                gear = pg
            else:  # 4.c
                g = gear
                t0 = times[i] + 5.01
                j = i + 1
                gen = enumerate(times[j:], j)
                while True:
                    try:
                        k, t = next(gen)
                    except StopIteration:
                        break
                    if t > t0:
                        break
                    g = get_next(k, g)
                    if g < gear:
                        gear = max(pg, g)
                        break
        # 4.e
        if gear == 1 and pg == 2 and power < 0:
            for k, p in enumerate(motive_powers[i + 1:], i + 1):
                if p >= 0:
                    if velocities[k] <= 1:
                        gear = 2
                    break

        # 4.e
        if gear and power < 0 and self.min_gear_vel[gear] > velocities[i]:
            gear = 0

        # 4.f
        if gear and power < 0 and pg > gear:
            j, g = i - 1, gear
            t0 = times.take(j, mode='clip')
            if (gears[np.searchsorted(times[:j], t0 - 3):j] == pg).all():
                t1, t2, t3 = times[i], t0 + 2, t0 + 5
                flag = gear, t1 + 3
                j = i + 1
                gen = enumerate(zip(times[j:], motive_powers[j:]), j)
                for k, (t, p) in gen:
                    if not (p <= 0 or t <= t2):
                        break
                    g = get_next(k, g)
                    if g != flag[0]:
                        flag = g, t + 3

                    if t >= flag[1]:
                        if flag[0] < gear:
                            gear = 0
                            self.next_gears = [(0, t0 + 1), (flag[0], t0 + 2)]
                        break
                    elif t > t3:
                        break
                    elif t > t2 and 2 <= pg - g <= 3:
                        gear = 0
                        self.next_gears = [(0, t0 + 1), (flag[0], t0 + 2)]
                        break

                if not gear and pg > flag[0] + 1:
                    v, g0, r = None, g, flag[0] - 1
                    t1, t2 = t0 + 2, t0 + 5
                    for k, (t, p) in gen:
                        g = get_next(k, g)
                        if p > 0 or g > g0:
                            break
                        g0 = g
                        if v is None and t >= t1:
                            v = matrix[g][k] - g <= 2
                        if t >= t2:
                            if g <= r:
                                if v:
                                    self.next_gears = [(0, t0 + 1), (r, t0 + 4)]
                                else:
                                    self.next_gears = [(0, t0 + 2), (g, t2)]
                            break

        # 4.f
        if power < 0 and gear and (pg > gear or pg == 0):
            j, g0 = i + 1, gear
            t0 = times.take(i - 1, mode='clip') + 2
            gen = enumerate(zip(times[j:], motive_powers[j:]), j)
            for k, (t, p) in gen:
                if p > 0:
                    break
                g1 = get_next(k, g0)
                if g1 == g0 and (g0 == 1 or t <= t0):
                    g0 = g1
                    continue
                if velocities[k] < 1:
                    self.next_gears = [(0, t)]
                    gear = 0
                elif not g1 and gear == 1:
                    g0 = g1
                    continue
                break

        # 3.2
        j = i + np.searchsorted(times[i:], times[i] + 1)
        if not gear and np.diff(velocities.take([j, j + 1], mode='clip')) > 0:
            gear = self.min_gear

        return gear

    def prepare(
            self, matrix, times, velocities, accelerations, motive_powers,
            engine_coolant_temperatures):
        for f in self.prepare_pipe:
            matrix = f(
                matrix, times, velocities, accelerations, motive_powers,
                engine_coolant_temperatures
            )
        return matrix

    def __call__(self, gear, i, gears, times, velocities, accelerations,
                 motive_powers, engine_coolant_temperatures, matrix):
        for f in self.pipe:
            gear = f(
                gear, i, gears, times, velocities, accelerations, motive_powers,
                engine_coolant_temperatures, matrix
            )
        return gear


def _upgrade_gsm(gsm, velocity_speed_ratios, cycle_type):
    if isinstance(gsm, (CMV, DTGS)):
        gsm = copy.deepcopy(gsm).convert(velocity_speed_ratios)
        if cycle_type == 'NEDC':
            if isinstance(gsm, MVL):
                par = defaults.dfl.functions.correct_constant_velocity
                gsm.correct_constant_velocity(
                    up_cns_vel=par.CON_VEL_DN_SHIFT,
                    dn_cns_vel=par.CON_VEL_UP_SHIFT
                )
            elif isinstance(gsm, CMV) and not isinstance(gsm, GSPV):
                par = defaults.dfl.functions.correct_constant_velocity
                gsm.correct_constant_velocity(
                    up_cns_vel=par.CON_VEL_UP_SHIFT,
                    up_window=par.VEL_UP_WINDOW,
                    up_delta=par.DV_UP_SHIFT, dn_cns_vel=par.CON_VEL_DN_SHIFT,
                    dn_window=par.VEL_DN_WINDOW, dn_delta=par.DV_DN_SHIFT
                )
    elif isinstance(gsm, GSMColdHot):
        for k, v in gsm.items():
            gsm[k] = _upgrade_gsm(v, velocity_speed_ratios, cycle_type)

    return gsm


def correct_gear_v0(
        cycle_type, velocity_speed_ratios, mvl, idle_engine_speed,
        full_load_curve, max_velocity_full_load_correction=float('inf'),
        plateau_acceleration=float('inf')):
    """
    Returns a function to correct the gear predicted according to
    :func:`correct_gear_mvl` and :func:`correct_gear_full_load`.

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

    :param max_velocity_full_load_correction:
        Maximum velocity to apply the correction due to the full load curve.
    :type max_velocity_full_load_correction: float

    :param plateau_acceleration:
        Maximum acceleration to be at constant velocity [m/s2].
    :type plateau_acceleration: float

    :return:
        A function to correct the predicted gear.
    :rtype: callable
    """

    mvl = _upgrade_gsm(mvl, velocity_speed_ratios, cycle_type)
    mvl.plateau_acceleration = plateau_acceleration

    correct_gear = CorrectGear(velocity_speed_ratios, idle_engine_speed)
    correct_gear.fit_correct_gear_mvl(mvl)
    correct_gear.fit_correct_gear_full_load(
        full_load_curve, max_velocity_full_load_correction
    )
    correct_gear.fit_basic_correct_gear()

    return correct_gear


def correct_gear_v1(
        cycle_type, velocity_speed_ratios, mvl, idle_engine_speed,
        plateau_acceleration=float('inf')):
    """
    Returns a function to correct the gear predicted according to
    :func:`correct_gear_mvl`.

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param mvl:
        Matrix velocity limits (upper and lower bound) [km/h].
    :type mvl: OrderedDict

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param plateau_acceleration:
        Maximum acceleration to be at constant velocity [m/s2].
    :type plateau_acceleration: float

    :return:
        A function to correct the predicted gear.
    :rtype: callable
    """

    mvl = _upgrade_gsm(mvl, velocity_speed_ratios, cycle_type)
    mvl.plateau_acceleration = plateau_acceleration

    correct_gear = CorrectGear(velocity_speed_ratios, idle_engine_speed)
    correct_gear.fit_correct_gear_mvl(mvl)
    correct_gear.fit_basic_correct_gear()

    return correct_gear


def correct_gear_v2(
        velocity_speed_ratios, idle_engine_speed, full_load_curve,
        max_velocity_full_load_correction=float('inf')):
    """
    Returns a function to correct the gear predicted according to
    :func:`correct_gear_full_load`.

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param max_velocity_full_load_correction:
        Maximum velocity to apply the correction due to the full load curve.
    :type max_velocity_full_load_correction: float

    :return:
        A function to correct the predicted gear.
    :rtype: callable
    """

    correct_gear = CorrectGear(velocity_speed_ratios, idle_engine_speed)
    correct_gear.fit_correct_gear_full_load(
        full_load_curve, max_velocity_full_load_correction
    )
    correct_gear.fit_basic_correct_gear()

    return correct_gear


def correct_gear_v3(velocity_speed_ratios, idle_engine_speed):
    """
    Returns a function that does not correct the gear predicted.

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        A function to correct the predicted gear.
    :rtype: callable
    """

    correct_gear = CorrectGear(velocity_speed_ratios, idle_engine_speed)
    correct_gear.fit_basic_correct_gear()
    return correct_gear


def identify_gear_shifting_velocity_limits(gears, velocities, stop_velocity):
    """
    Identifies gear shifting velocity matrix.

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Gear shifting velocity matrix.
    :rtype: dict
    """

    limits = {}

    for v, (g0, g1) in zip(velocities, sh.pairwise(gears)):
        if v >= stop_velocity and g0 != g1:
            limits[g0] = limits.get(g0, [[], []])
            limits[g0][g0 < g1].append(v)

    def _rjt_out(x, default):
        if x:
            x = np.asarray(x)

            # noinspection PyTypeChecker
            m, (n, s) = np.median(x), (len(x), np.std(x))

            y = 2 > (abs(x - m) / s)

            if y.any():
                y = x[y]

                # noinspection PyTypeChecker
                m, (n, s) = np.median(y), (len(y), np.std(y))

            return m, (n, s)
        else:
            return default

    max_gear = max(limits)
    gsv = collections.OrderedDict()
    for k in range(max_gear + 1):
        v0, v1 = limits.get(k, [[], []])
        gsv[k] = [_rjt_out(v0, (-1, (0, 0))),
                  _rjt_out(v1, (defaults.dfl.INF, (0, 0)))]

    return correct_gsv(gsv, stop_velocity)


def define_gear_filter(
        change_gear_window_width=defaults.dfl.values.change_gear_window_width):
    """
    Defines a gear filter function.

    :param change_gear_window_width:
        Time window used to apply gear change filters [s].
    :type change_gear_window_width: float

    :return:
        Gear filter function.
    :rtype: callable
    """

    def gear_filter(times, gears):
        """
        Filter the gears to remove oscillations.

        :param times:
            Time vector [s].
        :type times: numpy.array

        :param gears:
            Gear vector [-].
        :type gears: numpy.array

        :return:
            Filtered gears [-].
        :rtype: numpy.array
        """

        gears = co2_utl.median_filter(
            times, gears.astype(float), change_gear_window_width
        )

        gears = co2_utl.clear_fluctuations(
            times, gears, change_gear_window_width
        )

        return np.asarray(gears, dtype=int)

    return gear_filter


# noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
class CMV(collections.OrderedDict):
    def __init__(self, *args, velocity_speed_ratios=None):
        super(CMV, self).__init__(*args)
        if args and isinstance(args[0], CMV):
            if velocity_speed_ratios:
                self.convert(velocity_speed_ratios)
            else:
                velocity_speed_ratios = args[0].velocity_speed_ratios

        self.velocity_speed_ratios = velocity_speed_ratios or {}

    def __repr__(self):
        name = self.__class__.__name__
        items = [(k, v) for k, v in self.items()]
        vsr = pprint.pformat(self.velocity_speed_ratios)
        s = '{}({}, velocity_speed_ratios={})'.format(name, items, vsr)
        return s.replace('inf', "float('inf')")

    def fit(self, correct_gear, gears, engine_speeds_out, times, velocities,
            accelerations, motive_powers, velocity_speed_ratios, stop_velocity):
        from .mechanical import calculate_gear_box_speeds_in
        self.clear()
        self.velocity_speed_ratios = velocity_speed_ratios
        self.update(identify_gear_shifting_velocity_limits(
            gears, velocities, stop_velocity
        ))
        if defaults.dfl.functions.CMV.ENABLE_OPT_LOOP:
            gear_id, velocity_limits = zip(*list(sorted(self.items()))[1:])
            max_gear, _inf, grp = gear_id[-1], float('inf'), co2_utl.grouper
            update, predict = self.update, self.predict

            def _update_gvs(vel_limits):
                self[0] = (0, vel_limits[0])
                self[max_gear] = (vel_limits[-1], _inf)
                update(dict(zip(gear_id, grp(vel_limits[1:-1], 2))))

            def _error_fun(vel_limits):
                _update_gvs(vel_limits)

                g_pre = predict(
                    times, velocities, accelerations, motive_powers,
                    correct_gear=correct_gear
                )

                speed_pred = calculate_gear_box_speeds_in(
                    g_pre, velocities, velocity_speed_ratios, stop_velocity)

                return np.float32(np.mean(np.abs(
                    speed_pred - engine_speeds_out
                )))

            x0 = [self[0][1]].__add__(
                list(itertools.chain(*velocity_limits))[:-1]
            )

            x = sci_opt.fmin(_error_fun, x0, disp=False)

            _update_gvs(x)

        return self

    def correct_constant_velocity(
            self, up_cns_vel=(), up_window=0.0, up_delta=0.0, dn_cns_vel=(),
            dn_window=0.0, dn_delta=0.0):
        """
        Corrects the gear shifting matrix velocity for constant velocities.

        :param up_cns_vel:
            Constant velocities to correct the upper limits [km/h].
        :type up_cns_vel: tuple[float]

        :param up_window:
            Window to identify if the shifting matrix has limits close to
            `up_cns_vel` [km/h].
        :type up_window: float

        :param up_delta:
            Delta to add to the limit if this is close to `up_cns_vel` [km/h].
        :type up_delta: float

        :param dn_cns_vel:
            Constant velocities to correct the bottom limits [km/h].
        :type dn_cns_vel: tuple[float]

        :param dn_window:
            Window to identify if the shifting matrix has limits close to
            `dn_cns_vel` [km/h].
        :type dn_window: float

        :param dn_delta:
            Delta to add to the limit if this is close to `dn_cns_vel` [km/h].
        :type dn_delta: float

        :return:
            A gear shifting velocity matrix corrected from NEDC velocities.
        :rtype: dict
        """

        def _set_velocity(velocity, const_steps, window, delta):
            for s in const_steps:
                if s < velocity < s + window:
                    return s + delta
            return velocity

        for k, v in sorted(self.items()):
            v = [
                _set_velocity(v[0], dn_cns_vel, dn_window, dn_delta),
                _set_velocity(v[1], up_cns_vel, up_window, up_delta)
            ]

            if v[0] >= v[1]:
                v[0] = v[1] + dn_delta

            try:
                if self[k - 1][1] <= v[0]:
                    v[0] = self[k - 1][1] + up_delta
            except KeyError:
                pass
            self[k] = tuple(v)

        return self

    def plot(self):
        import matplotlib.pylab as plt
        for k, v in self.items():
            kv = {}
            for (s, l), x in zip((('down', '--'), ('up', '-')), v):
                if x < defaults.dfl.INF:
                    kv['label'] = 'Gear %d:%s-shift' % (k, s)
                    kv['linestyle'] = l
                    kv['color'] = plt.plot([x] * 2, [0, 1], **kv)[0]._color
        plt.legend(loc='best')
        plt.xlabel('Velocity [km/h]')

    def _prepare(self, times, velocities, accelerations, motive_powers,
                 engine_coolant_temperatures):
        keys = sorted(self.keys())
        matrix, r, c = {}, velocities.shape[0], len(keys) - 1
        for i, g in enumerate(keys):
            down, up = self[g]
            matrix[g] = p = np.tile(g, r)
            p[velocities < down] = keys[max(0, i - 1)]
            p[velocities >= up] = keys[min(i + 1, c)]
        return matrix

    def predict(self, times, velocities, accelerations, motive_powers,
                engine_coolant_temperatures=None,
                correct_gear=lambda i, g, *args: g[i],
                gear_filter=define_gear_filter(), index=0, gears=None):
        if gears is None:
            gears = np.zeros_like(times, int)

        for _ in self.yield_gear(
                times, velocities, accelerations, motive_powers,
                engine_coolant_temperatures, correct_gear, index, gears):
            pass

        # if gear_filter is not None:
        #    gears[index:times.shape[0]] = gear_filter(times, gears)

        return gears[index:times.shape[0]]

    @staticmethod
    def get_gear(gear, index, gears, times, velocities, accelerations,
                 motive_powers, engine_coolant_temperatures, matrix):
        return matrix[gear][index]

    def yield_gear(self, times, velocities, accelerations, motive_powers,
                   engine_coolant_temperatures=None,
                   correct_gear=lambda i, g, *args: g[i], index=0, gears=None):

        matrix = self._prepare(
            times, velocities, accelerations, motive_powers,
            engine_coolant_temperatures
        )
        if hasattr(correct_gear, 'prepare'):
            matrix = correct_gear.prepare(
                matrix, times, velocities, accelerations, motive_powers,
                engine_coolant_temperatures
            )

        valid_gears = np.array(list(getattr(self, 'gears', self)))

        def get_valid_gear(g):
            if g in valid_gears:
                return g
            return valid_gears[np.abs(np.subtract(valid_gears, g)).argmin()]

        gear = valid_gears.min()
        if gears is None:
            gears = np.zeros_like(times, int)
        else:
            gear = get_valid_gear(gears[index])

        args = (
            gears, times, velocities, accelerations, motive_powers,
            engine_coolant_temperatures, matrix
        )

        for i in np.arange(index, times.shape[0], dtype=int):
            gear = gears[i] = get_valid_gear(correct_gear(
                self.get_gear(gear, i, *args), i, *args
            ))
            yield gear

    def yield_speed(self, stop_velocity, gears, velocities, *args, **kwargs):
        vsr = self.velocity_speed_ratios
        for g, v in zip(gears, velocities):
            r = v > stop_velocity and vsr.get(g, 0)
            yield v / r if r else 0

    # noinspection PyPep8Naming
    def convert(self, velocity_speed_ratios):
        if velocity_speed_ratios != self.velocity_speed_ratios:

            vsr, n_vsr = self.velocity_speed_ratios, velocity_speed_ratios
            it = [(vsr.get(k, 0), v[0], v[1]) for k, v in self.items()]

            K, X = zip(*[(k, v) for k, v in sorted(n_vsr.items())])

            L, U = _convert_limits(it, X)

            self.clear()

            for k, l, u in sorted(zip(K, L, U), reverse=it[0][0] > it[1][0]):
                self[k] = (l, u)

            self.velocity_speed_ratios = n_vsr

        return self


# noinspection PyMissingOrEmptyDocstring
class GSMColdHot(collections.OrderedDict):
    def __init__(self, *args, time_cold_hot_transition=0.0):
        super(GSMColdHot, self).__init__(*args)
        self.time_cold_hot_transition = time_cold_hot_transition

    def __repr__(self):
        name = self.__class__.__name__
        items = [(k, v) for k, v in self.items()]
        s = '{}({}, time_cold_hot_transition={})'.format(
            name, items, self.time_cold_hot_transition
        )
        return s.replace('inf', "float('inf')")

    def fit(self, model_class, times, *args):
        self.clear()

        b = times <= self.time_cold_hot_transition

        for i in ['cold', 'hot']:
            a = (v[b] if isinstance(v, np.ndarray) else v for v in args)
            self[i] = model_class().fit(*a)
            b = ~b
        return self

    # noinspection PyTypeChecker,PyCallByClass
    def predict(self, *args, **kwargs):
        return CMV.predict(self, *args, **kwargs)

    def yield_gear(self, times, velocities, accelerations, motive_powers,
                   engine_coolant_temperatures=None,
                   correct_gear=lambda i, g, *args: g[i], index=0, gears=None):

        if gears is None:
            gears = np.zeros_like(times, int)

        n = index + np.searchsorted(
            times[index:], self.time_cold_hot_transition
        )

        flag, temp = engine_coolant_temperatures is not None, None
        for i, j, k in [(index, n, 'cold'), (n, times.shape[0], 'hot')]:
            if flag:
                temp = engine_coolant_temperatures[:j]
            yield from self[k].yield_gear(
                times[:j], velocities[:j], accelerations[:j], motive_powers[:j],
                temp, correct_gear=correct_gear, index=int(i), gears=gears
            )

    def yield_speed(self, *args, **kwargs):
        yield from self['hot'].yield_speed(*args, **kwargs)


# noinspection PyPep8Naming
def _convert_limits(it, X):
    it = sorted(it)
    x, l, u = zip(*it[1:])

    _inf = u[-1]
    x = np.asarray(x)
    l, u = np.asarray(l) / x, np.asarray(u) / x
    Spline = sci_itp.InterpolatedUnivariateSpline
    L = Spline(x, l, k=1)(X) * X
    U = np.append(Spline(x[:-1], u[:-1], k=1)(X[:-1]) * X[:-1], [_inf])
    L[0], U[0] = it[0][1:]

    return L, U


def calibrate_gear_shifting_cmv(
        correct_gear, gears, engine_speeds_out, times, velocities,
        accelerations, motive_powers, velocity_speed_ratios, stop_velocity):
    """
    Calibrates a corrected matrix velocity to predict gears.

    :param correct_gear:
        A function to correct the predicted gear.
    :type correct_gear: callable

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

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

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :returns:
        A corrected matrix velocity to predict gears.
    :rtype: dict
    """

    cmv = CMV().fit(
        correct_gear, gears, engine_speeds_out, times, velocities,
        accelerations, motive_powers, velocity_speed_ratios, stop_velocity
    )

    return cmv


def calibrate_gear_shifting_cmv_cold_hot(
        correct_gear, times, gears, engine_speeds_out, velocities,
        accelerations, motive_powers, velocity_speed_ratios,
        time_cold_hot_transition, stop_velocity):
    """
    Calibrates a corrected matrix velocity for cold and hot phases to predict
    gears.

    :param correct_gear:
        A function to correct the predicted gear.
    :type correct_gear: callable

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param time_cold_hot_transition:
        Time at cold hot transition phase [s].
    :type time_cold_hot_transition: float

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :returns:
        Two corrected matrix velocities for cold and hot phases.
    :rtype: dict
    """
    model = GSMColdHot(time_cold_hot_transition=time_cold_hot_transition).fit(
        CMV, times, correct_gear, gears, engine_speeds_out, times, velocities,
        accelerations, motive_powers, velocity_speed_ratios, stop_velocity
    )

    return model


def correct_gsv(gsv, stop_velocity):
    """
    Corrects gear shifting velocity matrix from unreliable limits.

    :param gsv:
        Gear shifting velocity matrix.
    :type gsv: dict

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Gear shifting velocity matrix corrected from unreliable limits.
    :rtype: dict
    """

    gsv[0] = [0, (stop_velocity, (defaults.dfl.INF, 0))]

    # noinspection PyMissingOrEmptyDocstring
    def func(x):
        return not x and float('inf') or 1 / x

    for v0, v1 in sh.pairwise(gsv.values()):
        up0, s0, down1, s1 = v0[1][0], v0[1][1][1], v1[0][0], v1[0][1][1]

        if down1 + s1 <= v0[0]:
            v0[1], v1[0] = up0 + s0, up0 - s0
        elif up0 >= down1:
            v0[1], v1[0] = up0 + s0, down1 - s1
            continue
        elif (v0[1][1][0], func(s0)) >= (v1[0][1][0], func(s1)):
            v0[1], v1[0] = up0 + s0, up0 - s0
        else:
            v0[1], v1[0] = down1 + s1, down1 - s1

        v0[1] += stop_velocity

    gsv[max(gsv)][1] = defaults.dfl.INF

    return gsv


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class GSPV(CMV):
    def __init__(self, *args, cloud=None, velocity_speed_ratios=None):
        super(GSPV, self).__init__(*args)
        if args and isinstance(args[0], GSPV):
            if not cloud:
                self.cloud = args[0].cloud
            if velocity_speed_ratios:
                self.convert(velocity_speed_ratios)
            else:
                velocity_speed_ratios = args[0].velocity_speed_ratios
        else:
            self.cloud = cloud or {}

        self.velocity_speed_ratios = velocity_speed_ratios or {}
        if cloud:
            self._fit_cloud()

    def __repr__(self):
        s = 'GSPV(cloud={}, velocity_speed_ratios={})'
        vsr = pprint.pformat(self.velocity_speed_ratios)
        s = s.format(pprint.pformat(self.cloud), vsr)
        return s.replace('inf', "float('inf')")

    # noinspection PyMethodOverriding
    def fit(self, gears, velocities, motive_powers, velocity_speed_ratios,
            stop_velocity):
        self.clear()

        self.velocity_speed_ratios = velocity_speed_ratios

        it = zip(velocities, motive_powers, sh.pairwise(gears))

        for v, p, (g0, g1) in it:
            if v > stop_velocity and g0 != g1:
                x = self.get(g0, [[], [[], []]])
                if g0 < g1 and p >= 0:
                    x[1][0].append(p)
                    x[1][1].append(v)
                elif g0 > g1 and p <= 0:
                    x[0].append(v)
                else:
                    continue
                self[g0] = x

        self[0] = [[0.0], [[0.0], [stop_velocity]]]

        self[max(self)][1] = [[0, 1], [defaults.dfl.INF] * 2]

        self.cloud = {k: copy.deepcopy(v) for k, v in self.items()}

        self._fit_cloud()

        return self

    def _fit_cloud(self):
        spl = sci_itp.InterpolatedUnivariateSpline

        def _line(n, m, i):
            x = np.mean(m[i]) if m[i] else None
            k_p = n - 1
            while k_p > 0 and k_p not in self:
                k_p -= 1
            x_up = self[k_p][not i](0) if k_p >= 0 else x

            if x is None or x > x_up:
                x = x_up
            return spl([0, 1], [x] * 2, k=1)

        self.clear()
        self.update(copy.deepcopy(self.cloud))

        for k, v in sorted(self.items()):
            v[0] = _line(k, v, 0)

            if len(v[1][0]) > 2:
                v[1] = _gspv_interpolate_cloud(*v[1])
            elif v[1][1]:
                v[1] = spl([0, 1], [np.mean(v[1][1])] * 2, k=1)
            else:
                v[1] = self[k - 1][0]

    @property
    def limits(self):
        limits = {}
        X = [defaults.dfl.INF, 0]
        for v in self.cloud.values():
            X[0] = min(min(v[1][0]), X[0])
            X[1] = max(max(v[1][0]), X[1])
        X = list(np.linspace(*X))
        X = [0] + X + [X[-1] * 1.1]
        for k, func in self.items():
            limits[k] = [(f(X), X) for f, x in zip(func, X)]
        return limits

    def plot(self):
        import matplotlib.pylab as plt
        for k, v in self.limits.items():
            kv = {}
            for (s, l), (x, y) in zip((('down', '--'), ('up', '-')), v):
                if x[0] < defaults.dfl.INF:
                    kv['label'] = 'Gear %d:%s-shift' % (k, s)
                    kv['linestyle'] = l
                    kv['color'] = plt.plot(x, y, **kv)[0]._color
            cy, cx = self.cloud[k][1]
            if cx[0] < defaults.dfl.INF:
                kv.pop('label')
                kv['linestyle'] = ''
                kv['marker'] = 'o'
                plt.plot(cx, cy, **kv)
        plt.legend(loc='best')
        plt.xlabel('Velocity [km/h]')
        plt.ylabel('Power [kW]')

    def _prepare(self, times, velocities, accelerations, motive_powers,
                 engine_coolant_temperatures):
        keys = sorted(self.keys())
        matrix, r, c = {}, times.shape[0], len(keys) - 1
        for i, g in enumerate(keys):
            down, up = [func(motive_powers) for func in self[g]]
            matrix[g] = p = np.tile(g, r)
            p[velocities < down] = keys[max(0, i - 1)]
            p[velocities >= up] = keys[min(i + 1, c)]
        return matrix

    def convert(self, velocity_speed_ratios):
        if velocity_speed_ratios != self.velocity_speed_ratios:

            vsr, n_vsr = self.velocity_speed_ratios, velocity_speed_ratios

            limits = [defaults.dfl.INF, 0]

            for v in self.cloud.values():
                limits[0] = min(min(v[1][0]), limits[0])
                limits[1] = max(max(v[1][0]), limits[1])

            K, X = zip(*[(k, v) for k, v in sorted(n_vsr.items())])
            cloud = self.cloud = {}

            for p in np.linspace(*limits):
                it = [[vsr.get(k, 0)] + [func(p) for func in v]
                      for k, v in self.items()]

                L, U = _convert_limits(it, X)

                for k, l, u in zip(K, L, U):
                    c = cloud[k] = cloud.get(k, [[], [[], []]])
                    c[0].append(l)
                    c[1][0].append(p)
                    c[1][1].append(u)

            cloud[0] = [[0.0], [[0.0], [self[0][1](0.0)]]]
            cloud[max(cloud)][1] = [[0, 1], [defaults.dfl.INF] * 2]

            self._fit_cloud()

            self.velocity_speed_ratios = n_vsr

        return self


def calibrate_gspv(
        gears, velocities, motive_powers, velocity_speed_ratios, stop_velocity):
    """
    Identifies gear shifting power velocity matrix.

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Gear shifting power velocity matrix.
    :rtype: dict
    """

    gspv = GSPV()

    gspv.fit(gears, velocities, motive_powers, velocity_speed_ratios,
             stop_velocity)

    return gspv


def _gspv_interpolate_cloud(powers, velocities):
    from sklearn.isotonic import IsotonicRegression
    regressor = IsotonicRegression()
    regressor.fit(powers, velocities)
    x = np.linspace(min(powers), max(powers))
    y = regressor.predict(x)
    return sci_itp.InterpolatedUnivariateSpline(x, y, k=1, ext=3)


def calibrate_gspv_cold_hot(
        times, gears, velocities, motive_powers, time_cold_hot_transition,
        velocity_speed_ratios, stop_velocity):
    """
    Identifies gear shifting power velocity matrices for cold and hot phases.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param time_cold_hot_transition:
        Time at cold hot transition phase [s].
    :type time_cold_hot_transition: float

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Gear shifting power velocity matrices for cold and hot phases.
    :rtype: dict
    """
    model = GSMColdHot(time_cold_hot_transition=time_cold_hot_transition).fit(
        GSPV, times, gears, velocities, motive_powers, velocity_speed_ratios,
        stop_velocity
    )

    return model


def prediction_gears_gsm(
        correct_gear, gear_filter, gsm, times, velocities, accelerations,
        motive_powers, cycle_type=None, velocity_speed_ratios=None,
        engine_coolant_temperatures=None):
    """
    Predicts gears with a gear shifting model (cmv or gspv or dtgs or mgs) [-].

    :param correct_gear:
        A function to correct the gear predicted.
    :type correct_gear: callable

    :param gear_filter:
        Gear filter function.
    :type gear_filter: callable

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param gsm:
        A gear shifting model (cmv or gspv or dtgs).
    :type gsm: GSPV | CMV | DTGS

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array, optional

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Predicted gears.
    :rtype: numpy.array
    """

    if velocity_speed_ratios is not None and cycle_type is not None:
        gsm = _upgrade_gsm(gsm, velocity_speed_ratios, cycle_type)

    # noinspection PyArgumentList
    gears = gsm.predict(
        times, velocities, accelerations, motive_powers,
        engine_coolant_temperatures,
        correct_gear=correct_gear, gear_filter=gear_filter
    )
    return gears


# noinspection PyMissingOrEmptyDocstring,PyCallByClass,PyUnusedLocal
# noinspection PyTypeChecker,PyPep8Naming
class DTGS:
    def __init__(self, velocity_speed_ratios):
        self.tree = sk_tree.DecisionTreeClassifier(random_state=0)
        self.model = self.gears = None
        self.velocity_speed_ratios = velocity_speed_ratios

    def fit(self, gears, velocities, accelerations, motive_powers,
            engine_coolant_temperatures):
        i = np.arange(-1, gears.shape[0] - 1)
        i[0] = 0
        from ..engine.thermal import _SelectFromModel
        model = self.tree
        self.model = sk_pip.Pipeline([
            ('feature_selection', _SelectFromModel(
                model, '0.8*median', in_mask=(0, 1)
            )),
            ('classification', model)
        ])
        X = np.column_stack((
            gears[i], velocities, accelerations, motive_powers,
            engine_coolant_temperatures
        ))
        self.model.fit(X, gears)

        self.gears = np.unique(gears)
        return self

    def _prepare(self, times, velocities, accelerations, motive_powers,
                 engine_coolant_temperatures):
        keys = sorted(self.velocity_speed_ratios.keys())
        matrix, r, c = {}, velocities.shape[0], len(keys) - 1
        func = self.model.predict
        for i, g in enumerate(keys):
            matrix[g] = func(np.column_stack((
                np.tile(g, r), velocities, accelerations, motive_powers,
                engine_coolant_temperatures
            )))
        return matrix

    @staticmethod
    def get_gear(gear, index, gears, times, velocities, accelerations,
                 motive_powers, engine_coolant_temperatures, matrix):
        return matrix[gear][index]

    def yield_gear(self, *args, **kwargs):
        return CMV.yield_gear(self, *args, **kwargs)

    def yield_speed(self, *args, **kwargs):
        return CMV.yield_speed(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return CMV.predict(self, *args, **kwargs)

    def convert(self, velocity_speed_ratios):
        self.velocity_speed_ratios = velocity_speed_ratios


def calibrate_gear_shifting_decision_tree(
        velocity_speed_ratios, gears, velocities, accelerations, motive_powers,
        engine_coolant_temperatures):
    """
    Calibrates a decision tree to predict gears.

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :returns:
        A decision tree to predict gears.
    :rtype: DTGS
    """

    model = DTGS(velocity_speed_ratios).fit(
        gears, velocities, accelerations, motive_powers,
        engine_coolant_temperatures
    )
    return model


def prediction_gears_decision_tree(
        correct_gear, gear_filter, dtgs, times, velocities, accelerations,
        motive_powers, engine_coolant_temperatures=None):
    """
    Predicts gears with a decision tree classifier [-].

    :param correct_gear:
        A function to correct the gear predicted.
    :type correct_gear: callable

    :param gear_filter:
        Gear filter function.
    :type gear_filter: callable

    :param dtgs:
        A decision tree to predict gears.
    :type dtgs: DTGS

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

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Predicted gears.
    :rtype: numpy.array
    """

    gears = prediction_gears_gsm(
        correct_gear, gear_filter, dtgs, times, velocities, accelerations,
        motive_powers, engine_coolant_temperatures=engine_coolant_temperatures
    )

    return gears


def calculate_error_coefficients(
        identified_gears, gears, engine_speeds, predicted_engine_speeds,
        velocities, stop_velocity):
    """
    Calculates the prediction's error coefficients.

    :param identified_gears:
        Identified gear vector [-].
    :type identified_gears: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param engine_speeds:
        Engine speed vector [RPM].
    :type engine_speeds: numpy.array

    :param predicted_engine_speeds:
        Predicted Engine speed vector [RPM].
    :type predicted_engine_speeds: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Correlation coefficient and mean absolute error.
    :rtype: dict
    """

    b = velocities > stop_velocity

    x = engine_speeds[b]
    y = predicted_engine_speeds[b]

    res = {
        'mean_absolute_error': sk_met.mean_absolute_error(x, y),
        'correlation_coefficient': np.corrcoef(x, y)[0, 1],
        'accuracy_score': sk_met.accuracy_score(identified_gears, gears)
    }

    return res


def calibrate_mvl(
        gears, velocities, velocity_speed_ratios, idle_engine_speed,
        stop_velocity):
    """
    Calibrates the matrix velocity limits (upper and lower bound) [km/h].

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Matrix velocity limits (upper and lower bound) [km/h].
    :rtype: MVL
    """

    mvl = MVL().fit(gears, velocities, velocity_speed_ratios, idle_engine_speed,
                    stop_velocity)

    return mvl


# not used
def correct_gear_mvl_v1(
        velocity, acceleration, gear, mvl, max_gear, min_gear,
        plateau_acceleration):
    """
    Corrects the gear predicted according to upper and lower bound velocity
    limits.

    :param velocity:
        Vehicle velocity [km/h].
    :type velocity: float

    :param acceleration:
        Vehicle acceleration [m/s2].
    :type acceleration: float

    :param gear:
        Predicted vehicle gear [-].
    :type gear: int

    :param max_gear:
        Maximum gear [-].
    :type max_gear: int

    :param min_gear:
        Minimum gear [-].
    :type min_gear: int

    :param plateau_acceleration:
        Maximum acceleration to be at constant velocity [m/s2].
    :type plateau_acceleration: float

    :param mvl:
        Matrix velocity limits (upper and lower bound) [km/h].
    :type mvl: OrderedDict

    :return:
        A gear corrected according to upper bound engine speed [-].
    :rtype: int
    """

    if abs(acceleration) < plateau_acceleration:

        while mvl[gear][1] < velocity and gear < max_gear:
            gear += 1

        while mvl[gear][0] > velocity and gear > min_gear:
            gear -= 1

    return gear


# noinspection PyMissingOrEmptyDocstring
class MVL(CMV):
    def __init__(self, *args,
                 plateau_acceleration=defaults.dfl.values.plateau_acceleration,
                 **kwargs):
        super(MVL, self).__init__(*args, **kwargs)
        self.plateau_acceleration = plateau_acceleration

    # noinspection PyMethodOverriding,PyMethodOverriding
    def fit(self, gears, velocities, velocity_speed_ratios, idle_engine_speed,
            stop_velocity):
        self.velocity_speed_ratios = velocity_speed_ratios
        idle = idle_engine_speed
        mvl = [np.array([idle[0] - idle[1], idle[0] + idle[1]])]
        for k in range(1, int(max(gears)) + 1):
            l, on, vsr = [], None, velocity_speed_ratios[k]

            for i, b in enumerate(itertools.chain(gears == k, [False])):
                if not b and on is not None:
                    v = velocities[on:i]
                    l.append([min(v), max(v)])
                    on = None

                elif on is None and b:
                    on = i

            if l:
                min_v, max_v = zip(*l)
                l = [sum(co2_utl.reject_outliers(min_v)), max(max_v)]
                mvl.append(np.array([max(idle[0], l / vsr) for l in l]))
            else:
                mvl.append(mvl[-1].copy())

        mvl = [[k, tuple(v * velocity_speed_ratios[k])]
               for k, v in reversed(list(enumerate(mvl[1:], 1)))]
        mvl[0][1] = (mvl[0][1][0], defaults.dfl.INF)
        mvl.append([0, (0, mvl[-1][1][0])])

        for i, v in enumerate(mvl[1:]):
            v[1] = (v[1][0], max(v[1][1], mvl[i][1][0] + stop_velocity))

        self.clear()
        self.update(collections.OrderedDict(mvl))

        return self

    # noinspection PyMethodOverriding
    def predict(self, velocity, acceleration, gear):
        if abs(acceleration) < self.plateau_acceleration:
            for k, v in self.items():
                if k <= gear:
                    break
                elif velocity > v[0]:
                    return k

        if gear:
            while velocity > self[gear][1]:
                gear += 1

        return gear


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def domain_fuel_saving_at_strategy(fuel_saving_at_strategy, *args):
    return fuel_saving_at_strategy


def default_specific_gear_shifting():
    """
    Returns the default value of specific gear shifting [-].

    :return:
        Specific gear shifting model.
    :rtype: str
    """

    d = defaults.dfl.functions.default_specific_gear_shifting
    return d.SPECIFIC_GEAR_SHIFTING


# noinspection PyMissingOrEmptyDocstring
def at_domain(method):
    # noinspection PyMissingOrEmptyDocstring
    def domain(kwargs):
        return kwargs['specific_gear_shifting'] in ('ALL', method)

    return domain


# noinspection PyMissingOrEmptyDocstring
def dt_domain(method):
    # noinspection PyMissingOrEmptyDocstring
    def domain(kwargs):
        s = 'specific_gear_shifting'
        dt = 'use_dt_gear_shifting'
        return kwargs[s] == method or (kwargs[dt] and kwargs[s] == 'ALL')

    return domain


def at_gear():
    """
    Defines the A/T gear shifting model.

    .. dispatcher:: d

        >>> d = at_gear()

    :return:
        The A/T gear shifting model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Automatic gear model',
        description='Defines an omni-comprehensive gear shifting model for '
                    'automatic vehicles.'
    )

    d.add_data(
        data_id='fuel_saving_at_strategy',
        default_value=defaults.dfl.values.fuel_saving_at_strategy,
        description='Apply the eco-mode gear shifting?'
    )

    d.add_data(
        data_id='plateau_acceleration',
        default_value=defaults.dfl.values.plateau_acceleration
    )

    d.add_function(
        function=calibrate_mvl,
        inputs=['gears', 'velocities', 'velocity_speed_ratios',
                'idle_engine_speed', 'stop_velocity'],
        outputs=['MVL']
    )

    d.add_data(
        data_id='change_gear_window_width',
        default_value=defaults.dfl.values.change_gear_window_width
    )

    d.add_function(
        function=define_gear_filter,
        inputs=['change_gear_window_width'],
        outputs=['gear_filter']
    )

    d.add_data(
        data_id='max_velocity_full_load_correction',
        default_value=defaults.dfl.values.max_velocity_full_load_correction
    )

    d.add_function(
        function=sh.add_args(correct_gear_v0),
        inputs=['fuel_saving_at_strategy', 'cycle_type',
                'velocity_speed_ratios', 'MVL', 'idle_engine_speed',
                'full_load_curve', 'max_velocity_full_load_correction',
                'plateau_acceleration'],
        outputs=['correct_gear'],
        input_domain=domain_fuel_saving_at_strategy
    )

    d.add_function(
        function=sh.add_args(correct_gear_v1),
        inputs=['fuel_saving_at_strategy', 'cycle_type',
                'velocity_speed_ratios', 'MVL', 'idle_engine_speed',
                'plateau_acceleration'],
        outputs=['correct_gear'],
        weight=50,
        input_domain=domain_fuel_saving_at_strategy
    )

    d.add_function(
        function=correct_gear_v2,
        inputs=['velocity_speed_ratios', 'idle_engine_speed',
                'full_load_curve', 'max_velocity_full_load_correction'],
        outputs=['correct_gear'],
        weight=50)

    d.add_function(
        function=correct_gear_v3,
        inputs=['velocity_speed_ratios', 'idle_engine_speed'],
        outputs=['correct_gear'],
        weight=100)

    d.add_function(
        function=default_specific_gear_shifting,
        outputs=['specific_gear_shifting']
    )

    d.add_data(
        data_id='specific_gear_shifting',
        description='Specific gear shifting model.'
    )

    d.add_dispatcher(
        dsp_id='cmv_model',
        dsp=at_cmv(),
        input_domain=at_domain('CMV'),
        inputs=(
            'CMV', 'accelerations', 'correct_gear', 'cycle_type',
            'engine_speeds_out', 'gear_filter', 'gears', 'stop_velocity',
            'times', 'velocities', 'velocity_speed_ratios', 'motive_powers',
            {'specific_gear_shifting': sh.SINK}),
        outputs=('CMV', 'gears')
    )

    d.add_dispatcher(
        include_defaults=True,
        dsp_id='cmv_ch_model',
        input_domain=at_domain('CMV_Cold_Hot'),
        dsp=at_cmv_cold_hot(),
        inputs=(
            'CMV_Cold_Hot', 'accelerations', 'correct_gear', 'cycle_type',
            'engine_speeds_out', 'gear_filter', 'gears', 'stop_velocity',
            'time_cold_hot_transition', 'times', 'velocities', 'motive_powers',
            'velocity_speed_ratios', {'specific_gear_shifting': sh.SINK}),
        outputs=('CMV_Cold_Hot', 'gears')
    )

    d.add_data(
        data_id='use_dt_gear_shifting',
        default_value=defaults.dfl.values.use_dt_gear_shifting,
        description='If to use decision tree classifiers to predict gears.'
    )

    d.add_dispatcher(
        dsp_id='dtgs_model',
        input_domain=dt_domain('DTGS'),
        dsp=at_dtgs(),
        inputs=(
            'velocity_speed_ratios', 'DTGS', 'accelerations', 'correct_gear',
            'engine_coolant_temperatures', 'gear_filter', 'gears',
            'motive_powers', 'times', 'velocities',
            {'specific_gear_shifting': sh.SINK,
             'use_dt_gear_shifting': sh.SINK}),
        outputs=('DTGS', 'gears')
    )

    d.add_dispatcher(
        dsp_id='gspv_model',
        dsp=at_gspv(),
        input_domain=at_domain('GSPV'),
        inputs=(
            'GSPV', 'accelerations', 'correct_gear', 'cycle_type',
            'gear_filter', 'gears', 'motive_powers', 'stop_velocity', 'times',
            'velocities', 'velocity_speed_ratios',
            {'specific_gear_shifting': sh.SINK}),
        outputs=('GSPV', 'gears')
    )

    d.add_dispatcher(
        include_defaults=True,
        dsp_id='gspv_ch_model',
        dsp=at_gspv_cold_hot(),
        input_domain=at_domain('GSPV_Cold_Hot'),
        inputs=(
            'GSPV_Cold_Hot', 'accelerations', 'correct_gear', 'cycle_type',
            'gear_filter', 'gears', 'motive_powers', 'stop_velocity',
            'time_cold_hot_transition', 'times', 'velocities',
            'velocity_speed_ratios', {'specific_gear_shifting': sh.SINK}),
        outputs=('GSPV_Cold_Hot', 'gears')
    )

    return d


def at_cmv():
    """
    Defines the corrected matrix velocity model.

    .. dispatcher:: d

        >>> d = at_cmv()

    :return:
        The corrected matrix velocity model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Corrected Matrix Velocity Approach',
    )

    d.add_data(
        data_id='stop_velocity',
        default_value=defaults.dfl.values.stop_velocity
    )

    # calibrate corrected matrix velocity
    d.add_function(
        function=calibrate_gear_shifting_cmv,
        inputs=['correct_gear', 'gears', 'engine_speeds_out', 'times',
                'velocities', 'accelerations', 'motive_powers',
                'velocity_speed_ratios', 'stop_velocity'],
        outputs=['CMV'])

    # predict gears with corrected matrix velocity
    d.add_function(
        function=prediction_gears_gsm,
        inputs=['correct_gear', 'gear_filter', 'CMV', 'times', 'velocities',
                'accelerations', 'motive_powers', 'cycle_type',
                'velocity_speed_ratios'],
        outputs=['gears'])

    return d


def at_cmv_cold_hot():
    """
    Defines the corrected matrix velocity with cold/hot model.

    .. dispatcher:: d

        >>> d = at_cmv_cold_hot()

    :return:
        The corrected matrix velocity with cold/hot model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Corrected Matrix Velocity Approach with Cold/Hot'
    )

    d.add_data(
        data_id='time_cold_hot_transition',
        default_value=defaults.dfl.values.time_cold_hot_transition
    )

    d.add_data(
        data_id='stop_velocity',
        default_value=defaults.dfl.values.stop_velocity
    )

    # calibrate corrected matrix velocity cold/hot
    d.add_function(
        function=calibrate_gear_shifting_cmv_cold_hot,
        inputs=['correct_gear', 'times', 'gears', 'engine_speeds_out',
                'velocities', 'accelerations', 'motive_powers',
                'velocity_speed_ratios', 'time_cold_hot_transition',
                'stop_velocity'],
        outputs=['CMV_Cold_Hot'])

    # predict gears with corrected matrix velocity
    d.add_function(
        function=prediction_gears_gsm,
        inputs=['correct_gear', 'gear_filter', 'CMV_Cold_Hot', 'times',
                'velocities', 'accelerations', 'motive_powers', 'cycle_type',
                'velocity_speed_ratios'],
        outputs=['gears'])

    return d


def at_dtgs():
    """
    Defines the decision tree with velocity, acceleration, temperature & power
    model.

    .. dispatcher:: d

        >>> d = at_dtgs()

    :return:
        The decision tree with velocity, acceleration, temperature & power
        model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Decision Tree with Velocity, Acceleration, Temperature, & Power'
    )

    d.add_data(
        data_id='accelerations',
        description='Acceleration vector [m/s2].'
    )

    d.add_data(
        data_id='engine_coolant_temperatures',
        description='Engine coolant temperature vector [°C].'
    )

    d.add_data(
        data_id='motive_powers',
        description='Motive power [kW].'
    )

    # calibrate decision tree with velocity, acceleration, temperature
    # & wheel power
    d.add_function(
        function=calibrate_gear_shifting_decision_tree,
        inputs=['velocity_speed_ratios', 'gears', 'velocities', 'accelerations',
                'motive_powers', 'engine_coolant_temperatures'],
        outputs=['DTGS']
    )

    # predict gears with decision tree with velocity, acceleration, temperature
    # & wheel power
    d.add_function(
        function=prediction_gears_decision_tree,
        inputs=['correct_gear', 'gear_filter', 'DTGS', 'times', 'velocities',
                'accelerations', 'motive_powers',
                'engine_coolant_temperatures'],
        outputs=['gears'])

    return d


def at_gspv():
    """
    Defines the gear shifting power velocity model.

    .. dispatcher:: d

        >>> d = at_gspv()

    :return:
        The gear shifting power velocity model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Gear Shifting Power Velocity Approach'
    )

    d.add_data(
        data_id='stop_velocity',
        default_value=defaults.dfl.values.stop_velocity
    )

    # calibrate corrected matrix velocity
    d.add_function(
        function=calibrate_gspv,
        inputs=['gears', 'velocities', 'motive_powers',
                'velocity_speed_ratios', 'stop_velocity'],
        outputs=['GSPV'])

    # predict gears with corrected matrix velocity
    d.add_function(
        function=prediction_gears_gsm,
        inputs=['correct_gear', 'gear_filter', 'GSPV', 'times', 'velocities',
                'accelerations', 'motive_powers', 'cycle_type',
                'velocity_speed_ratios'],
        outputs=['gears'])

    return d


def at_gspv_cold_hot():
    """
    Defines the gear shifting power velocity with cold/hot model.

    .. dispatcher:: d

        >>> d = at_gspv_cold_hot()

    :return:
        The gear shifting power velocity with cold/hot model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Gear Shifting Power Velocity Approach with Cold/Hot'
    )

    d.add_data(
        data_id='time_cold_hot_transition',
        default_value=defaults.dfl.values.time_cold_hot_transition
    )

    d.add_data(
        data_id='stop_velocity',
        default_value=defaults.dfl.values.stop_velocity
    )

    # calibrate corrected matrix velocity
    d.add_function(
        function=calibrate_gspv_cold_hot,
        inputs=['times', 'gears', 'velocities',
                'motive_powers', 'time_cold_hot_transition',
                'velocity_speed_ratios', 'stop_velocity'],
        outputs=['GSPV_Cold_Hot'])

    # predict gears with corrected matrix velocity
    d.add_function(
        function=prediction_gears_gsm,
        inputs=['correct_gear', 'gear_filter', 'GSPV_Cold_Hot', 'times',
                'velocities', 'accelerations', 'motive_powers', 'cycle_type',
                'velocity_speed_ratios'],
        outputs=['gears'])

    return d
