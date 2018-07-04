# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions that model the basic mechanics of the engine.

Sub-Modules:

.. currentmodule:: co2mpas.model.physical.engine

.. autosummary::
    :nosignatures:
    :toctree: engine/

    thermal
    co2_emission
    cold_start
    start_stop
"""

import math
import co2mpas.model.physical.defaults as defaults
import numpy as np
import sklearn.metrics as sk_met
from sklearn.cluster import DBSCAN
import schedula as sh
import co2mpas.utils as co2_utl
import functools
import scipy.interpolate as sci_itp


def calculate_engine_mass(ignition_type, engine_max_power):
    """
    Calculates the engine mass [kg].

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :param engine_max_power:
        Engine nominal power [kW].
    :type engine_max_power: float

    :return:
       Engine mass [kg].
    :rtype: float
    """

    par = defaults.dfl.functions.calculate_engine_mass.PARAMS
    _mass_coeff = par['mass_coeff']
    m, q = par['mass_reg_coeff']
    # Engine mass empirical formula based on web data found for engines weighted
    # according DIN 70020-GZ
    # kg
    return (m * engine_max_power + q) * _mass_coeff[ignition_type]


def calculate_engine_heat_capacity(engine_mass):
    """
    Calculates the engine heat capacity [kg*J/K].

    :param engine_mass:
        Engine mass [kg].
    :type engine_mass: float

    :return:
       Engine heat capacity [kg*J/K].
    :rtype: float
    """

    par = defaults.dfl.functions.calculate_engine_heat_capacity.PARAMS
    mp, hc = par['heated_mass_percentage'], par['heat_capacity']

    return engine_mass * np.sum(hc[k] * v for k, v in mp.items())


def identify_engine_speed_at_max_power(full_load_speeds, full_load_powers):
    """
    Identifies engine nominal speed at engine nominal power [RPM].

    :param full_load_speeds:
        T1 map speed vector [RPM].
    :type full_load_speeds: numpy.array

    :param full_load_powers:
        T1 map power vector [kW].
    :type full_load_powers: numpy.array

    :return:
        Engine speed at engine nominal power [RPM].
    :rtype: float
    """
    return full_load_speeds[np.argmax(full_load_powers)]


def calculate_engine_max_power(full_load_curve, engine_speed_at_max_power):
    """
    Calculates engine nominal power [kW].

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param engine_speed_at_max_power:
        Engine speed at engine nominal power [RPM].
    :type engine_speed_at_max_power: float

    :return:
        Engine nominal power [kW].
    :rtype: float
    """
    return full_load_curve(engine_speed_at_max_power)


def define_full_load_curve(
        full_load_speeds, full_load_powers, idle_engine_speed,
        engine_max_speed):
    """
    Calculates the full load curve.

    :param full_load_speeds:
        T1 map speed vector [RPM].
    :type full_load_speeds: numpy.array

    :param full_load_powers:
        T1 map power vector [kW].
    :type full_load_powers: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :return:
        Vehicle full load curve.
    :rtype: function
    """
    xp = [idle_engine_speed[0] - idle_engine_speed[1], engine_max_speed]
    xp.extend(full_load_speeds)
    xp = np.unique(xp)

    fp = sci_itp.InterpolatedUnivariateSpline(
        full_load_speeds, full_load_powers, k=1
    )(xp)

    return functools.partial(np.interp, xp=xp, fp=fp, left=0, right=0)


def default_engine_max_speed(
        ignition_type, idle_engine_speed, engine_speed_at_max_power):
    """
    Returns the default maximum allowed engine speed [RPM].

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_speed_at_max_power:
        Engine speed at engine nominal power [RPM].
    :type engine_speed_at_max_power: float

    :return:
        Maximum allowed engine speed [RPM].
    :rtype: float
    """
    dfl = defaults.dfl.functions.default_full_load_speeds_and_powers
    idle, r = idle_engine_speed[0], max(dfl.FULL_LOAD[ignition_type][0])
    return idle + r * (engine_speed_at_max_power - idle)


def default_full_load_speeds_and_powers(
        ignition_type, engine_max_power, engine_speed_at_max_power,
        idle_engine_speed, engine_max_speed):
    """
    Returns the defaults full load speeds and powers [RPM, kW].

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :param engine_max_power:
        Engine nominal power [kW].
    :type engine_max_power: float

    :param engine_speed_at_max_power:
        Engine speed at engine nominal power [RPM].
    :type engine_speed_at_max_power: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :return:
         T1 map speed [RPM] and power [kW] vectors.
    :rtype: (numpy.array, numpy.array)
    """
    dfl = defaults.dfl.functions.default_full_load_speeds_and_powers
    xp, fp = dfl.FULL_LOAD[ignition_type]

    idle = idle_engine_speed[0]

    full_load_speeds = np.unique(np.append(
        [engine_speed_at_max_power], np.linspace(idle, engine_max_speed)
    ))

    full_load_powers = sci_itp.InterpolatedUnivariateSpline(xp, fp, k=1)(
        (full_load_speeds - idle) / (engine_speed_at_max_power - idle)
    ) * engine_max_power

    return full_load_speeds, full_load_powers


def identify_on_idle(
        velocities, engine_speeds_out, gears, stop_velocity,
        min_engine_on_speed):
    """
    Identifies when the engine is on idle [-].

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        If the engine is on idle [-].
    :rtype: numpy.array
    """

    on_idle = engine_speeds_out > min_engine_on_speed
    on_idle &= (gears == 0) | (velocities <= stop_velocity)

    return on_idle


# noinspection PyPep8Naming,PyMissingOrEmptyDocstring
class _IdleDetector(DBSCAN):
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 algorithm='auto', leaf_size=30, p=None):
        super(_IdleDetector, self).__init__(
            eps=eps, min_samples=min_samples, metric=metric,
            algorithm=algorithm, leaf_size=leaf_size, p=p
        )
        self.cluster_centers_ = None
        self.min, self.max = None, None

    def fit(self, X, y=None, sample_weight=None):
        super(_IdleDetector, self).fit(X, y=y, sample_weight=sample_weight)

        c, l = self.components_, self.labels_[self.core_sample_indices_]
        self.cluster_centers_ = np.array(
            [np.mean(c[l == i]) for i in range(l.max() + 1)]
        )
        self.min, self.max = c.min(), c.max()
        return self

    def predict(self, X, set_outliers=True):
        y = sk_met.pairwise_distances_argmin(X, self.cluster_centers_[:, None])
        if set_outliers:
            y[((X > self.max) | (X < self.min))[:, 0]] = -1
        return y


def define_idle_model_detector(
        velocities, engine_speeds_out, stop_velocity, min_engine_on_speed):
    """
    Defines idle engine speed model detector.

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Idle engine speed model detector.
    :rtype: sklearn.cluster.DBSCAN
    """

    b = (velocities < stop_velocity) & (engine_speeds_out > min_engine_on_speed)
    if not b.any():
        return sh.NONE
    x = engine_speeds_out[b, None]
    eps = defaults.dfl.functions.define_idle_model_detector.EPS
    model = _IdleDetector(eps=eps)
    model.fit(x)

    return model


def identify_idle_engine_speed_median(idle_model_detector):
    """
    Identifies idle engine speed [RPM].

    :param idle_model_detector:
        Idle engine speed model detector.
    :type idle_model_detector: _IdleDetector

    :return:
        Idle engine speed [RPM].
    :rtype: float
    """
    imd = idle_model_detector
    return np.median(imd.cluster_centers_[imd.labels_])


def identify_idle_engine_speed_std(
        idle_model_detector, engine_speeds_out, idle_engine_speed_median,
        min_engine_on_speed):
    """
    Identifies standard deviation of idle engine speed [RPM].

    :param idle_model_detector:
        Idle engine speed model detector.
    :type idle_model_detector: _IdleDetector

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param idle_engine_speed_median:
        Idle engine speed [RPM].
    :type idle_engine_speed_median: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Standard deviation of idle engine speed [RPM].
    :rtype: float
    """
    b = idle_model_detector.predict([(idle_engine_speed_median,)],
                                    set_outliers=False)
    b = idle_model_detector.predict(engine_speeds_out[:, None]) == b
    b &= (engine_speeds_out > min_engine_on_speed)
    idle_std = defaults.dfl.functions.identify_idle_engine_speed_std.MIN_STD
    if not b.any():
        return idle_std

    s = np.sqrt(np.mean((engine_speeds_out[b] - idle_engine_speed_median) ** 2))

    p = defaults.dfl.functions.identify_idle_engine_speed_std.MAX_STD_PERC
    return min(max(s, idle_std), idle_engine_speed_median * p)


# not used.
def identify_upper_bound_engine_speed(
        gears, engine_speeds_out, idle_engine_speed):
    """
    Identifies upper bound engine speed [RPM].

    It is used to correct the gear prediction for constant accelerations (see
    :func:`co2mpas.model.physical.at_gear.
    correct_gear_upper_bound_engine_speed`).

    This is evaluated as the median value plus 0.67 standard deviation of the
    filtered cycle engine speed (i.e., the engine speeds when engine speed >
    minimum engine speed plus 0.67 standard deviation and gear < maximum gear).

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param engine_speeds_out:
         Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :returns:
        Upper bound engine speed [RPM].
    :rtype: float

    .. note:: Assuming a normal distribution then about 68 percent of the data
       values are within 0.67 standard deviation of the mean.
    """

    max_gear = max(gears)

    idle_speed = idle_engine_speed[1]

    dom = (engine_speeds_out > idle_speed) & (gears < max_gear)

    m, sd = co2_utl.reject_outliers(engine_speeds_out[dom])

    return m + sd * 0.674490


def calculate_engine_max_torque(
        engine_max_power, engine_speed_at_max_power, ignition_type):
    """
    Calculates engine nominal torque [N*m].

    :param engine_max_power:
        Engine nominal power [kW].
    :type engine_max_power: float

    :param engine_speed_at_max_power:
        Engine speed at engine nominal power [RPM].
    :type engine_speed_at_max_power: float

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :return:
        Engine nominal torque [N*m].
    :rtype: float
    """

    c = defaults.dfl.functions.calculate_engine_max_torque.PARAMS[ignition_type]
    pi = math.pi
    return engine_max_power / engine_speed_at_max_power * 30000.0 / pi * c


def calculate_engine_max_power_v1(
        engine_max_torque, engine_speed_at_max_power, ignition_type):
    """
    Calculates engine nominal power [kW].

    :param engine_max_torque:
        Engine nominal torque [N*m].
    :type engine_max_torque: float

    :param engine_speed_at_max_power:
        Engine speed at engine nominal power [RPM].
    :type engine_speed_at_max_power: float

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :return:
        Engine nominal power [kW].
    :rtype: float
    """

    c = calculate_engine_max_torque(1, engine_speed_at_max_power, ignition_type)

    return engine_max_torque / c


def calculate_engine_speeds_out_hot(
        gear_box_speeds_in, on_engine, idle_engine_speed):
    """
    Calculates the engine speed at hot condition [RPM].

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array, float

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array, bool

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Engine speed at hot condition [RPM].
    :rtype: numpy.array, float
    """

    if isinstance(gear_box_speeds_in, float):
        s = max(idle_engine_speed[0], gear_box_speeds_in) if on_engine else 0
    else:
        s = np.where(
            on_engine, np.maximum(gear_box_speeds_in, idle_engine_speed[0]), 0
        )

    return s


def calculate_engine_speeds_out(
        on_engine, idle_engine_speed, engine_speeds_out_hot, *delta_speeds):
    """
    Calculates the engine speed [RPM].

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param delta_speeds:
        Delta engine speed [RPM].
    :type delta_speeds: (numpy.array,)

    :return:
        Engine speed [RPM].
    :rtype: numpy.array
    """

    speeds = engine_speeds_out_hot.copy()
    s = speeds[on_engine]
    for delta in delta_speeds:
        s += delta[on_engine]

    dn = idle_engine_speed[0]

    s[s < dn] = dn

    speeds[on_engine] = s

    return speeds


def calculate_uncorrected_engine_powers_out(
        times, engine_moment_inertia, clutch_tc_powers, engine_speeds_out,
        on_engine, auxiliaries_power_losses, gear_box_type, on_idle,
        alternator_powers_demand=None):
    """
    Calculates the uncorrected engine power [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_moment_inertia:
        Engine moment of inertia [kg*m2].
    :type engine_moment_inertia: float

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param auxiliaries_power_losses:
        Engine torque losses due to engine auxiliaries [N*m].
    :type auxiliaries_power_losses: numpy.array

    :param gear_box_type:
        Gear box type (manual or automatic or cvt).
    :type gear_box_type: str

    :param on_idle:
        If the engine is on idle [-].
    :type on_idle: numpy.array

    :param alternator_powers_demand:
        Alternator power demand to the engine [kW].
    :type alternator_powers_demand: numpy.array, optional

    :return:
        Uncorrected engine power [kW].
    :rtype: numpy.array
    """

    p, b = np.zeros_like(clutch_tc_powers, dtype=float), on_engine
    p[b] = clutch_tc_powers[b]

    if gear_box_type == 'manual':
        p[on_idle & (p < 0)] = 0.0

    p[b] += auxiliaries_power_losses[b]

    if alternator_powers_demand is not None:
        p[b] += alternator_powers_demand[b]

    p_inertia = engine_moment_inertia / 2000 * (2 * math.pi / 60) ** 2
    p += p_inertia * co2_utl.derivative(times, engine_speeds_out) ** 2

    return p


def calculate_min_available_engine_powers_out(
        engine_stroke, engine_capacity, friction_params, engine_speeds_out):
    """
    Calculates the minimum available engine power (i.e., engine motoring curve).

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param friction_params:
        Engine initial friction params l & l2 [-].
    :type friction_params: float, float

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array | float

    :return:
        Minimum available engine power [kW].
    :rtype: numpy.array | float
    """

    l, l2 = np.array(friction_params) * (engine_capacity / 1200000.0)
    l2 *= (engine_stroke / 30000.0) ** 2

    return (l2 * engine_speeds_out * engine_speeds_out + l) * engine_speeds_out


def calculate_max_available_engine_powers_out(
        full_load_curve, engine_speeds_out):
    """
    Calculates the maximum available engine power [kW].

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array | float

    :return:
        Maximum available engine power [kW].
    :rtype: numpy.array | float
    """

    return full_load_curve(engine_speeds_out)


def correct_engine_powers_out(
        max_available_engine_powers_out, min_available_engine_powers_out,
        uncorrected_engine_powers_out):
    """
    Corrects the engine powers out according to the available powers and
    returns the missing and brake power [kW].

    :param max_available_engine_powers_out:
        Maximum available engine power [kW].
    :type max_available_engine_powers_out: numpy.array

    :param min_available_engine_powers_out:
        Minimum available engine power [kW].
    :type min_available_engine_powers_out: numpy.array

    :param uncorrected_engine_powers_out:
        Uncorrected engine power [kW].
    :type uncorrected_engine_powers_out: numpy.array

    :return:
        Engine, missing, and braking powers [kW].
    :rtype: numpy.array, numpy.array, numpy.array
    """

    ul, dl = max_available_engine_powers_out, min_available_engine_powers_out
    p = uncorrected_engine_powers_out

    up, dn = ul < p, dl > p

    missing_powers, brake_powers = np.zeros_like(p), np.zeros_like(p)
    missing_powers[up], brake_powers[dn] = p[up] - ul[up], dl[dn] - p[dn]

    return np.where(up, ul, np.where(dn, dl, p)), missing_powers, brake_powers


def calculate_braking_powers(
        engine_speeds_out, engine_torques_in, friction_powers):
    """
    Calculates braking power [kW].

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_torques_in:
        Engine torque out [N*m].
    :type engine_torques_in: numpy.array

    :param friction_powers:
        Friction power [kW].
    :type friction_powers: numpy.array

    :return:
        Braking powers [kW].
    :rtype: numpy.array
    """

    bp = engine_torques_in * engine_speeds_out * (math.pi / 30000.0)

    # noinspection PyUnresolvedReferences
    bp[bp < friction_powers] = 0

    return bp


def calculate_friction_powers(
        engine_speeds_out, piston_speeds, engine_loss_parameters,
        engine_capacity):
    """
    Calculates friction power [kW].

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param piston_speeds:
        Piston speed [m/s].
    :type piston_speeds: numpy.array

    :param engine_loss_parameters:
        Engine parameter (loss, loss2).
    :type engine_loss_parameters: (float, float)

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :return:
        Friction powers [kW].
    :rtype: numpy.array
    """

    loss, loss2 = engine_loss_parameters
    cap, es = engine_capacity, engine_speeds_out

    # indicative_friction_powers
    return (loss2 * piston_speeds ** 2 + loss) * es * (cap / 1200000.0)


def calculate_mean_piston_speeds(engine_speeds_out, engine_stroke):
    """
    Calculates mean piston speed [m/sec].

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :return:
        Mean piston speed vector [m/s].
    :rtype: numpy.array | float
    """

    return (engine_stroke / 30000.0) * engine_speeds_out


def calculate_engine_type(ignition_type, engine_is_turbo):
    """
    Calculates the engine type (gasoline turbo, gasoline natural aspiration,
    diesel).

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :param engine_is_turbo:
        If the engine is equipped with any kind of charging.
    :type engine_is_turbo: bool

    :return:
        Engine type (positive turbo, positive natural aspiration, compression).
    :rtype: str
    """

    engine_type = ignition_type

    if ignition_type == 'positive':
        engine_type = 'turbo' if engine_is_turbo else 'natural aspiration'
        engine_type = '%s %s' % (ignition_type, engine_type)

    return engine_type


def calculate_engine_moment_inertia(engine_capacity, ignition_type):
    """
    Calculates engine moment of inertia [kg*m2].

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :return:
        Engine moment of inertia [kg*m2].
    :rtype: float
    """
    # noinspection PyPep8Naming
    PARAMS = defaults.dfl.functions.calculate_engine_moment_inertia.PARAMS

    return (0.05 + 0.1 * engine_capacity / 1000.0) * PARAMS[ignition_type]


def calculate_auxiliaries_torque_losses(
        times, auxiliaries_torque_loss, engine_capacity):
    """
    Calculates engine torque losses due to engine auxiliaries [N*m].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param auxiliaries_torque_loss:
        Constant torque loss due to engine auxiliaries [N*m].
    :type auxiliaries_torque_loss: (float, float)

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :return:
        Engine torque losses due to engine auxiliaries [N*m].
    :rtype: numpy.array
    """
    m, q = auxiliaries_torque_loss
    return np.ones_like(times, dtype=float) * (m * engine_capacity / 1000.0 + q)


def calculate_auxiliaries_power_losses(
        auxiliaries_torque_losses, engine_speeds_out, on_engine,
        auxiliaries_power_loss):
    """
    Calculates engine power losses due to engine auxiliaries [kW].

    :param auxiliaries_torque_losses:
        Engine torque losses due to engine auxiliaries [N*m].
    :type auxiliaries_torque_losses: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param auxiliaries_power_loss:
        Constant power loss due to engine auxiliaries [kW].
    :type auxiliaries_power_loss: float

    :return:
        Engine power losses due to engine auxiliaries [kW].
    :rtype: numpy.array
    """

    from ..wheels import calculate_wheel_powers
    p = calculate_wheel_powers(auxiliaries_torque_losses, engine_speeds_out)
    if auxiliaries_power_loss:
        p[on_engine] += auxiliaries_power_loss
    return p


def check_vehicle_has_sufficient_power(times, missing_powers):
    """
    Checks if the vehicle has sufficient power.

    :param times:
        Time vector [s].
    :type times: numpy.array
    
    :param missing_powers:
        Missing powers [kW].
    :type missing_powers: numpy.array

    :return:
        The cycle's percentage in which the vehicle has sufficient power [%].
    :rtype: float
    """
    w = np.zeros_like(times, dtype=float)
    t = (times[:-1] + times[1:]) / 2
    # noinspection PyUnresolvedReferences
    w[0], w[1:-1], w[-1] = t[0] - times[0], np.diff(t), times[-1] - t[-1]
    return 1 - np.average(missing_powers != 0, weights=w)


def default_ignition_type_v1(fuel_type):
    """
    Returns the default ignition type according to the fuel type.

    :param fuel_type:
        Fuel type (diesel, gasoline, LPG, NG, ethanol, methanol, biodiesel,
        propane).
    :type fuel_type: str

    :return:
        Engine ignition type (positive or compression).
    :rtype: str
    """

    if 'diesel' in fuel_type:
        return 'compression'
    return 'positive'


def default_ignition_type(engine_type):
    """
    Returns the default ignition type according to the fuel type.

    :param engine_type:
        Engine type (positive turbo, positive natural aspiration, compression).
    :type engine_type: str

    :return:
        Engine ignition type (positive or compression).
    :rtype: str
    """

    if 'compression' in engine_type:
        return 'compression'
    return 'positive'


def identify_engine_max_speed(full_load_speeds):
    """
    Identifies the maximum allowed engine speed [RPM].

    :param full_load_speeds:
        T1 map speed vector [RPM].
    :type full_load_speeds: numpy.array

    :return:
        Maximum allowed engine speed [RPM].
    :rtype: float
    """
    return np.max(full_load_speeds)


def define_full_bmep_curve(
        full_load_speeds, full_load_curve, min_engine_on_speed,
        engine_capacity, engine_stroke, idle_engine_speed, engine_max_speed):
    """
    Defines the vehicle full bmep curve.

    :param full_load_speeds:
        T1 map speed vector [RPM].
    :type full_load_speeds: numpy.array

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :return:
        Vehicle full bmep curve.
    :rtype: function
    """

    speeds = np.unique(np.append(
        full_load_speeds, [idle_engine_speed[0], engine_max_speed]
    ))
    from .co2_emission import calculate_brake_mean_effective_pressures
    p = calculate_brake_mean_effective_pressures(
        speeds, full_load_curve(speeds), engine_capacity,
        min_engine_on_speed)

    s = calculate_mean_piston_speeds(speeds, engine_stroke)

    return functools.partial(np.interp, xp=s, fp=p, left=0, right=0)


def identify_idle_engine_speed():
    """
    Defines the model to identify idle engine speed median and std.

    .. dispatcher:: d

        >>> d = identify_idle_engine_speed()

    :return:
        The model to identify idle engine speed median and std.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='calculate_idle_engine_speed',
        description='Identify idle engine speed median and std.'
    )

    d.add_function(
        function=define_idle_model_detector,
        inputs=['velocities', 'engine_speeds_out', 'stop_velocity',
                'min_engine_on_speed'],
        outputs=['idle_model_detector']
    )

    # identify idle engine speed
    d.add_function(
        function=identify_idle_engine_speed_median,
        inputs=['idle_model_detector'],
        outputs=['idle_engine_speed_median']
    )

    # identify idle engine speed
    d.add_function(
        function=identify_idle_engine_speed_std,
        inputs=['idle_model_detector', 'engine_speeds_out',
                'idle_engine_speed_median', 'min_engine_on_speed'],
        outputs=['idle_engine_speed_std']
    )
    return d


# noinspection PyMissingOrEmptyDocstring
class EngineModel:
    key_outputs = [
        'on_engine', 'engine_starts', 'engine_speeds_out_hot',
        'engine_coolant_temperatures'
    ]

    types = {
        float: {'engine_speeds_out_hot', 'engine_coolant_temperatures'},
        bool: {'on_engine', 'engine_starts'}
    }

    def __init__(self,
                 start_stop_prediction_model=None, idle_engine_speed=None,
                 engine_temperature_prediction_model=None, outputs=None):
        self.start_stop_prediction_model = start_stop_prediction_model
        self.idle_engine_speed = idle_engine_speed
        self.engine_temperature_prediction_model = \
            engine_temperature_prediction_model
        self._outputs = outputs or {}
        self.outputs = None

    def __call__(self, times, *args, **kwargs):
        self.set_outputs(times.shape[0])
        for _ in self.yield_results(times, *args, **kwargs):
            pass
        return sh.selector(self.key_outputs, self.outputs, output_type='list')

    def yield_on_start(self, times, velocities, accelerations,
                       engine_coolant_temperatures, state_of_charges, gears):
        yield from self.start_stop_prediction_model.yield_results(
            times, velocities, accelerations, engine_coolant_temperatures,
            state_of_charges, gears
        )

    def yield_speed(self, on_engine, gear_box_speeds_in):
        key = 'engine_speeds_out_hot'
        if self._outputs is not None and key in self._outputs:
            yield from self._outputs[key]
        else:
            idle = self.idle_engine_speed
            for on_eng, gb_s in zip(on_engine, gear_box_speeds_in):
                yield calculate_engine_speeds_out_hot(gb_s, on_eng, idle)

    def yield_thermal(self, times, accelerations, final_drive_powers_in,
                      engine_speeds_out_hot):
        yield from self.engine_temperature_prediction_model.yield_results(
            times, accelerations, final_drive_powers_in, engine_speeds_out_hot
        )

    def set_outputs(self, n, outputs=None):
        if outputs is None:
            outputs = {}
        self.engine_temperature_prediction_model.set_outputs(n, outputs)
        self.start_stop_prediction_model.set_outputs(n, outputs)
        outputs.update(self._outputs or {})
        for t, names in self.types.items():
            names = names - set(outputs)
            if names:
                outputs.update(zip(names, np.empty((len(names), n), dtype=t)))

        self.outputs = outputs

    def yield_results(self, times, velocities, accelerations,
                      final_drive_powers_in, gears, gear_box_speeds_in):
        outputs = self.outputs

        ss_gen = self.yield_on_start(
            times, velocities, accelerations,
            outputs['engine_coolant_temperatures'], outputs['state_of_charges'],
            gears
        )

        s_gen = self.yield_speed(outputs['on_engine'], gear_box_speeds_in)

        t_gen = self.yield_thermal(
            times, accelerations, final_drive_powers_in,
            outputs['engine_speeds_out_hot']
        )
        eng_temp = outputs['engine_coolant_temperatures']
        eng_temp[0] = next(t_gen)
        for i, on_eng in enumerate(ss_gen):
            # if e[-1] < min_soc and not on_eng[0]:
            #    on_eng[0], on_eng[1] = True, not eng[0]

            outputs['on_engine'][i], outputs['engine_starts'][i] = on_eng

            outputs['engine_speeds_out_hot'][i] = eng_s = next(s_gen)
            try:
                eng_temp[i + 1] = next(t_gen)
            except IndexError:
                pass
            yield on_eng[0], on_eng[1], eng_s, eng_temp[i]


def define_engine_prediction_model(
        start_stop_prediction_model, idle_engine_speed,
        engine_temperature_prediction_model):
    """
    Defines the engine prediction model.

    :param start_stop_prediction_model:
        Engine start/stop prediction model.
    :type start_stop_prediction_model: EngineStartStopModel

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_temperature_prediction_model:
        Engine temperature prediction model.
    :type engine_temperature_prediction_model: .thermal.EngineTemperatureModel

    :return:
        Engine prediction model.
    :rtype: EngineModel
    """
    model = EngineModel(
        start_stop_prediction_model, idle_engine_speed,
        engine_temperature_prediction_model
    )

    return model


def define_fuel_type_and_is_hybrid(obd_fuel_type_code):
    """
    Defines engine fuel type and if the vehicle is hybrid.

    :param obd_fuel_type_code:
        OBD fuel type of the vehicle [-].
    :type obd_fuel_type_code: int

    :return:
        Engine fuel type and if the vehicle is hybrid.
    :rtype: (str, bool)
    """

    i = int(obd_fuel_type_code)
    dfl = defaults.dfl.functions.define_fuel_type_and_is_hybrid
    return dfl.fuel_type.get(i, sh.NONE), dfl.is_hybrid.get(i, sh.NONE)


def engine():
    """
    Defines the engine model.

    .. dispatcher:: d

        >>> d = engine()

    :return:
        The engine model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Engine',
        description='Models the vehicle engine.'
    )

    d.add_function(
        function=define_fuel_type_and_is_hybrid,
        inputs=['obd_fuel_type_code'],
        outputs=['fuel_type', 'is_hybrid']
    )

    d.add_function(
        function=calculate_engine_mass,
        inputs=['ignition_type', 'engine_max_power'],
        outputs=['engine_mass']
    )

    d.add_function(
        function=calculate_engine_heat_capacity,
        inputs=['engine_mass'],
        outputs=['engine_heat_capacity']
    )

    d.add_function(
        function=default_ignition_type,
        inputs=['engine_type'],
        outputs=['ignition_type']
    )

    d.add_function(
        function=default_ignition_type_v1,
        inputs=['fuel_type'],
        outputs=['ignition_type'],
        weight=1
    )

    d.add_function(
        function=define_full_bmep_curve,
        inputs=['full_load_speeds', 'full_load_curve', 'min_engine_on_speed',
                'engine_capacity', 'engine_stroke', 'idle_engine_speed',
                'engine_max_speed'],
        outputs=['full_bmep_curve']
    )

    d.add_data(
        data_id='is_cycle_hot',
        default_value=defaults.dfl.values.is_cycle_hot
    )

    from ..wheels import calculate_wheel_powers, calculate_wheel_torques
    d.add_function(
        function_id='calculate_full_load_powers',
        function=calculate_wheel_powers,
        inputs=['full_load_torques', 'full_load_speeds'],
        outputs=['full_load_powers']
    )

    d.add_function(
        function_id='calculate_full_load_speeds',
        function=calculate_wheel_torques,
        inputs=['full_load_powers', 'full_load_torques'],
        outputs=['full_load_speeds']
    )

    d.add_function(
        function=default_engine_max_speed,
        inputs=['ignition_type', 'idle_engine_speed',
                'engine_speed_at_max_power'],
        outputs=['engine_max_speed'],
        weight=20
    )

    d.add_function(
        function=default_full_load_speeds_and_powers,
        inputs=['ignition_type', 'engine_max_power',
                'engine_speed_at_max_power', 'idle_engine_speed',
                'engine_max_speed'],
        outputs=['full_load_speeds', 'full_load_powers']
    )

    d.add_function(
        function=identify_engine_speed_at_max_power,
        inputs=['full_load_speeds', 'full_load_powers'],
        outputs=['engine_speed_at_max_power']
    )

    d.add_function(
        function=calculate_engine_max_power,
        inputs=['full_load_curve', 'engine_speed_at_max_power'],
        outputs=['engine_max_power']
    )

    d.add_function(
        function=define_full_load_curve,
        inputs=['full_load_speeds', 'full_load_powers', 'idle_engine_speed',
                'engine_max_speed'],
        outputs=['full_load_curve']
    )

    d.add_function(
        function=identify_engine_max_speed,
        inputs=['full_load_speeds'],
        outputs=['engine_max_speed']
    )

    # Idle engine speed
    d.add_data(
        data_id='idle_engine_speed_median',
        description='Idle engine speed [RPM].'
    )

    # default value
    d.add_data(
        data_id='idle_engine_speed_std',
        default_value=defaults.dfl.values.idle_engine_speed_std,
        initial_dist=20,
        description='Standard deviation of idle engine speed [RPM].'
    )

    d.add_dispatcher(
        dsp=identify_idle_engine_speed(),
        inputs=(
            'engine_speeds_out', 'idle_engine_speed_median',
            'min_engine_on_speed', 'stop_velocity', 'velocities'),
        outputs=('idle_engine_speed_median', 'idle_engine_speed_std')
    )

    # set idle engine speed tuple
    d.add_function(
        function=sh.bypass,
        inputs=['idle_engine_speed_median', 'idle_engine_speed_std'],
        outputs=['idle_engine_speed']
    )

    # set idle engine speed tuple
    d.add_function(
        function=sh.bypass,
        inputs=['idle_engine_speed'],
        outputs=['idle_engine_speed_median', 'idle_engine_speed_std']
    )

    from .thermal import thermal
    d.add_dispatcher(
        include_defaults=True,
        dsp=thermal(),
        dsp_id='thermal',
        inputs=(
            'accelerations', 'engine_coolant_temperatures',
            'engine_speeds_out_hot', 'engine_temperature_regression_model',
            'engine_thermostat_temperature',
            'engine_thermostat_temperature_window', 'final_drive_powers_in',
            'idle_engine_speed', 'initial_engine_temperature',
            'max_engine_coolant_temperature', 'on_engine', 'times'),
        outputs=(
            'engine_temperature_derivatives',
            'engine_temperature_regression_model',
            'engine_thermostat_temperature',
            'engine_thermostat_temperature_window',
            'initial_engine_temperature', 'max_engine_coolant_temperature',
            'engine_temperature_prediction_model')
    )

    d.add_function(
        function=calculate_engine_max_torque,
        inputs=['engine_max_power', 'engine_speed_at_max_power',
                'ignition_type'],
        outputs=['engine_max_torque']
    )

    d.add_function(
        function_id='calculate_engine_max_power_v2',
        function=calculate_engine_max_torque,
        inputs=['engine_max_torque', 'engine_speed_at_max_power',
                'ignition_type'],
        outputs=['engine_max_power']
    )

    from .start_stop import start_stop
    d.add_dispatcher(
        include_defaults=True,
        dsp=start_stop(),
        dsp_id='start_stop',
        inputs=(
            'accelerations', 'correct_start_stop_with_gears',
            'engine_coolant_temperatures', 'engine_speeds_out', 'engine_starts',
            'gear_box_type', 'gears', 'has_start_stop', 'idle_engine_speed',
            'is_hybrid', 'min_time_engine_on_after_start', 'on_engine',
            'start_stop_activation_time', 'start_stop_model',
            'state_of_charges', 'times', 'use_basic_start_stop', 'velocities'),
        outputs=(
            'correct_start_stop_with_gears', 'engine_starts', 'on_engine',
            'start_stop_model', 'use_basic_start_stop',
            'start_stop_prediction_model', 'start_stop_activation_time')
    )

    d.add_data(
        data_id='plateau_acceleration',
        default_value=defaults.dfl.values.plateau_acceleration
    )

    d.add_function(
        function=calculate_engine_speeds_out_hot,
        inputs=['gear_box_speeds_in', 'on_engine', 'idle_engine_speed'],
        outputs=['engine_speeds_out_hot']
    )

    d.add_function(
        function=identify_on_idle,
        inputs=['velocities', 'engine_speeds_out_hot', 'gears', 'stop_velocity',
                'min_engine_on_speed'],
        outputs=['on_idle']
    )

    from .cold_start import cold_start
    d.add_dispatcher(
        dsp=cold_start(),
        inputs=(
            'cold_start_speed_model', 'cold_start_speeds_phases',
            'engine_coolant_temperatures', 'engine_speeds_out',
            'engine_speeds_out_hot', 'engine_thermostat_temperature',
            'idle_engine_speed', 'on_engine', 'on_idle'),
        outputs=(
            'cold_start_speed_model', 'cold_start_speeds_delta',
            'cold_start_speeds_phases')
    )

    d.add_function(
        function=calculate_engine_speeds_out,
        inputs=['on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
                'cold_start_speeds_delta', 'clutch_tc_speeds_delta'],
        outputs=['engine_speeds_out']
    )

    d.add_function(
        function=calculate_uncorrected_engine_powers_out,
        inputs=['times', 'engine_moment_inertia', 'clutch_tc_powers',
                'engine_speeds_out', 'on_engine', 'auxiliaries_power_losses',
                'gear_box_type', 'on_idle', 'alternator_powers_demand'],
        outputs=['uncorrected_engine_powers_out']
    )

    d.add_function(
        function=calculate_min_available_engine_powers_out,
        inputs=['engine_stroke', 'engine_capacity', 'initial_friction_params',
                'engine_speeds_out'],
        outputs=['min_available_engine_powers_out']
    )

    d.add_function(
        function=calculate_max_available_engine_powers_out,
        inputs=['full_load_curve', 'engine_speeds_out'],
        outputs=['max_available_engine_powers_out']
    )

    d.add_function(
        function=correct_engine_powers_out,
        inputs=['max_available_engine_powers_out',
                'min_available_engine_powers_out',
                'uncorrected_engine_powers_out'],
        outputs=['engine_powers_out', 'missing_powers', 'brake_powers']
    )

    d.add_function(
        function=check_vehicle_has_sufficient_power,
        inputs=['times', 'missing_powers'],
        outputs=['has_sufficient_power']
    )

    d.add_function(
        function=calculate_mean_piston_speeds,
        inputs=['engine_speeds_out', 'engine_stroke'],
        outputs=['mean_piston_speeds']
    )

    d.add_data(
        data_id='engine_is_turbo',
        default_value=defaults.dfl.values.engine_is_turbo
    )

    d.add_function(
        function=calculate_engine_type,
        inputs=['ignition_type', 'engine_is_turbo'],
        outputs=['engine_type']
    )

    d.add_function(
        function=calculate_engine_moment_inertia,
        inputs=['engine_capacity', 'ignition_type'],
        outputs=['engine_moment_inertia']
    )

    d.add_data(
        data_id='auxiliaries_torque_loss',
        default_value=defaults.dfl.values.auxiliaries_torque_loss
    )

    d.add_data(
        data_id='auxiliaries_power_loss',
        default_value=defaults.dfl.values.auxiliaries_power_loss
    )

    d.add_function(
        function=calculate_auxiliaries_torque_losses,
        inputs=['times', 'auxiliaries_torque_loss', 'engine_capacity'],
        outputs=['auxiliaries_torque_losses']
    )

    d.add_function(
        function=calculate_auxiliaries_power_losses,
        inputs=['auxiliaries_torque_losses', 'engine_speeds_out', 'on_engine',
                'auxiliaries_power_loss'],
        outputs=['auxiliaries_power_losses']
    )

    from .co2_emission import co2_emission
    d.add_dispatcher(
        include_defaults=True,
        dsp=co2_emission(),
        dsp_id='CO2_emission_model',
        inputs=(
            'accelerations', 'active_cylinder_ratios', 'angle_slopes',
            'calibration_status', 'co2_emission_EUDC', 'co2_emission_UDC',
            'co2_emission_extra_high', 'co2_emission_high', 'co2_emission_low',
            'co2_emission_medium', 'co2_emissions',
            'co2_normalization_references', 'co2_params',
            'enable_phases_willans', 'enable_willans', 'engine_capacity',
            'engine_coolant_temperatures', 'engine_fuel_lower_heating_value',
            'engine_has_cylinder_deactivation',
            'engine_has_variable_valve_actuation', 'engine_max_speed',
            'engine_powers_out', 'engine_speeds_out', 'engine_stroke',
            'engine_thermostat_temperature',
            'engine_thermostat_temperature_window', 'engine_type',
            'fuel_carbon_content', 'fuel_carbon_content_percentage',
            'fuel_consumptions', 'fuel_density', 'fuel_type', 'full_bmep_curve',
            'has_exhausted_gas_recirculation', 'has_lean_burn',
            'has_periodically_regenerating_systems',
            'has_selective_catalytic_reduction', 'idle_engine_speed',
            'initial_engine_temperature', 'is_cycle_hot', 'ki_additive',
            'ki_multiplicative',
            'mean_piston_speeds', 'min_engine_on_speed', 'missing_powers',
            'motive_powers', 'engine_n_cylinders', 'on_engine',
            'phases_integration_times', 'stop_velocity', 'times', 'velocities',
            {'co2_params_calibrated': ('co2_params_calibrated', 'co2_params'),
             'engine_idle_fuel_consumption': (
                 'engine_idle_fuel_consumption',
                 'idle_fuel_consumption_initial_guess')}
        ),
        outputs=(
            'active_cylinders', 'active_exhausted_gas_recirculations',
            'active_lean_burns', 'active_variable_valves',
            'after_treatment_temperature_threshold', 'calibration_status',
            'co2_emission_value', 'co2_emissions', 'co2_emissions_model',
            'co2_error_function_on_emissions', 'co2_error_function_on_phases',
            'co2_params_calibrated', 'co2_params_initial_guess',
            'declared_co2_emission_value', 'engine_fuel_lower_heating_value',
            'engine_idle_fuel_consumption', 'extended_phases_co2_emissions',
            'extended_phases_integration_times', 'fuel_carbon_content',
            'fuel_carbon_content_percentage', 'fuel_consumptions',
            'fuel_density', 'has_exhausted_gas_recirculation',
            'identified_co2_emissions', 'initial_friction_params',
            'ki_additive', 'ki_multiplicative',
            'optimal_efficiency', 'phases_co2_emissions',
            'phases_fuel_consumptions', 'phases_willans_factors',
            'willans_factors', 'co2_rescaling_scores'),
        inp_weight={'co2_params': defaults.dfl.EPS}
    )

    d.add_function(
        function=define_engine_prediction_model,
        inputs=['start_stop_prediction_model', 'idle_engine_speed',
                'engine_temperature_prediction_model'],
        outputs=['engine_prediction_model'],
        weight=4000
    )

    return d
