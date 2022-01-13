# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the engine.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.engine

.. autosummary::
    :nosignatures:
    :toctree: engine/

    fc
    idle
    thermal
"""
import math
import functools
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
from .idle import dsp as _idle
from .thermal import dsp as _thermal
from .fc import dsp as _fc

dsp = sh.BlueDispatcher(name='Engine', description='Models the vehicle engine.')


@sh.add_function(dsp, outputs=['fuel_type', 'is_hybrid'])
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
    d = dfl.functions.define_fuel_type_and_is_hybrid
    return d.fuel_type.get(i, sh.NONE), d.is_hybrid.get(i, sh.NONE)


@sh.add_function(dsp, outputs=['engine_mass'])
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

    par = dfl.functions.calculate_engine_mass.PARAMS
    _mass_coeff = par['mass_coeff']
    m, q = par['mass_reg_coeff']
    # Engine mass empirical formula based on web data found for engines weighted
    # according DIN 70020-GZ
    # kg
    return (m * engine_max_power + q) * _mass_coeff[ignition_type]


@sh.add_function(dsp, outputs=['engine_heat_capacity'])
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

    par = dfl.functions.calculate_engine_heat_capacity.PARAMS
    mp, hc = par['heated_mass_percentage'], par['heat_capacity']

    return engine_mass * np.sum([hc[k] * v for k, v in mp.items()])


@sh.add_function(dsp, outputs=['ignition_type'])
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


@sh.add_function(dsp, outputs=['ignition_type'], weight=1)
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


@sh.add_function(dsp, outputs=['full_bmep_curve'])
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
    from .fc import calculate_brake_mean_effective_pressures
    p = calculate_brake_mean_effective_pressures(
        speeds, full_load_curve(speeds), engine_capacity,
        min_engine_on_speed)

    s = calculate_mean_piston_speeds(speeds, engine_stroke)

    return functools.partial(np.interp, xp=s, fp=p, left=0, right=0)


dsp.add_data('is_cycle_hot', dfl.values.is_cycle_hot)


@sh.add_function(dsp, outputs=['full_load_powers'])
def calculate_full_load_powers(full_load_torques, full_load_speeds):
    """
    Calculates T1 map power vector [kW].

    :param full_load_torques:
        T1 map torque vector [N*m].
    :type full_load_torques: numpy.array

    :param full_load_speeds:
        T1 map speed vector [RPM].
    :type full_load_speeds: numpy.array

    :return:
        T1 map power vector [kW].
    :rtype: numpy.array
    """
    from ..wheels import calculate_wheel_powers
    return calculate_wheel_powers(full_load_torques, full_load_speeds)


@sh.add_function(dsp, outputs=['full_load_speeds'])
def calculate_full_load_speeds(full_load_powers, full_load_torques):
    """
    Calculates T1 map speed vector [RPM].

    :param full_load_powers:
        T1 map power vector [kW].
    :type full_load_powers: numpy.array

    :param full_load_torques:
        T1 map torque vector [N*m].
    :type full_load_torques: numpy.array

    :return:
        T1 map speed vector [RPM].
    :rtype: numpy.array
    """
    from ..wheels import calculate_wheel_torques
    return calculate_wheel_torques(full_load_powers, full_load_torques)


@sh.add_function(dsp, outputs=['engine_max_speed'], weight=20)
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
    fl = dfl.functions.default_full_load_speeds_and_powers.FULL_LOAD
    idl, r = idle_engine_speed[0], max(fl[ignition_type][0])
    return idl + r * (engine_speed_at_max_power - idl)


@sh.add_function(dsp, outputs=['full_load_speeds', 'full_load_powers'])
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
    from scipy.interpolate import InterpolatedUnivariateSpline as Spline

    idl = idle_engine_speed[0]

    full_load_speeds = np.unique(np.append(
        [engine_speed_at_max_power], np.linspace(idl, engine_max_speed)
    ))

    d = dfl.functions.default_full_load_speeds_and_powers
    full_load_powers = Spline(*d.FULL_LOAD[ignition_type], k=1)(
        (full_load_speeds - idl) / (engine_speed_at_max_power - idl)
    ) * engine_max_power

    return full_load_speeds, full_load_powers


@sh.add_function(dsp, outputs=['engine_speed_at_max_power'])
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


@sh.add_function(dsp, outputs=['engine_max_power'])
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


@sh.add_function(dsp, outputs=['full_load_curve'])
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
    fp = np.interp(xp, full_load_speeds, full_load_powers)
    return functools.partial(np.interp, xp=xp, fp=fp, left=0, right=0)


@sh.add_function(dsp, outputs=['engine_max_speed'])
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


@sh.add_function(dsp, outputs=['engine_max_torque'])
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

    c = dfl.functions.calculate_engine_max_torque.PARAMS[ignition_type]
    pi = math.pi
    return engine_max_power / engine_speed_at_max_power * (30000.0 / pi * c)


@sh.add_function(dsp, outputs=['engine_max_power'])
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


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_idle,
    inputs=(
        'idle_engine_speed_median', 'min_engine_on_speed', 'engine_speeds_out',
        'idle_engine_speed_std', 'idle_engine_speed', 'velocities',
        'stop_velocity'
    ),
    outputs=(
        'idle_engine_speed_median', 'idle_engine_speed_std',
        'idle_engine_speed'
    )
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_thermal,
    dsp_id='thermal',
    inputs=(
        'max_engine_coolant_temperature', 'initial_engine_temperature', 'times',
        'engine_thermostat_temperature_window', 'engine_thermostat_temperature',
        'engine_temperature_regression_model', 'idle_engine_speed', 'on_engine',
        'engine_coolant_temperatures', 'gear_box_powers_out', 'accelerations',
        'engine_speeds_out', 'velocities', 'after_treatment_warm_up_phases',
        'temperature_shift'
    ),
    outputs=(
        'engine_thermostat_temperature_window', 'engine_thermostat_temperature',
        'engine_temperature_regression_model', 'max_engine_coolant_temperature',
        'engine_temperature_derivatives', 'engine_coolant_temperatures',
        'initial_engine_temperature', 'engine_temperatures', 'temperature_shift'
    )
)


@sh.add_function(
    dsp,
    inputs=[
        'on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
        'after_treatment_speeds_delta', 'clutch_tc_speeds_delta'
    ],
    outputs=['engine_speeds_out']
)
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


@sh.add_function(dsp, outputs=['engine_inertia_powers_losses'])
def calculate_engine_inertia_powers_losses(
        times, engine_speeds_out, engine_moment_inertia):
    """
    Calculates the engine power losses due to inertia [kW].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_moment_inertia:
        Engine moment of inertia [kg*m2].
    :type engine_moment_inertia: float

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :return:
        Engine power losses due to inertia [kW].
    :rtype: numpy.array
    """
    t = times[:, None] + np.array([-1, 1])
    c = engine_moment_inertia / 2000 * (2 * math.pi / 60) ** 2 / 4
    return c * np.diff(np.interp(t, times, engine_speeds_out)).ravel() ** 2


dsp.add_data('belt_efficiency', dfl.values.belt_efficiency)
dsp.add_function(
    function=sh.bypass,
    inputs=['belt_efficiency'],
    outputs=['belt_mean_efficiency']
)


@sh.add_function(dsp, outputs=['gross_engine_powers_out'])
def calculate_gross_engine_powers_out(
        clutch_tc_powers, belt_efficiency, on_engine, gear_box_type, on_idle,
        alternator_powers, motor_p0_powers, motor_p1_powers,
        motor_p2_planetary_powers):
    """
    Calculates the gross engine power (pre-losses) [kW].

    :param clutch_tc_powers:
        Clutch or torque converter power [kW].
    :type clutch_tc_powers: numpy.array

    :param belt_efficiency:
        Belt efficiency [-].
    :type belt_efficiency: float

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param gear_box_type:
        Gear box type (manual or automatic or cvt).
    :type gear_box_type: str

    :param on_idle:
        If the engine is on idle [-].
    :type on_idle: numpy.array

    :param alternator_powers:
        Alternator power [kW].
    :type alternator_powers: numpy.array

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array

    :param motor_p1_powers:
        Power at motor P1 [kW].
    :type motor_p1_powers: numpy.array

    :param motor_p2_planetary_powers:
        Power at planetary motor P2 [kW].
    :type motor_p2_planetary_powers: numpy.array

    :return:
        Gross engine power (pre-losses) [kW].
    :rtype: numpy.array
    """

    p, b = np.zeros_like(clutch_tc_powers, dtype=float), on_engine
    p[b] = clutch_tc_powers[b]

    if gear_box_type == 'manual':
        p[on_idle & (p < 0)] = 0.0
    eff = np.where(motor_p0_powers[b] < 0, belt_efficiency, 1 / belt_efficiency)
    p[b] -= alternator_powers[b] + motor_p0_powers[b] * eff
    p[b] -= motor_p1_powers[b] + motor_p2_planetary_powers[b]
    return p


@sh.add_function(dsp, outputs=['clutch_tc_powers'])
def calculate_clutch_tc_powers(
        gross_engine_powers_out, belt_efficiency, alternator_powers,
        motor_p0_powers, motor_p1_powers, motor_p2_planetary_powers):
    """
    Calculates the clutch or torque converter power [kW] from gross engine kW.

    :param gross_engine_powers_out:
        Gross engine power (pre-losses) [kW].
    :type gross_engine_powers_out: numpy.array

    :param belt_efficiency:
        Belt efficiency [-].
    :type belt_efficiency: float

    :param alternator_powers:
        Alternator power [kW].
    :type alternator_powers: numpy.array

    :param motor_p0_powers:
        Power at motor P0 [kW].
    :type motor_p0_powers: numpy.array

    :param motor_p1_powers:
        Power at motor P1 [kW].
    :type motor_p1_powers: numpy.array

    :param motor_p2_planetary_powers:
        Power at planetary motor P2 [kW].
    :type motor_p2_planetary_powers: numpy.array

    :return:
        Clutch or torque converter power [kW].
    :rtype: numpy.array
    """
    p = np.where(motor_p0_powers < 0, belt_efficiency, 1 / belt_efficiency)
    p *= motor_p0_powers
    p += gross_engine_powers_out + alternator_powers + motor_p1_powers
    p += motor_p2_planetary_powers
    return p


@sh.add_function(dsp, outputs=['min_available_engine_powers_out'])
def calculate_min_available_engine_powers_out(
        engine_stroke, engine_capacity, initial_friction_params,
        engine_speeds_out):
    """
    Calculates the minimum available engine power (i.e., engine motoring curve).

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param initial_friction_params:
        Engine initial friction params l & l2 [-].
    :type initial_friction_params: float, float

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array | float

    :return:
        Minimum available engine power [kW].
    :rtype: numpy.array | float
    """

    l, l2 = np.array(initial_friction_params) * (engine_capacity / 1200000.0)
    l2 *= (engine_stroke / 30000.0) ** 2

    return (l2 * engine_speeds_out * engine_speeds_out + l) * engine_speeds_out


@sh.add_function(dsp, outputs=['max_available_engine_powers_out'])
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


@sh.add_function(dsp, outputs=['gross_engine_powers_out'])
def calculate_gross_engine_powers_out_v1(
        engine_powers_out, auxiliaries_power_losses,
        engine_inertia_powers_losses):
    """
    Calculates the gross engine power (pre-losses) [kW].

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param engine_inertia_powers_losses:
        Engine power losses due to inertia [kW].
    :type engine_inertia_powers_losses: numpy.array

    :param auxiliaries_power_losses:
        Engine power losses due to engine auxiliaries [kW].
    :type auxiliaries_power_losses: numpy.array

    :return:
        Gross engine power (pre-losses) [kW].
    :rtype: numpy.array
    """
    gross_engine_powers_out = engine_powers_out - auxiliaries_power_losses
    gross_engine_powers_out -= engine_inertia_powers_losses
    return gross_engine_powers_out


@sh.add_function(
    dsp, outputs=['engine_powers_out', 'missing_powers', 'brake_powers']
)
def correct_engine_powers_out(
        max_available_engine_powers_out, min_available_engine_powers_out,
        gross_engine_powers_out, auxiliaries_power_losses,
        engine_inertia_powers_losses):
    """
    Corrects the engine powers out according to the available powers and
    returns the missing and brake power [kW].

    :param max_available_engine_powers_out:
        Maximum available engine power [kW].
    :type max_available_engine_powers_out: numpy.array

    :param min_available_engine_powers_out:
        Minimum available engine power [kW].
    :type min_available_engine_powers_out: numpy.array

    :param engine_inertia_powers_losses:
        Engine power losses due to inertia [kW].
    :type engine_inertia_powers_losses: numpy.array

    :param auxiliaries_power_losses:
        Engine power losses due to engine auxiliaries [kW].
    :type auxiliaries_power_losses: numpy.array

    :param gross_engine_powers_out:
        Gross engine power (pre-losses) [kW].
    :type gross_engine_powers_out: numpy.array

    :return:
        Engine, missing, and braking powers [kW].
    :rtype: numpy.array, numpy.array, numpy.array
    """

    ul, dl = max_available_engine_powers_out, min_available_engine_powers_out
    p = gross_engine_powers_out + auxiliaries_power_losses
    p += engine_inertia_powers_losses
    up, dn = ul < p, dl > p

    missing_powers, brake_powers = np.zeros_like(p), np.zeros_like(p)
    missing_powers[up], brake_powers[dn] = p[up] - ul[up], dl[dn] - p[dn]

    return np.where(up, ul, np.where(dn, dl, p)), missing_powers, brake_powers


@sh.add_function(dsp, outputs=['has_sufficient_power'])
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


@sh.add_function(dsp, outputs=['mean_piston_speeds'])
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


dsp.add_data('engine_is_turbo', dfl.values.engine_is_turbo)


@sh.add_function(dsp, outputs=['engine_type'])
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


@sh.add_function(dsp, outputs=['engine_moment_inertia'])
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
    par = dfl.functions.calculate_engine_moment_inertia.PARAMS[ignition_type]
    return (0.05 + 0.1 * engine_capacity / 1000.0) * par


dsp.add_data(
    'auxiliaries_torque_loss_factors',
    dfl.values.auxiliaries_torque_loss_factors
)
dsp.add_data('auxiliaries_power_loss', dfl.values.auxiliaries_power_loss)


@sh.add_function(dsp, outputs=['auxiliaries_torque_loss'])
def calculate_auxiliaries_torque_loss(
        auxiliaries_torque_loss_factors, engine_capacity):
    """
    Calculates engine torque losses due to engine auxiliaries [N*m].

    :param auxiliaries_torque_loss_factors:
        Constant torque loss factors due to engine auxiliaries [N/m2, N*m].
    :type auxiliaries_torque_loss_factors: (float, float)

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :return:
        Engine torque losses due to engine auxiliaries [N*m].
    :rtype: numpy.array
    """
    m, q = auxiliaries_torque_loss_factors
    return m * engine_capacity / 1000.0 + q


@sh.add_function(dsp, outputs=['auxiliaries_torque_losses'])
def calculate_auxiliaries_torque_losses(times, auxiliaries_torque_loss):
    """
    Calculates engine torque losses due to engine auxiliaries [N*m].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param auxiliaries_torque_loss:
        Constant torque loss due to engine auxiliaries [N*m].
    :type auxiliaries_torque_loss: float

    :return:
        Engine torque losses due to engine auxiliaries [N*m].
    :rtype: numpy.array
    """

    return np.tile(auxiliaries_torque_loss, len(times))


@sh.add_function(dsp, outputs=['auxiliaries_power_losses'])
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


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_fc,
    dsp_id='fc_model',
    inputs=(
        'engine_has_variable_valve_actuation', 'engine_max_speed', 'velocities',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_n_cylinders', 'min_engine_on_speed', 'phases_integration_times',
        'has_selective_catalytic_reduction', 'engine_has_cylinder_deactivation',
        'engine_fuel_lower_heating_value', 'idle_engine_speed', 'co2_emissions',
        'calibration_status', 'fuel_consumptions', 'engine_powers_out', 'times',
        'engine_speeds_out', 'full_load_powers', 'full_bmep_curve', 'on_engine',
        'full_load_speeds', 'engine_capacity', 'stop_velocity', 'has_lean_burn',
        'has_exhausted_gas_recirculation', 'fuel_carbon_content', 'engine_type',
        'engine_temperatures', 'mean_piston_speeds', 'phases_distances',
        'co2_normalization_references', 'phases_co2_emissions', 'is_cycle_hot',
        'active_cylinder_ratios', 'engine_stroke', 'co2_params', 'fuel_type',
        'after_treatment_warm_up_phases', 'engine_inertia_powers_losses',
        'fuel_consumptions_liters', 'is_hybrid', 'fuel_density', {
            'co2_params_calibrated': ('co2_params_calibrated', 'co2_params'),
            'engine_idle_fuel_consumption': (
                'engine_idle_fuel_consumption',
                'idle_fuel_consumption_initial_guess'
            )
        }),
    outputs=(
        'active_lean_burns', 'co2_rescaling_scores', 'co2_params_initial_guess',
        'engine_idle_fuel_consumption',
        'identified_co2_emissions', 'co2_emissions_model', 'calibration_status',
        'initial_friction_params', 'active_variable_valves', 'active_cylinders',
        'fuel_carbon_content', 'fuel_consumptions', 'co2_emissions', 'fuel_map',
        'extended_phases_integration_times', 'mean_piston_speeds', 'fmep_model',
        'active_exhausted_gas_recirculations', 'extended_phases_co2_emissions',
        'has_exhausted_gas_recirculation', 'engine_fuel_lower_heating_value',
        'co2_params_calibrated', 'fuel_consumptions_liters'
    ),
    inp_weight={'co2_params': dfl.EPS}
)
