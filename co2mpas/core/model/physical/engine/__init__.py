# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the engine.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.engine

.. autosummary::
    :nosignatures:
    :toctree: engine/

    co2_emission
    cold_start
    idle
    start_stop
    thermal
"""
import math
import functools
import numpy as np
import schedula as sh
from ..defaults import dfl
from .idle import dsp as _idle
from co2mpas.utils import BaseModel
from .thermal import dsp as _thermal
from .start_stop import dsp as _start_stop
from .cold_start import dsp as _cold_start
from .co2_emission import dsp as _co2_emission

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


@sh.add_function(dsp, outputs=['ignition_type'], weigth=1)
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
    from .co2_emission import calculate_brake_mean_effective_pressures
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


@sh.add_function(dsp, outputs=['full_load_torques'])
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


@sh.add_function(dsp, outputs=['engine_speeds_out_hot'])
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


@sh.add_function(dsp, outputs=['on_idle'])
def identify_on_idle(
        velocities, engine_speeds_out_hot, gears, stop_velocity,
        min_engine_on_speed):
    """
    Identifies when the engine is on idle [-].

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

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

    on_idle = engine_speeds_out_hot > min_engine_on_speed
    on_idle &= (gears == 0) | (velocities <= stop_velocity)

    return on_idle


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_idle,
    inputs=(
        'idle_engine_speed_median', 'idle_engine_speed_std',
        'min_engine_on_speed', 'stop_velocity', 'velocities',
        'engine_speeds_out', 'idle_engine_speed',
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

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_start_stop,
    dsp_id='start_stop',
    inputs=(
        'accelerations', 'correct_start_stop_with_gears', 'start_stop_model',
        'engine_coolant_temperatures', 'engine_starts', 'use_basic_start_stop',
        'has_start_stop', 'is_hybrid', 'state_of_charges', 'idle_engine_speed',
        'min_time_engine_on_after_start', 'on_engine', 'gears', 'gear_box_type',
        'start_stop_activation_time', 'times', 'velocities', 'engine_speeds_out'
    ),
    outputs=(
        'use_basic_start_stop', 'on_engine', 'start_stop_prediction_model',
        'start_stop_model', 'engine_starts', 'correct_start_stop_with_gears',
        'start_stop_activation_time'
    )
)

dsp.add_dispatcher(
    dsp=_cold_start,
    inputs=(
        'cold_start_speed_model', 'cold_start_speeds_phases', 'on_engine',
        'engine_coolant_temperatures', 'engine_speeds_out', 'idle_engine_speed',
        'engine_speeds_out_hot', 'engine_thermostat_temperature', 'on_idle'
    ),
    outputs=(
        'cold_start_speed_model', 'cold_start_speeds_delta',
        'cold_start_speeds_phases'
    )
)


@sh.add_function(
    dsp,
    inputs=[
        'on_engine', 'idle_engine_speed', 'engine_speeds_out_hot',
        'cold_start_speeds_delta', 'clutch_tc_speeds_delta'
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


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['uncorrected_engine_powers_out']
)
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
    from scipy.misc import derivative
    from scipy.interpolate import InterpolatedUnivariateSpline as Spline

    p, b = np.zeros_like(clutch_tc_powers, dtype=float), on_engine
    p[b] = clutch_tc_powers[b]

    if gear_box_type == 'manual':
        p[on_idle & (p < 0)] = 0.0

    p[b] += auxiliaries_power_losses[b]

    if alternator_powers_demand is not None:
        p[b] += alternator_powers_demand[b]

    p_ine = engine_moment_inertia / 2000 * (2 * math.pi / 60) ** 2
    p += p_ine * derivative(Spline(times, engine_speeds_out, k=1), times) ** 2

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


@sh.add_function(
    dsp, outputs=['engine_powers_out', 'missing_powers', 'brake_powers']
)
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


dsp.add_data('auxiliaries_torque_loss', dfl.values.auxiliaries_torque_loss)
dsp.add_data('auxiliaries_power_loss', dfl.values.auxiliaries_power_loss)


@sh.add_function(dsp, outputs=['auxiliaries_torque_losses'])
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
    dsp=_co2_emission,
    dsp_id='CO2_emission_model',
    inputs=(
        'accelerations', 'active_cylinder_ratios', 'angle_slopes',
        'calibration_status', 'co2_emission_EUDC', 'co2_emission_UDC', 'times',
        'co2_emission_extra_high', 'co2_emission_high', 'co2_emission_low',
        'co2_emission_medium', 'co2_emissions', 'co2_normalization_references',
        'co2_params', 'enable_phases_willans', 'enable_willans', 'ki_additive',
        'engine_capacity', 'engine_coolant_temperatures', 'idle_engine_speed',
        'engine_fuel_lower_heating_value', 'engine_has_cylinder_deactivation',
        'engine_has_variable_valve_actuation', 'engine_max_speed', 'velocities',
        'engine_powers_out', 'engine_speeds_out', 'engine_stroke', 'on_engine',
        'engine_thermostat_temperature', 'engine_thermostat_temperature_window',
        'engine_type', 'fuel_carbon_content', 'fuel_carbon_content_percentage',
        'fuel_consumptions', 'fuel_density', 'full_bmep_curve', 'is_cycle_hot',
        'has_exhausted_gas_recirculation', 'has_lean_burn', 'missing_powers',
        'has_periodically_regenerating_systems', 'initial_engine_temperature',
        'has_selective_catalytic_reduction', 'phases_integration_times',
        'stop_velocity', 'ki_multiplicative', 'mean_piston_speeds', 'fuel_type',
        'motive_powers', 'engine_n_cylinders', 'min_engine_on_speed',
        {'co2_params_calibrated': ('co2_params_calibrated', 'co2_params'),
         'engine_idle_fuel_consumption': (
             'engine_idle_fuel_consumption',
             'idle_fuel_consumption_initial_guess')}
    ),
    outputs=(
        'active_cylinders', 'fuel_consumptions', 'optimal_efficiency',
        'active_lean_burns', 'active_variable_valves', 'co2_rescaling_scores',
        'after_treatment_temperature_threshold', 'calibration_status',
        'co2_emission_value', 'co2_emissions', 'co2_emissions_model',
        'co2_error_function_on_emissions', 'co2_error_function_on_phases',
        'co2_params_calibrated', 'co2_params_initial_guess', 'ki_additive',
        'declared_co2_emission_value', 'engine_fuel_lower_heating_value',
        'engine_idle_fuel_consumption', 'fuel_carbon_content', 'fuel_density',
        'extended_phases_integration_times', 'extended_phases_co2_emissions',
        'fuel_carbon_content_percentage', 'active_exhausted_gas_recirculations',
        'has_exhausted_gas_recirculation', 'identified_co2_emissions',
        'initial_friction_params', 'ki_multiplicative', 'phases_co2_emissions',
        'phases_fuel_consumptions', 'phases_willans_factors', 'willans_factors',
    ),
    inp_weight={'co2_params': dfl.EPS}
)


# noinspection PyMissingOrEmptyDocstring
class EngineModel(BaseModel):
    key_outputs = (
        'on_engine', 'engine_starts', 'engine_speeds_out_hot',
        'engine_coolant_temperatures'
    )
    contract_outputs = 'engine_coolant_temperatures',
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
        super(EngineModel, self).__init__(outputs)

    def set_outputs(self, outputs=None):
        super(EngineModel, self).set_outputs(outputs)

        if self.start_stop_prediction_model:
            self.start_stop_prediction_model.set_outputs(outputs)
        if self.engine_temperature_prediction_model:
            self.engine_temperature_prediction_model.set_outputs(outputs)

    def init_on_start(self, times, velocities, accelerations,
                      engine_coolant_temperatures, state_of_charges, gears):
        return self.start_stop_prediction_model.init_results(
            times, velocities, accelerations, engine_coolant_temperatures,
            state_of_charges, gears
        )

    def init_speed(self, on_engine, gear_box_speeds_in):
        key = 'engine_speeds_out_hot'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]

        def _next(i):
            return calculate_engine_speeds_out_hot(
                gear_box_speeds_in[i], on_engine[i], self.idle_engine_speed
            )

        return _next

    def init_thermal(self, times, accelerations, final_drive_powers_in,
                     engine_speeds_out_hot):
        return self.engine_temperature_prediction_model.init_results(
            times, accelerations, final_drive_powers_in, engine_speeds_out_hot
        )

    def init_results(self, times, velocities, accelerations, state_of_charges,
                     final_drive_powers_in, gears, gear_box_speeds_in):
        outputs = self.outputs
        on_engine, temp, starts, speeds = (
            outputs['on_engine'], outputs['engine_coolant_temperatures'],
            outputs['engine_starts'], outputs['engine_speeds_out_hot']
        )
        ss_gen = self.init_on_start(
            times, velocities, accelerations, temp, state_of_charges, gears
        )
        s_gen = self.init_speed(on_engine, gear_box_speeds_in)
        t_gen = self.init_thermal(
            times, accelerations, final_drive_powers_in, speeds
        )

        def _next(i):
            on_engine[i], starts[i] = on, start = ss_gen(i)
            speeds[i] = eng_s = s_gen(i)
            try:
                temp[i + 1] = t_gen(i)
            except IndexError:
                pass
            return on, start, eng_s, temp[i]

        return _next


@sh.add_function(dsp, outputs=['engine_prediction_model'], weight=4000)
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