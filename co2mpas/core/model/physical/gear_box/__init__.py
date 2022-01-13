# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the gear box.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.gear_box

.. autosummary::
    :nosignatures:
    :toctree: gear_box/

    at_gear
    cvt
    manual
    mechanical
    planet
"""

import math
import functools
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
from .cvt import dsp as _cvt_model
from .at_gear import dsp as _at_gear
from .mechanical import dsp as _mechanical
from .manual import dsp as _manual
from .planet import dsp as _planet_model
from co2mpas.utils import reject_outliers

dsp = sh.BlueDispatcher(
    name='Gear box model', description='Models the gear box.'
)


@sh.add_function(dsp, outputs=['gear_box_powers_in'])
def calculate_gear_box_powers_in_v1(gear_box_torques_in, gear_box_speeds_in):
    """
    Calculates gear box power [kW].

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array | float

    :param gear_box_speeds_in:
        Rotating speed of the wheel [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :return:
        Gear box power [kW].
    :rtype: numpy.array | float
    """
    from ..wheels import calculate_wheel_powers
    return calculate_wheel_powers(gear_box_torques_in, gear_box_speeds_in)


@sh.add_function(dsp, outputs=['gear_box_torques_in'])
def calculate_gear_box_torques_in(gear_box_powers_in, gear_box_speeds_in):
    """
    Calculates torque required vector [N*m].

    :param gear_box_powers_in:
        Gear box power [kW].
    :type gear_box_powers_in: numpy.array | float

    :param gear_box_speeds_in:
        Rotating speed of the wheel [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :return:
        Torque required vector [N*m].
    :rtype: numpy.array | float
    """
    from ..wheels import calculate_wheel_torques
    return calculate_wheel_torques(gear_box_powers_in, gear_box_speeds_in)


@sh.add_function(dsp, outputs=['gear_shifts'])
def calculate_gear_shifts(gears):
    """
    Returns when there is a gear shifting [-].

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        When there is a gear shifting [-].
    :rtype: numpy.array
    """
    return np.ediff1d(gears, to_begin=[0]) != 0


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['gear_box_efficiency_constants'])
def get_gear_box_efficiency_constants(has_torque_converter, gear_box_type):
    """
    Returns vehicle gear box efficiency constants (gbp00, gbp10, and gbp01).

    :param has_torque_converter:
        Does the vehicle use torque converter?
    :type has_torque_converter: bool

    :param gear_box_type:
        Gear box type (manual or automatic or cvt).
    :type gear_box_type: str

    :return:
        Vehicle gear box efficiency constants (gbp00, gbp10, and gbp01).
    :rtype: dict
    """
    PARAMS = dfl.functions.get_gear_box_efficiency_constants.PARAMS
    return PARAMS[has_torque_converter and gear_box_type != 'cvt']


def _linear(x, m, q):
    return x * m + q


def _get_par(obj, key, default=None):
    if default is None:
        default = obj

    try:
        return obj.get(key, default)
    except AttributeError:
        return default


@sh.add_function(dsp, outputs=['gear_box_efficiency_parameters_cold_hot'])
def calculate_gear_box_efficiency_parameters_cold_hot(
        gear_box_efficiency_constants, engine_max_torque):
    """
    Calculates the parameters of gear box efficiency model for cold/hot phases.

    :param gear_box_efficiency_constants:
        Vehicle gear box efficiency constants.
    :type gear_box_efficiency_constants: dict

    :param engine_max_torque:
        Engine Max Torque [N*m].
    :type engine_max_torque: float

    :return:
        Parameters of gear box efficiency model for cold/hot phases:

            - 'hot': `gbp00`, `gbp10`, `gbp01`
            - 'cold': `gbp00`, `gbp10`, `gbp01`
    :rtype: dict
    """

    par = {'hot': {}, 'cold': {}}

    for p in ['hot', 'cold']:
        for k, v in gear_box_efficiency_constants.items():
            m = _get_par(_get_par(v, 'm', default=0.0), p)
            q = _get_par(_get_par(v, 'q', default=0.0), p)
            par[p][k] = _linear(engine_max_torque, m, q)

    return par


dsp.add_function(
    function=functools.partial(sh.replicate_value, copy=False),
    inputs=['gear_box_powers_in'],
    outputs=['gear_box_powers_in_hot', 'gear_box_powers_in_cold']
)


@sh.add_function(dsp, outputs=['gear_box_powers_in_hot'])
def calculate_gear_box_powers_in_hot(
        gear_box_efficiency_parameters_cold_hot, gear_box_powers_out,
        gear_box_speeds_out, gear_box_speeds_in, phase='hot'):
    """
    Calculates the gear box powers in for cold/hot phases.

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model for cold/hot phases:

            - 'hot': `gbp00`, `gbp10`, `gbp01`
            - 'cold': `gbp00`, `gbp10`, `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :param gear_box_powers_out:
        Gear box power out vector [kW].
    :type gear_box_powers_out: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param phase:
        Cold or hot phase.
    :type phase: str

    :return:
        Gear box powers in for cold/hot phases [kW].
    :rtype: numpy.array
    """
    p, c = gear_box_efficiency_parameters_cold_hot[phase], math.pi / 30000.0
    eff, m, q = p['gbp01'], p['gbp10'] * c, p['gbp00'] * c
    po, wo, wi = gear_box_powers_out, gear_box_speeds_out, gear_box_speeds_in
    pi = np.maximum(po + wi * (m * wi + q), 0) / eff
    pi = np.where(po > 0, pi, np.minimum(0, eff * po + wo * (m * wo + q)))
    return np.where(gear_box_speeds_out == gear_box_speeds_in, po, pi)


dsp.add_func(
    functools.partial(calculate_gear_box_powers_in_hot, phase='cold'),
    function_id='calculate_gear_box_powers_in_cold',
    outputs=['gear_box_powers_in_cold']
)

dsp.add_function(
    function=functools.partial(sh.replicate_value, copy=False),
    inputs=['gear_box_powers_out'],
    outputs=['gear_box_powers_out_hot', 'gear_box_powers_out_cold']
)


@sh.add_function(dsp, outputs=['gear_box_powers_out_hot'])
def calculate_gear_box_powers_out_hot(
        gear_box_efficiency_parameters_cold_hot, gear_box_powers_in,
        gear_box_speeds_out, gear_box_speeds_in, phase='hot'):
    """
    Calculates the gear box powers out for cold/hot phases.

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model for cold/hot phases:

            - 'hot': `gbp00`, `gbp10`, `gbp01`
            - 'cold': `gbp00`, `gbp10`, `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :param gear_box_powers_in:
        Gear box power in vector [kW].
    :type gear_box_powers_in: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param phase:
        Cold or hot phase.
    :type phase: str

    :return:
        Gear box powers out for cold/hot phases [kW].
    :rtype: numpy.array
    """
    p, c = gear_box_efficiency_parameters_cold_hot[phase], math.pi / 30000.0
    eff, m, q = p['gbp01'], p['gbp10'] * c, p['gbp00'] * c
    pi, wo, wi = gear_box_powers_in, gear_box_speeds_out, gear_box_speeds_in
    po = np.minimum(pi - wo * (m * wo + q), 0) / eff
    po = np.where(pi < 0, po, np.maximum(0, eff * pi - wi * (m * wi + q)))
    return np.where(gear_box_speeds_out == gear_box_speeds_in, pi, po)


dsp.add_func(
    functools.partial(calculate_gear_box_powers_out_hot, phase='cold'),
    function_id='calculate_gear_box_powers_out_cold',
    outputs=['gear_box_powers_out_cold']
)

dsp.add_data(
    'has_gear_box_thermal_management',
    dfl.values.has_gear_box_thermal_management
)


@sh.add_function(dsp, outputs=['equivalent_gear_box_heat_capacity'])
def calculate_equivalent_gear_box_heat_capacity(
        engine_mass, has_gear_box_thermal_management):
    """
    Calculates the equivalent gear box heat capacity [kg*J/K].

    :param engine_mass:
        Engine mass [kg].
    :type engine_mass: str

    :param has_gear_box_thermal_management:
        Does the gear box have some additional technology to heat up faster?
    :type has_gear_box_thermal_management: bool

    :return:
       Equivalent gear box heat capacity [kg*J/K].
    :rtype: float
    """

    par = dfl.functions.calculate_engine_heat_capacity.PARAMS

    heated_eng_mass = engine_mass * sum(par['heated_mass_percentage'].values())

    par = dfl.functions.calculate_equivalent_gear_box_heat_capacity
    par = par.PARAMS

    heated_gear_box_mass = heated_eng_mass * par['gear_box_mass_engine_ratio']

    if has_gear_box_thermal_management:
        heated_gear_box_mass *= par['thermal_management_factor']

    return par['heat_capacity']['oil'] * heated_gear_box_mass


dsp.add_data(
    'gear_box_temperature_references',
    dfl.values.gear_box_temperature_references
)


@sh.add_function(dsp, function_id='calculate_gear_box_temperatures', inputs=[
    'gear_box_powers_out', 'gear_box_powers_in_hot', 'gear_box_powers_in_cold',
    'initial_gear_box_temperature', 'engine_thermostat_temperature', 'times',
    'equivalent_gear_box_heat_capacity', 'gear_box_temperature_references'
], outputs=['gear_box_temperatures'])
@sh.add_function(dsp, function_id='calculate_gear_box_temperatures_v1', inputs=[
    'gear_box_powers_in', 'gear_box_powers_out_hot', 'gear_box_powers_out_cold',
    'initial_gear_box_temperature', 'engine_thermostat_temperature', 'times',
    'equivalent_gear_box_heat_capacity', 'gear_box_temperature_references'
], outputs=['gear_box_temperatures'])
def calculate_gear_box_temperatures(
        gear_box_powers, gear_box_powers_hot, gear_box_powers_cold,
        initial_gear_box_temperature, engine_thermostat_temperature, times,
        equivalent_gear_box_heat_capacity, gear_box_temperature_references):
    """
    Calculates the gear box temperatures [°C].

    :param gear_box_powers:
        Gear box power out/in vector [kW].
    :type gear_box_powers: numpy.array

    :param gear_box_powers_hot:
        Gear box powers in/out for hot phase [kW].
    :type gear_box_powers_hot: numpy.array

    :param gear_box_powers_in_cold:
        Gear box powers in/out for cold phase [kW].
    :type gear_box_powers_cold: numpy.array

    :param initial_gear_box_temperature:
        Initial gear box temperature [°C].
    :type initial_gear_box_temperature: float

    :param engine_thermostat_temperature:
        Engine thermostat temperature [°C].
    :type engine_thermostat_temperature: float

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param equivalent_gear_box_heat_capacity:
        Equivalent gear box heat capacity [kg*J/K].
    :type equivalent_gear_box_heat_capacity: float

    :param gear_box_temperature_references:
        Reference temperature [°C].
    :type gear_box_temperature_references: (float, float)

    :return:
        Temperature vector [°C].
    :rtype: numpy.array
    """
    (tc, th), t = gear_box_temperature_references, initial_gear_box_temperature

    temp = np.empty((times.shape[0] + 1,), float)
    temp[0] = t
    tm = temp[1:] = engine_thermostat_temperature - 5.0

    c = 1000.0 / equivalent_gear_box_heat_capacity * np.ediff1d(times, to_end=0)
    ph, pc, p = gear_box_powers_hot, gear_box_powers_cold, gear_box_powers

    it = zip(ph - p, pc - p, (ph - pc) / ((th - tc) or 1), c)
    for i, (dph, dpc, m, c) in enumerate(it, start=1):
        if t >= th or m == 0:
            dp = dph
        elif t <= tc:
            dp = dpc
        else:
            dp = dph + (th - t) * m
        t += abs(dp) * c
        if t >= tm:
            break
        temp[i] = t
    return temp[:-1]


@sh.add_function(dsp, function_id='calculate_gear_box_powers_in', inputs=[
    'gear_box_temperatures', 'gear_box_powers_in_hot',
    'gear_box_powers_in_cold', 'gear_box_temperature_references'
], outputs=['gear_box_powers_in'])
@sh.add_function(dsp, function_id='calculate_gear_box_powers_out', inputs=[
    'gear_box_temperatures', 'gear_box_powers_out_hot',
    'gear_box_powers_out_cold', 'gear_box_temperature_references'
], outputs=['gear_box_powers_out'])
def calculate_gear_box_powers(
        gear_box_temperatures, gear_box_powers_hot, gear_box_powers_cold,
        gear_box_temperature_references):
    """
    Calculates gear box power in/out [kW].

    :param gear_box_powers_hot:
        Gear box powers in/out for hot phase [kW].
    :type gear_box_powers_hot: numpy.array

    :param gear_box_powers_cold:
        Gear box powers in/out for cold phase [kW].
    :type gear_box_powers_cold: numpy.array

    :param gear_box_temperatures:
        Temperature vector [°C].
    :type gear_box_temperatures: numpy.array

    :param gear_box_temperature_references:
        Reference temperature [°C].
    :type gear_box_temperature_references: (float, float)

    :return:
        Gear box power in/out [kW].
    :rtype: numpy.array
    """
    (tc, th), t = gear_box_temperature_references, gear_box_temperatures
    p = gear_box_powers_hot.copy()
    b = (tc < t) & (t < th)
    p[b] += (th - t[b]) / (th - tc) * (p[b] - gear_box_powers_cold[b])
    b = t <= tc
    p[b] = gear_box_powers_cold[b]
    return p


@sh.add_function(dsp, outputs=['gear_box_efficiencies'])
def calculate_gear_box_efficiencies(gear_box_powers_out, gear_box_powers_in):
    """
    Calculates gear box efficiency vector [-].

    :param gear_box_powers_out:
        Gear box power out vector [kW].
    :type gear_box_powers_out: numpy.array

    :param gear_box_powers_in:
        Gear box power in vector [kW].
    :type gear_box_powers_in: numpy.array

    :return:
        Gear box efficiency vector [-].
    :rtype: numpy.array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = gear_box_powers_in / gear_box_powers_out
    eff[np.isnan(eff) & ~np.isfinite(eff)] = 1
    b = eff > 1
    eff[b] = 1 / eff[b]
    return eff


@sh.add_function(dsp, outputs=['gear_box_mean_efficiency'])
def identify_gear_box_mean_efficiency(gear_box_efficiencies):
    """
    Identify gear box mean efficiency [-].

    :param gear_box_efficiencies:
        Gear box efficiency vector [-].
    :type gear_box_efficiencies: numpy.array

    :return:
        Gear box mean efficiency [-].
    :rtype: float
    """
    return reject_outliers(gear_box_efficiencies)[0]


dsp.add_function(
    function=sh.bypass,
    inputs=['gear_box_mean_efficiency'],
    outputs=['gear_box_mean_efficiency_guess']
)


@sh.add_function(
    dsp, outputs=['gear_box_mean_efficiency_guess'], weight=sh.inf(10, 30)
)
def calculate_gear_box_mean_efficiency_guess(
        times, motive_powers, final_drive_mean_efficiency, gear_box_speeds_in,
        equivalent_gear_box_heat_capacity, gear_box_temperature_references,
        gear_box_efficiency_parameters_cold_hot, initial_gear_box_temperature,
        gear_box_speeds_out):
    """
    Calculate gear box mean efficiency guess [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param final_drive_mean_efficiency:
        Final drive mean efficiency [-].
    :type final_drive_mean_efficiency: float

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param equivalent_gear_box_heat_capacity:
        Equivalent gear box heat capacity [kg*J/K].
    :type equivalent_gear_box_heat_capacity: float

    :param gear_box_temperature_references:
        Reference temperature [°C].
    :type gear_box_temperature_references: (float, float)

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model for cold/hot phases:

            - 'hot': `gbp00`, `gbp10`, `gbp01`
            - 'cold': `gbp00`, `gbp10`, `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :param initial_gear_box_temperature:
        Initial gear box temperature [°C].
    :type initial_gear_box_temperature: float

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :return:
        Gear box mean efficiency guess [-].
    :rtype: float
    """
    from ..electrics.motors.p4 import calculate_motor_p4_electric_powers as func
    p = func(motive_powers, final_drive_mean_efficiency)
    func = functools.partial(
        calculate_gear_box_powers_in_hot,
        gear_box_efficiency_parameters_cold_hot, p, gear_box_speeds_out,
        gear_box_speeds_in
    )
    h, c = func(), func(phase='cold')
    return identify_gear_box_mean_efficiency(calculate_gear_box_efficiencies(
        p, calculate_gear_box_powers(calculate_gear_box_temperatures(
            p, h, c, initial_gear_box_temperature, 100, times,
            equivalent_gear_box_heat_capacity, gear_box_temperature_references
        ), h, c, gear_box_temperature_references)
    ))


# noinspection PyMissingOrEmptyDocstring
def is_automatic(kwargs):
    return kwargs.get('gear_box_type') == 'automatic'


# noinspection PyMissingOrEmptyDocstring
def is_manual(kwargs):
    b = kwargs.get('gear_box_type') == 'manual'
    return b and kwargs.get('cycle_type', 'NEDC') != 'NEDC'


# noinspection PyMissingOrEmptyDocstring
def is_cvt(kwargs):
    return kwargs.get('gear_box_type') == 'cvt'


# noinspection PyMissingOrEmptyDocstring
def is_planetary(kwargs):
    return kwargs.get('gear_box_type') == 'planetary'


# noinspection PyMissingOrEmptyDocstring
def is_manual_or_automatic(kwargs):
    return kwargs.get('gear_box_type') in ('manual', 'automatic')


dsp.add_dispatcher(
    include_defaults=True,
    dsp=_mechanical,
    inputs=(
        'accelerations', 'change_gear_window_width', 'engine_max_torque',
        'engine_speed_at_max_velocity', 'engine_speeds_out', 'f0', 'on_engine',
        'final_drive_ratios', 'first_gear_box_ratio', 'full_load_curve',
        'gear_box_ratios', 'gear_box_speeds_out', 'gears', 'idle_engine_speed',
        'last_gear_box_ratio', 'maximum_vehicle_laden_mass', 'maximum_velocity',
        'n_gears', 'plateau_acceleration', 'r_dynamic', 'road_loads',
        'stop_velocity', 'times', 'velocities', 'velocity_speed_ratios',
        'motive_powers', 'correct_gear', 'gear_box_speeds_in',
        {'gear_box_type': sh.SINK}
    ),
    outputs=(
        'engine_speed_at_max_velocity', 'first_gear_box_ratio', 'max_gear',
        'gear_box_ratios', 'gear_box_speeds_in', 'gears', 'last_gear_box_ratio',
        'maximum_velocity', 'speed_velocity_ratios', 'n_gears',
        'velocity_speed_ratios'
    ),
    input_domain=is_manual_or_automatic
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_at_gear,
    dsp_id='at_gear_shifting',
    inputs=(
        'CMV', 'CMV_Cold_Hot', 'DTGS', 'GSPV', 'GSPV_Cold_Hot', 'MVL', 'times',
        'accelerations', 'change_gear_window_width', 'cycle_type', 'gears',
        'engine_temperatures', 'engine_speeds_out', 'full_load_curve',
        'fuel_saving_at_strategy', 'max_velocity_full_load_correction',
        'idle_engine_speed', 'stop_velocity', 'time_cold_hot_transition',
        'motive_powers', 'plateau_acceleration', 'specific_gear_shifting',
        'use_dt_gear_shifting', 'velocities', 'velocity_speed_ratios',
        {'gear_box_type': sh.SINK}
    ),
    outputs=(
        {
            'CMV': ('CMV', 'gear_shifting_model_raw'),
            'CMV_Cold_Hot': ('CMV_Cold_Hot', 'gear_shifting_model_raw'),
            'DTGS': ('DTGS', 'gear_shifting_model_raw'),
            'GSPV': ('GSPV', 'gear_shifting_model_raw'),
            'GSPV_Cold_Hot': ('GSPV_Cold_Hot', 'gear_shifting_model_raw')
        }, 'MVL', 'gears', 'specific_gear_shifting', 'correct_gear'),
    input_domain=is_automatic
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_manual,
    dsp_id='manual_gear_shifting',
    inputs=(
        'cycle_type', 'full_load_speeds', 'idle_engine_speed', 'motive_powers',
        'engine_max_speed', 'full_load_curve', 'engine_max_power', 'road_loads',
        'velocity_speed_ratios', 'engine_speed_at_max_power', 'velocities',
        'accelerations', 'times', {'gear_box_type': sh.SINK}
    ),
    outputs=('gears', 'correct_gear', {
        'MGS': ('MGS', 'gear_shifting_model_raw')
    }),
    input_domain=is_manual
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_cvt_model,
    dsp_id='cvt_model',
    inputs=(
        'CVT', 'accelerations', 'engine_speeds_out', 'gear_box_powers_out',
        'idle_engine_speed', 'on_engine', 'stop_velocity', 'velocities',
        'gear_box_speeds_in', {'gear_box_type': sh.SINK}
    ),
    outputs=(
        'gear_box_speeds_in', 'correct_gear', 'gears', 'max_gear',
        'max_speed_velocity_ratio', {'CVT': ('CVT', 'gear_shifting_model')}
    ),
    input_domain=is_cvt
)

dsp.add_dispatcher(
    include_defaults=True,
    dsp=_planet_model,
    dsp_id='no_model',
    inputs=(
        'accelerations', 'engine_speeds_out', 'idle_engine_speed', 'velocities',
        'stop_velocity', 'gear_box_speeds_in', 'gear_box_speeds_out',
        {'gear_box_type': sh.SINK}
    ),
    outputs=(
        'max_speed_velocity_ratio', 'gear_shifting_model', 'gear_box_speeds_in',
        'correct_gear', 'max_gear', 'gears',
    ),
    input_domain=is_planetary
)


@sh.add_function(dsp, outputs=['gear_shifting_model'])
def initialize_gear_shifting_model(
        gear_shifting_model_raw, velocity_speed_ratios, cycle_type):
    """
    Initialize the gear shifting model.

    :param gear_shifting_model_raw:
        A gear shifting model (cmv or gspv or dtgs).
    :type gear_shifting_model_raw: GSPV | CMV | DTGS

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :return:
        Initialized gear shifting model (cmv or gspv or dtgs).
    :rtype: GSPV | CMV | DTGS
    """
    # noinspection PyProtectedMember
    from .at_gear import _upgrade_gsm
    gsm = gear_shifting_model_raw
    return _upgrade_gsm(gsm, velocity_speed_ratios, cycle_type)
