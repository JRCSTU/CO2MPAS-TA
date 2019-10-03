# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
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
    manual
    cvt
    mechanical
    thermal
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
from co2mpas.utils import List, reject_outliers

dsp = sh.BlueDispatcher(
    name='Gear box model', description='Models the gear box.'
)


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


@sh.add_function(dsp, outputs=['gear_box_mean_efficiency'])
def identify_gear_box_mean_efficiency(gear_box_powers_in, gear_box_powers_out):
    """
    Identify gear box mean efficiency [-].

    :param gear_box_powers_in:
        Gear box power in vector [kW].
    :type gear_box_powers_in: numpy.array

    :param gear_box_powers_out:
        Gear box power out vector [kW].
    :type gear_box_powers_out: numpy.array

    :return:
        Gear box mean efficiency [-].
    :rtype: float
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = gear_box_powers_out / gear_box_powers_in
        b = eff > 1
        eff[b] = 1 / eff[b]
        return reject_outliers(eff[np.isfinite(eff) & (eff >= 0)])[0]


dsp.add_function(
    function=sh.bypass,
    inputs=['gear_box_mean_efficiency'],
    outputs=['gear_box_mean_efficiency_guess']
)


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['gear_box_mean_efficiency_guess']
)
@sh.add_function(dsp, outputs=['gear_box_mean_efficiency_guess'], weight=90)
def calculate_gear_box_mean_efficiency_guess(
        motive_powers, final_drive_mean_efficiency, gear_box_loss_model, times,
        gear_box_speeds_in, gear_box_speeds_out, min_engine_on_speed,
        gears=None):
    """
    Calculate gear box mean efficiency guess [-].

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param final_drive_mean_efficiency:
        Final drive mean efficiency [-].
    :type final_drive_mean_efficiency: float

    :param gear_box_loss_model:
        Gear box loss model.
    :type gear_box_loss_model: GearBoxLosses

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param gears:
        Gear vector [-].
    :type gears: numpy.array, optional

    :return:
        Gear box mean efficiency guess [-].
    :rtype: float
    """
    from ..electrics.motors.p4 import calculate_motor_p4_electric_powers as func
    powers_out = func(motive_powers, final_drive_mean_efficiency)
    torques = calculate_gear_box_torques(
        powers_out, gear_box_speeds_out, gear_box_speeds_in, min_engine_on_speed
    )
    torques_in = calculate_gear_box_efficiencies_torques_temperatures(
        gear_box_loss_model, times, powers_out, gear_box_speeds_in,
        gear_box_speeds_out, torques, 100, gears
    )[1]
    powers_in = calculate_gear_box_powers_in(torques_in, gear_box_speeds_in)
    return identify_gear_box_mean_efficiency(powers_in, powers_out)


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


dsp.add_data('min_engine_on_speed', dfl.values.min_engine_on_speed)


@sh.add_function(dsp, outputs=['gear_box_torques'])
def calculate_gear_box_torques(
        gear_box_powers_out, gear_box_speeds_out, gear_box_speeds_in,
        min_engine_on_speed):
    """
    Calculates torque entering the gear box [N*m].

    :param gear_box_powers_out:
        Gear box power vector [kW].
    :type gear_box_powers_out: numpy.array | float

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array | float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Torque gear box vector [N*m].
    :rtype: numpy.array | float

    .. note:: Torque entering the gearbox can be from engine side
       (power mode or from wheels in motoring mode)
    """
    if isinstance(gear_box_speeds_in, float):
        if gear_box_powers_out > 0:
            x = gear_box_speeds_in
        else:
            x = gear_box_speeds_out
        if x <= min_engine_on_speed:
            return 0
        return gear_box_powers_out / x * 30000.0 / math.pi
    else:
        x = np.where(
            gear_box_powers_out > 0, gear_box_speeds_in, gear_box_speeds_out
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            y = gear_box_powers_out / x
        y *= 30000.0 / math.pi

        return np.where(x <= min_engine_on_speed, 0, y)


dsp.add_data(
    'gear_box_temperature_references',
    dfl.values.gear_box_temperature_references
)


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['gear_box_torques_in<0>'])
def calculate_gear_box_torques_in(
        gear_box_torques, gear_box_speeds_in, gear_box_speeds_out,
        gear_box_temperatures, gear_box_efficiency_parameters_cold_hot,
        gear_box_temperature_references, min_engine_on_speed):
    """
    Calculates torque required according to the temperature profile [N*m].

    :param gear_box_torques:
        Torque gear box vector [N*m].
    :type gear_box_torques: numpy.array

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_temperatures:
        Temperature vector [°C].
    :type gear_box_temperatures: numpy.array

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model for cold/hot phases:

            - 'hot': `gbp00`, `gbp10`, `gbp01`
            - 'cold': `gbp00`, `gbp10`, `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :param gear_box_temperature_references:
        Cold and hot reference temperatures [°C].
    :type gear_box_temperature_references: tuple

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Torque required vector according to the temperature profile [N*m].
    :rtype: numpy.array
    """

    par = gear_box_efficiency_parameters_cold_hot
    T_cold, T_hot = gear_box_temperature_references
    t_out, e_s, gb_s = gear_box_torques, gear_box_speeds_in, gear_box_speeds_out
    fun = functools.partial(_gear_box_torques_in, min_engine_on_speed)

    t = fun(t_out, e_s, gb_s, par['hot'])

    if not T_cold == T_hot:
        gbt = gear_box_temperatures

        b = gbt <= T_hot

        t_cold = fun(t_out[b], e_s[b], gb_s[b], par['cold'])

        t[b] += (T_hot - gbt[b]) / (T_hot - T_cold) * (t_cold - t[b])

    return t


def _gear_box_torques_in(
        min_engine_on_speed, gear_box_torques, gear_box_speeds_in,
        gear_box_speeds_out, gear_box_efficiency_parameters_cold_hot):
    """
    Calculates torque required according to the temperature profile [N*m].

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param gear_box_torques:
        Torque gear_box vector [N*m].
    :type gear_box_torques: numpy.array

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model:

            - `gbp00`,
            - `gbp10`,
            - `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :return:
        Torque required vector [N*m].
    :rtype: numpy.array
    """
    tgb, es, ws = gear_box_torques, gear_box_speeds_in, gear_box_speeds_out

    b = (tgb < 0) & (es != 0)

    y = np.zeros_like(tgb)

    par = gear_box_efficiency_parameters_cold_hot

    y[b] = (par['gbp01'] * tgb[b] - par['gbp10'] * ws[b] - par['gbp00']) * ws[b]
    y[b] /= es[b]

    b = ~b & (es > min_engine_on_speed)
    b &= (ws > min_engine_on_speed)

    y[b] = (tgb[b] - par['gbp10'] * es[b] - par['gbp00']) / par['gbp01']

    return y


@sh.add_function(
    dsp, inputs=['gear_box_torques', 'gear_box_torques_in<0>', 'gears',
                 'gear_box_ratios'], outputs=['gear_box_torques_in'])
def correct_gear_box_torques_in(
        gear_box_torques, gear_box_torques_in, gears, gear_box_ratios):
    """
    Corrects the torque when the gear box ratio is equal to 1.

    :param gear_box_torques:
        Torque gear_box vector [N*m].
    :type gear_box_torques: numpy.array

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int]

    :return:
        Corrected Torque required vector [N*m].
    :rtype: numpy.array
    """
    b = np.zeros_like(gears, dtype=bool)

    for k, v in gear_box_ratios.items():
        if v == 1:
            b |= gears == k

    return np.where(b, gear_box_torques, gear_box_torques_in)


dsp.add_function(
    function=sh.bypass, inputs=['gear_box_torques_in<0>'],
    outputs=['gear_box_torques_in'], weight=100,
)


@sh.add_function(dsp, outputs=['gear_box_efficiencies'])
def calculate_gear_box_efficiencies(
        gear_box_powers_out, gear_box_speeds_in, gear_box_torques,
        gear_box_torques_in, min_engine_on_speed):
    """
    Calculates gear box efficiency [-].

    :param gear_box_powers_out:
        Power at wheels vector [kW].
    :type gear_box_powers_out: numpy.array

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_torques:
        Torque gear_box vector [N*m].
    :type gear_box_torques: numpy.array

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Gear box efficiency vector [-].
    :rtype: numpy.array
    """
    wp = gear_box_powers_out
    tgb = gear_box_torques
    tr = gear_box_torques_in
    es = gear_box_speeds_in

    eff = np.zeros_like(wp)

    b0 = tr * tgb >= 0
    b1 = b0 & (wp >= 0) & (es > min_engine_on_speed) & (tr != 0)
    b = ((b0 & (wp < 0)) | b1)

    eff[b] = es[b] * tr[b] / wp[b] * (math.pi / 30000)

    eff[b1] = 1 / eff[b1]

    return np.nan_to_num(eff)


@sh.add_function(dsp, outputs=['gear_box_torque_losses'])
def calculate_torques_losses(gear_box_torques_in, gear_box_torques):
    """
    Calculates gear box torque losses [N*m].

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array | float

    :param gear_box_torques:
        Torque gear_box vector [N*m].
    :type gear_box_torques: numpy.array | float

    :return:
        Gear box torques losses [N*m].
    :rtype: numpy.array | float
    """

    return gear_box_torques_in - gear_box_torques


# noinspection PyMissingOrEmptyDocstring
class GearBoxLosses:
    def __init__(self, gear_box_efficiency_parameters_cold_hot,
                 equivalent_gear_box_heat_capacity,
                 gear_box_temperature_references, initial_gear_box_temperature,
                 min_engine_on_speed, gear_box_ratios=None):
        par = gear_box_efficiency_parameters_cold_hot
        base = dict(
            equivalent_gear_box_heat_capacity=equivalent_gear_box_heat_capacity,
            gear_box_efficiency_parameters_cold_hot=par,
            gear_box_temperature_references=gear_box_temperature_references,
            gear_box_ratios=gear_box_ratios,
            min_engine_on_speed=min_engine_on_speed
        )
        self.initial_gear_box_temperature = initial_gear_box_temperature

        # noinspection PyProtectedMember
        from .thermal import _thermal
        self._thermal = functools.partial(_thermal, **base)

    def init_losses(self, gear_box_temperatures, times, gear_box_powers_out,
                    gear_box_speeds_out, gear_box_speeds_in, gears,
                    gear_box_torques=None):
        gear_box_temperatures[0] = self.initial_gear_box_temperature
        if gear_box_torques is None:
            # noinspection PyUnusedLocal
            def get_gear_box_torque(i):
                return None
        else:
            def get_gear_box_torque(i):
                return gear_box_torques[i]

        def _next(i):
            j = i + 1
            dt = len(times) > j and times[j] - times[i] or 0
            return self._thermal(
                gear_box_temperatures[i], get_gear_box_torque(i), gears[i], dt,
                gear_box_powers_out[i], gear_box_speeds_out[i],
                gear_box_speeds_in[i]
            )

        return _next


@sh.add_function(dsp, inputs_kwargs=True, outputs=['gear_box_loss_model'])
@sh.add_function(dsp, outputs=['gear_box_loss_model'], weight=10)
def define_gear_box_loss_model(
        gear_box_efficiency_parameters_cold_hot,
        equivalent_gear_box_heat_capacity,
        gear_box_temperature_references, initial_gear_box_temperature,
        min_engine_on_speed, gear_box_ratios=None):
    """
    Defines the gear box loss model.

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model for cold/hot phases:

            - 'hot': `gbp00`, `gbp10`, `gbp01`
            - 'cold': `gbp00`, `gbp10`, `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :param equivalent_gear_box_heat_capacity:
        Equivalent gear box heat capacity [kg*J/K].
    :type equivalent_gear_box_heat_capacity: float

    :param gear_box_temperature_references:
        Reference temperature [°C].
    :type gear_box_temperature_references: (float, float)

    :param initial_gear_box_temperature:
        Initial gear box temperature [°C].
    :type initial_gear_box_temperature: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int, float | int], optional

    :return:
        Gear box loss model.
    :rtype: GearBoxLosses

    .. note:: Torque entering the gearbox can be from engine side
       (power mode or from wheels in motoring mode).
    """

    model = GearBoxLosses(
        gear_box_efficiency_parameters_cold_hot,
        equivalent_gear_box_heat_capacity,
        gear_box_temperature_references, initial_gear_box_temperature,
        min_engine_on_speed, gear_box_ratios=gear_box_ratios
    )

    return model


_o = 'gear_box_temperatures', 'gear_box_torques_in', 'gear_box_efficiencies'


@sh.add_function(dsp, inputs_kwargs=True, outputs=_o, weight=40)
@sh.add_function(dsp, outputs=_o, weight=90)
def calculate_gear_box_efficiencies_torques_temperatures(
        gear_box_loss_model, times, gear_box_powers_out, gear_box_speeds_in,
        gear_box_speeds_out, gear_box_torques, engine_thermostat_temperature,
        gears=None):
    """
    Calculates gear box efficiency [-], torque in [N*m], and temperature [°C].

    :param gear_box_loss_model:
        Gear box loss model.
    :type gear_box_loss_model: GearBoxLosses

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param gear_box_powers_out:
        Power at wheels vector [kW].
    :type gear_box_powers_out: numpy.array

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_torques:
        Torque gear_box vector [N*m].
    :type gear_box_torques: numpy.array

    :param engine_thermostat_temperature:
        Engine thermostat temperature [°C].
    :type engine_thermostat_temperature: float

    :param gears:
        Gear vector [-].
    :type gears: numpy.array, optional

    :return:
        Gear box efficiency [-], torque in [N*m], and temperature [°C] vectors.
    :rtype: (numpy.array, numpy.array, numpy.array)

    .. note:: Torque entering the gearbox can be from engine side
       (power mode or from wheels in motoring mode).
    """
    temp, to_in, eff = List(), List(), List()

    func = gear_box_loss_model.init_losses(
        temp, times, gear_box_powers_out, gear_box_speeds_out,
        gear_box_speeds_in, gears, gear_box_torques
    )
    for i in range(times.shape[0]):
        temp[i + 1], to_in[i], eff[i] = func(i)
    temp = np.minimum(engine_thermostat_temperature - 5, temp[:-1].toarray())
    return temp, to_in.toarray(), eff.toarray()


@sh.add_function(dsp, outputs=['gear_box_powers_in'])
def calculate_gear_box_powers_in(gear_box_torques_in, gear_box_speeds_in):
    """
    Calculates gear box power [kW].

    :param gear_box_torques_in:
        Torque at the wheel [N*m].
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
        'engine_coolant_temperatures', 'engine_speeds_out', 'full_load_curve',
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
