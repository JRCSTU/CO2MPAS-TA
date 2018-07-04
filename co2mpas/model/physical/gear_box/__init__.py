# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions that model the basic mechanics of the gear box.

Sub-Modules:

.. currentmodule:: co2mpas.model.physical.gear_box

.. autosummary::
    :nosignatures:
    :toctree: gear_box/

    thermal
    at_gear
    cvt
    mechanical
"""

import schedula as sh
import math
import co2mpas.model.physical.defaults as defaults
import functools
import numpy as np
import collections


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
    PARAMS = defaults.dfl.functions.get_gear_box_efficiency_constants.PARAMS
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

        y = gear_box_powers_out / x
        y *= 30000.0 / math.pi

        return np.where(x <= min_engine_on_speed, 0, y)


# noinspection PyPep8Naming
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
        min_engine_on_speed, gear_box_torques_out, gear_box_speeds_in,
        gear_box_speeds_out, gear_box_efficiency_parameters_cold_hot):
    """
    Calculates torque required according to the temperature profile [N*m].

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param gear_box_torques_out:
        Torque gear_box vector [N*m].
    :type gear_box_torques_out: numpy.array

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

    tgb, es, ws = gear_box_torques_out, gear_box_speeds_in, gear_box_speeds_out

    b = tgb < 0

    y = np.zeros_like(tgb)

    par = gear_box_efficiency_parameters_cold_hot

    y[b] = (par['gbp01'] * tgb[b] - par['gbp10'] * ws[b] - par['gbp00']) * ws[b]
    y[b] /= es[b]

    b = ~b & (es > min_engine_on_speed)
    b &= (ws > min_engine_on_speed)

    y[b] = (tgb[b] - par['gbp10'] * es[b] - par['gbp00']) / par['gbp01']

    return y


def correct_gear_box_torques_in(
        gear_box_torques_out, gear_box_torques_in, gears, gear_box_ratios):
    """
    Corrects the torque when the gear box ratio is equal to 1.

    :param gear_box_torques_out:
        Torque gear_box vector [N*m].
    :type gear_box_torques_out: numpy.array

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int | float]

    :return:
        Corrected Torque required vector [N*m].
    :rtype: numpy.array
    """

    b = np.zeros_like(gears, dtype=bool)

    for k, v in gear_box_ratios.items():
        if v == 1:
            b |= gears == k

    return np.where(b, gear_box_torques_out, gear_box_torques_in)


def calculate_gear_box_efficiencies_v2(
        gear_box_powers_out, gear_box_speeds_in, gear_box_torques_out,
        gear_box_torques_in, min_engine_on_speed):
    """
    Calculates gear box efficiency [-].

    :param gear_box_powers_out:
        Power at wheels vector [kW].
    :type gear_box_powers_out: numpy.array

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_torques_out:
        Torque gear_box vector [N*m].
    :type gear_box_torques_out: numpy.array

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
    tgb = gear_box_torques_out
    tr = gear_box_torques_in
    es = gear_box_speeds_in

    eff = np.zeros_like(wp)

    b0 = tr * tgb >= 0
    b1 = b0 & (wp >= 0) & (es > min_engine_on_speed) & (tr != 0)
    b = ((b0 & (wp < 0)) | b1)

    eff[b] = es[b] * tr[b] / wp[b] * (math.pi / 30000)

    eff[b1] = 1 / eff[b1]

    return np.nan_to_num(eff)


def calculate_torques_losses(gear_box_torques_in, gear_box_torques_out):
    """
    Calculates gear box torque losses [N*m].

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array | float

    :param gear_box_torques_out:
        Torque gear_box vector [N*m].
    :type gear_box_torques_out: numpy.array | float

    :return:
        Gear box torques losses [N*m].
    :rtype: numpy.array | float
    """

    return gear_box_torques_in - gear_box_torques_out


# noinspection PyMissingOrEmptyDocstring
class GearBoxLosses(object):
    def __init__(self, gear_box_efficiency_parameters_cold_hot,
                 equivalent_gear_box_heat_capacity, thermostat_temperature,
                 gear_box_temperature_references, gear_box_ratios=None):
        base = collections.OrderedDict()
        base['gear_box_ratios'] = gear_box_ratios
        base['thermostat_temperature'] = thermostat_temperature
        base['equivalent_gear_box_heat_capacity'] = \
            equivalent_gear_box_heat_capacity
        base['gear_box_efficiency_parameters_cold_hot'] = \
            gear_box_efficiency_parameters_cold_hot
        base['gear_box_temperature_references'] = \
            gear_box_temperature_references

        # from .thermal import thermal
        # inputs = (
        #     'gear_box_temperature', 'delta_time', 'gear_box_power_out',
        #     'gear_box_speed_out','gear_box_speed_in', 'gear_box_torque_out',
        #     'gear'
        # )
        # _thermal = dsp_utl.SubDispatchPipe(
        #     dsp=thermal(),
        #     function_id='thermal',
        #     inputs=tuple(base) + inputs,
        #     outputs=('gear_box_temperature', 'gear_box_torque_in',
        #              'gear_box_efficiency')
        # )
        from .thermal import _thermal

        self._thermal = functools.partial(_thermal, *tuple(base.values()))

    def predict(self, *args, **kwargs):
        return np.array(list(self.yield_losses(*args, **kwargs))).T

    def yield_losses(self, times, gear_box_powers_out, gear_box_speeds_in,
                     gear_box_speeds_out, gear_box_torques_out,
                     initial_gear_box_temperature, gears=None,
                     min_engine_on_speed=None):
        delta_times = np.zeros_like(times, dtype=float)
        delta_times[:-1] = np.diff(times)

        if min_engine_on_speed is None:
            # noinspection PyUnusedLocal
            def gb_tor(*args):
                return gear_box_torques_out[index]
        else:
            gb_tor = functools.partial(
                calculate_gear_box_torques,
                min_engine_on_speed=min_engine_on_speed
            )
            if gear_box_torques_out is None:
                gear_box_torques_out = np.empty_like(gear_box_powers_out, float)

        o, func = [initial_gear_box_temperature], self._thermal
        it = enumerate(zip(
            delta_times, gear_box_powers_out, gear_box_speeds_out,
            gear_box_speeds_in, gears
        ))
        for index, (dt, po, so, si, g) in it:
            temp = o[0]
            tr = gear_box_torques_out[index] = gb_tor(po, so, si)
            o = func(temp, dt, po, so, si, tr, g)
            yield [temp] + o[1:]


def define_gear_box_loss_model(
        gear_box_efficiency_parameters_cold_hot,
        equivalent_gear_box_heat_capacity, thermostat_temperature,
        gear_box_temperature_references, gear_box_ratios=None):
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

    :param thermostat_temperature:
        Engine thermostat temperature [°C].
    :type thermostat_temperature: float

    :param gear_box_temperature_references:
        Reference temperature [°C].
    :type gear_box_temperature_references: (float, float)

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int | float], optional

    :return:
        Gear box loss model.
    :rtype: GearBoxLosses

    .. note:: Torque entering the gearbox can be from engine side
       (power mode or from wheels in motoring mode).
    """

    model = GearBoxLosses(
        gear_box_efficiency_parameters_cold_hot,
        equivalent_gear_box_heat_capacity, thermostat_temperature,
        gear_box_temperature_references, gear_box_ratios=gear_box_ratios
    )

    return model


def calculate_gear_box_efficiencies_torques_temperatures(
        gear_box_loss_model, times, gear_box_powers_out, gear_box_speeds_in,
        gear_box_speeds_out, gear_box_torques_out, initial_gear_box_temperature,
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

    :param gear_box_torques_out:
        Torque gear_box vector [N*m].
    :type gear_box_torques_out: numpy.array

    :param initial_gear_box_temperature:
        initial_gear_box_temperature [°C].
    :type initial_gear_box_temperature: float

    :param gears:
        Gear vector [-].
    :type gears: numpy.array, optional

    :return:
        Gear box efficiency [-], torque in [N*m], and temperature [°C] vectors.
    :rtype: (numpy.array, numpy.array, numpy.array)

    .. note:: Torque entering the gearbox can be from engine side
       (power mode or from wheels in motoring mode).
    """

    temp, to_in, eff = gear_box_loss_model.predict(
        times, gear_box_powers_out, gear_box_speeds_in, gear_box_speeds_out,
        gear_box_torques_out, initial_gear_box_temperature, gears=gears
    )

    return temp, to_in, eff


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

    par = defaults.dfl.functions.calculate_engine_heat_capacity.PARAMS

    heated_eng_mass = engine_mass * sum(par['heated_mass_percentage'].values())

    par = defaults.dfl.functions.calculate_equivalent_gear_box_heat_capacity
    par = par.PARAMS

    heated_gear_box_mass = heated_eng_mass * par['gear_box_mass_engine_ratio']

    if has_gear_box_thermal_management:
        heated_gear_box_mass *= par['thermal_management_factor']

    return par['heat_capacity']['oil'] * heated_gear_box_mass


# noinspection PyMissingOrEmptyDocstring
def is_automatic(kwargs):
    return kwargs['gear_box_type'] == 'automatic'


# noinspection PyMissingOrEmptyDocstring
def is_manual(kwargs):
    b = kwargs['gear_box_type'] == 'manual'
    return b and kwargs['cycle_type'] != 'NEDC'


# noinspection PyMissingOrEmptyDocstring
def is_cvt(kwargs):
    return kwargs['gear_box_type'] == 'cvt'


# noinspection PyMissingOrEmptyDocstring
def not_cvt(kwargs):
    return kwargs['gear_box_type'] != 'cvt'


# noinspection PyMissingOrEmptyDocstring
class GearBoxModel:
    key_outputs = [
        'gears',
        'gear_box_speeds_in',
        'gear_box_temperatures',
        'gear_box_torques_in',
        'gear_box_efficiencies',
        'gear_box_powers_in'
    ]

    types = {
        float: {
            # Gear box model outputs.
            'gear_box_speeds_in', 'gear_box_powers_in',
            'gear_box_torques_in',
            'gear_box_temperatures', 'gear_box_efficiencies'
        },
        int: {
            # Gear box model outputs.
            'gears'
        }
    }

    def __init__(self, stop_velocity=None, min_engine_on_speed=None,
                 gear_shifting_model=None, gear_box_loss_model=None,
                 initial_gear_box_temperature=None, correct_gear=None,
                 outputs=None):
        self.stop_velocity = stop_velocity
        self.min_engine_on_speed = min_engine_on_speed
        self.gear_shifting_model = gear_shifting_model
        self.gear_box_loss_model = gear_box_loss_model
        self.initial_gear_box_temperature = initial_gear_box_temperature
        self.correct_gear = correct_gear
        self._outputs = outputs
        self.outputs = None

    def __call__(self, times, *args, **kwargs):
        self.set_outputs(times.shape[0])
        for _ in self.yield_results(times, *args, **kwargs):
            pass
        return sh.selector(self.key_outputs, self.outputs, output_type='list')

    def yield_gear(self, times, velocities, accelerations, motive_powers,
                   engine_coolant_temperatures, gears):
        if self._outputs is not None and 'gears' in self._outputs:
            yield from self._outputs['gears']
        else:
            yield from self.gear_shifting_model.yield_gear(
                times, velocities, accelerations, motive_powers,
                engine_coolant_temperatures, correct_gear=self.correct_gear,
                gears=gears
            )

    def yield_speed(self, gears, velocities, accelerations,
                    final_drive_powers_in):
        if self._outputs is not None and 'gear_box_speeds_in' in self._outputs:
            yield from self._outputs['gear_box_speeds_in']
        else:
            yield from self.gear_shifting_model.yield_speed(
                self.stop_velocity, gears, velocities, accelerations,
                final_drive_powers_in
            )

    def yield_losses(self, times, final_drive_powers_in, gear_box_speeds_in,
                     final_drive_speeds_in, gears):
        keys = [
            'gear_box_temperatures', 'gear_box_torques_in',
            'gear_box_efficiencies'
        ]

        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            yield from zip(*sh.selector(
                keys, self._outputs, output_type='list'
            ))
        else:
            yield from self.gear_box_loss_model.yield_losses(
                times, final_drive_powers_in, gear_box_speeds_in,
                final_drive_speeds_in, self.outputs['gear_box_torques_in'],
                self.initial_gear_box_temperature,
                gears=gears, min_engine_on_speed=self.min_engine_on_speed
            )

    def yield_power(self, gear_box_torques_in, gear_box_speeds_in):
        if self._outputs is not None and 'gear_box_powers_in' in self._outputs:
            yield from self._outputs['gear_box_powers_in']
        else:
            for t, s in zip(gear_box_torques_in, gear_box_speeds_in):
                yield calculate_gear_box_powers_in(t, s)

    def set_outputs(self, n, outputs=None):
        if outputs is None:
            outputs = {}
        outputs.update(self._outputs or {})

        for t, names in self.types.items():
            names = names - set(outputs)
            if names:
                outputs.update(zip(names, np.empty((len(names), n), dtype=t)))
            if 'gears' in names:
                outputs['gears'][0] = 0
        self.outputs = outputs

    def yield_results(self, times, velocities, accelerations, motive_powers,
                      final_drive_speeds_in, final_drive_powers_in):
        outputs = self.outputs
        g_gen = self.yield_gear(
            times, velocities, accelerations, motive_powers,
            outputs['engine_coolant_temperatures'], outputs['gears']
        )

        s_gen = self.yield_speed(
            outputs['gears'], velocities, accelerations, final_drive_powers_in
        )

        l_gen = self.yield_losses(
            times, final_drive_powers_in, outputs['gear_box_speeds_in'],
            final_drive_speeds_in, outputs['gears']
        )

        p_gen = self.yield_power(
            outputs['gear_box_torques_in'], outputs['gear_box_speeds_in']
        )

        for i in range(0, times.shape[0]):
            outputs['gears'][i] = g = next(g_gen)
            outputs['gear_box_speeds_in'][i] = gb_s = next(s_gen)
            gb_temp, gb_tor, gb_eff = next(l_gen)
            outputs['gear_box_temperatures'][i] = gb_temp
            outputs['gear_box_torques_in'][i] = gb_tor
            outputs['gear_box_efficiencies'][i] = gb_eff
            outputs['gear_box_powers_in'][i] = gb_p = next(p_gen)
            yield g, gb_s, gb_temp, gb_tor, gb_eff, gb_p


def define_fake_gear_box_prediction_model(
        gears, gear_box_speeds_in, gear_box_temperatures, gear_box_torques_in,
        gear_box_efficiencies, gear_box_powers_in):
    """
    Defines a fake gear box prediction model.

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_temperatures:
        Temperature vector [°C].
    :type gear_box_temperatures: numpy.array

    :param gear_box_torques_in:
        Torque required vector [N*m].
    :type gear_box_torques_in: numpy.array

    :param gear_box_efficiencies:
        Gear box efficiency vector [-].
    :type gear_box_efficiencies: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :return:
        Gear box prediction model.
    :rtype: GearBoxModel
    """
    model = GearBoxModel(outputs={
        'gears': gears,
        'gear_box_speeds_in': gear_box_speeds_in,
        'gear_box_temperatures': gear_box_temperatures,
        'gear_box_torques_in': gear_box_torques_in,
        'gear_box_efficiencies': gear_box_efficiencies,
        'gear_box_powers_in': gear_box_powers_in
    })

    return model


def initialize_gear_shifting_model(gsm, velocity_speed_ratios, cycle_type):
    """
    Initialize the gear shifting model.

    :param gsm:
        A gear shifting model (cmv or gspv or dtgs).
    :type gsm: GSPV | CMV | DTGS

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
    from .at_gear import _upgrade_gsm
    return _upgrade_gsm(gsm, velocity_speed_ratios, cycle_type)


def define_gear_box_prediction_model(
        stop_velocity, min_engine_on_speed, gear_shifting_model,
        gear_box_loss_model, initial_gear_box_temperature, correct_gear):
    """
    Defines the gear box prediction model.

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param gear_shifting_model:
        Gear shifting model.
    :type gear_shifting_model: CMV | GSPV | GSM | DTGS

    :param gear_box_loss_model:
        Gear box loss model.
    :type gear_box_loss_model: GearBoxLosses

    :param initial_gear_box_temperature:
        initial_gear_box_temperature [°C].
    :type initial_gear_box_temperature: float

    :param correct_gear:
        A function to correct the predicted gear.
    :type correct_gear: callable

    :return:
        Gear box prediction model.
    :rtype: GearBoxModel
    """
    model = GearBoxModel(
        stop_velocity, min_engine_on_speed, gear_shifting_model,
        gear_box_loss_model, initial_gear_box_temperature, correct_gear
    )

    return model


def gear_box():
    """
    Defines the gear box model.

    .. dispatcher:: d

        >>> d = gear_box()

    :return:
        The gear box model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Gear box model',
        description='Models the gear box.'
    )

    d.add_function(
        function=define_fake_gear_box_prediction_model,
        inputs=[
            'gears', 'gear_box_speeds_in', 'gear_box_temperatures',
            'gear_box_torques_in', 'gear_box_efficiencies',
            'gear_box_powers_in'
        ],
        outputs=['gear_box_prediction_model']
    )

    d.add_function(
        function=initialize_gear_shifting_model,
        inputs=['gear_shifting_model_raw', 'velocity_speed_ratios',
                'cycle_type'],
        outputs=['gear_shifting_model']
    )

    d.add_function(
        function=define_gear_box_prediction_model,
        inputs=['stop_velocity', 'min_engine_on_speed', 'gear_shifting_model',
                'gear_box_loss_model', 'initial_gear_box_temperature',
                'correct_gear'],
        outputs=['gear_box_prediction_model'],
        weight=4000
    )

    d.add_function(
        function=calculate_gear_shifts,
        inputs=['gears'],
        outputs=['gear_shifts']
    )

    d.add_function(
        function=get_gear_box_efficiency_constants,
        inputs=['has_torque_converter', 'gear_box_type'],
        outputs=['gear_box_efficiency_constants'],
    )

    d.add_function(
        function=calculate_gear_box_efficiency_parameters_cold_hot,
        inputs=['gear_box_efficiency_constants', 'engine_max_torque'],
        outputs=['gear_box_efficiency_parameters_cold_hot'],
    )

    d.add_data(
        data_id='min_engine_on_speed',
        default_value=defaults.dfl.values.min_engine_on_speed
    )

    d.add_function(
        function=calculate_gear_box_torques,
        inputs=['gear_box_powers_out', 'gear_box_speeds_out',
                'gear_box_speeds_in', 'min_engine_on_speed'],
        outputs=['gear_box_torques'],
    )

    d.add_data(
        data_id='gear_box_temperature_references',
        default_value=defaults.dfl.values.gear_box_temperature_references
    )

    d.add_function(
        function=calculate_gear_box_torques_in,
        inputs=['gear_box_torques', 'gear_box_speeds_in',
                'gear_box_speeds_out', 'gear_box_temperatures',
                'gear_box_efficiency_parameters_cold_hot',
                'gear_box_temperature_references', 'min_engine_on_speed'],
        outputs=['gear_box_torques_in<0>']
    )

    d.add_function(
        function=correct_gear_box_torques_in,
        inputs=['gear_box_torques', 'gear_box_torques_in<0>', 'gears',
                'gear_box_ratios'],
        outputs=['gear_box_torques_in'],
    )

    d.add_function(
        function=sh.bypass,
        inputs=['gear_box_torques_in<0>'],
        outputs=['gear_box_torques_in'],
        weight=100,
    )

    d.add_function(
        function=calculate_gear_box_efficiencies_v2,
        inputs=['gear_box_powers_out', 'gear_box_speeds_in', 'gear_box_torques',
                'gear_box_torques_in', 'min_engine_on_speed'],
        outputs=['gear_box_efficiencies'],
    )

    d.add_function(
        function=calculate_torques_losses,
        inputs=['gear_box_torques_in', 'gear_box_torques'],
        outputs=['gear_box_torque_losses'],
    )

    d.add_function(
        function=define_gear_box_loss_model,
        inputs=['gear_box_efficiency_parameters_cold_hot',
                'equivalent_gear_box_heat_capacity',
                'engine_thermostat_temperature',
                'gear_box_temperature_references', 'gear_box_ratios'],
        outputs=['gear_box_loss_model']
    )

    d.add_function(
        function=define_gear_box_loss_model,
        inputs=['gear_box_efficiency_parameters_cold_hot',
                'equivalent_gear_box_heat_capacity',
                'engine_thermostat_temperature',
                'gear_box_temperature_references'],
        outputs=['gear_box_loss_model'],
        weight=10
    )

    d.add_function(
        function=calculate_gear_box_efficiencies_torques_temperatures,
        inputs=['gear_box_loss_model', 'times', 'gear_box_powers_out',
                'gear_box_speeds_in', 'gear_box_speeds_out', 'gear_box_torques',
                'initial_gear_box_temperature', 'gears'],
        outputs=['gear_box_temperatures', 'gear_box_torques_in',
                 'gear_box_efficiencies'],
        weight=40
    )

    d.add_function(
        function=calculate_gear_box_efficiencies_torques_temperatures,
        inputs=['gear_box_loss_model', 'times', 'gear_box_powers_out',
                'gear_box_speeds_in', 'gear_box_speeds_out', 'gear_box_torques',
                'initial_gear_box_temperature'],
        outputs=['gear_box_temperatures', 'gear_box_torques_in',
                 'gear_box_efficiencies'],
        weight=90
    )

    d.add_function(
        function=calculate_gear_box_powers_in,
        inputs=['gear_box_torques_in', 'gear_box_speeds_in'],
        outputs=['gear_box_powers_in']
    )

    d.add_data(
        data_id='has_gear_box_thermal_management',
        default_value=defaults.dfl.values.has_gear_box_thermal_management
    )

    d.add_function(
        function=calculate_equivalent_gear_box_heat_capacity,
        inputs=['engine_mass', 'has_gear_box_thermal_management'],
        outputs=['equivalent_gear_box_heat_capacity']
    )

    from .mechanical import mechanical
    d.add_dispatcher(
        include_defaults=True,
        dsp=mechanical(),
        inputs=(
            'accelerations', 'change_gear_window_width', 'engine_max_torque',
            'engine_speed_at_max_velocity', 'engine_speeds_out', 'f0',
            'final_drive_ratios', 'first_gear_box_ratio', 'full_load_curve',
            'gear_box_ratios', 'gear_box_speeds_out', 'gears',
            'idle_engine_speed', 'last_gear_box_ratio',
            'maximum_vehicle_laden_mass', 'maximum_velocity', 'n_gears',
            'plateau_acceleration', 'r_dynamic', 'road_loads', 'stop_velocity',
            'times', 'velocities', 'velocity_speed_ratios',
            {'gear_box_type': sh.SINK}),
        outputs=(
            'engine_speed_at_max_velocity', 'first_gear_box_ratio',
            'gear_box_ratios', 'gear_box_speeds_in', 'gears',
            'last_gear_box_ratio', 'max_gear', 'maximum_velocity', 'n_gears',
            'speed_velocity_ratios', 'velocity_speed_ratios'),
        input_domain=not_cvt
    )

    from .at_gear import at_gear
    d.add_dispatcher(
        include_defaults=True,
        dsp=at_gear(),
        dsp_id='at_gear_shifting',
        inputs=(
            'CMV', 'CMV_Cold_Hot', 'DTGS', 'GSPV', 'GSPV_Cold_Hot', 'MVL',
            'accelerations', 'change_gear_window_width', 'cycle_type',
            'engine_coolant_temperatures', 'engine_speeds_out',
            'fuel_saving_at_strategy', 'full_load_curve', 'gears',
            'idle_engine_speed', 'max_velocity_full_load_correction',
            'motive_powers', 'plateau_acceleration',
            'specific_gear_shifting', 'stop_velocity',
            'time_cold_hot_transition', 'times', 'use_dt_gear_shifting',
            'velocities', 'velocity_speed_ratios',
            {'gear_box_type': sh.SINK}),
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

    from .manual_gear import manual_gear
    d.add_dispatcher(
        include_defaults=True,
        dsp=manual_gear(),
        dsp_id='manual_gear_shifting',
        inputs=(
            'cycle_type',
            'full_load_speeds', 'idle_engine_speed', 'engine_max_speed',
            'full_load_curve', 'engine_max_power', 'road_loads',
            'velocity_speed_ratios', 'engine_speed_at_max_power',
            'velocities', 'accelerations', 'times', 'motive_powers',
            {'gear_box_type': sh.SINK}),
        outputs=('gears', 'correct_gear', {
            'MGS': ('MGS', 'gear_shifting_model_raw')
        }),
        input_domain=is_manual
    )

    from .cvt import cvt_model
    d.add_dispatcher(
        include_defaults=True,
        dsp=cvt_model(),
        dsp_id='cvt_model',
        inputs=(
            'CVT', 'accelerations', 'engine_speeds_out', 'gear_box_powers_out',
            'idle_engine_speed', 'on_engine', 'stop_velocity', 'velocities',
            {'gear_box_type': sh.SINK}),
        outputs=(
            {
                'CVT': ('CVT', 'gear_shifting_model')
            },
            'gear_box_speeds_in', 'correct_gear',
            'gears', 'max_gear',
            'max_speed_velocity_ratio'),
        input_domain=is_cvt
    )

    return d
