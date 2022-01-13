# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to predict the fuel consumptions.
"""
import copy
import functools
import itertools
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl
import co2mpas.utils as co2_utl
from numbers import Number

dsp = sh.BlueDispatcher(
    name='Engine fuel consumption sub model',
    description='Calculates fuel consumptions.'
)


@sh.add_function(dsp, outputs=['has_exhausted_gas_recirculation'])
def default_engine_has_exhausted_gas_recirculation(fuel_type):
    """
    Returns the default engine has exhaust gas recirculation value [-].

    :param fuel_type:
        Fuel type (diesel, gasoline, LPG, NG, ethanol, methanol, biodiesel,
        propane).
    :type fuel_type: str

    :return:
        Does the engine have exhaust gas recirculation technology?
    :rtype: bool
    """

    return fuel_type in ('diesel', 'biodiesel')


@sh.add_function(dsp, outputs=['brake_mean_effective_pressures'])
def calculate_brake_mean_effective_pressures(
        engine_speeds_out, engine_powers_out, engine_capacity,
        min_engine_on_speed):
    """
    Calculates engine brake mean effective pressure [bar].

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Engine brake mean effective pressure vector [bar].
    :rtype: numpy.array
    """

    b = engine_speeds_out > min_engine_on_speed

    p = np.zeros_like(engine_powers_out)
    p[b] = engine_powers_out[b] / engine_speeds_out[b]
    p[b] *= 1200000.0 / engine_capacity

    return np.nan_to_num(p)


@sh.add_function(dsp, outputs=['extended_integration_times'])
def calculate_extended_integration_times(
        times, velocities, on_engine, phases_integration_times, stop_velocity,
        after_treatment_warm_up_phases):
    """
    Calculates the extended integration times [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param phases_integration_times:
        Cycle phases integration times [s].
    :type phases_integration_times: tuple

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Extended cycle phases integration times [s].
    :rtype: tuple
    """
    from ..gear_box.mechanical import _shift
    lv, pit = np.zeros(velocities.size + 2), np.unique(phases_integration_times)
    lv[1:-1] = np.asarray(velocities <= stop_velocity, int)
    indices = np.where(np.diff(lv) != 0)[0].reshape(-1, 2)
    split_points = []
    for i, j in indices:
        t0, t1 = times[i], times[j - 1]
        if t1 - t0 < 20 or any(t0 <= x <= t1 for x in pit):
            continue

        b = ~on_engine[i:j]
        if b.any() and not b.all():
            t = np.median(times[i:j][b])
        else:
            t = (t0 + t1) / 2
        split_points.append(t)
    try:
        for i in _shift(after_treatment_warm_up_phases):
            if not lv[i + 1]:
                split_points.append(times[i])
    except IndexError:
        pass

    return sorted(split_points)


@sh.add_function(dsp, outputs=[
    'extended_cumulative_co2_emissions', 'extended_phases_integration_times'
])
def calculate_extended_cumulative_co2_emissions(
        times, velocities, on_engine, extended_integration_times,
        co2_normalization_references, phases_integration_times,
        phases_co2_emissions, phases_distances, stop_velocity):
    """
    Calculates the extended cumulative CO2 of cycle phases [CO2g].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param extended_integration_times:
        Extended cycle phases integration times [s].
    :type extended_integration_times: tuple

    :param co2_normalization_references:
        CO2 normalization references (e.g., engine loads) [-].
    :type co2_normalization_references: numpy.array

    :param phases_integration_times:
        Cycle phases integration times [s].
    :type phases_integration_times: tuple

    :param phases_co2_emissions:
        CO2 emission of cycle phases [CO2g/km].
    :type phases_co2_emissions: numpy.array

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Extended cumulative CO2 of cycle phases [CO2g].
    :rtype: numpy.array
    """

    r = co2_normalization_references.copy()
    r[~on_engine] = 0
    lv = np.asarray(velocities <= stop_velocity, int)
    _cco2, phases = [], []
    cco2 = phases_co2_emissions * phases_distances
    from scipy.integrate import trapz

    def _stops(n, m):
        return trapz(lv[n:m], times[n:m]) / (times[m] - times[n])

    for cco2, (t0, t1) in zip(cco2, phases_integration_times):
        i, j = np.searchsorted(times, (t0, t1))
        if i == j:
            continue
        v = trapz(r[i:j], times[i:j])
        c, k0 = [0.0], i

        p = [t for t in extended_integration_times if t0 < t < t1]
        for k, t in zip(np.searchsorted(times, p), p):
            if k < j and _stops(k0, k) < 0.5 and _stops(k, j) < 0.5:
                phases.append((t0, t))
                t0, k0 = t, k
                c.append(trapz(r[i:k], times[i:k]) / v)
        phases.append((t0, t1))
        c.append(1.0)

        _cco2.extend(np.diff(c) * cco2)

    return np.array(_cco2), phases


# noinspection PyMissingOrEmptyDocstring
class IdleFuelConsumptionModel:
    def __init__(self, fc=None):
        self.fc = fc
        self.n_s = None
        self.c = None
        self.fmep_model = None

    def fit(self, idle_engine_speed, engine_capacity, engine_stroke, lhv,
            fmep_model):
        idle = idle_engine_speed[0]
        from . import calculate_mean_piston_speeds
        self.n_s = calculate_mean_piston_speeds(idle, engine_stroke)
        self.c = idle * (engine_capacity / (lhv * 1200))
        self.fmep_model = fmep_model
        return self

    def consumption(self, params=None, ac_phases=None):
        import lmfit
        if isinstance(params, lmfit.Parameters):
            params = params.valuesdict()

        if self.fc is not None:
            ac = params.get('acr', self.fmep_model.base_acr)
            avv = params.get('avv', 0)
            lb = params.get('lb', 0)
            egr = params.get('egr', 0)
            fc = self.fc
        else:
            fc, _, ac, avv, lb, egr = self.fmep_model(params, self.n_s, 0, 1)

            if not (ac_phases is None or ac_phases.all() or 'acr' in params):
                p = params.copy()
                p['acr'] = self.fmep_model.base_acr
                _fc, _, _ac, _avv, _lb, _egr = self.fmep_model(p, self.n_s, 0,
                                                               1)
                fc = np.where(ac_phases, fc, _fc)
                ac = np.where(ac_phases, ac, _ac)
                avv = np.where(ac_phases, avv, _avv)
                lb = np.where(ac_phases, lb, _lb)
                egr = np.where(ac_phases, egr, _egr)

            fc *= self.c  # [g/sec]
        return fc, ac, avv, lb, egr


dsp.add_data('stop_velocity', dfl.values.stop_velocity)
dsp.add_data('min_engine_on_speed', dfl.values.min_engine_on_speed)


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True,
    outputs=['idle_fuel_consumption_model']
)
def define_idle_fuel_consumption_model(
        idle_engine_speed, engine_capacity, engine_stroke,
        engine_fuel_lower_heating_value, fmep_model,
        idle_fuel_consumption_initial_guess=None):
    """
    Defines the idle fuel consumption model.

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :param idle_fuel_consumption_initial_guess:
        Initial guess of fuel consumption at hot idle engine speed [g/s].
    :type idle_fuel_consumption_initial_guess: float, optional

    :return:
        Idle fuel consumption model.
    :rtype: IdleFuelConsumptionModel
    """
    return IdleFuelConsumptionModel(idle_fuel_consumption_initial_guess).fit(
        idle_engine_speed, engine_capacity, engine_stroke,
        engine_fuel_lower_heating_value, fmep_model
    )


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['engine_idle_fuel_consumption']
)
def calculate_engine_idle_fuel_consumption(
        idle_fuel_consumption_model, co2_params_calibrated=None):
    """
    Calculates fuel consumption at hot idle engine speed [g/s].

    :param idle_fuel_consumption_model:
        Idle fuel consumption model.
    :type idle_fuel_consumption_model: IdleFuelConsumptionModel

    :param co2_params_calibrated:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type co2_params_calibrated: lmfit.Parameters

    :return:
        Fuel consumption at hot idle engine speed [g/s].
    :rtype: float
    """

    return idle_fuel_consumption_model.consumption(co2_params_calibrated)[0]


def _yield_factors(param_id, factor):
    try:
        for k, v in factor.get(param_id, {}).items():
            yield k, v, 1
    except TypeError:
        p = {}

        def _defaults():
            return (np.zeros_like(param_id, dtype=float),
                    np.zeros_like(param_id, dtype=int))

        for m in np.unique(param_id):
            if not isinstance(m, np.ma.core.MaskedConstant):
                b = m == param_id
                for k, v in factor.get(m, {}).items():
                    j, i = sh.get_nested_dicts(p, k, default=_defaults)
                    j[b], i[b] = v, 1

        for k, (j, n) in p.items():
            b = n == 0
            j[b], n[b] = 1, 1
            yield k, j, n


def _tech_mult_factors(**params):
    p = {}
    # noinspection PyProtectedMember
    factors = dfl.functions._tech_mult_factors.factors
    for k, v in factors.items():
        for i, j, n in _yield_factors(params.get(k, 0), v):
            s = sh.get_nested_dicts(p, i, default=lambda: [0, 0])
            s[0] += j
            s[1] += n

    for k, (n, d) in p.items():
        m = n / d
        params[k] = m * params[k]

    return params


# noinspection PyUnusedLocal,PyPep8Naming
def _ABC(
        n_speeds, n_powers=0, n_temperatures=1,
        a2=0, b2=0, a=0, b=0, c=0, t=0, l=0, l2=0, acr=1, **kw):
    acr2 = (acr ** 2)
    A = a2 / acr2 + (b2 / acr2) * n_speeds
    B = a / acr + (b / acr + (c / acr) * n_speeds) * n_speeds
    C = l + l2 * n_speeds ** 2
    if not isinstance(n_temperatures, Number) or n_temperatures != 1:
        C *= np.power(n_temperatures, -t)
    C -= n_powers / acr

    return A, B, C


# noinspection PyPep8Naming
def _fuel_ABC(n_speeds, **kw):
    with np.errstate(under='ignore'):
        return _ABC(n_speeds, **_tech_mult_factors(**kw))


# noinspection PyPep8Naming
def _calculate_fc(A, B, C):
    with np.errstate(under='ignore'):
        b = np.array(A, dtype=bool)
        if b.all():
            v = np.sqrt(np.abs(B ** 2 - 4.0 * A * C))
            return (-B + v) / (2 * A), v
        elif ~b.all():
            return -C / B, B
        else:
            fc, v = np.zeros_like(C), np.zeros_like(C)
            fc[b], v[b] = _calculate_fc(A[b], B[b], C[b])
            b = ~b
            fc[b], v[b] = _calculate_fc(A[b], B[b], C[b])
            return fc, v


# noinspection PyMissingOrEmptyDocstring
class FMEP:
    def __init__(self, full_bmep_curve, active_cylinder_ratios=(1.0,),
                 has_cylinder_deactivation=False,
                 acr_full_bmep_curve_percentage=0.5,
                 acr_max_mean_piston_speeds=12.0,
                 acr_min_mean_piston_speeds=3.0,
                 has_variable_valve_actuation=False,
                 has_lean_burn=False,
                 lb_max_mean_piston_speeds=12.0,
                 lb_full_bmep_curve_percentage=0.4,
                 has_exhausted_gas_recirculation=False,
                 has_selective_catalytic_reduction=False,
                 egr_max_mean_piston_speeds=12.0,
                 egr_full_bmep_curve_percentage=0.7,
                 engine_type=None):

        if active_cylinder_ratios:
            self.base_acr = max(active_cylinder_ratios)
        else:
            self.base_acr = 1.0
        self.active_cylinder_ratios = set(active_cylinder_ratios)
        self.active_cylinder_ratios -= {self.base_acr}
        self.fbc = full_bmep_curve

        self.has_cylinder_deactivation = has_cylinder_deactivation
        self.acr_max_mean_piston_speeds = float(acr_max_mean_piston_speeds)
        self.acr_min_mean_piston_speeds = float(acr_min_mean_piston_speeds)
        self.acr_fbc_percentage = acr_full_bmep_curve_percentage

        self.has_variable_valve_actuation = has_variable_valve_actuation

        self.has_lean_burn = has_lean_burn
        self.lb_max_mean_piston_speeds = float(lb_max_mean_piston_speeds)
        self.lb_fbc_percentage = lb_full_bmep_curve_percentage
        self.lb_n_temp_min = 0.5

        self.has_exhausted_gas_recirculation = has_exhausted_gas_recirculation
        self.has_selective_catalytic_reduction = \
            has_selective_catalytic_reduction
        self.egr_max_mean_piston_speeds = float(egr_max_mean_piston_speeds)
        self.egr_fbc_percentage = egr_full_bmep_curve_percentage
        self.engine_type = engine_type

    def vva(self, params, n_powers, a=None):
        a = a or {}
        if self.has_variable_valve_actuation and 'vva' not in params:
            a['vva'] = ((0, True), (1, n_powers >= 0))
        return a

    def lb(self, params, n_speeds, n_powers, n_temp, a=None):
        a = a or {}
        if self.has_lean_burn and 'lb' not in params:
            b = n_temp >= self.lb_n_temp_min
            b &= n_speeds < self.lb_max_mean_piston_speeds
            b &= n_powers <= (self.fbc(n_speeds) * self.lb_fbc_percentage)
            a['lb'] = ((0, True), (1, b))
        return a

    def egr(self, params, n_speeds, n_powers, n_temp, a=None):
        a = a or {}
        if self.has_exhausted_gas_recirculation and 'egr' not in params:
            # b = n_speeds < self.egr_max_mean_piston_speeds
            # b &= n_powers <= (self.fbc(n_speeds) * self.egr_fbc_percentage)
            k = self.engine_type, self.has_selective_catalytic_reduction
            egr = dfl.functions.FMEP_egr.egr_fact_map[k]
            if k[0] == 'compression':
                if k[1]:
                    b = n_temp < 1
                else:
                    b = n_speeds < self.egr_max_mean_piston_speeds
                    b &= n_powers <= (
                            self.fbc(n_speeds) * self.egr_fbc_percentage
                    )

                if b is True:
                    a['egr'] = (egr, True),
                elif b is False:
                    a['egr'] = (0, True),
                else:
                    a['egr'] = (np.where(b, egr, 0), True),
            else:
                a['egr'] = ((0, True), (egr, True))

        return a

    def acr(self, params, n_speeds, n_powers, n_temp, a=None, acr_valid=None):
        a = a or {'acr': [(self.base_acr, True)]}
        if self.has_cylinder_deactivation and self.active_cylinder_ratios and \
                'acr' not in params:
            l = a['acr']
            b = n_powers > 0
            b &= (self.acr_min_mean_piston_speeds < n_speeds)
            b &= (n_speeds < self.acr_max_mean_piston_speeds)
            if acr_valid is not None:
                b &= acr_valid
            ac = self.fbc(n_speeds) * self.acr_fbc_percentage
            ac = n_powers / np.maximum(dfl.EPS, ac)
            for acr in sorted(self.active_cylinder_ratios):
                l.append((acr, b & (ac < acr)))
        return a

    @staticmethod
    def _check_combinations(a):
        out = {}
        for k, v in a.items():
            for i in v:
                try:
                    if i[1] is True or i[1].any():
                        sh.get_nested_dicts(out, k, default=list).append(i)
                except AttributeError:
                    pass
        return out

    def combination(self, params, n_speeds, n_powers, n_temp, acr_valid=None):
        a = self.acr(params, n_speeds, n_powers, n_temp, acr_valid=acr_valid)
        a = self.lb(params, n_speeds, n_powers, n_temp, a=a)
        a = self.vva(params, n_powers, a=a)
        a = self.egr(params, n_speeds, n_powers, n_temp, a=a)
        a = self._check_combinations(a)

        keys, c = zip(*sorted(a.items()))
        p = params.copy()
        p.update({
            'n_speeds': n_speeds,
            'n_powers': n_powers,
            'n_temperatures': n_temp
        })
        for s in itertools.product(*c):
            b, d = True, {}

            for k, (v, n) in zip(keys, s):
                b &= n
                d[k] = v
            try:
                if b is False or not b.any():
                    continue
                if b.all():
                    b = True
            except AttributeError:
                pass

            p.update(d)
            yield {k: self.g(v, b) for k, v in p.items()}, d, b

    @staticmethod
    def g(data, b):
        if b is not True:
            try:
                return np.ma.masked_where(~b, data, copy=False)
            except (IndexError, TypeError):
                pass
        return data

    def __call__(self, params, n_speeds, n_powers, n_temp, acr_valid=None):
        it = self.combination(params, n_speeds, n_powers, n_temp, acr_valid)
        s = None
        for p, d, n in it:
            d['fmep'], d['v'] = _calculate_fc(*_fuel_ABC(**p))
            if s is None:
                s = d
            else:
                b = d['fmep'] < s['fmep']
                if n is True and b is True:
                    s = d
                elif b is not False:
                    n &= b
                    if n.all():
                        s = d
                    elif n.any():
                        for k, v in d.items():
                            s[k] = np.where(n, v, s[k])

        acr = s.get('acr', params.get('acr', self.base_acr))
        vva = s.get('vva', params.get('vva', 0))
        lb = s.get('lb', params.get('lb', 0))
        egr = s.get('egr', params.get('egr', 0))

        return s['fmep'], s['v'], acr, vva, lb, egr


@sh.add_function(dsp, outputs=['cylinder_deactivation_valid_phases'])
def calculate_cylinder_deactivation_valid_phases(engine_inertia_powers_losses):
    """
    Calculates valid activation phases for cylinder deactivation.

    :param engine_inertia_powers_losses:
        Engine power losses due to inertia [kW].
    :type engine_inertia_powers_losses: numpy.array

    :return:
        Valid activation phases for cylinder deactivation.
    :rtype: numpy.array
    """
    p = dfl.functions.calculate_cylinder_deactivation_valid_phases.LIMIT
    return engine_inertia_powers_losses <= p


dsp.add_data('active_cylinder_ratios', dfl.values.active_cylinder_ratios)
dsp.add_data(
    'engine_has_cylinder_deactivation',
    dfl.values.engine_has_cylinder_deactivation
)
dsp.add_data(
    'engine_has_variable_valve_actuation',
    dfl.values.engine_has_variable_valve_actuation
)
dsp.add_data('has_lean_burn', dfl.values.has_lean_burn)
dsp.add_data(
    'has_selective_catalytic_reduction',
    dfl.values.has_selective_catalytic_reduction
)


@sh.add_function(dsp, outputs=['fmep_model'])
def define_fmep_model(
        full_bmep_curve, engine_max_speed, engine_stroke,
        active_cylinder_ratios, engine_has_cylinder_deactivation,
        engine_has_variable_valve_actuation, has_lean_burn,
        has_exhausted_gas_recirculation, has_selective_catalytic_reduction,
        engine_type, idle_engine_speed):
    """
    Defines the vehicle FMEP model.

    :param full_bmep_curve:
        Vehicle full bmep curve.
    :type full_bmep_curve: function

    :param engine_max_speed:
        Maximum allowed engine speed [RPM].
    :type engine_max_speed: float

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param active_cylinder_ratios:
        Possible active cylinder ratios [-].
    :type active_cylinder_ratios: tuple[float]

    :param engine_has_cylinder_deactivation:
        Does the engine have cylinder deactivation technology?
    :type engine_has_cylinder_deactivation: bool

    :param engine_has_variable_valve_actuation:
        Does the engine feature variable valve actuation? [-].
    :type engine_has_variable_valve_actuation: bool

    :param has_lean_burn:
        Does the engine have lean burn technology?
    :type has_lean_burn: bool

    :param has_exhausted_gas_recirculation:
        Does the engine have exhaust gas recirculation technology?
    :type has_exhausted_gas_recirculation: bool

    :param has_selective_catalytic_reduction:
        Does the engine have selective catalytic reduction technology?
    :type has_selective_catalytic_reduction: bool

    :param engine_type:
        Engine type (positive turbo, positive natural aspiration, compression).
    :type engine_type: str

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Vehicle FMEP model.
    :rtype: FMEP
    """

    d = dfl.functions.define_fmep_model
    acr_fbcp = d.acr_full_bmep_curve_percentage
    lb_fbcp = d.lb_full_bmep_curve_percentage
    egr_fbcp = d.egr_full_bmep_curve_percentage

    acr_maps = d.acr_max_mean_piston_speeds_percentage * engine_max_speed
    acr_mips = d.acr_min_mean_piston_speeds_percentage * idle_engine_speed[0]

    lb_mps = d.lb_max_mean_piston_speeds_percentage * engine_max_speed
    egr_mps = d.egr_max_mean_piston_speeds_percentage * engine_max_speed

    from . import calculate_mean_piston_speeds
    bmep = calculate_mean_piston_speeds

    model = FMEP(
        full_bmep_curve,
        active_cylinder_ratios=active_cylinder_ratios,
        has_cylinder_deactivation=engine_has_cylinder_deactivation,
        acr_full_bmep_curve_percentage=acr_fbcp,
        acr_max_mean_piston_speeds=bmep(acr_maps, engine_stroke),
        acr_min_mean_piston_speeds=bmep(acr_mips, engine_stroke),
        has_variable_valve_actuation=engine_has_variable_valve_actuation,
        has_lean_burn=has_lean_burn,
        lb_max_mean_piston_speeds=bmep(lb_mps, engine_stroke),
        lb_full_bmep_curve_percentage=lb_fbcp,
        has_exhausted_gas_recirculation=has_exhausted_gas_recirculation,
        has_selective_catalytic_reduction=has_selective_catalytic_reduction,
        egr_max_mean_piston_speeds=bmep(egr_mps, engine_stroke),
        egr_full_bmep_curve_percentage=egr_fbcp,
        engine_type=engine_type
    )

    return model


# noinspection PyUnresolvedReferences
@sh.add_function(dsp, outputs=['fuel_map'])
def define_fuel_map(
        idle_engine_speed, engine_capacity, co2_params_calibrated, fmep_model,
        engine_fuel_lower_heating_value, engine_stroke, full_load_speeds,
        full_load_powers):
    """
    Define fuel consumption map [RPM, kW, g/s].

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param full_load_speeds:
        T1 map speed vector [RPM].
    :type full_load_speeds: numpy.array

    :param full_load_powers:
        T1 map power vector [kW].
    :type full_load_powers: numpy.array

    :param co2_params_calibrated:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type co2_params_calibrated: lmfit.Parameters

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :return:
        Fuel consumption map [RPM, kW, g/s].
    :rtype: dict
    """
    from . import calculate_mean_piston_speeds
    speed = np.linspace(full_load_speeds[0], full_load_speeds[-1], 100).tolist()
    speed = np.unique(speed + list(full_load_speeds))
    p = co2_params_calibrated.valuesdict()
    lhv = engine_fuel_lower_heating_value
    par = dfl.functions.calculate_co2_emissions
    idle_cutoff = idle_engine_speed[0] * par.cutoff_idle_ratio
    ec_p0 = _calculate_p0(
        fmep_model, p, engine_capacity, engine_stroke, idle_cutoff, lhv
    )
    flp = np.interp(speed, full_load_speeds, full_load_powers)
    power = np.linspace(ec_p0, np.max(full_load_powers), 100).tolist()
    power = np.unique(power + list(full_load_powers) + list(flp) + [0])

    e_s, e_p = np.meshgrid(speed, power, indexing='ij')
    n_s = calculate_mean_piston_speeds(e_s, engine_stroke)
    n_p = calculate_brake_mean_effective_pressures(e_s, e_p, engine_capacity, 0)

    fc = np.maximum(0, fmep_model(p, n_s, n_p, 1)[0])
    fc *= e_s * (engine_capacity / (lhv * 1200))  # [g/sec]
    fc[np.argmax(fc, axis=1)[:, None] < np.arange(fc.shape[1])] = np.nan
    b = ~np.isnan(fc).all(0)
    return dict(
        speed=speed.tolist(), power=power[b].tolist(), fuel=fc[:, b].tolist()
    )


def _calculate_p0(
        fmep_model, params, engine_capacity, engine_stroke,
        idle_engine_speed_median, engine_fuel_lower_heating_value):
    """
    Calculates the engine power threshold limit [kW].

    :param params:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type params: dict

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param idle_engine_speed_median:
        Engine speed idle median [RPM].
    :type idle_engine_speed_median: float

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :return:
        Engine power threshold limit [kW].
    :rtype: float
    """

    engine_cm_idle = idle_engine_speed_median * engine_stroke / 30000.0

    lhv = engine_fuel_lower_heating_value

    wfb_idle, wfa_idle = fmep_model(params, engine_cm_idle, 0, 1)[:2]
    wfa_idle = (3600000.0 / lhv) / wfa_idle
    wfb_idle *= (3.0 * engine_capacity / lhv * idle_engine_speed_median)
    return -wfb_idle / wfa_idle


def _apply_ac_phases(func, fmep_model, params, *args, ac_phases=None):
    res_with_ac = func(fmep_model, params, *args)
    if ac_phases is None or ac_phases.all() or not (
            fmep_model.has_cylinder_deactivation and
            fmep_model.active_cylinder_ratios):
        return res_with_ac
    else:
        p = params.copy()
        p['acr'] = fmep_model.base_acr
        return np.where(ac_phases, res_with_ac, func(fmep_model, p, *args))


def _normalized_engine_temperatures(
        engine_temperatures, temperature_target):
    """
    Calculates the normalized engine coolant temperatures [-].

    ..note::
        Engine coolant temperatures are first converted in kelvin and then
        normalized. The results is between ``[0, 1]``.

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :param temperature_target:
        Normalization temperature [°C].
    :type temperature_target: float

    :return:
        Normalized engine coolant temperature [-].
    :rtype: numpy.array
    """
    temp = (engine_temperatures + 273.0) / (temperature_target + 273.0)
    if isinstance(temp, np.ndarray):
        temp[np.searchsorted(temp, (1,))[0]:] = 1
    return np.minimum(1, temp)


def _calculate_co2_emissions(
        time_series, engine_fuel_lower_heating_value, idle_engine_speed,
        engine_stroke, engine_capacity, idle_fuel_consumption_model,
        fuel_carbon_content, min_engine_on_speed, fmep_model, params,
        sub_values=None):
    """
    Calculates CO2 emissions [CO2g/s].

    :param time_series:
        Engine speed vector [RPM], Engine power vector [kW], Engine coolant
        temperature vector [°C], Mean piston speed vector [m/s], and Engine
        brake mean effective pressure vector [bar].
    :type time_series: numpy.array

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param idle_fuel_consumption_model:
        Model of fuel consumption at hot idle engine speed.
    :type idle_fuel_consumption_model: IdleFuelConsumptionModel

    :param fuel_carbon_content:
        Fuel carbon content [CO2g/g].
    :type fuel_carbon_content: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :param params:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type params: lmfit.Parameters

    :param sub_values:
        Boolean vector.
    :type sub_values: numpy.array, optional

    :return:
        CO2 emissions vector [CO2g/s].
    :rtype: numpy.array
    """

    p = params.valuesdict()
    # namespace shortcuts
    if sub_values is not None:
        e_s, e_p, e_t, n_s, n_p, acr_v, wu_p = time_series[:, sub_values]
    else:
        e_s, e_p, e_t, n_s, n_p, acr_v, wu_p = time_series
    lhv, acr_v = engine_fuel_lower_heating_value, acr_v.astype(bool)
    wu_p = wu_p.astype(bool)
    idle_fc_model = idle_fuel_consumption_model.consumption
    fc, ac, vva, lb, egr = np.zeros((5, len(e_p)), dtype=float)
    ac[:] = 1
    # Idle fc correction for temperature
    n = (e_s < idle_engine_speed[0] + min_engine_on_speed)
    _b = (e_s >= min_engine_on_speed)
    par = dfl.functions.calculate_co2_emissions
    idle_cutoff = idle_engine_speed[0] * par.cutoff_idle_ratio

    if p['t0'] == 0 and p['t1'] == 0:
        ac_phases, n_t = np.ones_like(e_p, dtype=bool), 1
        ec_p0 = _calculate_p0(
            fmep_model, p, engine_capacity, engine_stroke, idle_cutoff, lhv
        )
        _b &= ~((e_p <= ec_p0) & (e_s > idle_cutoff))
        b = n & _b
        fc[b], ac[b], vva[b], lb[b], egr[b] = idle_fc_model(p)
        b = ~n & _b
    else:
        p['t'] = np.where(wu_p, p['t0'], p['t1'])
        n_t = _normalized_engine_temperatures(e_t, p['trg'])
        ac_phases = ~wu_p & acr_v
        ec_p0 = _apply_ac_phases(
            _calculate_p0, fmep_model, p, engine_capacity, engine_stroke,
            idle_cutoff, lhv, ac_phases=ac_phases
        )
        _b &= ~((e_p <= ec_p0) & (e_s > idle_cutoff))
        b = n & _b
        # noinspection PyUnresolvedReferences
        idle_fc, ac[b], vva[b], lb[b], egr[b] = idle_fc_model(p, ac_phases[b])
        fc[b] = idle_fc * np.power(n_t[b], -p['t'][b])
        b = ~n & _b
        p['t'] = p['t'][b]
        n_t = n_t[b]

    fc[b], _, ac[b], vva[b], lb[b], egr[b] = fmep_model(
        p, n_s[b], n_p[b], n_t, acr_valid=acr_v[b]
    )
    fc[b] *= e_s[b] * (engine_capacity / (lhv * 1200))  # [g/sec]
    fc[fc < 0] = 0

    co2 = fc * fuel_carbon_content

    return np.nan_to_num(co2), ac, vva, lb, egr


# noinspection PyUnusedLocal
@sh.add_function(dsp, outputs=['co2_emissions_model'])
def define_co2_emissions_model(
        engine_speeds_out, engine_powers_out, mean_piston_speeds,
        brake_mean_effective_pressures, engine_temperatures, on_engine,
        cylinder_deactivation_valid_phases, after_treatment_warm_up_phases,
        engine_fuel_lower_heating_value, idle_engine_speed, engine_stroke,
        engine_capacity, idle_fuel_consumption_model, fuel_carbon_content,
        min_engine_on_speed, fmep_model):
    """
    Returns CO2 emissions model (see :func:`calculate_co2_emissions`).

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param mean_piston_speeds:
        Mean piston speed vector [m/s].
    :type mean_piston_speeds: numpy.array

    :param brake_mean_effective_pressures:
        Engine brake mean effective pressure vector [bar].
    :type brake_mean_effective_pressures: numpy.array

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param cylinder_deactivation_valid_phases:
        Valid activation phases for cylinder deactivation.
    :type cylinder_deactivation_valid_phases: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param idle_fuel_consumption_model:
        Idle fuel consumption model.
    :type idle_fuel_consumption_model: IdleFuelConsumptionModel

    :param fuel_carbon_content:
        Fuel carbon content [CO2g/g].
    :type fuel_carbon_content: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :return:
        CO2 emissions model (co2_emissions = models(params)).
    :rtype: callable
    """

    ts = (
        engine_speeds_out, engine_powers_out, engine_temperatures,
        mean_piston_speeds, brake_mean_effective_pressures,
        cylinder_deactivation_valid_phases, after_treatment_warm_up_phases
    )

    model = functools.partial(
        _calculate_co2_emissions, np.array(ts, copy=False),
        engine_fuel_lower_heating_value, idle_engine_speed, engine_stroke,
        engine_capacity, idle_fuel_consumption_model, fuel_carbon_content,
        min_engine_on_speed, fmep_model
    )

    return model


# noinspection PyUnusedLocal
@sh.add_function(
    dsp, outputs=['co2_emissions_model'], input_domain=co2_utl.check_first_arg
)
def define_co2_emissions_model_hybrid_calibration(
        is_hybrid, engine_speeds_out, engine_powers_out, mean_piston_speeds,
        brake_mean_effective_pressures, engine_temperatures, on_engine,
        cylinder_deactivation_valid_phases, times,
        engine_thermostat_temperature, is_cycle_hot,
        engine_fuel_lower_heating_value, idle_engine_speed, engine_stroke,
        engine_capacity, idle_fuel_consumption_model, fuel_carbon_content,
        min_engine_on_speed, fmep_model):
    """
    Returns CO2 emissions model (see :func:`calculate_co2_emissions`).

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param mean_piston_speeds:
        Mean piston speed vector [m/s].
    :type mean_piston_speeds: numpy.array

    :param brake_mean_effective_pressures:
        Engine brake mean effective pressure vector [bar].
    :type brake_mean_effective_pressures: numpy.array

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param cylinder_deactivation_valid_phases:
        Valid activation phases for cylinder deactivation.
    :type cylinder_deactivation_valid_phases: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_thermostat_temperature:
        Engine thermostat temperature [°C].
    :type engine_thermostat_temperature: float

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param idle_fuel_consumption_model:
        Idle fuel consumption model.
    :type idle_fuel_consumption_model: IdleFuelConsumptionModel

    :param fuel_carbon_content:
        Fuel carbon content [CO2g/g].
    :type fuel_carbon_content: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :return:
        CO2 emissions model (co2_emissions = models(params)).
    :rtype: callable
    """
    # noinspection PyProtectedMember
    from ..control.hybrid import _filter_warm_up
    after_treatment_warm_up_phases = _filter_warm_up(
        times, on_engine, on_engine, engine_thermostat_temperature,
        is_cycle_hot, engine_temperatures
    )
    return define_co2_emissions_model(
        engine_speeds_out, engine_powers_out, mean_piston_speeds,
        brake_mean_effective_pressures, engine_temperatures, on_engine,
        cylinder_deactivation_valid_phases, after_treatment_warm_up_phases,
        engine_fuel_lower_heating_value, idle_engine_speed, engine_stroke,
        engine_capacity, idle_fuel_consumption_model, fuel_carbon_content,
        min_engine_on_speed, fmep_model
    )


def _select_initial_friction_params(co2_params_initial_guess):
    """
    Selects initial guess of friction params l & l2 for the calculation of
    the motoring curve.

    :param co2_params_initial_guess:
        Initial guess of CO2 emission model params.
    :type co2_params_initial_guess: lmfit.Parameters

    :return:
        Initial guess of friction params l & l2.
    :rtype: float, float
    """

    params = co2_params_initial_guess.valuesdict()

    return sh.selector(('l', 'l2'), params, output_type='list')


# noinspection PyUnusedLocal
def _missing_co2_params(params, *args, _not=False):
    """
    Checks if all co2_params are not defined.

    :param params:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).
    :type params: dict | lmfit.Parameters

    :param _not:
        If True the function checks if not all co2_params are defined.
    :type _not: bool

    :return:
        If is missing some parameter.
    :rtype: bool
    """

    s = {'a', 'b', 'c', 'a2', 'b2', 'l', 'l2', 't0', 'dt', 'trg'}

    if _not:
        return set(params).issuperset(s)

    return not set(params).issuperset(s)


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True,
    outputs=['co2_params_initial_guess', 'initial_friction_params']
)
def define_initial_co2_emission_model_params_guess(
        co2_params, engine_type, engine_thermostat_temperature,
        engine_thermostat_temperature_window,
        engine_n_cylinders=dfl.values.engine_n_cylinders,
        is_cycle_hot=dfl.values.is_cycle_hot):
    """
    Selects initial guess and bounds of co2 emission model params.

    :param co2_params:
        CO2 emission model params (a2, b2, a, b, c, l, l2, t, trg).
    :type co2_params: dict

    :param engine_type:
        Engine type (positive turbo, positive natural aspiration, compression).
    :type engine_type: str

    :param engine_thermostat_temperature:
        Engine normalization temperature [°C].
    :type engine_thermostat_temperature: float

    :param engine_thermostat_temperature_window:
        Thermostat engine temperature limits [°C].
    :type engine_thermostat_temperature_window: (float, float)

    :param engine_n_cylinders:
        Number of engine cylinders [-].
    :type engine_n_cylinders: int

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool, optional

    :return:
        Initial guess of co2 emission model params and of friction params.
    :rtype: lmfit.Parameters, list[float]
    """
    import lmfit
    import collections
    bounds = {}  # Parameters bounds.
    par = dfl.functions.define_initial_co2_emission_model_params_guess
    default = collections.OrderedDict(
        copy.deepcopy(par.CO2_PARAMS[engine_type])
    )
    default['trg'] = {
        'value': engine_thermostat_temperature,
        'min': engine_thermostat_temperature_window[0],
        'max': engine_thermostat_temperature_window[1],
        'vary': False
    }

    keys, n = ('l', 'l2'), engine_n_cylinders / default.pop('n_cylinders', 4)
    for d in sh.selector(keys, default, allow_miss=True).values():
        for k in {'value', 'min', 'max'}.intersection(d):
            d[k] *= n

    if is_cycle_hot:
        default['t1'].update({'value': 0.0, 'vary': False})
        default['dt'].update({'value': 0.0, 'vary': False})

    p = lmfit.Parameters()
    eps = dfl.EPS
    for k, kw in default.items():
        kw['name'] = k

        kw['value'] = co2_params.get(k, kw.get('value', None))

        if k in bounds:
            b = bounds[k]
            kw['min'] = b.get('min', kw.get('min', None))
            kw['max'] = b.get('max', kw.get('max', None))
            kw['vary'] = b.get('vary', kw.get('vary', True))
        elif 'vary' not in kw:
            kw['vary'] = k not in co2_params

        if kw['value'] is not None:
            if 'min' in kw and kw['value'] < kw['min']:
                kw['min'] = kw['value'] - eps
            if 'max' in kw and kw['value'] > kw['max']:
                kw['max'] = kw['value'] + eps

        if 'min' in kw and 'max' in kw and kw['min'] == kw['max']:
            kw['vary'] = False
            # noinspection PyTypeChecker
            kw['max'] = kw['min'] = None

        kw['min'] = kw.get('min', None)
        kw['max'] = kw.get('max', None)
        p.add(**kw)

    friction_params = _select_initial_friction_params(p)
    if not _missing_co2_params(co2_params):
        p = sh.NONE

    return p, friction_params


@sh.add_function(dsp, outputs=['extended_phases_distances'])
def calculate_extended_phases_distances(
        times, extended_phases_integration_times, velocities):
    """
    Calculates extended cycle phases distances [km].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param extended_phases_integration_times:
        Extended cycle phases integration times [s].
    :type extended_phases_integration_times: tuple

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :return:
        Extended cycle phases distances [km].
    :rtype: numpy.array
    """
    from ..co2 import calculate_phases_distances, identify_phases_indices
    indices = identify_phases_indices(times, extended_phases_integration_times)
    return calculate_phases_distances(indices, velocities)


@sh.add_function(dsp, outputs=['extended_phases_co2_emissions'])
def calculate_extended_phases_co2_emissions(
        extended_cumulative_co2_emissions, extended_phases_distances):
    """
    Calculates the extended CO2 emission of cycle phases [CO2g/km].

    :param extended_cumulative_co2_emissions:
        Extended cumulative CO2 of cycle phases [CO2g].
    :type extended_cumulative_co2_emissions: numpy.array

    :param extended_phases_distances:
        Extended cycle phases distances [km].
    :type extended_phases_distances: numpy.array

    :return:
        Extended CO2 emission of cycle phases [CO2g/km].
    :rtype: numpy.array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        x = extended_cumulative_co2_emissions
        return np.nan_to_num(x / extended_phases_distances)


dsp.add_function(function=sh.bypass, weight=sh.inf(10, 300), inputs=[
    'phases_integration_times', 'cumulative_co2_emissions', 'phases_distances'
], outputs=[
    'extended_phases_integration_times', 'extended_cumulative_co2_emissions',
    'extended_phases_distances'
])


@sh.add_function(dsp, outputs=['cumulative_co2_emissions'])
def calculate_cumulative_co2_v1(phases_co2_emissions, phases_distances):
    """
    Calculates cumulative CO2 of cycle phases [CO2g].

    :param phases_co2_emissions:
        CO2 emission of cycle phases [CO2g/km].
    :type phases_co2_emissions: numpy.array

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array

    :return:
        Cumulative CO2 of cycle phases [CO2g].
    :rtype: numpy.array
    """

    return phases_co2_emissions * phases_distances


def _define_rescaling_function(
        co2_emissions_model, cumulative_co2_emissions, phases_integration_times,
        times, rescaling_matrix):
    dx, it = np.append(np.diff(times), [0]), []
    for ii, jj in np.searchsorted(times, phases_integration_times):
        d = dx[ii:jj].copy()
        d[1:-1] = d[1:-1] + d[:-2]
        it.append((ii, jj, d[:, None] * rescaling_matrix[ii:jj, :] / 2))

    def _rescaling_function(params_initial_guess):
        co2_emissions = co2_emissions_model(params_initial_guess)[0]
        mtx = [np.sum(co2_emissions[i:j, None] * m, 0) for i, j, m in it]
        k_factors = np.linalg.lstsq(mtx, cumulative_co2_emissions, rcond=-1)[0]
        co2_emissions *= np.dot(rescaling_matrix, k_factors)
        return co2_emissions, k_factors

    return _rescaling_function


def _rescaling_matrix(
        phases_integration_times, times, velocities, stop_velocity):
    from scipy.interpolate import interp1d
    # noinspection PyProtectedMember
    d, eps = dfl.functions._rescaling_matrix, dfl.EPS
    a, b = np.array([-1, 1]) * d.a / 2, d.b
    pit = np.array(phases_integration_times)
    mean = np.mean(pit, 1)
    points = np.zeros((len(phases_integration_times), 4), float)
    points[0, 0], points[-1, 3] = -np.inf, np.inf
    points[1:, 0] = (pit[1:, 0] - mean[:-1]) * (1 - b) + mean[:-1] - eps
    points[:, 1:3] = np.column_stack((mean,) * 2) + np.diff(pit, axis=1) * a
    points[:-1, 3] = (mean[1:] - pit[:-1, 1]) * b + pit[:-1, 1]

    r, y = [], (0, 1, 1, 0)
    for x in points:
        r.append(interp1d(x, y, bounds_error=False, fill_value=0)(times))
    r = np.column_stack(r)
    r[np.isnan(r)] = 1
    b = np.asarray(velocities <= stop_velocity, int)
    # noinspection PyTypeChecker
    it = np.split(range(b.size), np.where(np.diff(b) != 0)[0] + 1)[1 - b[0]::2]
    for x in it:
        i = list(itertools.chain(*(np.where(v > 0)[0] for v in r[x])))
        r[x] = 0
        r[x, int(np.median(i))] = 1
    return r / np.sum(r, 1)[:, None]


def _rescaling_score(times, rescaling_matrix, k):
    x = np.dot(rescaling_matrix, k)
    m = np.trapz(x, times) / (times[-1] - times[0])
    std = np.sqrt(np.trapz((x - m) ** 2, times) / (times[-1] - times[0]))
    return m, std


@sh.add_function(dsp, weight=5, outputs=[
    'identified_co2_emissions', 'co2_rescaling_scores', 'co2_params_identified'
])
def identify_co2_emissions(
        co2_emissions_model, co2_params_initial_guess, times,
        extended_phases_integration_times, extended_cumulative_co2_emissions,
        engine_temperatures, is_cycle_hot, velocities, stop_velocity):
    """
    Identifies instantaneous CO2 emission vector [CO2g/s].

    :param co2_emissions_model:
        CO2 emissions model (co2_emissions = models(params)).
    :type co2_emissions_model: callable

    :param co2_params_initial_guess:
        Initial guess of co2 emission model params.
    :type co2_params_initial_guess: dict

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param extended_phases_integration_times:
        Extended cycle phases integration times [s].
    :type extended_phases_integration_times: tuple

    :param extended_cumulative_co2_emissions:
        Extended cumulative CO2 of cycle phases [CO2g].
    :type extended_cumulative_co2_emissions: numpy.array

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        The instantaneous CO2 emission vector [CO2g/s], rescaling scores 
        (i.e., mean, std, and number of perturbations) [-], and the identified 
        initial guess of co2 emission model params.
    :rtype: numpy.array, tuple[float], dict
    """

    p = co2_params_initial_guess
    rescaling_matrix = _rescaling_matrix(
        extended_phases_integration_times, times, velocities, stop_velocity
    )
    rescale = _define_rescaling_function(
        co2_emissions_model, extended_cumulative_co2_emissions,
        extended_phases_integration_times, times, rescaling_matrix
    )

    d = dfl.functions.identify_co2_emissions
    n, (co2, k0) = 0, rescale(p)

    if d.enable_first_step or d.enable_second_step or d.enable_third_step:
        calibrate = functools.partial(
            calibrate_co2_params, is_cycle_hot, engine_temperatures,
            co2_emissions_model, _1st_step=d.enable_first_step,
            _2nd_step=d.enable_second_step, _3rd_step=d.enable_third_step,
        )
        xatol = d.xatol
        for n in range(d.n_perturbations):
            p = calibrate(co2, p)[0]
            co2, k1 = rescale(p)
            if np.max(np.abs(k1 - k0)) <= xatol:
                k0 = k1
                break
            k0 = k1

    return co2, _rescaling_score(times, rescaling_matrix, k0) + (n,), p


@sh.add_function(dsp, outputs=[
    'identified_co2_emissions', 'co2_rescaling_scores', 'co2_params_identified'
])
def fake_identification_co2_emissions(co2_emissions, co2_params_initial_guess):
    """
    Identifies instantaneous CO2 emission vector [CO2g/s].

    :param co2_emissions:
        CO2 instantaneous emissions vector [CO2g/s].
    :type co2_emissions: numpy.array

    :param co2_params_initial_guess:
        Initial guess of co2 emission model params.
    :type co2_params_initial_guess: dict
    
    :return:
        The instantaneous CO2 emission vector [CO2g/s], rescaling scores 
        (i.e., mean, std, and number of perturbations) [-], and the identified 
        initial guess of co2 emission model params.
    :rtype: numpy.array, tuple[float], dict
    """
    return co2_emissions, (1.0, 0, 0), co2_params_initial_guess


def _define_co2_error(co2_emissions_model, co2_emissions):
    def _error_func(params, sub_values=None):
        x = co2_emissions if sub_values is None else co2_emissions[sub_values]
        y = co2_emissions_model(params, sub_values=sub_values)[0]
        return co2_utl.mae(x, y)

    return _error_func


def _set_attr(params, data, default=False, attr='vary'):
    """
    Set attribute to CO2 emission model parameters.

    :param params:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).
    :type params: lmfit.Parameters

    :param data:
        Parameter ids to be set or key/value to set.
    :type data: iterable | dict

    :param default:
        Default value to set if a list of parameters ids is provided.
    :type default: bool | float

    :param attr:
        Parameter attribute to set.
    :type attr: str

    :return:
        CO2 emission model parameters.
    :rtype: lmfit.Parameters
    """
    import lmfit
    if not isinstance(data, dict):
        data = dict.fromkeys(data, default)

    d = {'min', 'max', 'value', 'vary', 'expr'} - {attr}

    for k, v in data.items():
        p = params[k]
        s = {i: getattr(p, i) for i in d}
        s[attr] = v
        p.set(**s)

        if lmfit.parameter.isclose(p.min, p.max, atol=1e-13, rtol=1e-13):
            p.set(value=np.mean((p.max, p.min)), min=None, max=None, vary=False)

    return params


def _identify_cold_phase(p, is_cycle_hot, engine_temperatures):
    cold = np.zeros_like(engine_temperatures, dtype=bool)
    if not is_cycle_hot:
        i = co2_utl.argmax(engine_temperatures >= p['trg'].value)
        cold[:i] = True
    return cold


def _calibrate_model_params(
        error_function, params, *args, method='nelder', **kws):
    """
    Calibrates the model params minimising the error_function.

    :param error_function:
        Model error function.
    :type error_function: callable

    :param params:
        Initial guess of model params.

        If not specified a brute force is used to identify the best initial
        guess with in the bounds.
    :type params: dict, optional

    :param method:
        Name of the fitting method to use.
    :type method: str, optional

    :return:
        Calibrated model params.
    :rtype: dict
    """

    if not any(p.vary for p in params.values()):
        return params, True

    import lmfit

    if callable(error_function):
        _error_f = error_function
    else:
        def _error_f(p, *a, **k):
            return sum(f(p, *a, **k) for f in error_function)

    min_e_and_p = [np.inf, copy.deepcopy(params)]

    def _err_func(p, *a, **kwargs):
        r = np.float32(_error_f(p, *a, **kwargs))

        if r < min_e_and_p[0]:
            min_e_and_p[0], min_e_and_p[1] = (r, copy.deepcopy(p))

        return r

    # See #7: Neither BFGS nor SLSQP fix "solution families".
    # leastsq: Improper input: N=6 must not exceed M=1.
    # nelder is stable (297 runs, 14 vehicles) [average time 181s/14 vehicles].
    # lbfgsb is unstable (2 runs, 4 vehicles) [average time 23s/4 vehicles].
    # cg is stable (20 runs, 4 vehicles) [average time 37s/4 vehicles].
    # newton: Jacobian is required for Newton-CG method
    # cobyla is unstable (8 runs, 4 vehicles) [average time 16s/4 vehicles].
    # tnc is unstable (6 runs, 4 vehicles) [average time 23s/4 vehicles].
    # dogleg: Jacobian is required for dogleg minimization.
    # slsqp is unstable (4 runs, 4 vehicles) [average time 18s/4 vehicles].
    # differential_evolution is unstable (1 runs, 4 vehicles)
    # [average time 270s/4 vehicles].
    res = lmfit.minimize(
        _err_func, params, args=args, kws=kws, method=method, nan_policy='omit'
    )

    # noinspection PyUnresolvedReferences
    return (res.params if res.success else min_e_and_p[1]), res.success


# noinspection PyUnresolvedReferences
@sh.add_function(dsp, outputs=['co2_params_calibrated', 'calibration_status'])
def calibrate_co2_params(
        is_cycle_hot, engine_temperatures, co2_emissions_model,
        identified_co2_emissions, co2_params_identified,
        _1st_step=dfl.functions.calibrate_co2_params.enable_first_step,
        _2nd_step=dfl.functions.calibrate_co2_params.enable_second_step,
        _3rd_step=dfl.functions.calibrate_co2_params.enable_third_step):
    """
    Calibrates the CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg
    ).

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :param co2_emissions_model:
        CO2 emissions model (co2_emissions = models(params)).
    :type co2_emissions_model: callable

    :param identified_co2_emissions:
        CO2 instantaneous emissions vector [CO2g/s].
    :type identified_co2_emissions: numpy.array

    :param co2_params_identified:
        Identified initial guess of co2 emission model params.
    :type co2_params_identified: Parameters

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool

    :param _1st_step:
        Enable first step in the co2_params calibration? [-]
    :type _1st_step: bool

    :param _2nd_step:
        Enable second step in the co2_params calibration? [-]
    :type _2nd_step: bool

    :param _3rd_step:
        Enable third step in the co2_params calibration? [-]
    :type _3rd_step: bool

    :return:
        Calibrated CO2 emission model parameters (a2, b2, a, b, c, l, l2, t,
        trg) and their calibration statuses.
    :rtype: (lmfit.Parameters, list)
    """
    err = _define_co2_error(co2_emissions_model, identified_co2_emissions)

    # Safety measure to not modify the initial guess.
    p = copy.deepcopy(co2_params_identified)

    # Identify cold and hot phases.
    cold = _identify_cold_phase(p, is_cycle_hot, engine_temperatures)
    hot = ~cold

    # Definition of thermal and willans parameters.
    thermal_p = {'t0', 't1', 'dt', 'trg'}
    willans_p = {'a2', 'b2', 'a', 'b', 'c', 'l', 'l2'}

    # Identification of all parameters that can vary.
    pvary = {k for k, v in p.items() if v.vary}

    # Zero step: Initialization of the statuses.
    statuses = [(True, copy.deepcopy(p))]

    # Definition of the optimization function.
    def _op(par, params2optimize, **kws):
        fixp = pvary - params2optimize
        _set_attr(par, fixp)
        if pvary - fixp:
            par, s = _calibrate_model_params(err, par, **kws)
        else:
            s = True
        statuses.append((s, copy.deepcopy(par)))
        _set_attr(par, pvary, True)
        return par

    # First step: Calibration of willans parameters using the hot phase.
    p = _op(p, _1st_step and hot.any() and willans_p or set(), sub_values=hot)

    # Second step: Calibration of thermal parameters using the cold phase.
    if not cold.any():
        # When the cycle has not cold phases, thermal parameters have no effect.
        # The third step will modify arbitrarily this parameters.
        # Hence, to avoid erroneous results, thermal parameters are fixed to
        # zero because they cannot be identified.
        _set_attr(p, thermal_p)
        _set_attr(p, pvary.intersection(('t1', 'dt')), 0, 'value')
        pvary -= thermal_p

    p = _op(p, _2nd_step and cold.any() and thermal_p or set(), sub_values=cold)

    # Third step: Calibration of all parameters.
    p = _op(p, _3rd_step and pvary or set())

    return p, statuses


@sh.add_function(
    dsp, outputs=['co2_params_calibrated', 'calibration_status'],
    input_domain=functools.partial(_missing_co2_params, _not=True)
)
def define_co2_params_calibrated(co2_params):
    """
    Defines the calibrated co2_params if all co2_params are given.

    :param co2_params:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).
    :type co2_params: dict | lmfit.Parameters

    :return:
        Calibrated CO2 emission model parameters (a2, b2, a, b, c, l, l2, t,
        trg) and their calibration statuses.
    :rtype: (lmfit.Parameters, list)
    """
    import lmfit
    if isinstance(co2_params, lmfit.Parameters):
        p = co2_params
    else:
        p = lmfit.Parameters()
        for k, v in co2_params.items():
            p.add(k, value=v, vary=False)

    success = [(None, copy.deepcopy(p))] * 4

    return p, success


@sh.add_function(dsp, outputs=[
    'co2_emissions', 'active_cylinders', 'active_variable_valves',
    'active_lean_burns', 'active_exhausted_gas_recirculations'
])
def predict_co2_emissions(co2_emissions_model, co2_params_calibrated):
    """
    Predicts CO2 instantaneous emissions vector [CO2g/s].

    :param co2_emissions_model:
        CO2 emissions model (co2_emissions = models(params)).
    :type co2_emissions_model: callable

    :param co2_params_calibrated:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type co2_params_calibrated: lmfit.Parameters

    :return:
        CO2 instantaneous emissions vector [CO2g/s].
    :rtype: numpy.array
    """

    return co2_emissions_model(co2_params_calibrated)


dsp.add_data('co2_params', dfl.values.co2_params.copy())


@sh.add_function(dsp, outputs=['fuel_consumptions'])
def calculate_fuel_consumptions(co2_emissions, fuel_carbon_content):
    """
    Calculates the instantaneous fuel consumption vector [g/s].

    :param co2_emissions:
        CO2 instantaneous emissions vector [CO2g/s].
    :type co2_emissions: numpy.array

    :param fuel_carbon_content:
        Fuel carbon content [CO2g/g].
    :type fuel_carbon_content: float

    :return:
        The instantaneous fuel consumption vector [g/s].
    :rtype: numpy.array
    """
    return co2_emissions / fuel_carbon_content


@sh.add_function(dsp, outputs=['fuel_consumptions'])
def calculate_fuel_consumptions(fuel_consumptions_liters, fuel_density):
    """
    Calculates the instantaneous fuel consumption vector [g/s].

    :param fuel_consumptions_liters:
        The instantaneous fuel consumption vector [L/h].
    :type fuel_consumptions_liters: numpy.array

    :param fuel_density:
        Fuel density [g/l].
    :type fuel_density: float

    :return:
        The instantaneous fuel consumption vector [g/s].
    :rtype: numpy.array
    """
    return fuel_consumptions_liters * (fuel_density / 3600.0)


@sh.add_function(dsp, outputs=['co2_emissions'])
def calculate_co2_emissions(fuel_consumptions, fuel_carbon_content):
    """
    Calculates instantaneous CO2 emission vector [CO2g/s].

    :param fuel_consumptions:
        The instantaneous fuel consumption vector [g/s].
    :type fuel_consumptions: numpy.array

    :param fuel_carbon_content:
        Fuel carbon content [CO2g/g].
    :type fuel_carbon_content: float

    :return:
        CO2 instantaneous emissions vector [CO2g/s].
    :rtype: numpy.array
    """
    return fuel_consumptions * fuel_carbon_content


@sh.add_function(dsp, outputs=['fuel_consumptions_liters'])
def calculate_fuel_consumptions_liters(fuel_consumptions, fuel_density):
    """
    Calculates the instantaneous fuel consumption vector [g/s].

    :param fuel_consumptions_liters:
        The instantaneous fuel consumption vector [L/h].
    :type fuel_consumptions_liters: numpy.array

    :param fuel_density:
        Fuel density [g/l].
    :type fuel_density: float

    :return:
        The instantaneous fuel consumption vector [g/s].
    :rtype: numpy.array
    """
    return fuel_consumptions / (fuel_density / 3600.0)
