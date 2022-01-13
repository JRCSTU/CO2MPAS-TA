# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the legislation corrections.
"""
import functools
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl
from co2mpas.defaults import dfl

dsp = sh.BlueDispatcher(
    name='Legislation', description='Models the legislation corrections.'
)


@sh.add_function(dsp, outputs=['phases_indices'])
def identify_phases_indices(times, phases_integration_times):
    """
    Identifies the indices of the cycle phases [-].

    If phases_distances is not given the result is the cumulative CO2 of cycle
    phases [CO2g] otherwise it is CO2 emission of cycle phases [CO2g/km].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param phases_integration_times:
        Cycle phases integration times [s].
    :type phases_integration_times: tuple

    :return:
        Indices of the cycle phases [-].
    :rtype: numpy.array
    """
    return np.searchsorted(times, phases_integration_times, side='right')


@sh.add_function(dsp, inputs_kwargs=True, outputs=['phases_co2_emissions'])
def calculate_phases_co2_emissions(
        times, phases_indices, co2_emissions, phases_distances=1.0):
    """
    Calculates CO2 emission or cumulative CO2 of cycle phases [CO2g/km or CO2g].

    If phases_distances is not given the result is the cumulative CO2 of cycle
    phases [CO2g] otherwise it is CO2 emission of cycle phases [CO2g/km].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param phases_indices:
        Indices of the cycle phases [-].
    :type phases_indices: numpy.array

    :param co2_emissions:
        CO2 instantaneous emissions vector [CO2g/s].
    :type co2_emissions: numpy.array

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array | float, optional

    :return:
        CO2 emission or cumulative CO2 of cycle phases [CO2g/km or CO2g].
    :rtype: numpy.array
    """
    from scipy.integrate import trapz
    co2 = []
    for i, j in phases_indices:
        co2.append(trapz(co2_emissions[i:j], times[i:j]))

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(np.array(co2) / phases_distances)


@sh.add_function(dsp, outputs=['co2_emission_value'])
def calculate_co2_emission_value(phases_co2_emissions, phases_distances):
    """
    Calculates the CO2 emission of the cycle [CO2g/km].

    :param phases_co2_emissions:
        CO2 emission of cycle phases [CO2g/km].
    :type phases_co2_emissions: numpy.array

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array | float

    :return:
        CO2 emission value of the cycle [CO2g/km].
    :rtype: float
    """

    n = sum(phases_co2_emissions * phases_distances)

    if isinstance(phases_distances, float):
        d = phases_distances * len(phases_co2_emissions)
    else:
        d = sum(phases_distances)

    return float(n / d)


dsp.add_data(
    'has_periodically_regenerating_systems',
    dfl.values.has_periodically_regenerating_systems
)
dsp.add_data('ki_additive', dfl.values.ki_additive)


@sh.add_function(dsp, outputs=['ki_multiplicative'])
def default_ki_multiplicative(
        has_periodically_regenerating_systems, ki_additive):
    """
    Returns the default ki multiplicative factor [-].

    :param has_periodically_regenerating_systems:
        Does the vehicle has periodically regenerating systems? [-].
    :type has_periodically_regenerating_systems: bool

    :param ki_additive:
        Additive correction for vehicles with periodically regenerating
        systems [CO2g/km].
    :type ki_additive: float

    :return:
        Multiplicative correction for vehicles with periodically regenerating
        systems [-].
    :rtype: float
    """
    if ki_additive:
        return 1.0
    par = dfl.functions.default_ki_multiplicative.ki_multiplicative
    return par.get(has_periodically_regenerating_systems, 1.0)


@sh.add_function(dsp, outputs=['phases_co2_emissions'])
def merge_wltp_phases_co2_emission(
        co2_emission_low, co2_emission_medium, co2_emission_high,
        co2_emission_extra_high):
    """
    Merges WLTP phases co2 emission.

    :param co2_emission_low:
        CO2 emission on low WLTP phase [CO2g/km].
    :type co2_emission_low: float

    :param co2_emission_medium:
        CO2 emission on medium WLTP phase [CO2g/km].
    :type co2_emission_medium: float

    :param co2_emission_high:
        CO2 emission on high WLTP phase [CO2g/km].
    :type co2_emission_high: float

    :param co2_emission_extra_high:
        CO2 emission on extra high WLTP phase [CO2g/km].
    :type co2_emission_extra_high: float

    :return:
        CO2 emission of cycle phases [CO2g/km].
    :rtype: tuple[float]
    """
    return (co2_emission_low, co2_emission_medium, co2_emission_high,
            co2_emission_extra_high)


dsp.add_function(
    function_id='split_wltp_phases_co2_emission',
    function=sh.add_args(sh.bypass),
    inputs=['cycle_type', 'phases_co2_emissions'],
    outputs=[
        'co2_emission_low', 'co2_emission_medium', 'co2_emission_high',
        'co2_emission_extra_high'
    ],
    input_domain=lambda cycle, phases: (cycle, len(phases)) == ('WLTP', 4)
)

dsp.add_function(
    function_id='split_nedc_phases_co2_emission',
    function=sh.add_args(sh.bypass),
    inputs=['cycle_type', 'phases_co2_emissions'],
    outputs=['co2_emission_UDC', 'co2_emission_EUDC'],
    input_domain=lambda cycle, phases: (cycle, len(phases)) == ('NEDC', 2)
)


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['phases_co2_emissions'])
def merge_nedc_phases_co2_emission(co2_emission_UDC, co2_emission_EUDC):
    """
    Merges NEDC phases co2 emission.

    :param co2_emission_UDC:
        CO2 emission on UDC NEDC phase [CO2g/km].
    :type co2_emission_UDC: float

    :param co2_emission_EUDC:
        CO2 emission on EUDC NEDC phase [CO2g/km].
    :type co2_emission_EUDC: float

    :return:
        CO2 emission of cycle phases [CO2g/km].
    :rtype: tuple[float]
    """
    return co2_emission_UDC, co2_emission_EUDC


def _rcb_correction(
        phases_delta_energy, phases_distances, fuel_type, engine_type,
        alternator_efficiency):
    # noinspection PyProtectedMember
    par = dfl.functions._rcb_correction.WILLANS
    dco2 = np.nan_to_num(-phases_delta_energy / phases_distances)
    dco2 *= par[fuel_type][engine_type] * 0.0036 / alternator_efficiency
    return dco2


@sh.add_function(dsp, outputs=['theoretical_phases_distances'])
def calculate_theoretical_phases_distances(
        times, theoretical_velocities, phases_indices):
    """
    Calculates theoretical cycle phases distances [km].

    :param times:
        Time vector.
    :type times: numpy.array

    :param theoretical_velocities:
        Theoretical velocity vector [km/h].
    :type theoretical_velocities: numpy.array

    :param phases_indices:
        Indices of the cycle phases [-].
    :type phases_indices: numpy.array

    :return:
        Theoretical cycle phases distances [km].
    :rtype:  numpy.array
    """
    from .vehicle import calculate_distances
    return calculate_phases_distances(
        phases_indices, calculate_distances(times, theoretical_velocities)
    )


dsp.add_data('rcb_correction', dfl.values.rcb_correction)


@sh.add_function(dsp, outputs=['speed_distance_correction'])
def default_speed_distance_correction(is_hybrid):
    """
    Returns if speed distance correction has to be applied.

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        Apply speed distance correction?
    :rtype: bool
    """
    return not is_hybrid


@sh.add_function(
    dsp, inputs_kwargs=True,
    outputs=['speed_distance_corrected_co2_emission_value']
)
def calculate_speed_distance_corrected_co2_emission_value(
        phases_co2_emissions, phases_times, batteries_phases_delta_energy,
        theoretical_phases_distances, phases_distances, alternator_efficiency,
        engine_type, fuel_type, motive_powers, theoretical_motive_powers, times,
        phases_indices, engine_max_power, speed_distance_correction=True,
        is_hybrid=False, cycle_type='WLTP'):
    """
    Calculates the CO2 emission value corrected for speed & distance [CO2g/km].

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param phases_co2_emissions:
        CO2 emission of cycle phases [CO2g/km].
    :type phases_co2_emissions: numpy.array

    :param phases_times:
        Cycle phases times [s].
    :type phases_times: numpy.array

    :param batteries_phases_delta_energy:
        Phases delta energy of the batteries [Wh].
    :type batteries_phases_delta_energy: numpy.array

    :param theoretical_phases_distances:
        Theoretical cycle phases distances [km].
    :type theoretical_phases_distances: numpy.array

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array

    :param alternator_efficiency:
        Alternator efficiency [-].
    :type alternator_efficiency: float

    :param engine_type:
        Engine type (positive turbo, positive natural aspiration, compression).
    :type engine_type: str

    :param fuel_type:
        Fuel type (diesel, gasoline, LPG, NG, ethanol, methanol, biodiesel,
        propane).
    :type fuel_type: str

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param theoretical_motive_powers:
        Theoretical motive power [kW].
    :type theoretical_motive_powers: numpy.array

    :param times:
        Time vector.
    :type times: numpy.array

    :param phases_indices:
        Indices of the cycle phases [-].
    :type phases_indices: numpy.array

    :param engine_max_power:
        Engine nominal power [kW].
    :type engine_max_power: float

    :param speed_distance_correction:
        Apply speed distance correction?
    :type speed_distance_correction: bool

    :return:
        CO2 emission value corrected for speed & distance [CO2g/km].
    :rtype: float
    """
    if is_hybrid:
        return sh.NONE
    p_co2, p_dist = np.array(phases_co2_emissions), phases_distances
    if cycle_type == 'WLTP' and speed_distance_correction:
        p_co2 = np.column_stack((p_co2, _rcb_correction(
            batteries_phases_delta_energy, phases_distances, fuel_type,
            engine_type, alternator_efficiency
        ))) * np.nan_to_num(phases_distances / phases_times)[:, None]

        from sklearn.linear_model import LinearRegression
        mdl = LinearRegression()
        cpmp = functools.partial(
            calculate_phases_co2_emissions, times, phases_indices,
            phases_distances=phases_times
        )
        y, p_co2 = p_co2.sum(axis=1).ravel(), p_co2[:, 0].ravel()
        x = cpmp(np.maximum(-.02 * engine_max_power, motive_powers))
        mdl.fit(x[:, None], y)
        p2 = -mdl.intercept_ / mdl.coef_ if mdl.coef_ else -float('inf')
        dp_mp = cpmp(np.maximum(p2, motive_powers))
        mdl.fit(dp_mp[:, None], y)
        dp_mp -= cpmp(np.maximum(p2, theoretical_motive_powers))
        p_co2 -= mdl.coef_ * dp_mp
        p_co2 *= np.nan_to_num(phases_times / theoretical_phases_distances)
        p_dist = theoretical_phases_distances
    return calculate_co2_emission_value(p_co2, p_dist)


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['rcb_corrected_co2_emission_value']
)
def calculate_rcb_corrected_co2_emission_value(
        speed_distance_corrected_co2_emission_value, engine_type, fuel_type,
        alternator_efficiency, phases_distances, theoretical_phases_distances,
        batteries_phases_delta_energy, speed_distance_correction=True,
        rcb_correction=True, is_hybrid=False, cycle_type='WLTP'):
    """
    Calculates the CO2 emission value corrected for RCB [CO2g/km].

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param speed_distance_corrected_co2_emission_value:
        CO2 emission value corrected for speed & distance [CO2g/km].
    :type speed_distance_corrected_co2_emission_value: float

    :param batteries_phases_delta_energy:
        Phases delta energy of the batteries [Wh].
    :type batteries_phases_delta_energy: numpy.array

    :param theoretical_phases_distances:
        Theoretical cycle phases distances [km].
    :type theoretical_phases_distances: numpy.array

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array

    :param alternator_efficiency:
        Alternator efficiency [-].
    :type alternator_efficiency: float

    :param engine_type:
        Engine type (positive turbo, positive natural aspiration, compression).
    :type engine_type: str

    :param fuel_type:
        Fuel type (diesel, gasoline, LPG, NG, ethanol, methanol, biodiesel,
        propane).
    :type fuel_type: str

    :param rcb_correction:
        Apply RCB correction?
    :type rcb_correction: bool

    :param speed_distance_correction:
        Apply speed distance correction?
    :type speed_distance_correction: bool

    :return:
        CO2 emission value corrected for RCB [CO2g/km].
    :rtype: float
    """
    if is_hybrid:
        return sh.NONE
    if cycle_type == 'WLTP' and rcb_correction:
        d = phases_distances
        if speed_distance_correction:
            d = theoretical_phases_distances
        return speed_distance_corrected_co2_emission_value + _rcb_correction(
            np.sum(batteries_phases_delta_energy), np.sum(d), fuel_type,
            engine_type, alternator_efficiency
        )
    return speed_distance_corrected_co2_emission_value


@sh.add_function(dsp, outputs=['batteries_phases_delta_energy'])
def calculate_batteries_phases_delta_energy(
        times, phases_indices, drive_battery_electric_powers,
        service_battery_electric_powers):
    """
    Calculates the phases delta energy of the batteries [Wh].

    :param times:
        Time vector.
    :type times: numpy.array

    :param phases_indices:
        Indices of the cycle phases [-].
    :type phases_indices: numpy.array

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param service_battery_electric_powers:
        Service battery electric power [kW].
    :type service_battery_electric_powers: numpy.array

    :return:
        Phases delta energy of the batteries [Wh].
    :rtype: numpy.array
    """
    from scipy.integrate import cumtrapz
    p = service_battery_electric_powers + drive_battery_electric_powers
    e = cumtrapz(p, times, initial=0)
    i = phases_indices.copy()
    i[:, 1] -= 1
    return np.diff(e[i], axis=1).ravel() / 3.6


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['kco2_wltp_correction_factor']
)
def identify_kco2_wltp_correction_factor(
        drive_battery_electric_powers, service_battery_electric_powers,
        co2_emissions, times, force_on_engine, after_treatment_warm_up_phases,
        velocities, is_hybrid=True):
    """
    Identifies the kco2 correction factor [g/Wh].

    :param drive_battery_electric_powers:
        Drive battery electric power [kW].
    :type drive_battery_electric_powers: numpy.array

    :param service_battery_electric_powers:
        Service battery electric power [kW].
    :type service_battery_electric_powers: numpy.array

    :param force_on_engine:
        Phases when engine is on because parallel mode is forced [-].
    :type force_on_engine: numpy.array

    :param after_treatment_warm_up_phases:
        Phases when engine speed is affected by the after treatment warm up [-].
    :type after_treatment_warm_up_phases: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param co2_emissions:
        CO2 instantaneous emissions vector [CO2g/s].
    :type co2_emissions: numpy.array

    :param times:
        Time vector.
    :type times: numpy.array

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :return:
        kco2 correction factor [g/Wh].
    :rtype: float
    """
    if not is_hybrid:
        return sh.NONE
    from scipy.integrate import cumtrapz
    from sklearn.linear_model import RANSACRegressor, Lasso
    b = ~(force_on_engine | after_treatment_warm_up_phases)
    e = np.where(
        b, drive_battery_electric_powers + service_battery_electric_powers, 0
    )
    e = cumtrapz(e, times, initial=0) / 3.6
    co2 = cumtrapz(np.where(b, co2_emissions, 0), times, initial=0)
    km = cumtrapz(np.where(b, velocities / 3.6, 0), times, initial=0) / 1000
    # noinspection PyTypeChecker
    it = co2_utl.sliding_window(list(zip(km, zip(km, e, co2))), 5)
    d = np.diff(np.array([(v[0][1], v[-1][1]) for v in it]), axis=1)[:, 0, :].T
    e, co2 = d[1:] / d[0]
    d0 = t0 = -float('inf')
    b, dkm = [], .5
    dt = dkm / np.mean(velocities) * 3600
    for i, (d, t) in enumerate(zip(km, times)):
        if d > d0 and t > t0:
            d0, t0 = d + dkm, t + dt
            b.append(i)
    b = np.array(b)
    m = RANSACRegressor(
        random_state=0,
        base_estimator=Lasso(random_state=0, positive=True)
    ).fit(e[b, None], co2[b])
    return float(m.estimator_.coef_)


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['rcb_corrected_co2_emission_value']
)
def calculate_rcb_corrected_co2_emission_value_v1(
        co2_emission_value, batteries_phases_delta_energy,
        kco2_wltp_correction_factor, phases_distances, rcb_correction=True,
        is_hybrid=True, cycle_type='WLTP'):
    """
    Calculates the CO2 emission value corrected for RCB [CO2g/km].

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param co2_emission_value:
        CO2 emission value of the cycle [CO2g/km].
    :type co2_emission_value: float

    :param batteries_phases_delta_energy:
        Phases delta energy of the batteries [Wh].
    :type batteries_phases_delta_energy: numpy.array

    :param kco2_wltp_correction_factor:
        kco2 WLTP correction factor [CO2g/Wh].
    :type kco2_wltp_correction_factor: float

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array

    :param rcb_correction:
        Apply RCB correction?
    :type rcb_correction: bool

    :return:
        CO2 emission value corrected for RCB [CO2g/km].
    :rtype: float
    """
    if not is_hybrid or cycle_type == 'NEDC':
        return sh.NONE

    if rcb_correction and kco2_wltp_correction_factor:
        de = np.sum(batteries_phases_delta_energy) / np.sum(phases_distances)
        return co2_emission_value - kco2_wltp_correction_factor * de
    return co2_emission_value


@sh.add_function(dsp, outputs=['kco2_nedc_correction_factor'])
def default_kco2_nedc_correction_factor(
        kco2_wltp_correction_factor, drive_battery_nominal_voltage, distances):
    """
    Returns the kco2 NEDC correction factor [CO2g/km/Ah].

    :param kco2_wltp_correction_factor:
        kco2 WLTP correction factor [CO2g/Wh].
    :type kco2_wltp_correction_factor: float

    :param drive_battery_nominal_voltage:
        Drive battery nominal voltage [V].
    :type drive_battery_nominal_voltage: float

    :param distances:
        Cumulative distance vector [m].
    :type distances: numpy.array

    :return:
        kco2 NEDC correction factor [CO2g/km/Ah].
    :rtype: float
    """
    d = distances[-1] / 1000
    return kco2_wltp_correction_factor * drive_battery_nominal_voltage / d


@sh.add_function(
    dsp, inputs_kwargs=True, outputs=['rcb_corrected_co2_emission_value']
)
def calculate_rcb_corrected_co2_emission_value_v2(
        co2_emission_value, drive_battery_delta_state_of_charge,
        drive_battery_capacity, kco2_nedc_correction_factor,
        rcb_correction=True, is_hybrid=True, cycle_type='NEDC'):
    """
    Calculates the CO2 emission value corrected for RCB [CO2g/km].

    :param is_hybrid:
        Is the vehicle hybrid?
    :type is_hybrid: bool

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param co2_emission_value:
        CO2 emission value of the cycle [CO2g/km].
    :type co2_emission_value: float

    :param drive_battery_delta_state_of_charge:
        Overall delta state of charge of the drive battery [%].
    :type drive_battery_delta_state_of_charge: numpy.array

    :param kco2_nedc_correction_factor:
        kco2 NEDC correction factor [CO2g/km/Ah].
    :type kco2_nedc_correction_factor: float

    :param drive_battery_capacity:
        Drive battery capacity [Ah].
    :type drive_battery_capacity: float

    :param rcb_correction:
        Apply RCB correction?
    :type rcb_correction: bool

    :return:
        CO2 emission value corrected for RCB [CO2g/km].
    :rtype: float
    """
    if not is_hybrid or cycle_type != 'NEDC':
        return sh.NONE

    if rcb_correction and kco2_nedc_correction_factor:
        di = drive_battery_delta_state_of_charge * drive_battery_capacity / 100
        return co2_emission_value - kco2_nedc_correction_factor * di
    return co2_emission_value


dsp.add_data(
    'atct_family_correction_factor', dfl.values.atct_family_correction_factor
)


def calculate_corrected_co2_emission(
        rcb_corrected_co2_emission_value, ki_multiplicative, ki_additive,
        atct_family_correction_factor=1.0):
    """
    Calculates the corrected CO2 emission of the cycle [CO2g/km].

    :param rcb_corrected_co2_emission_value:
        CO2 emission value corrected for RCB [CO2g/km].
    :type rcb_corrected_co2_emission_value: float

    :param ki_multiplicative:
        Multiplicative correction for vehicles with periodically regenerating
        systems [-].
    :type ki_multiplicative: float

    :param ki_additive:
        Additive correction for vehicles with periodically regenerating
        systems [CO2g/km].
    :type ki_multiplicative: float

    :param atct_family_correction_factor:
        Family correction factor for representative regional temperatures [-].
    :type atct_family_correction_factor: float

    :return:
        Corrected CO2 emission value of the cycle [CO2g/km].
    :rtype: float
    """
    v = rcb_corrected_co2_emission_value * ki_multiplicative + ki_additive
    return v * atct_family_correction_factor


@sh.add_function(dsp, True, True, outputs=['is_plugin'])
def default_is_plugin(input_type=None):
    """
    Returns if the vehicle is a plugin.

    :param input_type:
        Input file type.
    :type input_type: str

    :return:
        Is the vehicle a plugin?
    :rtype: bool
    """

    return str(input_type).upper() == 'OVC-HEV' or dfl.values.is_plugin


dsp.add_function(
    function=sh.add_args(calculate_corrected_co2_emission),
    inputs=[
        'is_plugin', 'rcb_corrected_co2_emission_value', 'ki_multiplicative',
        'ki_additive', 'atct_family_correction_factor'
    ],
    outputs=['corrected_co2_emission_value'],
    input_domain=co2_utl.check_first_arg_false
)
dsp.add_function(
    function=sh.add_args(calculate_corrected_co2_emission),
    inputs=[
        'is_plugin', 'rcb_corrected_co2_emission_value', 'ki_multiplicative',
        'ki_additive', 'atct_family_correction_factor'
    ],
    outputs=['corrected_sustaining_co2_emission_value'],
    input_domain=co2_utl.check_first_arg
)


# noinspection PyUnusedLocal
def _domain_calculate_corrected_co2_emission_for_conventional_nedc(
        cycle_type, is_hybrid, *args):
    return not is_hybrid and cycle_type == 'NEDC'


dsp.add_function(
    function_id='calculate_corrected_co2_emission_for_conventional_nedc',
    function=sh.add_args(calculate_corrected_co2_emission, n=2),
    inputs=['cycle_type', 'is_hybrid', 'co2_emission_value',
            'ki_multiplicative', 'ki_additive',
            'atct_family_correction_factor'],
    outputs=['corrected_co2_emission_value'],
    input_domain=_domain_calculate_corrected_co2_emission_for_conventional_nedc
)
dsp.add_function(
    function=sh.bypass,
    inputs=['corrected_co2_emission_value'],
    outputs=['declared_co2_emission_value']
)
dsp.add_function(
    function=sh.bypass,
    inputs=['corrected_sustaining_co2_emission_value'],
    outputs=['declared_sustaining_co2_emission_value']
)


def calculate_willans_factors(
        co2_params_calibrated, engine_fuel_lower_heating_value, engine_stroke,
        engine_capacity, min_engine_on_speed, fmep_model, engine_speeds_out,
        engine_powers_out, times, velocities, accelerations, motive_powers,
        engine_temperatures, missing_powers, angle_slopes):
    """
    Calculates the Willans factors.

    :param co2_params_calibrated:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type co2_params_calibrated: lmfit.Parameters

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

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

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :param missing_powers:
        Missing engine power [kW].
    :type missing_powers: numpy.array

    :param angle_slopes:
        Angle slope vector [rad].
    :type angle_slopes: numpy.array

    :return:
        Willans factors:

        - av_velocities                         [km/h]
        - av_slope                              [rad]
        - distance                              [km]
        - init_temp                             [°C]
        - av_temp                               [°C]
        - end_temp                              [°C]
        - av_vel_pos_mov_pow                    [kw/h]
        - av_pos_motive_powers                  [kW]
        - sec_pos_mov_pow                       [s]
        - av_neg_motive_powers                  [kW]
        - sec_neg_mov_pow                       [s]
        - av_pos_accelerations                  [m/s2]
        - av_engine_speeds_out_pos_pow          [RPM]
        - av_pos_engine_powers_out              [kW]
        - engine_bmep_pos_pow                   [bar]
        - mean_piston_speed_pos_pow             [m/s]
        - fuel_mep_pos_pow                      [bar]
        - fuel_consumption_pos_pow              [g/sec]
        - willans_a                             [g/kWh]
        - willans_b                             [g/h]
        - specific_fuel_consumption             [g/kWh]
        - indicated_efficiency                  [-]
        - willans_efficiency                    [-]

    :rtype: dict
    """

    from .engine import calculate_mean_piston_speeds
    from .engine.fc import calculate_brake_mean_effective_pressures
    av = np.average

    w = np.zeros_like(times, dtype=float)
    t = (times[:-1] + times[1:]) / 2
    # noinspection PyUnresolvedReferences
    w[0], w[1:-1], w[-1] = t[0] - times[0], np.diff(t), times[-1] - t[-1]

    f = {
        'av_velocities': av(velocities, weights=w),  # [km/h]
        'av_slope': av(angle_slopes, weights=w),
        'has_sufficient_power': 1 - av(missing_powers != 0, weights=w),
        'max_power_required': max(engine_powers_out + missing_powers)
    }

    f['distance'] = f['av_velocities'] * (times[-1] - times[0]) / 3600.0  # [km]

    b = engine_powers_out >= 0
    if b.any():
        p = co2_params_calibrated.valuesdict()
        _w = w[b]
        av_s = av(engine_speeds_out[b], weights=_w)
        av_p = av(engine_powers_out[b], weights=_w)
        av_mp = av(missing_powers[b], weights=_w)

        n_p = calculate_brake_mean_effective_pressures(
            av_s, av_p, engine_capacity, min_engine_on_speed
        )
        n_s = calculate_mean_piston_speeds(av_s, engine_stroke)

        f_mep, wfa = fmep_model(p, n_s, n_p, 1)[:2]

        c = engine_capacity / engine_fuel_lower_heating_value * av_s
        fc = f_mep * c / 1200.0
        ieff = av_p / (fc * engine_fuel_lower_heating_value) * 1000.0

        willans_a = 3600000.0 / engine_fuel_lower_heating_value / wfa
        willans_b = fmep_model(p, n_s, 0, 1)[0] * c * 3.0

        sfc = willans_a + willans_b / av_p

        willans_eff = 3600000.0 / (sfc * engine_fuel_lower_heating_value)

        f.update({
            'av_engine_speeds_out_pos_pow': av_s,  # [RPM]
            'av_pos_engine_powers_out': av_p,  # [kW]
            'av_missing_powers_pos_pow': av_mp,  # [kW]
            'engine_bmep_pos_pow': n_p,  # [bar]
            'mean_piston_speed_pos_pow': n_s,  # [m/s]
            'fuel_mep_pos_pow': f_mep,  # [bar]
            'fuel_consumption_pos_pow': fc,  # [g/sec]
            'willans_a': willans_a,  # [g/kW]
            'willans_b': willans_b,  # [g]
            'specific_fuel_consumption': sfc,  # [g/kWh]
            'indicated_efficiency': ieff,  # [-]
            'willans_efficiency': willans_eff  # [-]
        })

    b = motive_powers > 0
    if b.any():
        _w = w[b]
        f['av_vel_pos_mov_pow'] = av(velocities[b], weights=_w)  # [km/h]
        f['av_pos_motive_powers'] = av(motive_powers[b], weights=_w)  # [kW]
        f['sec_pos_mov_pow'] = np.sum(_w)  # [s]

    b = accelerations > 0
    if b.any():
        _w = w[b]
        f['av_pos_accelerations'] = av(accelerations[b], weights=_w)  # [m/s2]

    b = motive_powers < 0
    if b.any():
        _w = w[b]
        f['av_neg_motive_powers'] = av(motive_powers[b], weights=_w)  # [kW]
        f['sec_neg_mov_pow'] = np.sum(_w)  # [s]

    f['init_temp'] = engine_temperatures[0]  # [°C]
    f['av_temp'] = av(engine_temperatures, weights=w)  # [°C]
    f['end_temp'] = engine_temperatures[-1]  # [°C]

    return f


dsp.add_data(
    'enable_willans', dfl.values.enable_willans,
    description='Enable the calculation of Willans coefficients for '
                'the cycle?'
)

dsp.add_function(
    function=sh.add_args(calculate_willans_factors),
    inputs=[
        'enable_willans', 'co2_params_calibrated',
        'engine_fuel_lower_heating_value', 'engine_stroke', 'engine_capacity',
        'min_engine_on_speed', 'fmep_model', 'engine_speeds_out',
        'engine_powers_out', 'times', 'velocities', 'accelerations',
        'motive_powers', 'engine_temperatures', 'missing_powers',
        'angle_slopes'
    ],
    outputs=['willans_factors'],
    input_domain=co2_utl.check_first_arg
)


def calculate_phases_willans_factors(
        params, engine_fuel_lower_heating_value, engine_stroke, engine_capacity,
        min_engine_on_speed, fmep_model, times, phases_indices,
        engine_speeds_out, engine_powers_out, velocities, accelerations,
        motive_powers, engine_temperatures, missing_powers,
        angle_slopes):
    """
    Calculates the Willans factors for each phase.

    :param params:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type params: lmfit.Parameters

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param engine_stroke:
        Engine stroke [mm].
    :type engine_stroke: float

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param fmep_model:
        Engine FMEP model.
    :type fmep_model: FMEP

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param phases_indices:
        Indices of the cycle phases [-].
    :type phases_indices: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param engine_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_temperatures: numpy.array

    :param missing_powers:
        Missing engine power [kW].
    :type missing_powers: numpy.array

    :param angle_slopes:
        Angle slope vector [rad].
    :type angle_slopes: numpy.array

    :return:
        Willans factors:

        - av_velocities                         [km/h]
        - av_slope                              [rad]
        - distance                              [km]
        - init_temp                             [°C]
        - av_temp                               [°C]
        - end_temp                              [°C]
        - av_vel_pos_mov_pow                    [kw/h]
        - av_pos_motive_powers                  [kW]
        - sec_pos_mov_pow                       [s]
        - av_neg_motive_powers                  [kW]
        - sec_neg_mov_pow                       [s]
        - av_pos_accelerations                  [m/s2]
        - av_engine_speeds_out_pos_pow          [RPM]
        - av_pos_engine_powers_out              [kW]
        - engine_bmep_pos_pow                   [bar]
        - mean_piston_speed_pos_pow             [m/s]
        - fuel_mep_pos_pow                      [bar]
        - fuel_consumption_pos_pow              [g/sec]
        - willans_a                             [g/kWh]
        - willans_b                             [g/h]
        - specific_fuel_consumption             [g/kWh]
        - indicated_efficiency                  [-]
        - willans_efficiency                    [-]

    :rtype: dict
    """

    factors = []

    for i, j in phases_indices:
        factors.append(calculate_willans_factors(
            params, engine_fuel_lower_heating_value, engine_stroke,
            engine_capacity, min_engine_on_speed, fmep_model,
            engine_speeds_out[i:j], engine_powers_out[i:j], times[i:j],
            velocities[i:j], accelerations[i:j], motive_powers[i:j],
            engine_temperatures[i:j], missing_powers[i:j],
            angle_slopes[i:j]
        ))

    return factors


dsp.add_data(
    'enable_phases_willans', dfl.values.enable_phases_willans,
    description='Enable the calculation of Willans coefficients for '
                'all phases?'
)

dsp.add_function(
    function=sh.add_args(calculate_phases_willans_factors),
    inputs=[
        'enable_phases_willans', 'co2_params_calibrated',
        'engine_fuel_lower_heating_value', 'engine_stroke', 'engine_capacity',
        'min_engine_on_speed', 'fmep_model', 'times', 'phases_indices',
        'engine_speeds_out', 'engine_powers_out', 'velocities', 'accelerations',
        'motive_powers', 'engine_temperatures', 'missing_powers',
        'angle_slopes'
    ],
    outputs=['phases_willans_factors'],
    input_domain=co2_utl.check_first_arg
)


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['optimal_efficiency'])
def calculate_optimal_efficiency(co2_params_calibrated, mean_piston_speeds):
    """
    Calculates the optimal efficiency [-] and t.

    :param co2_params_calibrated:
        CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).

        The missing parameters are set equal to zero.
    :type co2_params_calibrated: lmfit.Parameters

    :param mean_piston_speeds:
        Mean piston speed vector [m/s].
    :type mean_piston_speeds: numpy.array

    :return:
        Optimal efficiency and the respective parameters:

        - mean_piston_speeds [m/s],
        - engine_bmep [bar],
        - efficiency [-].

    :rtype: dict[str | tuple]
    """
    # noinspection PyProtectedMember
    from .engine.fc import _fuel_ABC
    n_s = np.linspace(mean_piston_speeds.min(), mean_piston_speeds.max(), 10)

    A, B, C = _fuel_ABC(n_s, **co2_params_calibrated)
    # noinspection PyTypeChecker
    b = np.isclose(A, 0.0)
    # noinspection PyTypeChecker
    A = np.where(b, np.sign(C) * dfl.EPS, A)
    ac4, B2 = 4 * A * C, B ** 2
    sabc = np.sqrt(ac4 * B2)
    n = sabc - ac4

    bmep = np.where(b, np.nan, 2 * C - sabc / (2 * A))
    eff = n / (B - np.sqrt(B2 - sabc - n))

    return dict(mean_piston_speeds=n_s, engine_bmep=bmep, efficiency=eff)


@sh.add_function(dsp, outputs=['phases_fuel_consumptions'])
def calculate_phases_fuel_consumptions(
        phases_co2_emissions, fuel_carbon_content, fuel_density):
    """
    Calculates cycle phases fuel consumption [l/100km].

    :param phases_co2_emissions:
        CO2 emission of cycle phases [CO2g/km].
    :type phases_co2_emissions: numpy.array

    :param fuel_carbon_content:
        Fuel carbon content [CO2g/g].
    :type fuel_carbon_content: float

    :param fuel_density:
        Fuel density [g/l].
    :type fuel_density: float

    :return:
        Fuel consumption of cycle phases [l/100km].
    :rtype: tuple
    """

    c = 100.0 / (fuel_density * fuel_carbon_content)

    return tuple(np.asarray(phases_co2_emissions) * c)


@sh.add_function(dsp, outputs=['fuel_consumption_value'])
def calculate_fuel_consumption_value(
        phases_fuel_consumptions, phases_distances):
    """
    Calculates the fuel consumption of the cycle [l/100km].

    :param phases_fuel_consumptions:
        Fuel consumption of cycle phases [l/100km].
    :type phases_fuel_consumptions: numpy.array

    :param phases_distances:
        Cycle phases distances [km].
    :type phases_distances: numpy.array | float

    :return:
        Fuel consumption of the cycle [l/100km].
    :rtype: float
    """
    return calculate_co2_emission_value(
        phases_fuel_consumptions, phases_distances
    )


@sh.add_function(dsp, outputs=['phases_distances'])
def calculate_phases_distances(phases_indices, distances):
    """
    Calculates cycle phases distances [km].

    :param phases_indices:
        Indices of the cycle phases [-].
    :type phases_indices: numpy.array

    :param distances:
        Cumulative distance vector [m].
    :type distances: numpy.array

    :return:
        Cycle phases distances [km].
    :rtype: numpy.array
    """
    i = phases_indices.copy()
    i[:, 1] -= 1
    return np.diff(distances[i], axis=1).ravel() / 1000.0


@sh.add_function(dsp, outputs=['phases_times'])
def calculate_phases_times(phases_indices, times):
    """
    Calculates cycle phases times [s].

    :param phases_indices:
        Indices of the cycle phases [-].
    :type phases_indices: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :return:
        Cycle phases times [s].
    :rtype: numpy.array
    """
    i = phases_indices.copy()
    i[:, 1] -= 1
    return np.diff(times[i], axis=1).ravel()


@sh.add_function(dsp, outputs=['fuel_density'])
def default_fuel_density(fuel_type):
    """
    Returns the default fuel density [g/l].

    :param fuel_type:
        Fuel type (diesel, gasoline, LPG, NG, ethanol, methanol, biodiesel,
        propane).
    :type fuel_type: str

    :return:
        Fuel density [g/l].
    :rtype: float
    """
    return dfl.functions.default_fuel_density.FUEL_DENSITY[fuel_type]


@sh.add_function(dsp, outputs=['engine_fuel_lower_heating_value'])
def default_engine_fuel_lower_heating_value(fuel_type):
    """
    Returns the default fuel lower heating value [kJ/kg].

    :param fuel_type:
        Fuel type (diesel, gasoline, LPG, NG, ethanol, methanol, biodiesel,
        propane).
    :type fuel_type: str

    :return:
        Fuel lower heating value [kJ/kg].
    :rtype: float
    """
    return dfl.functions.default_fuel_lower_heating_value.LHV[fuel_type]


@sh.add_function(dsp, outputs=['fuel_carbon_content'], weight=3)
def default_fuel_carbon_content(fuel_type):
    """
    Returns the default fuel carbon content [CO2g/g].

    :param fuel_type:
        Fuel type (diesel, gasoline, LPG, NG, ethanol, methanol, biodiesel,
        propane).
    :type fuel_type: str

    :return:
        Fuel carbon content [CO2g/g].
    :rtype: float
    """
    return dfl.functions.default_fuel_carbon_content.CARBON_CONTENT[fuel_type]


@sh.add_function(dsp, outputs=['fuel_carbon_content'])
def calculate_fuel_carbon_content(fuel_carbon_content_percentage):
    """
    Calculates the fuel carbon content as CO2g/g.

    :param fuel_carbon_content_percentage:
        Fuel carbon content [%].
    :type fuel_carbon_content_percentage: float

    :return:
        Fuel carbon content [CO2g/g].
    :rtype: float
    """
    return fuel_carbon_content_percentage / 100.0 * 44.0 / 12.0


@sh.add_function(dsp, outputs=['fuel_carbon_content_percentage'])
def calculate_fuel_carbon_content_percentage(fuel_carbon_content):
    """
    Calculates the fuel carbon content as %.

    :param fuel_carbon_content:
        Fuel carbon content [CO2g/g].
    :type fuel_carbon_content: float

    :return:
        Fuel carbon content [%].
    :rtype: float
    """

    return fuel_carbon_content / calculate_fuel_carbon_content(1.0)


@sh.add_function(dsp, outputs=['fuel_heating_value'])
def calculate_fuel_heating_value(engine_fuel_lower_heating_value, fuel_density):
    """
    Calculates the fuel heating value as kWh/l.

    :param engine_fuel_lower_heating_value:
        Fuel lower heating value [kJ/kg].
    :type engine_fuel_lower_heating_value: float

    :param fuel_density:
        Fuel density [g/l].
    :type fuel_density: float

    :return:
        Fuel heating value [kWh/l].
    :rtype: float
    """
    return engine_fuel_lower_heating_value * fuel_density / 36e5


@sh.add_function(dsp, outputs=['fuel_consumptions_liters_value'])
def calculate_fuel_consumptions_liters_value(times, fuel_consumptions_liters):
    """

    :param times:
    :param fuel_consumptions_liters:
    :return:
    """
    from scipy.integrate import cumtrapz
    return cumtrapz(fuel_consumptions_liters / 3600, times, initial=0)
