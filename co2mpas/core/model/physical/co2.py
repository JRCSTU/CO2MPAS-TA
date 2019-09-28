# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the legislation corrections.
"""
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl
from .defaults import dfl

dsp = sh.BlueDispatcher(
    name='Legislation', description='Models the legislation corrections.'
)


@sh.add_function(dsp, inputs_kwargs=True, outputs=['phases_co2_emissions'])
def calculate_phases_co2_emissions(
        times, phases_integration_times, co2_emissions, phases_distances=1.0):
    """
    Calculates CO2 emission or cumulative CO2 of cycle phases [CO2g/km or CO2g].

    If phases_distances is not given the result is the cumulative CO2 of cycle
    phases [CO2g] otherwise it is CO2 emission of cycle phases [CO2g/km].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param phases_integration_times:
        Cycle phases integration times [s].
    :type phases_integration_times: tuple

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
    for p in phases_integration_times:
        i, j = np.searchsorted(times, p)
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


@sh.add_function(dsp, outputs=['declared_co2_emission_value'])
def calculate_declared_co2_emission(
        co2_emission_value, ki_multiplicative, ki_additive):
    """
    Calculates the declared CO2 emission of the cycle [CO2g/km].

    :param co2_emission_value:
        CO2 emission value of the cycle [CO2g/km].
    :type co2_emission_value: float

    :param ki_multiplicative:
        Multiplicative correction for vehicles with periodically regenerating
        systems [-].
    :type ki_multiplicative: float

    :param ki_additive:
        Additive correction for vehicles with periodically regenerating
        systems [CO2g/km].
    :type ki_multiplicative: float

    :return:
        Declared CO2 emission value of the cycle [CO2g/km].
    :rtype: float
    """

    return co2_emission_value * ki_multiplicative + ki_additive


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


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['phases_co2_emissions'])
def merge_wltp_phases_co2_emission(co2_emission_UDC, co2_emission_EUDC):
    """
    Merges WLTP phases co2 emission.

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


@sh.add_function(dsp, outputs=['corrected_co2_emission_value'])
def calculate_corrected_co2_emission_value(
        is_hybrid, rcb_correction, cycle_type, service_battery_delta_energy,
        driven_distance, alternator_efficiency, engine_type, fuel_type):
    par = dfl.functions.calculate_corrected_co2_emission_value.WILLANS
    n = par[fuel_type][engine_type] * service_battery_delta_energy * 0.0036
    return n / (driven_distance * alternator_efficiency)


def calculate_willans_factors(
        co2_params_calibrated, engine_fuel_lower_heating_value, engine_stroke,
        engine_capacity, min_engine_on_speed, fmep_model, engine_speeds_out,
        engine_powers_out, times, velocities, accelerations, motive_powers,
        engine_coolant_temperatures, missing_powers, angle_slopes):
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

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

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

    f['init_temp'] = engine_coolant_temperatures[0]  # [°C]
    f['av_temp'] = av(engine_coolant_temperatures, weights=w)  # [°C]
    f['end_temp'] = engine_coolant_temperatures[-1]  # [°C]

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
        'motive_powers', 'engine_coolant_temperatures', 'missing_powers',
        'angle_slopes'
    ],
    outputs=['willans_factors'],
    input_domain=co2_utl.check_first_arg
)


def calculate_phases_willans_factors(
        params, engine_fuel_lower_heating_value, engine_stroke, engine_capacity,
        min_engine_on_speed, fmep_model, times, phases_integration_times,
        engine_speeds_out, engine_powers_out, velocities, accelerations,
        motive_powers, engine_coolant_temperatures, missing_powers,
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

    :param phases_integration_times:
        Cycle phases integration times [s].
    :type phases_integration_times: tuple

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

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

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

    for p in phases_integration_times:
        i, j = np.searchsorted(times, p)

        factors.append(calculate_willans_factors(
            params, engine_fuel_lower_heating_value, engine_stroke,
            engine_capacity, min_engine_on_speed, fmep_model,
            engine_speeds_out[i:j], engine_powers_out[i:j], times[i:j],
            velocities[i:j], accelerations[i:j], motive_powers[i:j],
            engine_coolant_temperatures[i:j], missing_powers[i:j],
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
        'min_engine_on_speed', 'fmep_model', 'times',
        'phases_integration_times', 'engine_speeds_out', 'engine_powers_out',
        'velocities', 'accelerations', 'motive_powers',
        'engine_coolant_temperatures', 'missing_powers', 'angle_slopes'
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


@sh.add_function(dsp, outputs=['phases_distances'])
def calculate_phases_distances(times, phases_integration_times, velocities):
    """
    Calculates cycle phases distances [km].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param phases_integration_times:
        Cycle phases integration times [s].
    :type phases_integration_times: tuple

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :return:
        Cycle phases distances [km].
    :rtype: numpy.array
    """

    vel = velocities / 3600.0

    return calculate_phases_co2_emissions(times, phases_integration_times, vel)


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
    d = dfl.functions
    if d.ENABLE_ALL_FUNCTIONS or d.default_fuel_lower_heating_value.ENABLE:
        return d.default_fuel_lower_heating_value.LHV[fuel_type]
    return sh.NONE


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
    d = dfl.functions
    if d.ENABLE_ALL_FUNCTIONS or d.default_fuel_carbon_content.ENABLE:
        return d.default_fuel_carbon_content.CARBON_CONTENT[fuel_type]
    return sh.NONE


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
