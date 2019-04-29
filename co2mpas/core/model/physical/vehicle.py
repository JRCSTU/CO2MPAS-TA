# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the mechanic of the vehicle.
"""
import numpy as np
import schedula as sh
from .defaults import dfl
from co2mpas.utils import BaseModel

dsp = sh.BlueDispatcher(
    name='Vehicle free body diagram',
    description='Calculates forces and power acting on the vehicle.'
)


@sh.add_function(dsp, outputs=['velocities'])
def calculate_velocities(times, obd_velocities):
    """
    Filters the obd velocities [km/h].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param obd_velocities:
        OBD velocity vector [km/h].
    :type obd_velocities: numpy.array

    :return:
        Velocity vector [km/h].
    :rtype: numpy.array
    """
    from co2mpas.utils import median_filter
    dt_window = dfl.functions.calculate_velocities.dt_window
    return median_filter(times, obd_velocities, dt_window, np.mean)


def _integral(x, y, y0=.0):
    from scipy.interpolate import InterpolatedUnivariateSpline as Spl
    return Spl(x, y).antiderivative()(x) + y0


@sh.add_function(dsp, outputs=['distances'])
def calculate_distances(times, velocities):
    """
    Calculates the cumulative distance vector [m].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :return:
        Cumulative distance vector [m].
    :rtype: numpy.array
    """
    return _integral(times, velocities / 3.6, 0)


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True, outputs=['velocities']
)
def calculate_velocities_v1(times, accelerations, initial_velocity=.0):
    """
    Calculates the velocity from acceleration time series [km/h].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param initial_velocity:
        Initial velocity [km/h].
    :type initial_velocity: float

    :return:
        Velocity vector [km/h].
    :rtype: numpy.array
    """
    return _integral(times, accelerations, initial_velocity / 3.6) * 3.6


@sh.add_function(dsp, outputs=['accelerations'])
def calculate_accelerations(times, velocities):
    """
    Calculates the acceleration from velocity time series [m/s2].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array

    :return:
        Acceleration vector [m/s2].
    :rtype: numpy.array
    """
    from scipy.interpolate import InterpolatedUnivariateSpline as Spl
    acc = Spl(times, velocities / 3.6).derivative()(times)
    b = (velocities[:-1] == 0) & (velocities[1:] == velocities[:-1])
    acc[:-1][b] = 0
    if b[-1]:
        acc[-1] = 0
    return acc


@sh.add_function(dsp, outputs=['aerodynamic_resistances'])
def calculate_aerodynamic_resistances(f2, velocities):
    """
    Calculates the aerodynamic resistances of the vehicle [N].

    :param f2:
        As used in the dyno and defined by respective guidelines [N/(km/h)^2].
    :type f2: float

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array | float

    :return:
        Aerodynamic resistance vector [N].
    :rtype: numpy.array | float
    """

    return f2 * velocities ** 2


dsp.add_data('n_passengers', dfl.values.n_passengers)
dsp.add_data('passenger_mass', dfl.values.passenger_mass)
dsp.add_data('cargo_mass', dfl.values.cargo_mass)
dsp.add_data('fuel_mass', dfl.values.fuel_mass)


@sh.add_function(dsp, outputs=['passengers_mass'])
def calculate_passengers_mass(n_passengers, passenger_mass):
    """
    Calculate passengers mass including driver [kg].

    :param n_passengers:
        Number of passengers including driver [-].
    :type n_passengers: int

    :param passenger_mass:
        Average passenger mass [kg].
    :type passenger_mass: float

    :return:
        Passengers mass including the driver [kg].
    :rtype: float
    """

    return passenger_mass * n_passengers


@sh.add_function(dsp, outputs=['unladen_mass'])
def calculate_unladen_mass(curb_mass, fuel_mass):
    """
    Calculate unladen mass [kg].

    :param curb_mass:
        Curb mass [kg].
    :type curb_mass: float

    :param fuel_mass:
        Fuel mass [kg].
    :type fuel_mass: float

    :return:
        Unladen mass [kg].
    :rtype: float
    """

    return curb_mass + fuel_mass


@sh.add_function(dsp, outputs=['unladen_mass'])
def calculate_unladen_mass_v1(vehicle_mass, passengers_mass, cargo_mass):
    """
    Calculate unladen mass [kg].

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param passengers_mass:
        Passengers mass including the driver [kg].
    :type passengers_mass: float

    :param cargo_mass:
        Cargo mass [kg].
    :type cargo_mass: float

    :return:
        Unladen mass [kg].
    :rtype: float
    """

    return vehicle_mass - passengers_mass - cargo_mass


@sh.add_function(dsp, outputs=['curb_mass'])
def calculate_curb_mass(unladen_mass, fuel_mass):
    """
    Calculate unladen mass [kg].

    :param unladen_mass:
        Unladen mass [kg].
    :type unladen_mass: float

    :param fuel_mass:
        Fuel mass [kg].
    :type fuel_mass: float

    :return:
        Curb mass [kg].
    :rtype: float
    """

    return unladen_mass - fuel_mass


@sh.add_function(dsp, outputs=['vehicle_mass'])
def calculate_vehicle_mass(unladen_mass, passengers_mass, cargo_mass):
    """
    Calculate vehicle_mass [kg].

    :param unladen_mass:
        Unladen mass [kg].
    :type unladen_mass: float

    :param passengers_mass:
        Passengers mass including the driver [kg].
    :type passengers_mass: float

    :param cargo_mass:
        Cargo mass [kg].
    :type cargo_mass: float

    :return:
        Vehicle mass [kg].
    :rtype: float
    """

    return unladen_mass + passengers_mass + cargo_mass


@sh.add_function(dsp, outputs=['raw_frontal_area'])
def calculate_raw_frontal_area(vehicle_height, vehicle_width):
    """
    Calculates raw frontal area of the vehicle [m2].

    :param vehicle_height:
        Vehicle height [m].
    :type vehicle_height: float

    :param vehicle_width:
        Vehicle width [m].
    :type vehicle_width: float

    :return:
        Raw frontal area of the vehicle [m2].
    :rtype: float
    """
    return vehicle_height * vehicle_width


@sh.add_function(dsp, outputs=['raw_frontal_area'], weight=5)
def calculate_raw_frontal_area_v1(vehicle_mass, vehicle_category):
    """
    Calculates raw frontal area of the vehicle [m2].

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param vehicle_category:
        Vehicle category (i.e., A, B, C, D, E, F, S, M, and J).
    :type vehicle_category: str

    :return:
        Raw frontal area of the vehicle [m2].
    :rtype: float
    """
    from asteval import Interpreter as Interp
    d = dfl.functions.calculate_raw_frontal_area_v1
    return Interp().eval(d.formulas[vehicle_category.upper()])(vehicle_mass)


@sh.add_function(dsp, outputs=['frontal_area'])
def calculate_frontal_area(raw_frontal_area):
    """
    Calculates the vehicle frontal area [m2].

    :param raw_frontal_area:
        Raw frontal area of the vehicle [m2].
    :type raw_frontal_area: float

    :return:
        Frontal area of the vehicle [m2].
    :rtype: float
    """
    d = dfl.functions.calculate_frontal_area.projection_factor
    return raw_frontal_area * d


@sh.add_function(dsp, outputs=['aerodynamic_drag_coefficient'])
def calculate_aerodynamic_drag_coefficient(vehicle_category):
    """
    Calculates the aerodynamic drag coefficient [-].

    :param vehicle_category:
        Vehicle category (i.e., A, B, C, D, E, F, S, M, and J).
    :type vehicle_category: str

    :return:
        Aerodynamic drag coefficient [-].
    :rtype: float
    """
    d = dfl.functions.calculate_aerodynamic_drag_coefficient
    return d.cw[vehicle_category.upper()]


@sh.add_function(dsp, outputs=['aerodynamic_drag_coefficient'])
def calculate_aerodynamic_drag_coefficient_v1(vehicle_body):
    """
    Calculates the aerodynamic drag coefficient [-].

    :param vehicle_body:
        Vehicle body (i.e., cabriolet, sedan, hatchback, stationwagon,
        suv/crossover, mpv, coupé, bus, bestelwagen, pick-up).
    :type vehicle_body: str

    :return:
        Aerodynamic drag coefficient [-].
    :rtype: float
    """
    d = dfl.functions.calculate_aerodynamic_drag_coefficient_v1
    return d.cw[vehicle_body.lower()]


dsp.add_data('air_temperature', dfl.values.air_temperature)
dsp.add_data('atmospheric_pressure', dfl.values.atmospheric_pressure)


@sh.add_function(dsp, outputs=['air_density'])
def calculate_air_density(air_temperature, atmospheric_pressure):
    """
    Calculates the air density [kg/m3].

    :param air_temperature:
        Air temperature [°C].
    :type air_temperature: float

    :param atmospheric_pressure:
        Atmospheric pressure [kPa].
    :type atmospheric_pressure: float

    :return:
        Air density [kg/m3].
    :rtype: float
    """
    # http://www.thecartech.com/subjects/auto_eng/Road_loads.htm
    return 3.48 * atmospheric_pressure / (273.16 + air_temperature)


dsp.add_data('has_roof_box', dfl.values.has_roof_box)


@sh.add_function(dsp, outputs=['f2'], weight=5)
def calculate_f2(
        air_density, aerodynamic_drag_coefficient, frontal_area, has_roof_box):
    """
    Calculates the f2 coefficient [N/(km/h)^2].

    :param air_density:
        Air density [kg/m3].
    :type air_density: float

    :param aerodynamic_drag_coefficient:
        Aerodynamic drag coefficient [-].
    :type aerodynamic_drag_coefficient: float

    :param frontal_area:
        Frontal area of the vehicle [m2].
    :type frontal_area: float
    
    :param has_roof_box:
         Has the vehicle a roof box? [-].
    :type has_roof_box: bool

    :return:
        As used in the dyno and defined by respective guidelines [N/(km/h)^2].
    :rtype: float
    """

    c = aerodynamic_drag_coefficient * frontal_area * air_density

    if has_roof_box:
        c *= dfl.functions.calculate_f2.roof_box

    return 0.5 * c / 3.6 ** 2


dsp.add_data('tyre_state', dfl.values.tyre_state)
dsp.add_data('road_state', dfl.values.road_state)


@sh.add_function(dsp, outputs=['static_friction'])
def default_static_friction(tyre_state, road_state):
    """
    Returns the default static friction coefficient [-].

    :param tyre_state:
        Tyre state (i.e., new or worm).
    :type tyre_state: str

    :param road_state:
        Road state (i.e., dry, wet, rainfall, puddles, ice).
    :type road_state: str

    :return:
        Static friction coefficient [-].
    :rtype: float
    """
    coeff = dfl.functions.default_static_friction.coeff
    return coeff[tyre_state][road_state]


@sh.add_function(dsp, outputs=['n_wheel'])
def default_n_wheel(n_wheel_drive):
    """
    Returns the default total number of wheels [-].

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :return:
        Total number of wheels [-].
    :rtype: int
    """
    return max(n_wheel_drive, dfl.functions.default_n_wheel.n_wheel)


@sh.add_function(dsp, outputs=['wheel_drive_load_fraction'])
def calculate_wheel_drive_load_fraction(n_wheel_drive, n_wheel=4):
    """
    Calculate the repartition of the load on wheel drive axles [-].

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :param n_wheel:
        Total number of wheels [-].
    :type n_wheel: int

    :return:
        Repartition of the load on wheel drive axles [-].
    :rtype: float
    """
    return n_wheel_drive / n_wheel


def _compile_traction_acceleration_limits(
        static_friction, wheel_drive_load_fraction):
    deceleration = -9.81 * static_friction
    acceleration = -deceleration * wheel_drive_load_fraction

    def _func(angle_slopes):
        slope = np.cos(angle_slopes)
        return deceleration * slope, acceleration * slope

    return _func


@sh.add_function(dsp, outputs=['traction_acceleration_limits'])
def calculate_traction_acceleration_limits(
        static_friction, wheel_drive_load_fraction, angle_slopes):
    """
    Calculates the traction acceleration limits [m/s2].

    :param static_friction:
        Static friction coefficient [-].
    :type static_friction: float

    :param wheel_drive_load_fraction:
        Repartition of the load on wheel drive axles [-].
    :type wheel_drive_load_fraction: float

    :param angle_slopes:
        Angle slope vector [rad].
    :type angle_slopes: numpy.array

    :return:
        Traction acceleration limits (i.e., deceleration, acceleration) [m/s2].
    :rtype: tuple[float]
    """
    return _compile_traction_acceleration_limits(
        static_friction, wheel_drive_load_fraction
    )(angle_slopes)


dsp.add_data('tyre_class', dfl.values.tyre_class)


@sh.add_function(dsp, outputs=['rolling_resistance_coeff'])
def calculate_rolling_resistance_coeff(tyre_class, tyre_category):
    """
    Calculates the rolling resistance coefficient [-].

    :param tyre_class: 
        Tyre class (i.e., C1, C2, and C3).
    :type tyre_class: str

    :param tyre_category: 
        Tyre category (i.e., A, B, C, D, E, F, and G).
    :type tyre_category: str

    :return: 
        Rolling resistance coefficient [-].
    :rtype: float
    """
    coeff = dfl.functions.calculate_rolling_resistance_coeff.coeff
    return coeff[tyre_class.upper()][tyre_category.upper()]


@sh.add_function(dsp, outputs=['f0'], weight=5)
def calculate_f0(vehicle_mass, rolling_resistance_coeff):
    """
    Calculates rolling resistance [N].

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param rolling_resistance_coeff:
        Rolling resistance coefficient [-].
    :type rolling_resistance_coeff: float

    :return:
        Rolling resistance force [N] when angle_slope == 0.
    :rtype: float
    """

    return vehicle_mass * 9.81 * rolling_resistance_coeff


@sh.add_function(dsp, outputs=['f1'], weight=5)
def calculate_f1(f2):
    """
    Calculates the f1 road load [N/(km/h)].

    :param f2:
        As used in the dyno and defined by respective guidelines [N/(km/h)^2].
    :type f2: float

    :return: 
        Defined by dyno procedure [N/(km/h)].
    :rtype: float
    """

    q, m = dfl.functions.calculate_f1.qm
    return m * f2 + q


dsp.add_data('angle_slope', dfl.values.angle_slope)
dsp.add_data('path_distances', wildcard=True)
dsp.add_data('path_elevations', wildcard=True)


@sh.add_function(dsp, outputs=['slope_model'])
@sh.add_function(
    dsp, inputs=['path_distances', 'path_elevations'], outputs=['slope_model']
)
def define_slope_model(distances, elevations):
    """
    Returns the angle slope model [rad].

    :param distances:
        Cumulative distance vector [m].
    :type distances: numpy.array

    :param elevations:
        Elevation vector [m].
    :type elevations: numpy.array

    :return:
        Angle slope model [rad].
    :rtype: function
    """
    from scipy.interpolate import InterpolatedUnivariateSpline as Spl
    i = np.append([0], np.where(np.diff(distances) > 0)[0] + 1)
    func = Spl(distances[i], elevations[i]).derivative()
    return lambda d: np.arctan(func(d))


@sh.add_function(dsp, outputs=['slope_model'], weight=5)
def define_slope_model_v1(angle_slope):
    """
    Returns the angle slope model [rad].

    :param angle_slope:
        Angle slope [rad].
    :type angle_slope: float

    :return:
        Angle slope model [rad].
    :rtype: function
    """
    return np.vectorize(lambda *args: angle_slope, otypes=[float])


@sh.add_function(dsp, outputs=['angle_slopes'])
def calculate_angle_slopes(slope_model, distances):
    """
    Returns the angle slope vector [rad].

    :param slope_model:
        Angle slope model [rad].
    :type slope_model: function

    :param distances:
        Cumulative distance vector [m].
    :type distances: numpy.array

    :return:
        Angle slope vector [rad].
    :rtype: numpy.array
    """
    return slope_model(distances)


@sh.add_function(dsp, outputs=['rolling_resistance'])
def calculate_rolling_resistance(f0, angle_slopes):
    """
    Calculates rolling resistance [N].

    :param f0:
        Rolling resistance force [N] when angle_slope == 0.
    :type f0: float

    :param angle_slopes:
        Angle slope vector [rad].
    :type angle_slopes: numpy.array

    :return:
        Rolling resistance force [N].
    :rtype: numpy.array
    """
    return f0 * np.cos(angle_slopes)


@sh.add_function(dsp, outputs=['velocity_resistances'])
def calculate_velocity_resistances(f1, velocities):
    """
    Calculates forces function of velocity [N].

    :param f1:
        Defined by dyno procedure [N/(km/h)].
    :type f1: float

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array | float

    :return:
        Forces function of velocity [N].
    :rtype: numpy.array | float
    """

    return f1 * velocities


@sh.add_function(dsp, outputs=['climbing_force'])
def calculate_climbing_force(vehicle_mass, angle_slopes):
    """
    Calculates the vehicle climbing resistance [N].

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param angle_slopes:
        Angle slope vector [rad].
    :type angle_slopes: numpy.array

    :return:
        Vehicle climbing resistance [N].
    :rtype: numpy.array
    """
    return vehicle_mass * 9.81 * np.sin(angle_slopes)


@sh.add_function(dsp, outputs=['n_dyno_axes'])
def select_default_n_dyno_axes(cycle_type, n_wheel_drive):
    """
    Selects the default number of dyno axes[-].

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :return:
        Number of dyno axes [-].
    :rtype: int
    """
    par = dfl.functions.select_default_n_dyno_axes

    try:
        return par.DYNO_AXES[cycle_type.upper()][n_wheel_drive]
    except KeyError:
        return n_wheel_drive // 2


@sh.add_function(dsp, outputs=['inertial_factor'])
def select_inertial_factor(n_dyno_axes):
    """
    Selects the inertia factor [%] according to the number of dyno axes.

    :param n_dyno_axes:
        Number of dyno axes [-].
    :type n_dyno_axes: int

    :return:
        Factor that considers the rotational inertia [%].
    :rtype: float
    """

    return 1.5 * n_dyno_axes


@sh.add_function(dsp, outputs=['rotational_inertia_forces'])
def calculate_rotational_inertia_forces(
        vehicle_mass, inertial_factor, accelerations):
    """
    Calculate rotational inertia forces [N].

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param inertial_factor:
        Factor that considers the rotational inertia [%].
    :type inertial_factor: float

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array | float

    :return:
        Rotational inertia forces [N].
    :rtype: numpy.array | float
    """

    return vehicle_mass * inertial_factor * accelerations / 100


@sh.add_function(dsp, outputs=['accelerations'])
def calculate_accelerations_v1(
        vehicle_mass, inertial_factor, motive_forces, climbing_force,
        aerodynamic_resistances, rolling_resistance, velocity_resistances):
    """
    Calculates the acceleration from motive forces [m/s2].

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param inertial_factor:
        Factor that considers the rotational inertia [%].
    :type inertial_factor: float

    :param motive_forces:
        Motive forces [N].
    :type motive_forces: numpy.array | float

    :param climbing_force:
        Vehicle climbing resistance [N].
    :type climbing_force: float | numpy.array

    :param aerodynamic_resistances:
        Aerodynamic resistance vector [N].
    :type aerodynamic_resistances: numpy.array | float

    :param rolling_resistance:
        Rolling resistance force [N].
    :type rolling_resistance: float | numpy.array

    :param velocity_resistances:
        Forces function of velocity [N].
    :type velocity_resistances: numpy.array | float

    :return:
        Acceleration vector [m/s2].
    :rtype: numpy.array
    """
    acc = motive_forces - climbing_force - aerodynamic_resistances
    acc -= rolling_resistance + velocity_resistances
    acc /= vehicle_mass * (1 + inertial_factor / 100)
    return acc


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['motive_forces'])
def calculate_motive_forces(
        vehicle_mass, accelerations, climbing_force, aerodynamic_resistances,
        rolling_resistance, velocity_resistances, rotational_inertia_forces):
    """
    Calculate motive forces [N].

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array | float

    :param climbing_force:
        Vehicle climbing resistance [N].
    :type climbing_force: float | numpy.array

    :param rolling_resistance:
        Rolling resistance force [N].
    :type rolling_resistance: float | numpy.array

    :param aerodynamic_resistances:
        Aerodynamic resistance vector [N].
    :type aerodynamic_resistances: numpy.array | float

    :param velocity_resistances:
        Forces function of velocity [N].
    :type velocity_resistances: numpy.array | float

    :param rotational_inertia_forces:
        Rotational inertia forces [N].
    :type rotational_inertia_forces: numpy.array | float

    :return:
        Motive forces [N].
    :rtype: numpy.array | float
    """

    # namespace shortcuts
    Frr = rolling_resistance
    Faero = aerodynamic_resistances
    Fclimb = climbing_force
    Fvel = velocity_resistances
    Finertia = rotational_inertia_forces

    return vehicle_mass * accelerations + Fclimb + Frr + Faero + Fvel + Finertia


@sh.add_function(dsp, outputs=['motive_powers'])
def calculate_motive_powers(motive_forces, velocities):
    """
    Calculates motive power [kW].

    :param motive_forces:
        Motive forces [N].
    :type motive_forces: numpy.array | float

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array | float

    :return:
        Motive power [kW].
    :rtype: numpy.array | float
    """

    return motive_forces * velocities / 3600


@sh.add_function(dsp, outputs=['motive_forces'])
def calculate_motive_forces_v1(motive_powers, velocities):
    """
    Calculate motive forces [N].

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array | float

    :param velocities:
        Velocity vector [km/h].
    :type velocities: numpy.array | float

    :return:
        Motive forces [N].
    :rtype: numpy.array | float
    """

    return motive_powers / velocities * 3600


dsp.add_data(
    'road_loads', description='Cycle road loads [N, N/(km/h), N/(km/h)^2].'
)
dsp.add_function('grouping', sh.bypass, ['f0', 'f1', 'f2'], ['road_loads'])
dsp.add_function('splitting', sh.bypass, ['road_loads'], ['f0', 'f1', 'f2'])
dsp.add_data('correct_f0', dfl.values.correct_f0)


@sh.add_function(dsp, outputs=['f0'])
def apply_f0_correction(f0_uncorrected, correct_f0):
    """
    Corrects the rolling resistance force [N] if a different preconditioning
    cycle was used for WLTP (WLTP precon) and NEDC (NEDC precon).

    :param f0_uncorrected:
        Uncorrected rolling resistance force [N] when angle_slope == 0.
    :type f0_uncorrected: float

    :param correct_f0:
        A different preconditioning cycle was used for WLTP and NEDC?
    :type correct_f0: bool

    :return:
        Rolling resistance force [N] when angle_slope == 0.
    :rtype: float
    """

    if correct_f0:
        return f0_uncorrected - 6.0
    return f0_uncorrected


# noinspection PyMissingOrEmptyDocstring
class VehicleModel(BaseModel):
    key_outputs = 'velocities', 'distances', 'angle_slopes', 'motive_powers'
    contract_outputs = 'velocities', 'distances', 'angle_slopes'
    types = {float: set(key_outputs)}

    def __init__(self, vehicle_mass=None, f0=None, f1=None, f2=None,
                 inertial_factor=None, slope_model=None, initial_velocity=0,
                 outputs=None):
        pars = vehicle_mass, f0, f1, f2, inertial_factor, slope_model
        self.initial_velocity = initial_velocity
        self.slope_model = slope_model
        if not any(v is None for v in pars):
            d = dsp.register(memo={})
            d.set_default_value('vehicle_mass', vehicle_mass)
            d.set_default_value('f0', f0)
            d.set_default_value('f1', f1)
            d.set_default_value('f2', f2)
            d.set_default_value('inertial_factor', inertial_factor)
            self.power = sh.DispatchPipe(
                d, inputs=('velocities', 'accelerations', 'angle_slopes'),
                outputs=('motive_powers',)
            )
        super(VehicleModel, self).__init__(outputs)

    def init_velocity(self, times, accelerations):
        key = 'velocities'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            n = len(out) - 1
            return lambda i: out[min(i + 1, n)]

        velocities = self.outputs[key]
        velocities[0] = self.initial_velocity

        def _next(i):
            j = max(i - 1, 0)
            dt = times[i + 1] - times[j]
            return velocities[j] + 3.6 * accelerations[i] * dt

        return _next

    def init_distance(self, times, velocities):
        keys = 'angle_slopes', 'distances'
        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            slp, dist = sh.selector(keys, self._outputs, output_type='list')
            n = len(dist) - 1

            def _next(i):
                j = min(i + 1, n)
                return dist[j], slp[j]

            return _next
        slp, dist = sh.selector(keys, self.outputs, output_type='list')
        dist[0], vel = 0, velocities
        slp[0] = self.slope_model(0)

        def _next(i):
            j = i + 1
            d = (vel[j] + vel[i]) / (2 * 3.6) * (times[j] - times[i])
            d += dist[i]
            return d, np.asscalar(self.slope_model(d))

        return _next

    def init_power(self, velocities, accelerations, angle_slopes):
        key = 'motive_powers'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]

        func = self.power
        return lambda i: func(velocities[i], accelerations[i], angle_slopes[i])

    def init_results(self, times, accelerations):
        vel, dist = self.outputs['velocities'], self.outputs['distances']
        slp, pws = self.outputs['angle_slopes'], self.outputs['motive_powers']

        v_gen = self.init_velocity(times, accelerations)
        d_gen = self.init_distance(times, vel)
        p_gen = self.init_power(vel, accelerations, slp)

        def _next(i):
            try:
                vel[i + 1] = v_gen(i)
                dist[i + 1], slp[i + 1] = d_gen(i)
            except IndexError:
                pass
            pws[i] = p = p_gen(i)
            return vel[i], dist[i], slp[i], p

        return _next


@sh.add_function(dsp, outputs=['vehicle_prediction_model'])
def define_fake_vehicle_prediction_model(
        velocities, distances, motive_powers, angle_slopes):
    """
    Defines a fake vehicle prediction model.

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array

    :param wheel_powers:
        Power at the wheels [kW].
    :type wheel_powers: numpy.array

    :param wheel_torques:
        Torque at the wheel [N*m].
    :type wheel_torques: numpy.array

    :return:
        Wheels prediction model.
    :rtype: WheelsModel
    """
    return VehicleModel(outputs=dict(
        velocities=velocities, distances=distances, motive_powers=motive_powers,
        angle_slopes=angle_slopes
    ))


@sh.add_function(dsp, outputs=['vehicle_prediction_model'], weight=4000)
def define_vehicle_prediction_model(
        vehicle_mass, f0, f1, f2, inertial_factor, slope_model,
        initial_velocity):
    """
    Defines the vehicle prediction model.

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Wheels prediction model.
    :rtype: WheelsModel
    """
    return VehicleModel(
        vehicle_mass, f0, f1, f2, inertial_factor, slope_model, initial_velocity
    )
