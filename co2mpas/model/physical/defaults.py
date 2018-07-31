#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides constants for the CO2MPAS formulas.
"""

import collections
import numpy as np
import co2mpas.utils as co2_utl


#: Container of node default values.
class Values(co2_utl.Constants):
    #: Number of engine cylinders [-].
    engine_n_cylinders = 4

    #: Default additive correction for vehicles with periodically regenerating
    #: systems [CO2g/km].
    ki_additive = 0

    #: Number of passengers including driver [-].
    n_passengers = 1

    #: Average passenger mass [kg].
    passenger_mass = 75

    #: Cargo mass [kg].
    cargo_mass = 0

    #: Fuel mass [kg].
    fuel_mass = 25

    #: Does the engine have selective catalytic reduction technology?
    has_selective_catalytic_reduction = False

    #: Does the engine have lean burn technology?
    has_lean_burn = False

    #: Does the gear box have some additional technology to heat up faster?
    has_gear_box_thermal_management = False

    #: Does the vehicle has periodically regenerating systems? [-].
    has_periodically_regenerating_systems = False

    #: Possible percentages of active cylinders [-].
    active_cylinder_ratios = (1.0,)

    #: Does the engine feature variable valve actuation? [-].
    engine_has_variable_valve_actuation = False

    #: NEDC cycle time [s].
    max_time_NEDC = 1180.0

    #: WLTP cycle time [s].
    max_time_WLTP = 1800.0

    #: Maximum velocity to consider the vehicle stopped [km/h].
    stop_velocity = 1.0 + np.finfo(np.float32).eps

    #: Maximum acceleration to be at constant velocity [m/s2].
    plateau_acceleration = 0.1 + np.finfo(np.float32).eps

    #: Does the vehicle have start/stop system?
    has_start_stop = True

    #: Does the engine have cylinder deactivation technology?
    engine_has_cylinder_deactivation = False

    #: Minimum vehicle engine speed [RPM].
    min_engine_on_speed = 10.0

    #: Minimum time of engine on after a start [s].
    min_time_engine_on_after_start = 4.0

    #: Time window used to apply gear change filters [s].
    change_gear_window_width = 4.0

    #: Alternator start window width [s].
    alternator_start_window_width = 4.0

    #: Maximum clutch window width [s].
    max_clutch_window_width = 2.0

    #: Threshold vehicle velocity for gear correction due to full load curve
    #: [km/h].
    max_velocity_full_load_correction = 100.0

    #: Maximum allowed negative current for the alternator being considered off
    #: [A].
    alternator_off_threshold = -1.0

    #: Air temperature [°C].
    air_temperature = 20

    #: Atmospheric pressure [kPa].
    atmospheric_pressure = 101.325

    #: Has the vehicle a roof box? [-].
    has_roof_box = False

    #: Tyre class (C1, C2, and C3).
    tyre_class = 'C1'

    #: Angle slope [rad].
    angle_slope = 0.0

    #: A different preconditioning cycle was used for WLTP and NEDC?
    correct_f0 = False

    #: Final drive ratio [-].
    final_drive_ratio = 1.0

    #: Final drive efficiency [-].
    final_drive_efficiency = 1.0

    #: Number of wheel drive [-].
    n_wheel_drive = 2

    #: Apply the eco-mode gear shifting?
    fuel_saving_at_strategy = True

    #: Is the vehicle hybrid?
    is_hybrid = False

    #: Cold and hot gear box reference temperatures [°C].
    gear_box_temperature_references = (40.0, 80.0)

    #: Constant torque loss due to engine auxiliaries [N*m].
    auxiliaries_torque_loss = (0.175, 0.2021)  # m, q

    #: Constant power loss due to engine auxiliaries [kW].
    auxiliaries_power_loss = 0.0213

    #: If the engine is equipped with any kind of charging.
    engine_is_turbo = True

    #: Start-stop activation time threshold [s].
    start_stop_activation_time = 30

    #: Standard deviation of idle engine speed [RPM].
    idle_engine_speed_std = 100.0

    #: Is an hot cycle?
    is_cycle_hot = False

    #: CO2 emission model params.
    co2_params = {}

    #: Enable the calculation of Willans coefficients for all phases?
    enable_phases_willans = False

    #: Enable the calculation of Willans coefficients for the cycle?
    enable_willans = False

    #: Alternator efficiency [-].
    alternator_efficiency = 0.67

    #: Time elapsed to turn on the engine with electric starter [s].
    delta_time_engine_starter = 1.0

    #: If to use decision tree classifiers to predict gears.
    use_dt_gear_shifting = False

    #: Does the vehicle have energy recuperation features?
    has_energy_recuperation = True

    #: A/T Time at cold hot transition phase [s].
    time_cold_hot_transition = 300.0

    #: Time frequency [1/s].
    time_sample_frequency = 1.0

    #: Initial temperature of the test cell of NEDC [°C].
    initial_temperature_NEDC = 25.0

    #: Initial temperature of the test cell of WLTP [°C].
    initial_temperature_WLTP = 23.0

    #: K1 NEDC parameter (first or second gear) [-].
    k1 = 1

    #: K2 NEDC parameter (first or second gear) [-].
    k2 = 2

    #: K5 NEDC parameter (first or second gear) [-].
    k5 = 2

    #: WLTP base model params.
    wltp_base_model = {}

    #: Velocity downscale factor threshold [-].
    downscale_factor_threshold = 0.01

    #: Calibration torque converter speeds delta threshold [RPM].
    calibration_tc_speed_threshold = 100.0

    #: Limits (vel, acc) when torque converter is active [km/h, m/s].
    lock_up_tc_limits = (48.0, 0.3)

    #: Empirical value in case of CVT [-].
    tyre_dynamic_rolling_coefficient = 3.05 / 3.14


#: Container of internal function parameters.
class Functions(co2_utl.Constants):
    ENABLE_ALL_FUNCTIONS = False

    class calculate_velocities(co2_utl.Constants):
        #: Time window for the moving average of obd velocities [s].
        dt_window = 5

    class calculate_engine_temperature_derivatives(co2_utl.Constants):
        #: Derivative spacing [s].
        dx = 4

        #: Degree of the smoothing spline [-].
        order = 7

    class default_clutch_window(co2_utl.Constants):
        #: Enable function?
        ENABLE = True

        #: Clutching time window [s].
        clutch_window = (-0.89230769, 0.95384615)

    class define_fuel_type_and_is_hybrid(co2_utl.Constants):
        #: Is the vehicle hybrid?
        is_hybrid = {
            # 0: False,  # Not available
            1: False,  # Gasoline
            2: False,  # Methanol
            3: False,  # Ethanol
            4: False,  # Diesel
            5: False,  # LPG
            6: False,  # CNG
            7: False,  # Propane
            8: True,  # Electric
            9: False,  # Bifuel running Gasoline
            10: False,  # Bifuel running Methanol
            11: False,  # Bifuel running Ethanol
            12: False,  # Bifuel running LPG
            13: False,  # Bifuel running CNG
            14: False,  # Bifuel running Propane
            15: True,  # Bifuel running Electricity
            16: True,  # Bifuel running electric and combustion engine
            17: True,  # Hybrid gasoline
            18: True,  # Hybrid Ethanol
            19: True,  # Hybrid Diesel
            20: True,  # Hybrid Electric
            21: True,  # Hybrid running electric and combustion engine
            22: True,  # Hybrid Regenerative
            23: False,  # Bifuel running diesel
        }
        #: The vehicle fuel type.
        fuel_type = {
            # 0: 'gasoline',  # Not available
            1: 'gasoline',  # Gasoline
            2: 'methanol',  # Methanol
            3: 'ethanol',  # Ethanol
            4: 'diesel',  # Diesel
            5: 'LPG',  # LPG
            6: 'NG',  # CNG
            7: 'propane',  # Propane
            8: None,  # Electric
            9: 'gasoline',  # Bifuel running Gasoline
            10: 'methanol',  # Bifuel running Methanol
            11: 'ethanol',  # Bifuel running Ethanol
            12: 'LPG',  # Bifuel running LPG
            13: 'NG',  # Bifuel running CNG
            14: 'propane',  # Bifuel running Propane
            15: None,  # Bifuel running Electricity
            16: 'gasoline',  # Bifuel running electric and combustion engine
            17: 'gasoline',  # Hybrid gasoline
            18: 'ethanol',  # Hybrid Ethanol
            19: 'diesel',  # Hybrid Diesel
            20: None,  # Hybrid Electric
            21: 'gasoline',  # Hybrid running electric and combustion engine
            22: None,  # Hybrid Regenerative
            23: 'diesel',  # Bifuel running diesel
        }

    class CMV(co2_utl.Constants):
        #: Enable optimization loop?
        ENABLE_OPT_LOOP = False

    class default_start_stop_activation_time(co2_utl.Constants):
        #: Enable function?
        ENABLE = False

        #: Start-stop activation time threshold [s].
        threshold = 30

    class define_idle_fuel_consumption_model(co2_utl.Constants):
        #: Enable function?
        ENABLE = False

    class calculate_last_gear_box_ratio_v1(co2_utl.Constants):
        #: Maximum admissible ratio for the last gear [-].
        MAX_RATIO = 2

        #: Minimum admissible ratio for the last gear [-].
        MIN_RATIO = 0.2

        #: Delta ratio for calculating the last gear [-].
        DELTA_RATIO = 0.1

    class calculate_maximum_velocity(co2_utl.Constants):
        #: Maximum admissible velocity for the vehicle maximum velocity [km/h].
        MAX_VEL = 1000

        #: Minimum admissible velocity for the vehicle maximum velocity [km/h].
        MIN_VEL = 1

        #: Delta ratio for calculating the vehicle maximum velocity [km/h].
        DELTA_VEL = 1

        #: Full load curve percentage fro calculating the available power [-].
        PREC_FLC = 0.9

    class MGS(co2_utl.Constants):
        #: Maximum admissible velocity for the vehicle maximum velocity [km/h].
        MAX_VEL = 1000

        #: Minimum admissible velocity for the vehicle maximum velocity [km/h].
        MIN_VEL = 1

        #: Delta ratio for calculating the vehicle maximum velocity [km/h].
        DELTA_VEL = 0.1

        #: Full load curve percentage fro calculating the available power [-].
        PREC_FLC = 0.9

    class calculate_maximum_velocity_v2(co2_utl.Constants):
        #: Maximum admissible velocity for the vehicle maximum velocity [km/h].
        MAX_VEL = 1000

        #: Minimum admissible velocity for the vehicle maximum velocity [km/h].
        MIN_VEL = 1

        #: Delta ratio for calculating the vehicle maximum velocity [km/h].
        DELTA_VEL = 1

    class calculate_first_gear_box_ratio(co2_utl.Constants):
        #: Starting slope [-].
        STARTING_SLOPE = 0.5  # --> 50%

        #: Percentage of maximum engine torque to calculate the first gear [-].
        MAX_TORQUE_PERCENTAGE = 0.95

    class design_gear_box_ratios(co2_utl.Constants):
        #: Two factor to design the gear box ratios [-].
        f_two = np.arange(1, 1.2, 0.01)

        #: Tuning factor to design the gear box ratios [-].
        f_tuning = np.arange(1, 1.1, 0.01)

    class _filter_temperature_samples(co2_utl.Constants):
        #: Max abs val of temperature derivatives during the cold phase [°C/s].
        max_abs_dt_cold = 0.7

    class _rescaling_matrix(co2_utl.Constants):
        #: Percentage width top base (i.e., short base) [-].
        a = 0.9902
        #: Percentage to define the bottom base (i.e., long base) from the phase
        #: corner [-].
        b = 0.1699

    class calculate_aerodynamic_drag_coefficient_v1(co2_utl.Constants):
        #: Aerodynamic drag coefficients function of vehicle body [-].
        cw = {
            'cabriolet': 0.28, 'sedan': 0.27, 'hatchback': 0.3,
            'stationwagon': 0.28, 'suv/crossover': 0.35, 'mpv': 0.3,
            'coupé': 0.27, 'bus': 0.35, 'bestelwagen': 0.35,
            'pick-up': 0.4  # estimated.
        }

    class calculate_aerodynamic_drag_coefficient(co2_utl.Constants):
        #: Aerodynamic drag coefficients function of vehicle category [-].
        cw = {
            'A': 0.34, 'B': 0.31, 'C': 0.29, 'D': 0.30, 'E': 0.30, 'F': 0.28,
            'S': 0.29, 'M': 0.32, 'J': 0.38
        }

    class calculate_f2(co2_utl.Constants):
        #: Deteriorating coefficient of the aerodynamic drag and frontal area
        #: due to the roof box [-].
        roof_box = 1.2

    class calculate_rolling_resistance_coeff(co2_utl.Constants):
        #: Rolling resistance coeff, function of tyre class and category [-].
        #: http://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:
        #: 02009R1222-20120530&from=EN
        #: (table A4/1 of eu legislation not world wide)
        coeff = {
            'C1': {
                'A': 5.9, 'B': 7.1, 'C': 8.4, 'D': 9.8, 'E': 9.8, 'F': 11.3,
                'G': 12.9
            },
            'C2': {
                'A': 4.9, 'B': 6.1, 'C': 7.4, 'D': 8.6, 'E': 8.6, 'F': 9.9,
                'G': 11.2
            },
            'C3': {
                'A': 3.5, 'B': 4.5, 'C': 5.5, 'D': 6.5, 'E': 7.5, 'F': 8.5,
                'G': 8.5
            }
        }

        coeff = {k: {i: j / 1000.0 for i, j in v.items()}
                 for k, v in coeff.items()}

    class calculate_f1(co2_utl.Constants):
        #: Linear model coefficients.
        qm = 2.7609 / 2, -71.735 / 2

    class calculate_raw_frontal_area_v1(co2_utl.Constants):
        #: Frontal area formulas function of vehicle_mass [function].
        formulas = dict.fromkeys(
            'ABCDEFS',
            'lambda vehicle_mass: (0.4041 * np.log(vehicle_mass) - 0.338)'
        )
        formulas['J'] = 'lambda vehicle_mass: 0.0007 * vehicle_mass + 1.8721'
        formulas['M'] = formulas['J']

    class calculate_frontal_area(co2_utl.Constants):
        #: Projection factor from the row frontal area (h * w) [-].
        projection_factor = 0.84

    class select_prediction_data(co2_utl.Constants):
        #: If True the theoretical WLTP will be predicted, otherwise the driven.
        theoretical = True

    class calibrate_clutch_prediction_model(co2_utl.Constants):
        #: Delta time that identifies the previous engine speed to calibrate
        #: the clutch model [s].
        prev_dt = 1

    class _tech_mult_factors(co2_utl.Constants):
        #: Multiplication factors of the engine parameters [-].
        factors = {
            'vva': {
                # 0: {},
                1: {'a': 0.98, 'l': 0.92},
            },
            'lb': {
                # 0: {},
                1: {'a': 1.1, 'b': 0.72, 'c': 0.76, 'a2': 1.25, 'l2': 2.85}
            },
            'egr': {
                # 0: {},
                1: {'a': 1.02, 'b': 1.1, 'c': 1.5, 'a2': 1.1},  # positive turbo
                2: {'a': 1.02, 'b': 1.1, 'c': 1.5, 'a2': 1.1},  # positive natural aspiration
                3: {'b': 1.08, 'c': 1.15, 'a2': 1.1},  # compression
                4: {'b': 1.08, 'c': 1.15, 'a2': 1.1}  # compression + scr
            }
        }

    class FMEP_egr(co2_utl.Constants):
        #: Exhausted gas recirculation multiplication factors ids [-].
        egr_fact_map = {
            ('positive turbo', False): 1,
            ('positive natural aspiration', False): 2,
            ('compression', False): 3,
            ('compression', True): 4
        }

    class identify_co2_emissions(co2_utl.Constants):
        #: Number of perturbations to identify the co2_emissions [-].
        n_perturbations = 100

        #: Enable first step in the co2_params calibration? [-]
        enable_first_step = True

        #: Enable second step in the co2_params calibration? [-]
        enable_second_step = True

        #: Enable third step co2_params calibration in perturbation loop? [-]
        enable_third_step = False

        #: Absolute error in k_refactor between iterations that is acceptable
        #: for convergence in perturbation loop [-].
        xatol = 1e-4

    class calibrate_co2_params(co2_utl.Constants):
        #: Enable first step in the co2_params calibration? [-]
        enable_first_step = False

        #: Enable second step in the co2_params calibration? [-]
        enable_second_step = False

        #: Enable third step in the co2_params calibration? [-]
        enable_third_step = True

    class Alternator_status_model(co2_utl.Constants):
        #: Minimum delta time to consider valid a charging state to fit charges
        #: boundaries [s].
        min_delta_time_boundaries = 5

        #: Minimum acceptance percentile to fit the bers threshold [%].
        min_percentile_bers = 90

        #: Minimum delta soc to set the charging boundaries [%].
        min_delta_soc = 8

    class default_ki_multiplicative(co2_utl.Constants):
        #: Multiplicative correction for vehicles with periodically regenerating
        #: systems [-].
        ki_multiplicative = {True: 1.05, False: 1.0}

    class define_fmep_model(co2_utl.Constants):
        #: Percentage of max full bmep curve used as limit in cylinder
        #: deactivation strategy [-].
        acr_full_bmep_curve_percentage = 0.45

        #: Percentage of max mean piston speeds used as limit in cylinder
        #: deactivation strategy [-].
        acr_max_mean_piston_speeds_percentage = 0.6

        #: Percentage of max full bmep curve used as limit in lean burn
        #: strategy [-].
        lb_full_bmep_curve_percentage = 0.4

        #: Percentage of max mean piston speeds used as limit in lean burn
        #: strategy [-].
        lb_max_mean_piston_speeds_percentage = 0.6

        #: Percentage of max full bmep curve used as limit in exhausted gas
        #: recirculation strategy [-].
        egr_full_bmep_curve_percentage = 0.5

        #: Percentage of max mean piston speeds used as limit in exhausted gas
        #: recirculation strategy [-].
        egr_max_mean_piston_speeds_percentage = 0.5

    class define_idle_model_detector(co2_utl.Constants):
        #: eps parameter of DBSCAN [RPM].
        EPS = 100.0

    class identify_idle_engine_speed_std(co2_utl.Constants):
        #: Min standard deviation value [RPM].
        MIN_STD = 100.0

        #: Max standard deviation percentage of median value [-].
        MAX_STD_PERC = 0.3

    class StartStopModel(co2_utl.Constants):
        #: Maximum allowed velocity to stop the engine [km/h].
        stop_velocity = 2.0

        #: Minimum acceleration to switch on the engine [m/s2].
        plateau_acceleration = 1 / 3.6

    class correct_constant_velocity(co2_utl.Constants):
        #: Constant velocities to correct the upper limits for NEDC [km/h].
        CON_VEL_UP_SHIFT = (15.0, 32.0, 50.0, 70.0)

        #: Window to identify if the shifting matrix has limits close to
        # `CON_VEL_UP_SHIFT` [km/h].
        VEL_UP_WINDOW = 3.5

        #: Delta to add to the limit if this is close to `CON_VEL_UP_SHIFT`
        # [km/h].
        DV_UP_SHIFT = -0.5

        #: Constant velocities to correct the bottom limits for NEDC[km/h].
        CON_VEL_DN_SHIFT = (35.0, 50.0)

        #: Window to identify if the shifting matrix has limits close to
        # `CON_VEL_DN_SHIFT` [km/h].
        VEL_DN_WINDOW = 3.5

        #: Delta to add to the limit if this is close to `CON_VEL_DN_SHIFT`
        # [km/h].
        DV_DN_SHIFT = -1

    class define_initial_co2_emission_model_params_guess(co2_utl.Constants):
        #: Initial guess CO2 emission model params.
        CO2_PARAMS = {
            'positive turbo': collections.OrderedDict([
                ('a', {'value': 0.468678, 'min': 0.0}),  # 'min': 0.398589, 'max': 0.538767},
                ('b', {'value': 0.011859}),  # 'min': 0.006558, 'max': 0.01716},
                ('c', {'value': -0.00069}),  # 'min': -0.00099, 'max': -0.00038},
                ('a2', {'value': -0.00266, 'max': 0.0}),  # 'min': -0.00354, 'max': -0.00179},
                ('b2', {'value': 0, 'min': -1, 'max': 1, 'vary': False}),
                ('l', {'value': -2.14063, 'max': 0.0}),  # 'min': -3.17876, 'max': -1.1025}),
                ('l2', {'value': -0.0025, 'max': 0.0}),  # 'min': -0.00796, 'max': 0.0}),b
                ('t1', {'value': 3.5, 'min': 0.0}),  # 'max': 8.0},
                ('dt', {'value': 1.0, 'min': 0.0}),
                ('t0', {'expr': 't1 + dt', 'value': 4.5}),  # 'max': 8.0},
            ]),
            'positive natural aspiration': collections.OrderedDict([
                ('a', {'value': 0.4851, 'min': 0.0}),  # 'min': 0.40065, 'max': 0.54315},
                ('b', {'value': 0.01193}),  # 'min': -0.00247, 'max': 0.026333},
                ('c', {'value': -0.00065}),  # 'min': -0.00138, 'max': 0.0000888},
                ('a2', {'value': -0.00385, 'max': 0.0}),  # 'min': -0.00663, 'max': -0.00107},
                ('b2', {'value': 0, 'min': -1, 'max': 1, 'vary': False}),
                ('l', {'value': -2.39882, 'max': 0.0}),  # 'min': -3.27698, 'max': -1.72066},
                ('l2', {'value': -0.00286, 'max': 0.0}),  # 'min': -0.00577, 'max': 0.0},
                ('t1', {'value': 3.5, 'min': 0.0}),  # 'max': 8.0},
                ('dt', {'value': 1.0, 'min': 0.0}),
                ('t0', {'expr': 't1 + dt', 'value': 4.5}),  # 'max': 8.0},
            ]),
            'compression': collections.OrderedDict([
                ('a', {'value': 0.391197, 'min': 0.0}),  # 'min': 0.346548, 'max': 0.435846},
                ('b', {'value': 0.028604}),  # 'min': 0.002519, 'max': 0.054688},
                ('c', {'value': -0.00196}),  # 'min': -0.00386, 'max': -0.000057},
                ('a2', {'value': -0.0012, 'max': 0.0}),  # 'min': -0.00233, 'max': -0.000064},
                ('b2', {'value': 0, 'min': -1, 'max': 1, 'vary': False}),
                ('l', {'value': -1.55291, 'max': 0.0}),  # 'min': -2.2856, 'max': -0.82022},
                ('l2', {'value': -0.0076, 'max': 0.0}),  # 'min': -0.01852, 'max': 0.0},
                ('t1', {'value': 3.5, 'min': 0.0}),  # 'max': 8.0},
                ('dt', {'value': 1.0, 'min': 0.0}),
                ('t0', {'expr': 't1 + dt', 'value': 4.5}),  # 'max': 8.0},
            ])
        }

    class identify_charging_statuses(co2_utl.Constants):
        time_window = 4

    class restrict_bounds(co2_utl.Constants):
        #: Multipliers applied into the `restrict_bounds` function.
        CO2_PARAMS_LIMIT_MULTIPLIERS = {
            't0': (0.5, 1.5), 't1': (0.5, 1.5), 'trg': (0.9, 1.1),
            'a': (0.8, 1.2), 'b': (0.8, 1.2), 'c': (1.2, 0.8),
            'a2': (1.2, 0.8), 'l': (1.2, 0.8), 'l2': (1.2, 0.0),
        }

    class default_specific_gear_shifting(co2_utl.Constants):
        #: Specific gear shifting model.
        SPECIFIC_GEAR_SHIFTING = 'ALL'

    class default_clutch_k_factor_curve(co2_utl.Constants):
        #: Torque ratio when speed ratio==0 for clutch model.
        STAND_STILL_TORQUE_RATIO = 1.0

        #: Minimum speed ratio where torque ratio==1 for clutch model.
        LOCKUP_SPEED_RATIO = 0.0

    class default_tc_k_factor_curve(co2_utl.Constants):
        #: Torque ratio when speed ratio==0 for torque converter model.
        STAND_STILL_TORQUE_RATIO = 1.9

        #: Minimum speed ratio where torque ratio==1 for torque converter model.
        LOCKUP_SPEED_RATIO = 0.87

    class select_default_n_dyno_axes(co2_utl.Constants):
        #: Number of dyno axes [-].
        DYNO_AXES = {
            'WLTP': {2: 1, 4: 2},
            'NEDC': collections.defaultdict(lambda: 1)
        }

    class select_phases_integration_times(co2_utl.Constants):
        #: Cycle phases integration times [s].
        INTEGRATION_TIMES = {
            'WLTP': (0.0, 590.0, 1023.0, 1478.0, 1800.0),
            'NEDC': (0.0, 780.0, 1180.0)
        }

    class get_gear_box_efficiency_constants(co2_utl.Constants):
        #: Vehicle gear box efficiency constants (gbp00, gbp10, and gbp01).
        PARAMS = {
            True: {
                'gbp00': {'m': -0.0043233434399999994,
                          'q': {'hot': -0.29823614099999995,
                                'cold': -0.695884329}},
                'gbp10': {'m': -2.4525999999999996e-06,
                          'q': {'hot': -0.0001547871,
                                'cold': -0.0005171569}},
                'gbp01': {'q': {'hot': 0.9793688500000001,
                                'cold': 0.96921995}},
            },
            False: {
                'gbp00': {'m': -0.0043233434399999994,
                          'q': {'hot': -0.29823614099999995,
                                'cold': -0.695884329}},
                'gbp10': {'m': -2.4525999999999996e-06,
                          'q': {'hot': -5.15957e-05,
                                'cold': -0.00012958919999999998}},
                'gbp01': {'q': {'hot': 0.9895177500000001,
                                'cold': 0.9793688500000001}},
            }
        }

    class calculate_engine_mass(co2_utl.Constants):
        #: Equivalent gear box heat capacity parameters.
        PARAMS = {
            'mass_coeff': {
                'compression': 1.1,
                'positive': 1.0
            },
            'mass_reg_coeff': (0.4208, 60)
        }

    class calculate_engine_heat_capacity(co2_utl.Constants):
        #: Equivalent gear box heat capacity parameters.
        PARAMS = {
            'heated_mass_percentage': {
                'coolant': 0.04,  # coolant: 50%/50% (0.85*4.186)
                'oil': 0.055,  # oil: lubricant
                'crankcase': 0.18,  # crankcase: cast iron
                'cyl_head': 0.09,  # cyl_head: aluminium
                'pistons': 0.025,  # pistons: aluminium
                'crankshaft': 0.08,  # crankshaft: steel
                'body': 0.1  # body: cast iron
            },
            # Cp in (J/kgK)
            'heat_capacity': {
                'coolant': 0.85 * 4186.0,
                'oil': 2090.0,
                'crankcase': 460.0,
                'cyl_head': 910.0,
                'pistons': 910.0,
                'crankshaft': 490.0,
                'body': 460.0
            }
        }

    class calculate_equivalent_gear_box_heat_capacity(co2_utl.Constants):
        #: Equivalent gear box heat capacity parameters.
        PARAMS = {
            'gear_box_mass_engine_ratio': 0.25,
            # Cp in (J/kgK)
            'heat_capacity': {
                'oil': 2090.0,
            },
            'thermal_management_factor': 0.5
        }

    class default_full_load_speeds_and_powers(co2_utl.Constants):
        #: Vehicle normalized full load curve.
        FULL_LOAD = {
            'positive': (
                [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2],
                [0.1, 0.198238659, 0.30313392, 0.410104642, 0.516920841,
                 0.621300767, 0.723313491, 0.820780368, 0.901750158,
                 0.962968496, 0.995867804, 0.953356174, 0.85]
            ),
            'compression': (
                [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2],
                [0.1, 0.278071182, 0.427366185, 0.572340499, 0.683251935,
                 0.772776746, 0.846217049, 0.906754984, 0.94977083, 0.981937981,
                 1, 0.937598144, 0.85]
            )
        }

    class calculate_engine_max_torque(co2_utl.Constants):
        #: Engine nominal torque params.
        PARAMS = {
            'positive': 1.25,
            'compression': 1.1
        }

    class calculate_engine_moment_inertia(co2_utl.Constants):
        #: Engine moment of inertia params.
        PARAMS = {
            'positive': 1,
            'compression': 2
        }

    class calculate_co2_emissions(co2_utl.Constants):
        # idle ratio to define the fuel cutoff [-].
        cutoff_idle_ratio = 1.1

    class default_initial_state_of_charge(co2_utl.Constants):
        # default initial state of charge of the battery [%].
        initial_state_of_charge = {
            'WLTP': 90,
            'NEDC': 99
        }

    class default_fuel_density(co2_utl.Constants):
        #: Enable function?
        ENABLE = False

        #: Fuel density [g/l].
        FUEL_DENSITY = {
            'gasoline': 745.0,
            'diesel': 832.0,
            'LPG': 43200.0 / 46000.0 * 745.0,  # Gasoline equivalent.
            'NG': 43200.0 / 45100.0 * 745.0,  # Gasoline equivalent.
            'ethanol': 794.0,
            'methanol': 791.0,
            'propane': 510.0,
            'biodiesel': 890.0,
        }

    class default_fuel_lower_heating_value(co2_utl.Constants):
        #: Enable function?
        ENABLE = False

        #: Fuel lower heating value [kJ/kg].
        LHV = {
            'gasoline': 43200.0,
            'diesel': 43100.0,
            'LPG': 46000.0,
            'NG': 45100.0,
            'ethanol': 26800.0,
            'methanol': 19800.0,
            'propane': 49680.0,
            'biodiesel': 37900.0,
        }

    class default_fuel_carbon_content(co2_utl.Constants):
        #: Enable function?
        ENABLE = False

        #: Fuel carbon content [CO2g/g].
        CARBON_CONTENT = {
            'gasoline': 3.17,
            'diesel': 3.16,
            'LPG': 1.35,
            'NG': 3.21,
            'ethanol': 1.91,
            'methanol': 1.37,
            'propane': 2.99,
            'biodiesel': 2.81,
        }

    class calibrate_cold_start_speed_model_v1(co2_utl.Constants):
        #: Cold start engine speed model v1 params.
        PARAMS = {
            'first_seconds': 10.0,  # [s]
            'delta_speed_limits': (0.05, 0.2),  # [-, -]
            'max_temperature': 30.0  # [°C]
        }

    class calculate_cold_start_speeds_delta(co2_utl.Constants):
        #: Maximum cold start speed delta percentage of idle [-].
        MAX_COLD_START_SPEED_DELTA_PERCENTAGE = 1.0

    class _yield_on_start(co2_utl.Constants):
        #: Minimum velocity that allow to switch off stop the engine after an
        #: off [km/h].
        VEL = 1.0

        #: Minimum acceleration that allow to switch off stop the engine after
        #: an off [m/s2].
        ACC = 0.05


class Defaults(co2_utl.Constants):
    values = Values()
    functions = Functions()

    #: Machine error.
    EPS = np.finfo(np.float32).eps

    #: Infinite value.
    INF = 10000.0


dfl = Defaults()
