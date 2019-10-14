Glossary
========
this page lists and explains the terms used in the input file.

DICE
----
.. glossary::

    ``extension``
        expansion of the interpolation line (i.e. extension of the |CO2| values).
        It cannot be performed for any other purposes (EVAP, etc.).
        It is defined in section 3 of Annex I of Regulation (EC) No 692/2008.

    ``bifuel``
        vehicle with multifuel engines, capable of running on two fuels.

    ``incomplete``
        a vehicle which must undergo at least one further stage of completion
        before it can be considered a completed.

    ``atct_family_correction_factor``
        family correction factor used to correct for representative regional
        temperature conditions (ATCT).

    ``wltp_retest``
        it indicates which test conditions have been subject to retesting
        (see point 2.2a of Annex I of Regulation (EU) 2018/2043). Input can have
        multiple letters combination, leave it empty if not applicable.

    ``parent_vehicle_family_id``
        the family identifier code of the parent.

    ``vehicle_family_id``
        the family identifier code shall consist of one unique string of
        n-characters and one unique WMI code followed by '1'.

    ``is_hybrid``
         hybrid vehicle where one of the propulsion energy converters is an
         electric machine.

    ``comments``
        you may add comments regarding the DICE procedure. In case of extension,
        or resubmission, kindly provide a detailed description.

Model Inputs
------------
.. glossary::
    ``fuel_type``
        it refers to the type of fuel used during the vehicle test.
        The user must choose the correct one among the following:

        - diesel,
        - gasoline,
        - LPG,
        - NG,
        - ethanol
        - methanol
        - biodiesel
        - propane

    ``engine_fuel_lower_heating_value``
        lower heating value of the fuel used in the test, expressed in [kJ/kg]
        of fuel.

    ``fuel_heating_value``
        fuel heating value in kwh/l: Value according to the Table A6.App2/1
        in Regulation (EU) No [2017/1151][WLTP].

    ``fuel_carbon_content_percentage``
        the amount of carbon present in the fuel by weight, expressed in [%].

    ``ignition_type``
        it indicates whether the engine of the vehicle is a *spark ignition*
        (= *positive ignition*) or a *compression ignition* one.

    ``engine_capacity``
        the total volume of all the cylinders of the engine, expressed in cubic
        centimeters [cc].

    ``engine_stroke``
        is the full travel of the piston along the cylinder, in both directions.
        It is expressed in [mm].

    ``idle_engine_speed_median``
        it represents the engine speed in warm conditions during idling,
        expressed in revolutions per minute [rpm].

    ``engine_n_cylinders``
        it specifies the maximum number of engine cylinder. The default is *4*.

    ``engine_idle_fuel_consumption``
        measures the fuel consumption of the vehicle in warm conditions during
        idling. The idling fuel consumption of the vehicle, expressed in grams
        of fuel per second [gFuel/sec] should be measured when:

        - velocity of the vehicle is 0,
        - the start-stop system is disengaged,
        - the battery state of charge is at balance conditions.

        For |co2mpas| purposes, the engine idle fuel consumption can be measured
        as follows: just after a WLTP physical test, when the engine is still
        warm, leave the car to idle for 3 minutes so that it stabilizes. Then
        make a constant measurement of fuel consumption for 2 minutes.
        Disregard the first minute, then calculate idle fuel consumption as the
        average fuel consumption of the vehicle during the subsequent 1 minute.

    ``final_drive_ratio``
        the ratio to be multiplied with all `gear_box_ratios`. If the car has
        more than 1 final drive ratio (eg, vehicles with dual/variable clutch),
        leave blank the final_drive_ratio cell in the Inputs tab and provide the
        appropriate final drive ratio for each gear in the gear_box_ratios tab.

    ``final_drive_ratios``
        See relevant column in sheet (`gear_box_ratios`).

    ``tyre_code``
        the code of the tyres used in the WLTP test (e.g., P195/55R16 85H).
        |co2mpas| does not require the full tyre code to work, however at
        least provide the following information:

        - nominal width of the tyre, in [mm];
        - ratio of height to width [%]; and
        - the load index (e.g., 195/55R16).

        In case that the front and rear wheels are equipped with tyres of
        different radius (tyres of different width do not affect |co2mpas|),
        then the size of the tyres fitted in the powered axle should be declared
        as input to |co2mpas|. For vehicles with different front and rear
        wheels tyres tested in 4x4 mode, then the size of the tyres from the
        wheels where the OBD/CAN vehicle speed signal is measured should be
        declared as input to |co2mpas|.

    ``gear_box_type``
        the type of gear box among automatic transmission, manual transmission,
        or continuously variable transmission (CVT).

    ``start_stop_activation_time``
        is the the time elapsed from the beginning of the NEDC test to the first
        time the Start-Stop system is enabled, expressed in seconds [s].

    ``alternator_nominal_voltage``
        Alternator nominal voltage [V].

    ``alternator_nominal_power``
        Alternator maximum power [kW].

    ``service_battery_capacity``
        Capacity [Ah] of the service battery, e.g. the low voltage battery.

    ``service_battery_nominal_voltage``
        for low voltage battery as described in Appendix 2 to Sub-Annex 6 to
        Annex XXI to Regulation (EU) No [2017/1151][WLTP].

    ``calibration.initial_temperature WLTP-H``
        Initial temperature of the test cell during the WLTP test. It is used
        to calibrate the thermal model. The default value =23 °C.

    ``calibration.initial_temperature WLTP-L``
        initial temperature of the test cell during the WLTP-L test. Default
        value =23 °C.

    ``alternator_efficiency``
        efficiency is the ratio of electrical power out of the alternator to
        the mechanical power put into it. If not expressed by the manufacturer,
        then it is by default =0.67

    ``gear_box_ratios``
        see relevant sheet (gear_box_ratios).

    ``full_load_speeds``
        T1 map speed. See relavant sheet (T1_map).

    ``full_load_powers``
        T1 map POWER. See relavant sheet (T1_map).

Road Loads
----------
    ``vehicle_mass WLTP-H``
        simulated inertia applied during the WLTP-H test on the dyno [kg].
        It should reflect correction for rotational mass |mr| as foreseen by
        WLTP regulation for 1-axle chassis dyno testing. (Regulation 2017/1151;
        Sub-Annex 4; paragraph 2.5.3)

    ``f0 WLTP-H``
         set the F0 road load coefficient for WLTP-H. This scalar corresponds
         to the rolling resistance force [N], when the angle slope is 0.

    ``f1 WLTP-H``
        set the F1 road load coefficient for WLTP-H. Defined by Dyno procedure
        :math:`[\frac{N}{kmh}]`.

    ``f2 WLTP-H``
        set the F2 road load coefficient for WLTP-H. As used in the Dyno and
        defined by the respective guideline :math:`[\frac{N}{{kmh}^2}]`.

    ``vehicle_mass NEDC-H``
        inertia class of NEDC-H - Do not correct for rotating parts [kg].

    ``f0 NEDC-H``
        set the F0 road load coefficient for NEDC-H. This scalar corresponds to
        the rolling resistance force [N], when the angle slope is 0.

    ``f1 NEDC-H``
        set the F1 road load coefficient for NEDC-H. Defined by Dyno procedure
        :math:`[\frac{N}{kmh}]`.

    ``f2 NEDC-H``
        set the F2 road load coefficient for NEDC-H. As used in the Dyno and
        defined by the respective guideline :math:`[\frac{N}{{kmh}^2}]`.


Targets
-------
    ``co2_emission_low WLTP-H``
        phase low, |CO2| emissions bag values [g|CO2|/km], not corrected for
        RCB, not rounded WLTP-H test measurements.

    ``co2_emission_medium WLTP-H``
        phase medium, |CO2| emissions bag values [g|CO2|/km], not corrected for
        RCB, not rounded WLTP-H test measurements.

    ``co2_emission_high WLTP-H``
        phase high, |CO2| emissions bag values [g|CO2|/km], not corrected for
        RCB, not rounded WLTP-H test measurements.

    ``co2_emission_extra_high WLTP-H``
        phase extra high, |CO2| emissions bag values [g|CO2|/km], not corrected
        for RCB, not rounded WLTP-H test measurements.

    ``target fuel_consumption_value WLTP-H``
        combined fuel consumption for WLTP-H test (l/100 km)

    ``rcb_correction WLTP-H``
        boolean value that signalises if a correction has been performed.

    ``speed_distance_correction WLTP-H``
        boolean value that signalises if a correction has been performed.

    ``target corrected_co2_emission_value WLTP-H``
        combined bag values corrected for RCB (if applicable), speed,
        distance(if applicable), Ki factor (if applicable), and ATCT (MCO2, C, 5
        values from appendix 4 to Annex I to Regulation (EU) 2017/1151).

    ``target declared_co2_emission_value NEDC-H``
        declared value for NEDC vehicle H [g|CO2|/km]. Value should be Ki factor
        corrected.

    ``target declared_co2_emission_value WLTP-H``
        declared value for WLTP vehicle H. Values should be Ki and ATCT factor
        corrected.


Drive Mode
----------
    ``n_wheel_drive WLTP-H``
        specify whether WLTP-H test is conducted on 2-wheel driving or 4-wheel
        driving. The default is 2-wheel drive.

    ``n_wheel_drive NEDC-H``
         specify whether the NEDC-H test is conducted on 2-wheel driving or
         4-wheel driving. The default is 2-wheel drive.


Technologies
------------
    ``engine_is_turbo``
        if the air intake of the engine is equipped with any kind of forced
        induction system set like a turbocharger or supercharger, then set it to
        1; otherwise set it to 0. The default value is 1.

    ``has_start_stop``
        the start-stop system shuts down the engine of the vehicle during idling
        to reduce fuel consumption and it restarts it again when the footbrake/
        clutch is pressed. If the vehicle has a *S-S* system, set it to 1,
        otherwise, set it to 0. The default is 1.

    ``has_energy_recuperation``
        it should be set to 1 if the vehicle is equipped with any kind of brake
        energy recuperation technology or regenerative breaking.
        Otherwise, to 0. The default is 1.

    ``has_torque_converter``
        set it to 1 if the vehicle is equipped with this technology otherwise,
        set it to 0.
        For manual transmission vehicles the default is 0.
        For automatic tranmission vehicles, the default is 1.
        For vehicles with continuously variable transmission, the default is 0.

    ``fuel_saving_at_strategy``
        setting it to 1 allows |co2mpas| to use a gear at constant speed driving
        higher than when in transient conditions, resulting in a reduction of
        the fuel consumption. The default is 1.

    ``has_periodically_regenerating_systems``
        if the vehicle is equipped with periodically regenerating systems
        (anti-pollution devices such as catalytic converter or particulate trap)
        a periodical regeneration process in less than 4000 km of normal vehicle
        operation is required, set it to 1; otherwise, set it to 0.
        The default is 0.

    ``ki_multiplicative ki_additive``
        for vehicles without `has_periodically_regenerating_systems`
        ``ki_multiplicative`` and ``ki_additive`` are set to 1 and 0.
        Otherwise, if not provided ``ki_multiplicative`` or ``ki_additive``,
        ``ki_multiplicative`` and ``ki_additive`` are set to 1.05 and 0. The
        ``ki_multiplicative`` or ``ki_additive`` to be used for |co2mpas| are
        the same value used for NEDC physical tests.

    ``engine_has_variable_valve_actuation``
        this input includes a range of technologies which are used to enable
        variable valve event timing, duration and/or lift. The term, as set, i
        ncludes Valve Timing Control (VTC)—also referred to as Variable Valve
        Timing (VVT) systems and Variable Valve Lift (VVL) or a combination of
        these systems (phasing, timing and lift variation). Set it to 1 if the
        vehicle is equipped with such a system; otherwise, set it to 0.
        The default is 0.

    ``engine_has_cylinder_deactivation``
        does the engine feature a cylinder deactivation system? If yes provide
        the active cylinder ratios in the tab `active_cylinder_ratios`.

    ``active_cylinder_ratios``
        This technology allows the deactivation of one or more cylinders under
        specific conditions predefined in the |co2mpas| code. The implementation
        in |co2mpas| allows to use different deactivation ratios.
        In the case of an 8-cylinder engine, a 50% deactivation (4 cylinders off
        ) or a 25% deactivation ratio (2 cylinders off) are plausible.
        |co2mpas| selects the optimal ratio at each point from the plausible
        deactivation ratios provided by the user. The user cannot alter the
        deactivation strategy. If the vehicle is equipped with a cylinder
        deactivation system, set it to 1 and indicate the deactivation ratios in
        the `active_cylinder_ratios` tab.
        Note that the `active_cylinder_ratios` always start with 1
        (all cylinders are active) and then the user can set the corresponding
        ratios.

        For example, if the vehicle has an engine with 6 cylinders and it has
        the possibility to deactivate 2 or 3 or 4 cylinders, you have to
        introduce the following ratios: 0.66 (4/6), 0.5 (3/6), and 0.33 (2/6).
        If the vehicle does not have cylinder deactivation set
        ``engine_has_cylinder_deactivation`` to 0.
        The default is 0.

    ``has_lean_burn``
        the lean burn (LB) technology refers to the burning of fuel with an
        excess of air in an internal combustion engine. All ``compression ignition``
        vehicles are supposed to be equipped with *LB* by default therefore for
        ``compression ignition`` this must be set to 0.
        For ``positive ignition`` engines set it to 1 if the vehicle is equipped
        with *LB*, otherwise set it to 0. The default is 0.

    ``has_gear_box_thermal_management``
        this specific technology option applies only to vehicles in which the
        temperature of the gearbox is regulated from the vehicle's cooling
        circuit using a heat-exchanger, heating storage system or other methods
        for directing engine waste-heat to the gearbox.
        Gearbox mounting and other passive systems (encapsulation) should not be
        considered. In case the vehicle is equipped with the described gear box
        thermal management system, set it to 1; otherwise, set it to 0.
        The default is 0.

    ``has_exhausted_gas_recirculation``
        EGR recirculates a portion of an engine's exhaust gas back to the engine
         cylinders to reduce |NOx| emissions. The technology does not concern
         internal (in-cylinder) EGR. Set it to 1 if the vehicle is equipped with
         external EGR (high-pressure, low-pressure, or a combination of the
         two); otherwise, set it to 0. The default is 0 for `positive ignition`,
         and 1 for `compression ignition` engines.

    ``has_selective_catalytic_reduction``
        on `compression ignition` vehicles, the Selective Catalytic Reduction
        (SCR) system uses Urea (active), or Ammonia (passive) to reduce |NOx|
        emissions.
        Therefore this technology is only applicable for `compression ignition`
        engines.
        If the vehicle is equipped with SCR set
        `has_selective_catalytic_reduction` to 1; otherwise, set it to 0.
        The default value is 0.

Dyne - Vehicle Configuration
----------------------------
    ``n_dyno_axes WLTP-H``
        the WLTP regulation states that WLTP tests should be performed using
        a dyno with 2 rotating axis. Therefore, the default value for this
        variable is 2. I can be set to 1 if one rotating axis dyno was used
        during the WLTP-H test.


Hybrids - Inputs
----------------
    ``planetary_ratio``
        the ratio existing between the planetary and the final drive rotation
        speed during electric drive (engine speed =0). The planetary speed is
        the rotational speed of the planetary gearset side that is not the
        engine nor the final drive side.

    ``drive_battery_initial_state_of_charge WLTP-H``
        initial state of charge of the high-voltage battery at the beginning of
        the WLTP-H test.

    ``drive_battery_n_cells``
        number of cells of the high-voltage battery.

    ``drive_battery_technology``
        the technology of the battery: e.g., NiMH, Li-NCA, etc.

    ``drive_battery_capacity``
        high voltage battery capacity.

    ``motor_p0_maximum_power``
        rated power of motor P0.

    ``motor_p0_maximum_torque``
        rated torque of motor P0.

    ``motor_p0_speed_ratio``
        ratio between motor P0 speed and engine speed.

    ``motor_p1_maximum_power``
        rated power of motor P1.

    ``motor_p1_maximum_torque``
        rated torque of motor P1.

    ``motor_p1_speed_ratio``
         ratio between motor P1 speed and engine speed.

    ``motor_p2_maximum_power``
        rated power of motor P2.

    ``motor_p2_maximum_torque``
        rated torque of motor P2.

    ``motor_p2_speed_ratio``
        ratio between motor P2 speed and gearbox speed.

    ``motor_p2_planetary_maximum_power``
        rated power of planetary motor P2.

    ``motor_p2_planetary_maximum_torque``
        rated torque of planetary motor P2.

    ``motor_p2_planetary_speed_ratio``
        ratio between planetary motor P2 speed and planetary speed (branch that
        goes to planetary motor P2).

    ``motor_p3_front_maximum_power``
        rated power of front motor P3.

    ``motor_p3_front_maximum_torque``
        rated torque of front motor P3.

    ``motor_p3_front_speed_ratio``
        ratio between front motor P3 speed and final drive speed in.

    ``motor_p3_rear_maximum_power``
        rated power of rear motor P3.

    ``motor_p3_rear_maximum_torque``
        rated torque of rear motor P3.

    ``motor_p3_rear_speed_ratio``
        ratio between rear motor P3 speed and final drive speed in.

    ``motor_p4_front_maximum_power``
        rated power of front motor P4.

    ``motor_p4_front_maximum_torque``
        rated torque of front motor P4.

    ``motor_p4_front_speed_ratio``
        ratio between front motor P4 speed and wheel speed.

    ``motor_p4_rear_maximum_power``
        rated power of rear motor P4.

    ``motor_p4_rear_maximum_torque``
        rated torque of rear motor P4.

    ``motor_p4_rear_speed_ratio``
        ratio between rear motor P4 speed and wheel speed.


Time Series
-----------

    ``times``
        qq

    ``velocities``
        qq

    ``obd_velocities``
        qq

    ``target.calibration.gears``
        qq

    ``bag_phases``
        qq

    ``engine_speeds_out``
        qq

    ``engine_coolant_temperatures``
        qq

    ``co2_normalization_references``
        qq

    ``alternator_currents``
        qq

    ``service_battery_currents``
        qq

    ``drive_battery_voltages``
        qq

    ``drive_battery_currents``
        qq

    ``dcdc_converter_currents``
        qq

General Terms
-------------
    ``type-approval``
        is the authority that grants that a vehicle is conform to the EU
        Regulation.
    ``EU legislations``
        COMMISSION IMPLEMENTING REGULATION (EU) 2017/1152: sets out a methodology
        for determining the correlation parameters necessary for reflecting the
        change in the regulatory test procedure with regard to light commercial
        vehicles.
        COMMISSION IMPLEMENTING REGULATION (EU) 2017/1153: sets out a methodology
        for determining the correlation parameters necessary for reflecting the
        change in the regulatory test procedure and amending Regulation (EU) No
        1014/2010.


.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. |NOx| replace:: NO\ :sub:`x`\
.. |mr| replace:: m\ :sub:`r`\

.. default-role:: obj
