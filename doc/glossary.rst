Glossary
========
this page lists and explains the terms used in the input file.

GENERAL
-------
.. glossary::

    ``extension``
        Expansion of the interpolation line (i.e. extension of the |CO2|
        values). It cannot be performed for any other purposes (EVAP, etc.).
        It is defined in section 3 of Annex I of Regulation (EC) No 692/2008.

    ``bifuel``
        A vehicle with multi-fuel engine, capable of running on two fuels.

    ``incomplete``
        A vehicle that must undergo at least one further stage of completion
        before it can be considered complete.

    ``atct_family_correction_factor``
        Family correction factor used to correct for representative regional
        temperature conditions (ATCT).

    ``wltp_retest``
        It indicates which test conditions have been subject to retesting
        (see point 2.2a of Annex I of Regulation (EU) 2018/2043). Input can have
        multiple letters combination, leave it empty if not applicable.

    ``parent_vehicle_family_id``
        The family identifier code of the parent.

    ``regulation``
        It indicates if WLTP test has been performed in accordance with point
        5.1.2 of Annex VIII to Regulation (EU) No 582/2011.

    ``vehicle_family_id``
        The family identifier code shall consist of one unique string of
        n-characters and one unique WMI code followed by '1'.

    ``input_type``
        It indicates the data input type, i.e. `Pure ICE`, `NOVC-HEV`, or
        `OVC-HEV.`

    ``comments``
        A field to add comments regarding the DICE procedure. In case of
        extension, or resubmission, kindly provide a detailed description.

Model Inputs
------------
.. glossary::
    ``fuel_type``
        It refers to the type of fuel used during the vehicle test. The user
        must choose the correct one among the following:

        - diesel,
        - gasoline,
        - LPG,
        - NG,
        - ethanol,
        - methanol,
        - biodiesel, and
        - propane.

    ``engine_fuel_lower_heating_value``
        Lower heating value of the fuel used in the test, expressed in [kJ/kg]
        of fuel.

    ``fuel_heating_value``
        Fuel heating value in kwh/l: Value according to Table A6.App2/1
        in Regulation (EU) No [2017/1151][WLTP].

    ``fuel_carbon_content_percentage``
        The amount of carbon present in the fuel by weight, expressed in [%].

    ``ignition_type``
        It indicates whether the engine of the vehicle is a *spark ignition*
        (= *positive ignition*) or a *compression ignition* one.

    ``engine_capacity``
        The total volume of all the cylinders of the engine expressed in cubic
        centimeters [cc].

    ``engine_stroke``
        It is the full travel of the piston along the cylinder, in both
        directions. It is expressed in [mm].

    ``idle_engine_speed_median``
        It represents the engine speed in warm conditions during idling,
        expressed in revolutions per minute [rpm]. It can be measured at the end
        of a WLTP test.

    ``engine_n_cylinders``
        It specifies the maximum number of the engine cylinders. The default is
        *4*.

    ``engine_idle_fuel_consumption``
        It measures the fuel consumption of the vehicle in warm conditions
        during idling. The idling fuel consumption of the vehicle, expressed in
        grams of fuel per second [gFuel/sec] should be measured when:

        - the velocity of the vehicle is 0,
        - the start-stop system is disengaged,
        - the battery state of charge is at balance conditions.

        For |co2mpas| purposes, the engine idle fuel consumption can be measured
        as follows: just after a WLTP physical test, when the engine is still
        warm, leave the vehicle to idle for 3 minutes so that it stabilizes.
        Then make a constant measurement of fuel consumption for 2 minutes.
        Disregard the first minute, then calculate idle fuel consumption as the
        average fuel consumption of the vehicle during the subsequent 1 minute.

    ``final_drive_ratio``
        It is the ratio of gearbox output shaft to driven wheel revolutions. If
        the vehicle has more than one final drive ratio, it has to be left blank
        and use the ``final_drive_ratios``.

    ``final_drive_ratios``
        It specifies the final-drive ratios for each gear.

    ``tyre_code``
        The code of the tyres used in the WLTP/NEDC test (e.g., P195/55R16 85H).
        |co2mpas| does not require the full tyre code to work, however at
        least provide the following information (e.g., 195/55R16):

        - nominal width of the tyre, in [mm];
        - the ratio of height to width [%]; and
        - the load index.

        In case that the front and rear wheels are equipped with tyres of
        different radius (tyres of different width do not affect |co2mpas|),
        then the size of the tyres fitted in the powered axle should be declared
        as input to |co2mpas|. For vehicles with different front and rear
        wheels tyres tested in 4x4 mode, then the size of the tyres from the
        wheels where the OBD/CAN vehicle speed signal is measured should be
        declared as input to |co2mpas|.

    ``gear_box_type``
        The type of gearbox among automatic transmission, manual transmission,
        continuously variable transmission (CVT) or planetary (exclusively for
        hybrid vehicles fitted with a planetary gearset).

    ``start_stop_activation_time``
        It is the time elapsed from the beginning of the NEDC test to the first
        time the Start-Stop system is enabled, expressed in seconds [s].

    ``alternator_nominal_voltage``
        Alternator nominal voltage [V], i.e. the nominal voltage of the service
        battery.

    ``alternator_nominal_power``
        Alternator maximum power [kW], i.e. the rated power of the electric
        machine.

    ``service_battery_capacity``
        Capacity [Ah] of the service battery, e.g. the low voltage battery.

    ``service_battery_nominal_voltage``
        Service battery nominal voltage [V] as described in Appendix 2 to
        Sub-Annex 6 to Annex XXI to Regulation (EU) No [2017/1151][WLTP].

    ``initial_temperature``
        The initial temperature of the test cell during the test. It is used to
        calibrate the thermal model. The default value is *23* °C for WLTP and
        *25* °C for NEDC.

    ``alternator_efficiency``
        It is the ratio of electrical power out of the alternator to the
        mechanical power put into it. If not expressed by the manufacturer,
        then it is by default *0.67*.

    ``gear_box_ratios``
        It defines the ratios of engine to gearbox output shaft revolutions.

    ``full_load_speeds``
        They are rotational speed setpoints defining the engine full load curve
        expressed in *RPM*.

    ``full_load_powers``
        They are power values defining the engine full load curve expressed in
        *kW*.

    ``vehicle_mass``
        - For the WLTP: it is the simulated inertia applied during the test on
          the dyno [kg]. It should reflect correction for rotational mass |mr|
          as foreseen by WLTP regulation for 1-axle chassis dyno testing.
          (Regulation 2017/1151; Sub-Annex 4; paragraph 2.5.3).
        - For the NEDC: it is the inertia [kg] class of NEDC without the
          correction for rotating parts [kg].

    ``f0``
        It corresponds to the rolling resistance force [N] when the angle slope
        is 0 applied to the Dyno during the test cycle. This is defined by Dyno
        procedure.

    ``f1``
        It corresponds to the resistance :math:`[\frac{N}{kmh}]` function of the
        velocity applied to the Dyno during the test cycle. This is defined by
        Dyno procedure.

    ``f2``
        It corresponds to the aerodynamic resistance :math:`[\frac{N}{{kmh}^2}]`
        applied to the Dyno during the test cycle. This is defined by Dyno
        procedure.

    ``co2_emission_low``
        It is the |CO2| emissions bag value [g|CO2|/km] of WLTP low phase, not
        corrected for RCB and not rounded.

    ``co2_emission_medium``
        It is the |CO2| emissions bag value [g|CO2|/km] of WLTP medium phase not
        corrected for RCB and not rounded.

    ``co2_emission_high``
        It is the |CO2| emissions bag value [g|CO2|/km] of WLTP high phase not
        corrected for RCB and not rounded.

    ``co2_emission_extra_high``
        It is the |CO2| emissions bag value [g|CO2|/km] of WLTP extra high phase
        not corrected for RCB and not rounded.

    ``depleting_co2_emission_value``
        It is the combined |CO2| emissions value [g|CO2|/km] of the charge
        depleting tests.

    ``fuel_consumption_value``
        It is the combined fuel consumption [l/100km] of the test not corrected.

    ``sustaining_fuel_consumption_value``
        It is the combined fuel consumption [l/100km] of the charge sustaining
        test not corrected.

    ``rcb_correction``
        It says if the RCB correction has to be (or has been) performed.

    ``speed_distance_correction``
        It says if the speed distance correction has to be (or has been)
        performed.

    ``corrected_co2_emission_value``
        It is the combined |CO2| emissions value [g|CO2|/km] corrected for RCB
        (if applicable), speed & distance (if applicable), Ki factor
        (if applicable), and ATCT (MCO2, C, 5 values from appendix 4 to Annex I
        to Regulation (EU) 2017/1151).

    ``corrected_sustaining_co2_emission_value``
        It is the combined |CO2| emissions value [g|CO2|/km] of the charge
        sustaining test corrected for RCB (if applicable), speed & distance
        (if applicable), Ki factor (if applicable), and ATCT (MCO2, C, 5 values
        from appendix 4 to Annex I to Regulation (EU) 2017/1151).

    ``declared_co2_emission_value``
        It is the declared |CO2| emissions value [g|CO2|/km]. Value should be
        corrected for RCB (if applicable), speed & distance (if applicable), Ki
        factor (if applicable), and ATCT (MCO2, C, 5 values from appendix 4 to
        Annex I to Regulation (EU) 2017/1151).

    ``declared_sustaining_co2_emission_value``
        It is the declared |CO2| emissions value [g|CO2|/km] of the charge
        sustaining test. Value should be corrected for RCB (if applicable),
        speed & distance (if applicable), Ki factor (if applicable), and ATCT
        (MCO2, C, 5 values from appendix 4 to Annex I to Regulation (EU)
        2017/1151).

    ``declared_depleting_co2_emission_value``
        It is the declared |CO2| emissions value [g|CO2|/km] of the charge
        depleting tests. Value should be corrected for RCB (if applicable),
        speed & distance (if applicable), Ki factor (if applicable), and ATCT
        (MCO2, C, 5 values from appendix 4 to Annex I to Regulation (EU)
        2017/1151).

    ``transition_cycle_index``
        Index of the transition cycle according to entry 2.1.1.4.1.4 of Appendix
        8a to Annex I to Regulation (EU) 2017/1151. The transition cycle is the
        cycle before the confirmation cycle (where the break-off criterion is
        satisfied) in the charge-depleting sequence. In the transition cycle the
        operation of the vehicle can be partly charge-depleting and partly
        charge-sustaining.

    ``relative_electric_energy_change``
        The Relative Electric Energy Change (REEC) is a measure of the discharge
        of the vehicle traction REESS during the Charge Depleting test. It is
        calculated as the energy battery balance over the cycle divided by cycle
        energy, according to paraghraph 3.2.4.5.2 of Sub-Annex 8 to Annex XXI to
        Regulation (EU) 2017/1151.

    ``wltp_electric_range``
        The cycle-specific equivalent all-electric range (EAER) is an indication
        of the distance that the vehicle can drive using electric energy,
        according to paraghraph 4.4.4 of Sub-Annex 8 to Annex XXI to Regulation
        (EU) 2017/1151.

    ``nedc_electric_range``
        The NEDC electric range, calculated according to paragraph 4.2.2.1 of
        Annex 9 to UN Regulation 101, is an indication of the distance that the
        vehicle can drive using electric energy.

    ``n_wheel_drive``
        It specifies whether the test is conducted on 2-wheel driving or 4-wheel
        driving.

    ``engine_is_turbo``
        It specifies if the air intake of the engine is equipped with any kind
        of forced induction system set like a turbocharger or supercharger.

    ``has_start_stop``
        It specifies if the start-stop system shuts down the engine of the
        vehicle during idling to reduce fuel consumption and it restarts it
        again when the footbrake/clutch is pressed.

    ``has_energy_recuperation``
        It specifies if the vehicle is equipped with any kind of brake
        energy recuperation technology or regenerative breaking.

    ``has_torque_converter``
        It specifies if the vehicle is equipped with a torque converter.

    ``fuel_saving_at_strategy``
        It allows |co2mpas| to use gear at constant speed driving higher than
        when in transient conditions, resulting in a reduction of the fuel
        consumption.

    ``has_periodically_regenerating_systems``
        It specifies if the vehicle is equipped with periodically regenerating
        systems (anti-pollution devices such as catalytic converter or
        particulate trap). During cycles where regeneration occurs, 
        emission standards need not apply. 
        If a periodic regeneration occurs at least once per Type 1 test 
        and has already occurred at least once during vehicle preparation 
        or the distance between two successive periodic regenerations 
        is more than 4000 km of driving repeated Type 1 tests, 
        it does not require a special test procedure. 
        In this case, Ki factor should be set to 1.0 (``ki_multiplicative``), 
        or 0.0 (``ki_additive``).

    ``engine_has_variable_valve_actuation``
        It specifies if the engine is equipped with technologies that are used
        to enable variable valve event timing, duration and/or lift.
        For example, Valve Timing Control (VTC) — also referred to as
        Variable Valve Timing (VVT) systems - and Variable Valve Lift (VVL) or a
        combination of these systems (phasing, timing and lift variation).

    ``has_engine_idle_coasting``
        It specifies if the engine is allowed to idle during vehicle coasting in
        order to save fuel.

    ``has_engine_off_coasting``
        It specifies if the engine is allowed to turn off during vehicle
        coasting in order to save fuel.

    ``engine_has_cylinder_deactivation``
        It specifies if the engine has a cylinder deactivation system. If yes
        provide the active cylinder ratios in the tab `active_cylinder_ratios`.

    ``active_cylinder_ratios``
        They are the plausible deactivation ratios. For example, in the case of
        an 8-cylinder engine, a 50% deactivation (4 cylinders off) or a 25%
        deactivation ratio (2 cylinders off) are plausible.

        Note that the `active_cylinder_ratios` always start with 1
        (all cylinders are active) and then the user can set the corresponding
        plausible ratios.

    ``has_lean_burn``
        It specifies if the vehicle has lean-burn (LB) technology. This
        technology refers to the burning of fuel with an excess of air in an
        internal combustion engine.

    ``has_gear_box_thermal_management``
        It specifies if the temperature of the gearbox is regulated from the
        vehicle's cooling circuit using a heat-exchanger, heating storage system
        or other methods for directing engine waste-heat to the gearbox.
        Gearbox mounting and other passive systems (encapsulation) should not be
        considered.

    ``has_exhausted_gas_recirculation``
        It specifies if a portion of an engine's exhaust gas back to the engine
        cylinders to reduce |NOx| emissions. The technology does not concern
        internal (in-cylinder) EGR.

    ``has_selective_catalytic_reduction``
        It specifies if the vehicle has the Selective Catalytic Reduction
        (SCR) system active (Urea), or passive (Ammonia) to reduce |NOx|
        emissions.

    ``n_dyno_axes``
        It defines the Dyno rotating axis used during the test.

    ``kco2_wltp_correction_factor``
        |CO2|-emission correction coefficient (KCO2) for charge sustaining
        battery energy balance correction. Paragraph 2.3.2 of Appendix 2 of
        Sub-Annex 8 to Annex XXI to Regulation (EU) 2017/1151.

    ``kco2_nedc_correction_factor``
        |CO2|-emission correction coefficient (KCO2) for charge sustaining
        battery energy balance correction. Paragraph 5.3.5 of Annex 8 of UNECE
        Regulation No. 101 Rev.3.

    ``planetary_ratio``
        It is the ratio existing between the planetary speed and the final
        drive speed during electric drive (engine speed =0). The planetary speed
        is the rotational speed of the planetary gearset side that is not the
        engine nor the final drive side (the branch that goes to the motor P2
        planetary, referred to as the planetary side in this documentation).

    ``initial_drive_battery_state_of_charge``
        It is the initial state of charge of the drive battery at the beginning
        of the test.

    ``drive_battery_n_cells``
        It is the number of cells of the drive battery.

    ``drive_battery_technology``
        If is the technology of the drive battery. The technologies included in
        |co2mpas| are:

        - NiMH: Nickel-metal hydride
        - Li-NCA (Li-Ni-Co-Al): Lithium Nickel Cobalt Aluminum Oxide
        - Li-NCM (Li-Ni-Mn-Co): Lithium Nickel Manganese Cobalt Oxide
        - Li-MO (Li-Mn): Lithium Manganese Oxide
        - Li-FP (Li-Fe-P): Lithium Iron Phosphate
        - Li-TO (Li-Ti): Lithium Titanate Oxide

    ``drive_battery_capacity``
        Capacity [Ah] of the drive battery, e.g. the high voltage battery.

    ``drive_battery_nominal_voltage``
        Drive battery nominal voltage [V], e.g. the nominal voltage of the high
        voltage battery.

    ``motor_p0_maximum_power``
        Maximum power (i.e., the rated power) output of motor P0 [kW].

    ``motor_p0_maximum_torque``
        Maximum torque output of motor P0 [Nm].

    ``motor_p0_speed_ratio``
        The ratio between motor P0 speed and engine speed [-] (e.g. motor P0
        connected to the engine belt with ratio equal to 3 is spinning three
        times faster than the engine).

    ``motor_p1_maximum_power``
        Maximum power (i.e., the rated power) output of motor P1 [kW].

    ``motor_p1_maximum_torque``
        Maximum torque output of motor P1 [Nm].

    ``motor_p1_speed_ratio``
        The ratio between motor P1 speed and engine speed [-] (e.g. motor P1
        connected to the engine crankshaft with ratio equal to 3 is spinning
        three times faster than the engine).

    ``motor_p2_maximum_power``
        Maximum power (i.e., the rated power) output of motor P2 [kW].

    ``motor_p2_maximum_torque``
        Maximum torque output of motor P2 [Nm].

    ``motor_p2_speed_ratio``
        The ratio between motor P2 speed and transmission input speed [-] (motor
        P2 speed is proportional to wheels rotational speed multiplied by the
        final drive ratio and the transmission gear ratio).

    ``motor_p2_planetary_maximum_power``
        Maximum power (i.e., the rated power) output of motor P2 planetary [kW].

    ``motor_p2_planetary_maximum_torque``
        Maximum torque output of motor P2 planetary [Nm].

    ``motor_p2_planetary_speed_ratio``
        The ratio between planetary motor P2 speed and planetary side (branch
        that goes to planetary motor P2) speed.

    ``motor_p3_front_maximum_power``
        Maximum power (i.e., the rated power) output of motor P3 front [kW].

    ``motor_p3_front_maximum_torque``
        Maximum torque output of motor P3 front [Nm].

    ``motor_p3_front_speed_ratio``
        The ratio between motor P3 front speed and final drive input speed [-]
        (motor P3 front speed is equal to wheels rotational speed multiplied by
        the final drive ratio and ), where final drive input speed is
        the rotational speed of the shaft downstream the gearbox (therefore it's
        part of the engine driveline).

    ``motor_p3_rear_maximum_power``
        Maximum power (i.e., the rated power) output of motor P3 rear [kW].

    ``motor_p3_rear_maximum_torque``
        Maximum torque output of motor P3 rear [Nm].

    ``motor_p3_rear_speed_ratio``
        The ratio between motor P3 rear speed and final drive input speed [-]
        (motor P3 rear speed is proportional to wheels rotational speed
        multiplied by the final drive ratio), where final drive input speed is
        the rotational speed of the shaft downstream the gearbox (therefore it's
        part of the engine driveline).

    ``motor_p4_front_maximum_power``
        Maximum power (i.e., the rated power) output of motor P4 front [kW].
        When two P4 motors are present on the same axle, their specifications
        have to be combined to obtain an equivalent single motor in P4 position.

    ``motor_p4_front_maximum_torque``
        Maximum torque output of motor P4 front [Nm]. When two P4 motors are
        present on the same axle, their specifications have to be combined to
        obtain an equivalent single motor in P4 position.

    ``motor_p4_front_speed_ratio``
        The ratio between motor P4 front speed and wheels speed [-] (motor P4
        front speed is proportional to wheels rotational speed).

    ``motor_p4_rear_maximum_power``
        Maximum power (i.e., the rated power) output of motor P4 rear [kW]. When
        two P4 motors are present on the same axle, their specifications have to
        be combined to obtain an equivalent single motor in P4 position.

    ``motor_p4_rear_maximum_torque``
        Maximum torque output of motor P4 rear [Nm]. When two P4 motors are
        present on the same axle, their specifications have to be combined to
        obtain an equivalent single motor in P4 position.

    ``motor_p4_rear_speed_ratio``
        The ratio between motor P4 rear speed and wheels speed [-] (motor P4
        rear speed is proportional to wheels rotational speed).


Time Series
-----------
.. glossary::
    ``times``
        It is the time vector [s].

    ``velocities``
        It is the actual vehicle speed vector [km/h] from the dynamometer.

    ``obd_velocities``
        It is the actual vehicle speed vector [km/h] from the OBD.

    ``gears``
        It is the actual gear vector [-]. If the name of the parameter is
        `target.calibration.gears` it refers to the theoretical gears calculated
        according to Heinz Steven tool [-].

    ``bag_phases``
        It is the array to associate time values with different bag phases (this
        can be used to modify the duration of the phases from the default
        values).

    ``engine_speeds_out``
        It is the actual engine rotational speed vector [rpm] from the OBD.

    ``engine_coolant_temperatures``
        It is the actual engine coolant temperature vector [°C] from the OBD.

    ``co2_normalization_references``
        It is the normalization reference for |CO2| emissions (e.g. engine load,
        engine power output).

    ``alternator_currents``
        It is the current vector produced by the alternator [A] (current is
        negative when the alternator is supplying power to the low-voltage
        electrical system).

    ``service_battery_currents``
        It is the current vector flowing through the service battery [A]
        (current is positive when the battery is being charged, negative when
        discharged).

    ``drive_battery_voltages``
        It is the voltage vector of the drive battery [V].

    ``drive_battery_currents``
        It is the current flowing through the drive battery [A] (current is
        positive when the battery is being charged, negative when discharged).

    ``dcdc_converter_currents``
        It is the current flowing through the DCDC converter measured on the
        low-voltage side [A] (current is negative when the DCDC converter is
        supplying power to the low-voltage electrical system).

General Terms
-------------
.. glossary::
    ``type-approval``
        It is the authority that grants that a vehicle conforms to the EU
        Regulation.

    ``EU legislation``
        COMMISSION IMPLEMENTING REGULATION (EU) 2017/1152: sets out a
        methodology for determining the correlation parameters necessary for
        reflecting the change in the regulatory test procedure with regard to
        light commercial vehicles.
        COMMISSION IMPLEMENTING REGULATION (EU) 2017/1153: sets out a
        methodology for determining the correlation parameters necessary for
        reflecting the change in the regulatory test procedure and amending
        Regulation (EU) No 1014/2010.


.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. |NOx| replace:: NO\ :sub:`x`\
.. |mr| replace:: m\ :sub:`r`\

.. default-role:: obj
