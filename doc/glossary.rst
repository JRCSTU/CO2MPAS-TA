########
Glossary
########

.. contents::
   :depth: 4

.. default-role:: term

.. Tip to the authors: Use this web-app to previes this page: https://sphinxed.wltp.io/


Generic terms
=============
.. glossary::

    regulation
    EU legislation
        All EU regulations related to the tool:

        - `(EU) 2017/1151 <https://eur-lex.europa.eu/eli/reg/2017/1151/oj>`_:
          Commission Regulation (EU) 2017/1151 of 1 June 2017
          supplementing Regulation (EC) No 715/2007 of the European Parliament
          and of the Council on `type-approval` of motor vehicles with respect to
          emissions from light passenger and commercial vehicles (Euro 5 and Euro 6)
          and on access to vehicle repair and maintenance information,
          amending Directive 2007/46/EC of the European Parliament and of the Council,
          Commission Regulation (EC) No 692/2008 and Commission Regulation (EU) No 1230/2012
          and repealing Commission Regulation (EC) No 692/2008 (Text with EEA relevance)

        - `(EU) 2017/1152 <https://eur-lex.europa.eu/eli/reg_impl/2017/1152/oj>`_:
          Commission Implementing Regulation (EU) 2017/1152 of 2 June 2017
          setting out a methodology for determining the correlation parameters
          necessary for reflecting the change in the regulatory test procedure
          with regard to light commercial vehicles and amending Implementing Regulation
          (EU) No 293/2012 (Text with EEA relevance)

        - `(EU) 2017/1153 <https://eur-lex.europa.eu/eli/reg_impl/2017/1153/oj>`_:
          Commission Implementing Regulation (EU) 2017/1153 of 2 June 2017
          setting out a methodology for determining the correlation parameters
          necessary for reflecting the change in the regulatory test procedure
          and amending Regulation (EU) No 1014/2010 (Text with EEA relevance)

    NEDC
        New European Driving Cycle

    WLTP
    type-approval
        Worldwide harmonized Light vehicles Test Procedures

    |co2mpas|
        May refer to the application, the correlation procedure, or
        to the `WLTP` --> `NEDC` simulator.

    repeatability
        The capability of |co2mpas| to duplicate the exact simulation results when running repeatedly
        **on the same** computer.
        This is guaranteed by using non-stochastic algorithms (or using always the same random-seed).

    reproducibility
    replicability
        The capability of |co2mpas| to duplicate the exact same simulation results on **a different computer**.
        This is guaranteed when using the All-in-One environment.

    e-file
    electronic-file
        Any piece of information stored in electronic form that constitutes
        the input or the output of some software application or IT procedure.

    Semantic Versioning
        Given a version number ``MAJOR.MINOR.PATCH``, increment the:

        - ``MAJOR`` version when you make incompatible API changes,
        - ``MINOR`` version when you add functionality in a backwards-compatible
          manner, and
        - ``PATCH`` version when you make backwards-compatible bug fixes.

        See https://semver.org/

    hash
    Hash-ID
        A very big number usually expressed in hexadecimal form (e.g. `SHA1`)
        that can be generated cryptographically from any kind of `e-file` based
        exclusively on its contents; even if a single bit of the file changes,
        its hash-id is guaranteed to be totally different.

    Git
        An open-source version control system use for software development that
        organizes files in versioned folders, stored based on their `hash`.
        It is distributed, in the sense that any Git installation can communicate and exchange
        files and versioned folders with any other installation.

    SHA1
        A fast and hashing algorithm with 160bit numbers (20 bytes, 40 hex digits),
        used, among others, by `Git`.

        Example::

               SHA1("CO2MPAS") = c5badbe95ad77c0ca66abed422c964aa080d8c07

    JSON
        JavaScript Object Notation:  a lightweight human-readable data-interchange
        data format, easy for machines to parse and generate.
        https://en.wikipedia.org/wiki/JSON

    YAML
        Ain't Markup Language: A human-friendly data serialization language,
        commonly used for configuration files and data exchnage.
        https://en.wikipedia.org/wiki/YAML

    IO
        Input/Output; when referring to a software application, we mean the internal interfaces
        that read and write files and streams of data from devices, databases or other external resources.

    OEM
        Original Equipment Manufacturers, eg. a Vehicle manufacturer

    TAA
        Type Approval Authority: the national supervision body for a `type-approval`
        procedure

    TS
        Technical service: the entity running the `WLTP` on behalf of the `OEM`,
        which reports to some `TAA`.  in some cases, the `TAA` might be also the *TS*.

    designated user
        Any organizational entity or person (usually a `TS`) running `type-approval`
        on behalf of some `OEM` and reporting to some `TAA`.

    Capped cycles
        For vehicles that cannot follow the standard NEDC/WLTP cycles (for example, because they have not enough power to attain the acceleration and maximum speed values required in the operating cycle) it is still possible to use the |co2mpas| tool to predict the NEDC |co2| emission. For these capped cycles, the vehicle has to be operated with the accelerator control fully depressed until they once again reach the required operating curve. Thus, the operated cycle may last more than the standard duration seconds and the subphases may vary in duration. Therefore there is a need to indicate the exact duration of each subphase. This can be done by filling in, the corresponding bag_phases vector in the input file which define the phases integration time [1,1,1,...,2,2,2,...,3,3,3,...,4,4,4]. Providing this input for WLTP cycles together with the other standard vectorial inputs such as speed,engine speed, etc. allows |co2mpas| to process a "modified" WLTP and get calibrated properly. The NEDC that is predicted corresponds to the respective NEDC velocity profile and gearshifting that applies to the capped cycle, which is provided in the appropriate tab. Note that, providing NEDC velocity and gear shifting profile is not allowed for normal vehicles.

    AIO
    ALLINONE
        The *All-In-One is a "fat" archive (~1.4GB when inflated) containing
        all *3rd-party* applications, `WinPython` and all python packages
        required to run |co2mpas| for `type-approval` purposes.

        The official version to download is specified at the top of
        |co2mpas| landing page: https://co2mpas.io

    polyvers
    polyversion
        A utility that versions python-projects accurately based on git commits
        & tags.

    WinPython
        The :term:`WinPython` distribution is just a collection of
        standard pre-compiled binaries for *Windows* containing all
        the scientific packages, and much more. It is not update-able,
        and has a quasi-regular release-cycle of 3 months.

        The `ALLINONE` for official `type-approval` is based on this distribution.

    conda
    Anaconda
        A python distribution & package-manager different from the "standard' one.
        It was crafted originally for scientific python libraries (`numpy/pandas`)
        but has now evolved to a full blown software delivery platform, that
        included native packages (e.g. `GCC` & `GLib`).

        Can be downloaded from: http://continuum.io/downloads

    MSYS2
    MinGW
    Cygwin
        Open-source command-line environments for *Windows*, providing a `POSIX`
        emulation layer and a software development framework (compilers, etc).
        *Cygwin* was shipped with older `ALLINONE` archives, `MSYS2
        <https://www.msys2.org/>`_ since `1.7.3`.

    Unix
    POSIX
        The `Portable Operating System Interface <https://en.wikipedia.org/wiki/POSIX>`_
        family of standards that all variants of *Unix* comply with.


Input file terminology
=========================
Vehicle general characteristics
-------------------------------
.. glossary::

    Rotational mass
        The rotational mass is defined in the WLTP GTR (ECE/TRANS/WP.29/GRPE/2016/3) as the equivalent effective mass of all
        the  wheels and vehicle components rotating with the wheels on the road while the gearbox is placed in neutral, in kg. It shall
        be measured or calculated using an appropriate technique agreed upon by the responsible authority. Alternatively, it may be
        estimated to be 3 per cent of the sum of the mass in running order and 25 kg.

    ``input_version``
        It corresponds to the version of the template file used for |co2mpas| -
        not to the |co2mpas| version of the code.
        Different versions of the file have been used throughout the development of the tool.
        Input files from version >= 2.2.5 can be used for type approving.

        Check the currently supported version with ``co2mpas -vV`` command, or visit
        the "about" help item of the GUI.

    ``IF_ID``
    ``VF_ID``
    ``vehicle_family_id``
        It corresponds to an individual code for each vehicle that is simulated with the |co2mpas| model.
        This ID does not affect the NEDC prediction.
        The ID is allocated in the `output report` and in the `dice report`.

        The new structure of the ID, as defined in paragraph 5.0 of Annex XXI of
        the *amended* `regulation`, is the following:

            FT-nnnnnnnnnnnnnnn-WMI-x

        Where:

        - ``FT`` (Family Type) is pinned to ``'IP'`` (Interpolation Family)
          from paragraph 5.6, Annex XXI.

        - ``nnnnnnnnnnnnnnn`` is a string with a maximum of fifteen characters,
          restricted to using the characters 0-9, A-Z and the underscore character '_'.

        - ``WMI`` (world manufacturer identifier) is a code that identifies
          the manufacturer in a unique manner and is defined in ISO 3780:2009.
          See also: https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/World_Manufacturer_Identifier_(WMI)

        - ``x``: shall be set to '1' or '0' in accordance with the following
          provisions:

          a. With the agreement of the approval authority and the owner of the WMI,
             the number shall be set to '1' where a vehicle family is defined
             for the purpose of covering vehicles of:

             1. a single manufacturer with one single WMI code;
             2. a manufacturer with several WMI codes, but only in cases when
                one WMI code is to be used;
             3. more than one manufacturer, but only in cases when one WMI code
                is to be used.

             In the cases (1), (2) and (3), the family identifier code shall consist
             of one unique string of n-characters and one unique WMI code followed by '1';

          b. With the agreement of the approval authority, the number shall be set
             to '0' in the case that a vehicle family is defined based on the same criteria
             as the corresponding vehicle family defined in accordance with point (a),
             but the manufacturer chooses to use a different WMI.
             In this case the family identifier code shall consist of the same string
             of n-characters as the one determined for the vehicle family defined
             in accordance with point (a) and a unique WMI code which shall be different
             from any of the WMI codes used under case (a), followed by '0'.

        .. Attention::
            The format has changed in the legislation since May 2018 and in co2mpas
            after version (and including) ``v1.8.x``.
            The old format **is still supported** i.e. for extensions
            (but cell-validations in the input excel file must be disabled)::

                FT-TA-WMI-yyyy-nnnn

            Where:

            - ``FT`` is the identifier of the Family Type according to this:

              - ``'IP'``: Interpolation family as defined in paragraph 5.6, Annex XXI.
              - ``'RL'``: Road load family as defined in paragraph 5.7, Annex XXI.
              - ``'RM'``: Road load matrix family as defined in paragraph 5.8, Annex XXI.
              - ``'PR'``: Periodically regenerating systems (Ki) family as defined
                in paragraph 5.9, Annex XXI.

            - ``TA`` is the distinguishing number of the EC Member State authority responsible for the family approval
              as defined in `section 1 of point 1 of Annex VII of Directive (EC) 2007/46
              <http://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32007L0046&from=EN>`_:

              - 1 for Germany;
              - 2 for France;
              - 3 for Italy;
              - 4 for the Netherlands;
              - 5 for Sweden;
              - 6 for Belgium;
              - 7 for Hungary;
              - 8 for the Czech Republic;
              - 9 for Spain;
              - 11 for the United Kingdom;
              - 12 for Austria;
              - 13 for Luxembourg;
              - 17 for Finland;
              - 18 for Denmark;
              - 19 for Romania;
              - 20 for Poland;
              - 21 for Portugal;
              - 23 for Greece;
              - 24 for Ireland;
              - 26 for Slovenia;
              - 27 for Slovakia;
              - 29 for Estonia;
              - 32 for Latvia;
              - 34 for Bulgaria;
              - 36 for Lithuania;
              - 49 for Cyprus;
              - 50 for Malta.

            - ``WMI`` (world manufacturer identifier) is a code that identifies the manufacturer
              in a unique manner and is defined in ISO 3780:2009.
              For a single manufacturers several WMI codes may be used.
            - ``yyyy`` is the year when the test for the family were concluded.
            - ``nnnn`` is a four digit sequence number.


    ``fuel_type``
        Used to indicate the type of fuel used by the vehicle during the test.
        The user must select one among the following options:

        - diesel,
        - gasoline,
        - LPG,
        - NG or biomethane,
        - ethanol(E85) or
        - biodiesel.

    ``engine fuel lower heating value``
        Lower heating value of the fuel used in the test, expressed in [kJ/kg] of fuel.

    ``fuel_heating_value``
        Fuel heating value in kwh/l: Value according to the Table A6.App2/1 
        in Regulation (EU) No [2017/1151][WLTP] (Optional).

    ``fuel_carbon_content_percentage``
        The amount of carbon present in the fuel by weight, expressed in [%].

    positive ignition
    compression ignition
    ``ignition_type``
        Indicate wether the engine of the vehicle is a *spark ignition* (= *positive ignition*) or
        a *compression ignition* one.

    ``engine_capacity``
        The total volume of all the cylinders of the engine, expressed in cubic centimeters [cc].

    ``engine_stroke``
        A stroke refers to the full travel of the piston along the cylinder, in either direction.
        Indicate the stroke of the engine, expressed in [mm].

    ``idle_engine_speed_median``
        Indicate the engine speed in warm conditions during idling, expressed in revolutions per minute [rpm].

    ``engine_idle_fuel_consumption``
        Provide the fuel consumption of the vehicle in warm conditions during idling. The idling fuel consumption
        of the vehicle, expressed in grams of fuel per second [gFuel/sec] should be measured when:

        - velocity of the vehicle is 0
        - the start-stop system is disengaged
        - the battery state of charge is at balance conditions.

        For |co2mpas| purposes, the engine idle fuel consumption can be measured as follows: just after a WLTP physical test,
        when the engine is still warm, leave the car to idle for 3 minutes so that it stabilizes. Then make a constant
        measurement of fuel consumption for 2 minutes. Disregard the first minute, then calculate idle fuel consumption as the
        average fuel consumption of the vehicle during the subsequent 1 minute.

    ``engine_n_cylinders``
        Specify the maximum number of engine cylinder. The default is 4.

    ``final_drive_ratio``
        Provide the ratio to be multiplied with all `gear_box_ratios`. If the car has more than 1 final drive ratio (eg,
        vehicles with dual/variable clutch), leave blank the final_drive_ratio cell in the Inputs tab and provide the
        appropriate final drive ratio for each gear in the gear_box_ratios tab.

    ``tyre_code``
        Tyre code of the tyres used in the WLTP test (e.g., P195/55R16 85H\).
        |co2mpas| does not require the full tyre code to work.
        But at least provide the following information:

        - nominal width of the tyre, in [mm];
        - ratio of height to width [%]; and
        - the load index (e.g., 195/55R16\).

        In case that the front and rear wheels are equipped with tyres of different radius (tyres of different width do not
        affect |co2mpas|), then the size of the tyres fitted in the powered axle should be declared as input to |co2mpas|.
        For vehicles with different front and rear wheels tyres tested in 4x4 mode, then the size of the tyres from the wheels
        where the OBD/CAN vehicle speed signal is measured should be declared as input to |co2mpas|.

    ``gear_box_type``
        Indicate the kind of gear box among automatic transmission, manual transmission, or
        continuously variable transmission (CVT).

    ``start_stop_activation_time``
        Indicate the time elapsed from the begining of the NEDC test to the first time the Start-Stop system is enabled,
        expressed in seconds [s].

    ``alternator_nomimal_voltage``
        Alternator nomimal voltage [V].

    ``alternator_nomimal_power``
        Alternator maximum power [kW].

    ``battery_capacity``
        Battery capacity [Ah].
        
    ``battery_voltage``
        For low voltage battery as described in Appendix 2 
        to Sub-Annex 6 to Annex XXI to Regulation (EU) No [2017/1151][WLTP] (Optional).

    ``atct_family_correction_factor``
        family correction factor for correcting for representative regional 
        temperature conditions (ATCT) (Optional).

    ``calibration.initial_temperature.WLTP-H``
        Initial temperature of the test cell during the WLTP-H test. It is used to calibrate the thermal model.
        The default value is 23 °C.

    ``calibration.initial_temperature.WLTP-L``
        Initial temperature of the test cell during the WLTP-L test. It is used to calibrate the thermal model.
        The default value is 23 °C.

    ``alternator_efficiency``
        Average alternator efficiency as declared by the manufacturer; if the value is not provided,
        the default value is = 0.67.

    ``gear_box_ratios``
        Insert in the ``gear_box_ratios`` tab of the input file the gear box ratios as an array
        ``[ratio gear 1, ratio gear 2, ...]``

    ``full_load_speeds``
        Insert in the ``T1_map`` tab of the input file the engine full load speeds. Input the engine speed [rpm] array used by
        the OEM to calculate the gearshifting in WLTP. The engine maximum speed, and the engine speed at maximum power are
        read from this array.

    ``full_load_powers``
        Insert in the ``T1_map`` tab of the input file the engine full load powers. Input the engine power [kW] array used by
        the OEM to calculate the gearshifting in WLTP. The engine maximum power is read from this array.


Road loads
----------
.. glossary::
    ``vehicle_mass.WLTP-H``
        Simulated inertia applied during the WLTP-H test on the dyno [kg].
        It should reflect correction for rotational mass |mr| as foreseen by WLTP regulation
        for 1-axle chassis dyno testing. (Regulation 2017/1151; Sub-Annex 4; paragraph 2.5.3)

    ``f0.WLTP-H``
        Set the F0 road load coefficient for WLTP-H. This scalar corresponds to the rolling resistance force [N], when the angle slope is 0.

    ``f1.WLTP-H``
        Set the F1 road load coefficient for WLTP-H. Defined by Dyno procedure :math:`[\frac{N}{kmh}]`.

    ``f2.WLTP-H``
        Set the F2 road load coefficient for WLTP-H. As used in the Dyno and defined by the respective guideline
        :math:`[\frac{N}{{kmh}^2}]`.

    ``vehicle_mass.WLTP-L``
        Simulated inertia applied during the WLTP-L test on the dyno [kg].
        It should reflect correction for rotational mass |mr| as foreseen by WLTP regulation
        for 1-axle chassis dyno testing. (Regulation 2017/1151; Sub-Annex 4; paragraph 2.5.3)

    ``f0.WLTP-L``
        Set the F0 road load coefficient for WLTP-L. This scalar corresponds to the rolling resistance force [N], when the angle slope is 0.

    ``f1.WLTP-L``
        Set the F1 road load coefficient for WLTP-L. Defined by Dyno procedure :math:`[\frac{N}{kmh}]`.

    ``f2.WLTP-L``
        Set the F2 road load coefficient for WLTP-L. As used in the Dyno and defined by the respective guideline
        :math:`[\frac{N}{{kmh}^2}]`.

    ``vehicle_mass.NEDC-H``
        Inertia class of NEDC-H - Do not correct for rotating parts [kg].

    ``f0.NEDC-H``
        Set the F0 road load coefficient for NEDC-H. This scalar corresponds to the rolling resistance force [N],
        when the angle slope is 0.

    ``f1.NEDC-H``
        Set the F1 road load coefficient for NEDC-H. Defined by Dyno procedure :math:`[\frac{N}{kmh}]`.

    ``f2.NEDC-H``
        Set the F2 road load coefficient for NEDC-H. As used in the Dyno and defined by the respective guideline
        :math:`[\frac{N}{{kmh}^2}]`.

    ``vehicle_mass.NEDC-L``
        Inertia class of NEDC-H - Do not correct for rotating parts. [kg]

    ``f0.NEDC-L``
        Set the F0 road load coefficient for NEDC-L. This scalar corresponds to the rolling resistance force [N],
        when the angle slope is 0.

    ``f1.NEDC-L``
        Set the F1 road load coefficient for NEDC-L. Defined by Dyno procedure :math:`[\frac{N}{kmh}]`.

    ``f2.NEDC-L``
        Set the F2 road load coefficient for NEDC-L. As used in the Dyno and defined by the respective guideline
        :math:`[\frac{N}{{kmh}^2}]`.



Targets
-------
.. glossary::
    ``co2_emissions_low.WLTP-H``
        Phase low, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB, not rounded WLTP-H test measurements.

    ``co2_emissions_medium.WLTP-H``
        Phase medium, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB, not rounded WLTP-H test measurements.

    ``co2_emissions_high.WLTP-H``
        Phase high, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB, not rounded WLTP-H test measurements.

    ``co2_emissions_extra_high.WLTP-H``
        Phase extra high, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB,
        not rounded WLTP-H test measurements.

    ``co2_emissions_low.WLTP-L``
        Phase low, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB, not rounded WLTP-L test measurements.

    ``co2_emissions_medium.WLTP-L``
        Phase medium, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB, not rounded WLTP-L test measurements.

    ``co2_emissions_high.WLTP-L``
        Phase high, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB, not rounded WLTP-L test measurements.

    ``co2_emissions_extra_high.WLTP-L``
        Phase extra high, |CO2| emissions bag values [g|CO2|/km], not corrected for RCB, not rounded WLTP-L test measurements.

    ``target declared_co2_emission_value.NEDC-H``
        Declared value for NEDC vehicle H [g|CO2|/km]. Value should be Ki factor corrected.

    ``target declared_co2_emission_value.NEDC-L``
        Declared value for NEDC vehicle L [g|CO2|/km]. Value should be Ki factor corrected.

    ``ta_certificate_number``
        Type approving body certificate number. This number is printed in the output file of |co2mpas|

Drive mode
----------
The |co2mpas| model can handle vehicles that have 2x4 and 4x4 wheel drive.
Provide in this section the driving mode used in the WLTP and NEDC tests.
The default value for all tests is 2x4 wheel drive.

.. glossary::
    ``n_wheel_drive.WLTP-H``
        Specify whether WLTP-H test is conducted on 2-wheel driving or 4-wheel driving. The default is 2-wheel drive.

    ``n_wheel_drive.WLTP-L``
        Specify whether the WLTP-L test is conducted on 2-wheel driving or 4-wheel driving. The default is 2-wheel drive.

    ``n_wheel_drive.NEDC-H``
        Specify whether the NEDC-H test is conducted on 2-wheel driving or 4-wheel driving. The default is 2-wheel drive.

    ``n_wheel_drive.NEDC-L``
        Specify whether NEDC-L test is conducted on 2-wheel driving or 4-wheel driving. The default is 2-wheel drive.


Vehicle technologies
--------------------
The |co2mpas| model calculates the NEDC |CO2| emission prediction considering the presence/absence
of a set of technologies in the vehicle.
For the following |co2mpas| inputs, 0 corresponds to the absence of the technology
whereas 1 is when the vehicle is equipped with the technology.
If no input is provided, the |co2mpas| model will use the default value.

.. glossary::

    turbo
    ``engine_is_turbo``
        If the air intake of the engine is equipped with any kind of forced induction system
        set like a turbocharger or supercharger, then set it to 1; otherwise set it to 0.
        The default value is 1.

    S-S
    ``has_start_stop``
        The start-stop system shuts down the engine of the vehicle during idling to reduce fuel consumption and
        it restarts it again when the footbrake/clutch is pressed.
        If the vehicle has a *S-S* system, set it to 1, otherwise, set it to 0.
        The default is 1.

    ``has_energy_recuperation``
        Set it to 1 if the vehicle is equipped with any kind of brake energy recuperation technology or
        regenerative breaking. Otherwise, set it to 0.
        The default is 1.

    torque converter
    ``has_torque_converter``
        Set it to 1 if the vehicle is equipped with this technology otherwise,
        set it to 0.
        For manual transmission vehicles the default is 0.
        For automatic tranmission vehicles, the default is 1.
        For vehicles with continuously variable transmission, the default is 0.

    ``fuel_saving_at_strategy``
    eco mode
        Setting it to 1 allows |co2mpas| to use a higher gear at constant speed driving
        than when in transient conditions, resulting in a reduction of fuel consumption.
        This technology was refered as ``eco_mode`` in previous releases of |co2mpas|.
        The default is 1.

    ``has_periodically_regenerating_systems``
        If the vehicle is equipped with periodically regenerating systems
        (anti-pollution devices such as catalytic converter or particulate trap)
        that require a periodical regeneration process in less than 4000 km of normal vehicle operation,
        set it to 1; otherwise, set it to 0.
        The default is 0.

    ``ki_factor``
    ``ki_multiplicative``
    ``ki_additive``
        For vehicles without `has_periodically_regenerating_systems`
        ``ki_multiplicative`` and ``ki_additive`` are set to 1 and 0.
        Otherwise, if not provided ``ki_multiplicative`` or ``ki_additive``,
        ``ki_multiplicative`` and ``ki_additive`` are set to 1.05 and 0. The
        ``ki_multiplicative`` or ``ki_additive`` to be used for |co2mpas| are
        the same value used for NEDC physical tests.

    VVA
    Variable Valve Actuation
    ``engine_has_variable_valve_actuation``
        This includes a range of technologies which are used to enable variable valve event timing,
        duration and/or lift. The term as set includes Valve Timing Control (VTC)—also referred to
        as Variable Valve Timing (VVT) systems and Variable Valve Lift (VVL) or
        a combination of these systems (phasing, timing and lift variation).
        Set it to 1 if the vehicle is equipped with such a system; otherwise, set it to 0.
        The default is 0.

    ``engine_has_cylinder_deactivation``
    ``active_cylinder_ratios``
        This technology allows the deactivation of one or more cylinders under specific conditions predefined
        in the |co2mpas| code. The implementation in |co2mpas| allows to use different deactivation ratios.
        So in the case of an 8-cylinder engine, a 50% deactivation (4 cylinders off) or
        a 25% deactivation ratio (2 cylinders off) are plausible. |co2mpas| selects the optimal ratio at each point
        from the plausible deactivation ratios provided by the user. The user cannot alter the deactivation strategy.
        If the vehicle is equipped with a cylinder deactivation system, set it to 1 and
        and indicate the deactivation ratios in the `active_cylinder_ratios` tab.
        Note that the `active_cylinder_ratios` always start with 1 (all cylinders are active) and then
        the user can set the corresponding ratios.

        For example, if the vehicle has an engine with 6 cylinders and it has the possibility
        to deactivate 2 or 3 or 4 cylinders, you have to introduce the following ratios:
        0.66 (4/6), 0.5 (3/6), and 0.33 (2/6).
        If the vehicle does not have cylinder deactivation set `engine_has_cylinder_deactivation` to 0.
        The default is 0.

        Note that **as of November 2016 this specific technology is in validation phase** due to
        lack of sufficient data to support its appropriate implementation in the code.
        For **Rally** release, this specific input is considered to be optional.

    lean burn
    LB
    ``has_lean_burn``
        The lean burn (LB) technology refers to the burning of fuel with an excess of air in an
        internal combustion engine. All `compression ignition` vehicles are supposed to be equipped with *LB*
        by default therefore for `compression ignition` this must be set to 0.
        For `positive ignition` engines set it to 1 if the vehicle is equipped with *LB*,
        otherwise set it to 0.
        The default is 0.

    ``has_gear_box_thermal_management``
        This specific technology option applies only to vehicles in which the temperature of the gearbox
        is regulated from the vehicle's cooling circuit using a heat-exchanger, heating storage system or
        other methods for directing engine waste-heat to the gearbox.
        Gearbox mounting and other passive systems (encapsulation) should not be considered.
        In case the vehicle is equipped with the described gear box thermal management system,
        set it to 1; otherwise, set it to 0.
        The default is 0.

        Note that **as of November 2016 this specific technology is in validation phase** due to
        lack of sufficient data to support its appropriate implementation in the code.
        For **Rally** release, this specific input is considered to be optional.


    EGR
    Exhaust gas recirculation
    ``has_exhausted_gas_recirculation``
         EGR recirculates a portion of an engine's exhaust gas back to the engine cylinders
         to reduce |NOx| emissions. The technology does not concern internal (in-cylinder) EGR.
         Set it to 1 if the vehicle is equipped with external EGR
         (high-pressure, low-pressure, or a combination of the two); otherwise, set it to 0.
         The default is 0 for `positive ignition`, and 1 for `compression ignition` engines.

    SCR
    ``has_selective_catalytic_reduction``
        On `compression ignition` vehicles, the Selective Catalytic Reduction (SCR) system uses Urea (active),
        or Ammonia (passive) to reduce |NOx|  emissions.
        Therefore this technology is only applicable for `compression ignition` engines.
        If the vehicle is equipped with SCR set `has_selective_catalytic_reduction` to 1; otherwise, set it to 0.
        The default value is 0.

        Note that **as of November 2016 this specific technology is in validation phase** due to
        lack of sufficient data to support its appropriate implementation in the code.
        For **Rally** release, this specific input is considered to be optional.


Dyno configuration
------------------
.. glossary::

    ``n_dyno_axes.WLTP-H``  
        The WLTP regulation states that WLTP tests should be performed using a dyno with 2 rotating axis.
        Therefore, the default value for this variable is 2.
        Setit to 1 in case a 1 rotating axis dyno was used during the WLTP-H test.

    ``n_dyno_axes.WLTP-L``
        The WLTP regulation states that WLTP tests should be performed using a dyno with 2 rotating axis.
        Therefore, the default value for this variable is 2.
        Set it to 1 in case a 1 rotating axis dyno was used during the WLTP-L test.


Meta
---------
.. glossary::

    ``fuel_consumption_combined``
        Combined fuel consumption for WLTP-H test [l/100 km].

    ``rcb_correction``
        Correction performed? (To be edited).
        
    ``speed_distance_correction``
        Correction performed? (To be edited).

DICE
====
.. glossary::

    co2dice
    dice
    dice command
    sampling procedure
        The |co2mpas| application, procedure or the ``co2dice`` console command(s)
        required to produce eventually the `decision flag` defining whether a
        `type-approval` procedure needs `double testing`:

        .. image:: _static/CO2MPAS-dice_overview.png

        Used also as a verb:

            "The simulation files have been **diced** as ``NOSAMPLE``."

    Git DB
    Hash DB
    Git repo
    Git repo DB
    projects DB
        The `Git` repository maintained by the `dice command` that manages `project`
        instances.

        All `hash` occurences are generated and/or retrieved against this repository.

    project
    dice project
    project id
    project archive
        The **project** corresponds one-to-one with the `vehicle_family_id`,
        and it is the entity under which all electronic artifacts of the
        `type-approval` are stored inside the local `hash DB` of each `dice`
        installation:

          | *ID* (**project**)  :=  `vehicle_family_id`

        It is created and managed by the `designated user` using `dice command`\s
        to step through successive `state`\s.
        Finally it is  **archived** and sent to the supervising `TAA`.

    state
    project state
    state transitions
        A `project` undergoes certain *state transitions* during its lifetime,
        reacting to various `dice command`\s:

        .. image:: _static/CO2MPAS-states_transitions_cmds-2.png

    dice report sheet
        A sheet in the output excel-file roughly derived from Input + Output files
        containing the non-confidential results of the simulation,
        labelled as "summary report" in the legislation:

            | **dice report sheet** := *non_confidential_data* (input-files + output-files + other-files)

        The `dice report` is derived from it.
        This sheet is called "summary report" in the `regulation`.

    output report
    output report sheet
        A sheet in the output excel-file containing they major simulation results.

    dice report
    dice request
    dice email
        The `dice report sheet` in textual form (`YAML`) stored in the `project` and
        signed with the electronic key of the `designated user`:

          |        **dice report**  :=  `dice report sheet` + *SIG* (`designated user` key)
          | *ID* (**dice report**)  :=  `HASH-1`

        It is cryptographically signed to guarantee the authenticity of the contained
        values.
        It sent through a `stamper` to prevent its repudiation, and returns
        as the `dice stamp`.

    stamp
    dice stamp
    stamp response
    stamp email
        The signed `dice report` as retuned from the `stamper`:

          | **stamp email**  :=  `dice report` + *SIG* (`stamper` key)

        .. image:: _static/CO2MPAS-stamp_elements.png
           :height: 120px

        The `decision flag` gets derived from its signature while the `project`
        parses it and generates the `decision report`.

    decision
    decision flag
    decision percent
    double testing
        A structure containing the ``'OK'``/``'SAMPLE'`` flag and the *percent*
        derived from the `dice stamp`'s signature (a random number), persisted in the
        `decision report` and in the `project` as a plain file.

        The meaning of the flag's values is the following:

        - ``'OK'`` means that the declared `NEDC` value is accepted
          (assuming |co2mpas| prediction does not deviate more than 4% of the
          declared *NEDC* value).
        - ``'SAMPLE'`` means that independently of the result of |co2mpas| prediction
          the vehicle has to undergo an *NEDC* physical test, "double testing";
          see *decision percent* below for which H/L vehicle to test under *NEDC*.

        The meaning of the *decision percent* is explained in the following table:

        .. image:: _static/dice_co2mpas_dev.PNG

    decision report
        Since |co2mpas| v1.7.x, this new textual report (`YAML`) is the final outcome
        of the `sampling procedure` containing the signed and timestamped data
        from all intermediate reports;

          |        **decision report**  :=  `dice stamp` + `decision` + *SIG* (`designated user` key)
          | *ID* (**decision report**)  :=  `HASH-2`

        It generated and stored internally in the `project`, and signed by the
        `designated user` to prevent tampering and repudiation.
        The final `HASH-2` contained in it may be communicated to the supervising
        `TAA` earlier that the `project archive`.

    HASH-1
        The cryptographic `hash` contained in the `dice report` which identifies
        unequivocally the `type-approval` procedure prior to stamping.

        It is generated by the `project` while parsing the `dice report sheet`.

    HASH-2
        The cryptographic `hash` contained in the `decision report` which
        unequivocally identifies a completed `sampling procedure`.

        It is generated by the `project` while importing the `dice stamp`.
        It may be sent to the `TAA` prior to sending them the `project archive`.

    TAA Report
        A "printed" PDF file that the `TS` have to send to the `TAA` to generate
        the Certificate which is unequivocally associated with all files & reports
        above:

          | **TAA Report**  :=  `output report sheet` + `decision` + `HASH-2`

    stamper
    timestamper
    timestamp service
        Either the `mail stamper` or the `web stamper` services that append
        a cryptographic signature on an "incoming" `dice report`, and sends it
        with an email to recipients to prevent repudiation at a later time.

    mail stamper
        A `stamper` mail-server that stamps and forwards all incoming e-mails to
        specified recipients.

        The trust on its certifications stems from the list of signatures published
        daily in its site.

    web stamper
    WebStamper
        JRC's user-friendly `stamper` web-application that uses a simple HTTP-form to
        timestamp a pasted `dice report` and return a `dice stamp`, emailing it also
        to any specified recipients, always including from CLIMA/JRC.


.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. |NOx| replace:: NO\ :sub:`x`\
.. |mr| replace:: m\ :sub:`r`\

.. default-role:: obj
