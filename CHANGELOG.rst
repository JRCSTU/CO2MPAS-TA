Changelog
=========

(unreleased)
------------

Known Limitations
-----------------
1) Certain programs (for example Skype) could be pre-empting (or reserving)
some tcp/ip ports and therefore could conflict with co2mpas graphical interface
that tries to launch a webserver on a port in the higher range (> 10000)

2) Certain antivirus (for example Avast) could include python in the list of
malicious software; however, this is not to be considered harmful. If this
happens the antivirs should be disabled when running co2mpas, or a special
exclusion should be granted to the co2mpas executable.

Feat
~~~~
- (doc) :gh:`533`: Rephrased one sentence.
- (write): Add extended information of packages to output file.
Fix
~~~
- (doc) :gh:`540`: Added question on drive_battery_technology.
- (selector): Ensure `enable_selector`.
- (thermal): Add engine is hot in thermal.
- (manual): Correct driveability rule.
- (gear_box): Correct gear identification in manual transmission.
- (gear_box): Correct gear identification.
- (write): Ensure JSON format.

v4.1.1 (2019-10-11)
-------------------
Feat
~~~~
- (doc) :gh:`533`: Added 2 terms in the glossary.
- (doc) :gh:`533`: Improved model.rst .
- (doc) :gh:`533`: Updated simulation plan in 'names'.
- (core): Add co2mpas as readable format.
- (cli): Add log config variable.
- (doc) :gh:`533`: Added icons explanation in installation.
- (dice): Update dice version 4.0.0.
- (doc) :gh:`533`: Add more comments to cli messages.
- (doc) :gh:`533`: Refactor and update whole doc.
- (doc): Rearrange documentation content.
Fix
~~~
- (cvt): Correct model selection for CVT.
- (doc) :gh:`533`: Correct indentation.
- (engine): Correct `alternator_powers` sign.
- (model): Correct error in default_value.
- (physical): Remove division warning.
- (final_drive): Correct function args.
- (doc) :gh:`533`: Add video download_demo.
- (final_drive): Correct `final_drive_ratios` calculation.
- (tutorial) :gh:`533`: Update output results and add model plot.
- (doc) :gh:`533`: Update sync part in tutorial.
- (doc) :gh:`533`: Add image gui_start_up.
- (co2) :gh:`539`: Correct p_overrun percentage.
- (tutorial) :gh:`533`: Update run ta section of tutorial and add links.
- (doc) :gh:`533`: Format glossary.
- (doc) :gh:`533`: Correct doc version.
- (tutorial) :gh:`533`: Update run section of tutorial.
- (sync): Create folder to save output.
- (setup): Set `pandalone` and `wltp` versions.
- (convert): Make `_get_installed_packages` stable.
- (load): Correct inputs.
- (doc) :gh:`533`: Add video to tutorial.
- (doc) :gh:`533`: Update videos & images.
- (control) :gh:`550`: Set a default `_start_stop_model` when it cannot
  be calibrated.
- (load): Remove double waring of input file version.
- (gear_box) :gh:`551`: Correct index error.
- (fc) :gh:`552`: Remove warning.
- (co2) :gh:`539`: Correct inputs of
  `calculate_corrected_co2_emission_for_conventional_nedc`.
- (template): Add missing link.
- (doc) :gh:`533`: Restructure format.
- (co2) :gh:`539`: Change default value of `speed_distance_correction`.
- (co2) :gh:`539`: Normalise formula for default Kco2 NEDC correction.
- (write): Remove pip warning.
- (doc) :gh:`533`: Update tutorial.
- (doc) :gh:`533`: Update glossary.
- (doc,faq) :gh:`533`: Update faq format.
- (hybrid): Model planetary as parallel.
Other
~~~~~
- , :gh:`533`: Update model description and move images.
v4.1.0 (2019-10-06)
-------------------
Feat
~~~~
- (template, demos) :gh:`544`: Update input template and demos.
- (co2): Add `calculate_fuel_heating_value` function.
- (dice): Update dice plugin version.
- (core): Add model configuration file log msg.
- (co2) :gh:`539`: Add RCB correction for hybrid in NEDC.
- (load): Read dice data from `.co2mpas.ta` file.
- (co2) :gh:`539`: Add formulas to correct the co2 emission according to
  the regulation.
- (co2): Add module to calculate co2 emission.
- (doc,faq) :gh:`533`: Add FAQ file.
- (motors): Add functions to calculate `motor_px_maximum_torque`.
- (validate) :gh:`542`: Add variable `is_hybrid` to `dice`.
- (thermal) :gh:`538`: Revert changes.
- (battery) :gh:`540`: Correct bug when `drive_battery_technology` is
  unknown.
- (template) :gh:`516`: Add hybrid inputs to template.
- (battery) :gh:`540`: Add functions to calculate
  `drive_battery_n_parallel_cells`.
- (schema) :gh:`540`: Add field `drive_battery_technology_type`.
- (driver) :gh:`509`: Add plugin configuration functions.
- (planetary, defaults) :gh:`536`: Add function to define if the vehicle
  `is_serial`.
- (planetary, defaults) :gh:`536`: Add defaults for planetary.
- (planetary) :gh:`536`: Add planetary model.
- (gui) :gh:`508`: Add plugin configuration for gui CLI.
- (hybrid) :gh:`516`: Split `p4_motor` in `p4_motor_front` and
  `p4_motor_rear`.
- (cli) :gh:`509`: Add `CO2MPAS_HOME` env.
- (driver) :gh:`509`: Add plugin configuration functions.
- (wheels) :gh:`507`: Add PAX tyre code.
- (hybrid) :gh:`516`: Include starter time calibration into
  `start_stop_hybrid_params`.
- (dcdc) :gh:`516`: Add calculation of `dcdc_converter_electric_powers`
  from `dcdc_converter_electric_powers_demand`.
- (catalyst) :gh:`516`: Add `catalyst_power_model`.
- (control) :gh:`516`: Add `is_serial` parameter.
- (utils) :gh:`516`: Add `index_phases` function in utils.
- (cold_start) :gh:`516`: Simplify cold start model, improve thermal,
  and remove `clutch_tc_speeds`.
- (driver) :gh:`509`: Remove unused module.
- (gear_box) :gh:`516`: Add calculation of the
  `gear_box_mean_efficiency_guess`.
- (ems) :gh:`516`: Search for serial optimal when battery current is >=
  0.
- (report): Add `delta_state_of_charge` for service and drive batteries.
- (ems) :gh:`516`: Add function to calculate `hybrid_modes` from
  `on_engine`.
- (dcdc) :gh:`516`: Add function to calculate `dcdc_currents`.
- (motors) :gh:`516`: Split p3 in front and rear.
- (engine) :gh:`498`: Filter unfeasible `engine_temperature_derivatives`
  in calibration.
- (setup) :gh:`523`: Add env `ENABLE_SETUP_LONG_DESCRIPTION`.
- (ems) :gh:`516`: Simplify identification of `catalyst_warm_up`.
- (engine) :gh:`516`: Add function to identify `engine_speeds_out_hot`
  for hybrids.
- (gear_box) :gh:`516`: Improve gear identification from engine speed.
- (dcdc) :gh:`516`: Add default current when vehicle is not hybrid.
- (alternator) :gh:`516`: Add default current when vehicle is hybrid.
- (starter) :gh:`516`: Add `delta_time_engine_starter` to
  `StarterModel`.
- (control) :gh:`516`: Add functions to identify motors power split.
- (thermal): Improve thermal model.
- (gear_box): Vectorize gear identification.
- (selector) :gh:`516`: Update selectors.
- (clutch_tc) :gh:`516`: Add data `clutch_tc_speeds`.
- (engine) :gh:`516`: Make thermal model function of
  `gross_engine_powers_out`.
- (electrics) :gh:`516`: Add variables `has_motor_px`.
- (starter) :gh:`516`: Add `StarterModel`.
- (plot): No truncation in rendering numpy arrays.
- (selector) :gh:`516`: Update for hybrids.
- (electrics) :gh:`516`: Add prediction functions for electrics and EMS.
- (electric, control) :gh:`516`: Include service battery in controller
  logic.
- (electric) :gh:`516`: Add DC/DC converter current model.
- (electric) :gh:`516`: Move alternator status model as service battery
  status model.
- (control) :gh:`516`: Add energy management strategy model.
- (gear_box) :gh:`516`: Add `gear_box_mean_efficiency`
- (engine) :gh:`516`: Add function `define_fuel_map` to create a rater
  `fuel_map`.
- (motors) :gh:`516`: Add functions to calculate
  `motor_pi_maximum_power`, `motor_pi_rated_speed`,
  `motor_pi_maximum_torque`, etc.
- (clutch_tc) :gh:`516`: Add `clutch_tc_mean_efficiency`.
- (final_drive) :gh:`516`: Add `final_drive_mean_efficiency`.
- (battery) :gh:`516`: Add `BatteryModel` class.
- (alternator) :gh:`516`: Use `clutch_tc_powers` instead
  `gear_box_powers_in`.
- (control) :gh:`516`: Add new control model.
- (motors) :gh:`516`: Add calculation of `engine_speeds_out`,
  `wheel_speeds`, `final_drive_speeds_in`, `gear_box_speeds_in` from
  motors speeds.
- (dcdc) :gh:`516`: Add calculation of `dcdc_converter_electric_powers`
  from currents.
- (physical) :gh:`516`: Add motors mechanical power to drive line.
- (electrics) :gh:`516`: Update inputs/outputs to physical model.
- (batteries) :gh:`516`: Add dcdc model.
- (electrics) :gh:`516`: Map batteries and motors model.
- (motors) :gh:`516`: Modify motors models outputs.
- (battery:drive) :gh:`516`: Add calculation of
  `motors_electric_powers`.
- (motors:alternator) :gh:`516`: Restructure alternator model.
- (motors) :gh:`516`: Add alternator model.
- (motors:starter) :gh:`516`: Add starter model.
- (alternator) :gh:`516`: Move
  `identify_service_battery_state_of_charge_balance_and_window` to
  alternator model.
- (motors) :gh:`516`: Add `calculate_motors_electric_powers` func.
- (motors) :gh:`516`: Add p1 model.
- (motors) :gh:`516`: Add p2 model.
- (motors) :gh:`516`: Add p4 model.
- (battery:drive) :gh:`516`: Add drive battery model.
- (battery:service) :gh:`516`: Reorganize the service battery model.
- (motors) :gh:`516`: Add p0 model.
- (motors) :gh:`516`: Add p3 model.
- (motors) :gh:`516`: Add p4 model.
- (clutch_tc) :gh:`515`: Simplify clutch model, implement VDI253 model
  for torque converter, and add flag to disable speed prediction.
- (driver) :gh:`509`: Add maximum velocity limitation.
- (driver) :gh:`509`: Add auxiliaries losses into logic.
- (driver) :gh:`509`: Add `clutch_tc_prediction_model`.
- (driver) :gh:`509`: Add clutch and alternator correction for driver
  max acceleration.
- (driver) :gh:`509`: Add `desired_velocities` to output.
- (exe) :gh:`513`: Script to build the executable.
- (vehicle) :gh:`509`: Add calculation for the
  `traction_acceleration_limits`.
- (cycle) :gh:`509`: Add `CycleModel` with driver logic.
- (vehicle, cycle) :gh:`509`: Add `VehicleModel` and `CycleModel`.
- (electrics) :gh:`509`: Update for unlimited steps `ElectricModel`.
- (engine) :gh:`509`: Update for unlimited steps `EngineModel`.
- (git): Add ignore for `DICE_KEYS` folder.
- (gear_box) :gh:`509`: Update for unlimited steps `GearBoxModel`.
- (final_drive) :gh:`509`: Update for unlimited steps `FinalDriveModel`.
- (wheel) :gh:`509`: Update for unlimited steps `WheelsModel`.
- (cli): Add test case for `syncing` cmd.
- (docker): Add Dockerfile to build windows exe.
- (cli): Add test case for `run` cmd.
- (plot): Add simulation id to solution name.
- (cli): Add `--template-type` option to `template` cmd.
- (cli): Add test cases for `template`, `demo`, `conf`.
Fix
~~~
- (hybrid): Remove warning.
- (co2): Correct calculation of corrected_co2_emission_value for nedc
  hybrid.
- (battery): Correct calculation flow of `drive_battery_voltages`.
- (hybrid): Add `default_start_stop_activation_time` function.
- (selector) :gh:`541`: Add `initial_drive_battery_state_of_charge` as
  model data.
- (fc) :gh:`517`: Correct rule safe numpy error.
- (co2) :gh:`539`: Correct indices of phases.
- (wltp): Correct calculation process of theoretical velocity.
- (selector): Add missing model parameter `kco2_wltp_correction_factor`.
- (utils): Remove deprecation warning for yaml.
- (fc) :gh:`517`: Add `cylinder_deactivation_valid_phases` for fc
  calculation.
- (fc) :gh:`517`: Correct format.
- (core) :gh:`546`: Correct import order for setting the defaults
  variable.
- (acr) :gh:`517`: Add `engine_inertia_powers_losses` for applying acr.
- (hybrid) :gh:`541`: Correct error all nan.
- (core) :gh:`546`: Correct import order for setting the defaults
  variable.
- (hybrid): Correct identification of warm up phases.
- (write): collect installed packs with pip & conda cmds, only if
  present...
- (hybrid) :gh:`541`: Correct hybrid serial/planetary power flow.
- (vehicle): Correct calculation of the distance.
- (write): Correct model output format.
- (model): Add missing prediction data.
- (write): Replace `pip` with `conda` to freeze pkgs names.
- (doc) :gh:`533`: Correct documentation.
- (doc) :gh:`533`: Remove un-valid references.
- (doc) :gh:`533`: Remove unused parameters.
- (load): Add flag validation for declaration mode.
- (doc,faq) :gh:`533`: Update faq.
- (doc,faq) :gh:`533`: Text enhancement.
- (doc,faq) :gh:`533`: Delete unneeded line.
- (doc) :gh:`533`: Update documentation skeleton.
- (core): Correct `output_template` option.
- (demos): Update demos for conventional vehicles.
- (template): Correct `service_battery_nominal_voltage` inputs.
- (load) :gh:`542`: Correct `service_battery` inputs.
- (load) :gh:`542`: Activate `enable_selector` flag.
- (output) :gh:`534`: Add dice data to output file.
- (output) :gh:`534`: Fix report layout.
- (output) :gh:`534`: Correct flags output.
- (demo) :gh:`538`: Correct declared co2 emission in demo file.
- (planetary) :gh:`536`: Correct Calculation of serial and electric
  powers.
- (batteries) :gh:`516`: Add limitation of charging currents.
- (planetary) :gh:`536`: Correct sign of maximum power of planetary
  motor P2.
- (planetary) :gh:`536`: Correct bug for NEDC speed profile.
- (selector): Correct error when `after_treatment_warm_up_phases` is
  missing.
- (driver) :gh:`509`: Revert all changes for driver model.
- (utils): Set dtype default value to `float`.
- (setup) :gh:`526`: Fix xgboost version to avoid `WARNING: reg:linear
  is now deprecated`.
- (after_treat): Ensure not nan.
- (hybrid) :gh:`516`: Change calibration limit.
- (git): Ignore venv.
- (hybrid,starter) :gh:`516`: Correct minor bugs.
- (conventional) :gh:`516`: Correct definition of `hybrid_modes`.
- (catalyst, hybrid) :gh:`516`: Correct identification of catalyst warm
  up.
- (hybrid) :gh:`516`: Remove unused variable.
- (control, catalyst) :gh:`516`: Unify catalyst parameters and
  calculation.
- (control) :gh:`516`: Correct reference.
- (control) :gh:`516`: Correct catalyst model name.
- PEP8.
- (electrics) :gh:`516`: Remove unused link.
- (defaults): Remove unused function defaults.
- (gear_box) :gh:`516`: Correct identification when there is only one
  gear.
- (cmv): Correct bug when only one gear.
- (electrics) :gh:`516`: Correct missing links and minor bugs.
- (ems) :gh:`516`: Correct broadcast error.
- (motors) :gh:`516`: Correct links.
- (thermal) :gh:`458`, :gh:`498`, :gh:`516`: Filter temperature for
  calculating derivatives + improve stability.
- (setup) :gh:`514`: Remove `nose` from `setup_requires`.
- (build): Improve cleaning.
- (requirements): Correct `beautifulsoup4` requirement.
- (report) :gh:`516`: Change chart `service_battery_powers`-->
  `service_battery_electric_powers`.
- (template) :gh:`516`: Add missing model scores in output file.
- (electrics) :gh:`516`: Correct service battery load vector [kW]..
- (electrics) :gh:`516`: Correct calculation order.
- (test): Correct test case for conf file.
- (load) :gh:`529`: Correct file loader.
- (engine): Improve identification of `on_idle`.
- (ems) :gh:`516`: Correct function to identify the `catalyst_warm_up`.
- (gear_box) :gh:`516`: Improve gear identification.
- (engine) :gh:`530`: Correct mean absolute error with weights.
- (batteries) :gh:`516`: Correct calculation of
  `drive_battery_voltages`.
- (batteries) :gh:`516`: Correct calculation of DC/DC current in
  `DriveBatteryModel`.
- (thermal): Remove warning.
- (ems) :gh:`516`: Correct calculation order of `engine_speeds_out_hot`.
- (ems) :gh:`516`: Avoid mode fluctuation in prediction.
- (ems) :gh:`516`: Compare parallel or serial excluding starter
  penalties.
- (ems) :gh:`516`: Improve hybrid modes identification.
- (ems) :gh:`516`: Use starter time to compute the penalties.
- (physical): Use customized `_XGBRegressor`.
- (ems) :gh:`516`: Use engine speeds out to compute the hypothetical
  engine speed in parallel mode.
- (ems) :gh:`516`: Remove warnings.
- (engine) :gh:`516`: Remove default value for `is_hybrid`.
- (electrics) :gh:`516`: Add missing links.
- (excel): Correct data parser when id starts with a space.
- (clutch_tc) :gh:`516`: Split calculation of `clutch_tc_powers`.
- (ems) :gh:`516`: Ensure AMPGO reproducibility.
- (co2mpas): Remove prediction loop.
- (ems): Improve speed performances of `StartStopHybrid.fit`.
- (ems): Add missing doc.
- (gear_box): Correct gear identification.
- (electrics) :gh:`516`: Update power calculation wit efficiency.
- (batteries) :gh:`516`: Correct missing inputs.
- (selector) :gh:`516`: Update selector for electrics and start/stop.
- (electrics) :gh:`516`: Simplify losses.
- (control) :gh:`516`: Add domains + correct `predict_hybrid_modes`.
- (battery) :gh:`516`: Correct ServiceBatteryModel for dcdc prediction.
- (batteries) :gh:`516`: Correct identification of
  `service_battery_capacity` and soc limits.
- (electric) :gh:`516`: Simplify status model of service battery.
- (electric) :gh:`516`: Simplify status model of service battery.
- (co2_emission) :gh:`516`: Correct definition of fuel map.
- (doc) :gh:`516`: Correct documentation.
- (engine) :gh:`516`: Update graph links.
- (load) :gh:`516`: Update schema for missing data model.
- (wheels): Extend `calculate_wheel_torques` function to `list`.
- (fina) :gh:`516`: Use.
- (core): Correct asteval formulas.
- (sync): Correct reference.
- (final_drive) :gh:`516`: Simplify and correct final drive model
  efficiency.
- (gear_box) :gh:`516`: Correct bug to identify gears.
- (physical) :gh:`516`: Use `gear_box_speeds_in` to identify the
  `r_dynamic`.
- (batteries) :gh:`516`: Add missing data connection.
- (batteries) :gh:`516`: Correct starter bugs.
- (batteries) :gh:`516`: Correct sign convention.
- (gear_box) :gh:`516`: Use `gear_box_speeds_in` to calibrate the gear
  box.
- (batteries:service) :gh:`516`: Add starter power to service battery.
- (electrics) :gh:`516`: Correct models inputs/outputs.
- (battery:drive) :gh:`516`: Correct calculation of
  `drive_battery_currents`.
- (battery:drive) :gh:`516`: Correct typo input name.
- (motors) :gh:`516`: Correct dsp after rebase.
- (motors:p4) :gh:`516`: Correct `motor_p4_speed_ratio` default value.
- (motors) :gh:`516`: Correct P3 input.
- (motors:p4) :gh:`516`: Correct format documentation.
- (driver) :gh:`509`: Remove unneeded equation.
- (gear_box) :gh:`509`: Correct gear box logic.
- (co2) :gh:`509`: Remove division warning.
- (co2mpas): Correct bug in `_yield_files` function.
- (driver) :gh:`509`: Enable `driver_style_ratio` and
  `acceleration_damping`.
- (driver) :gh:`509`: Correct calculation of engine inertia power to
  driver model.
- (driver) :gh:`509`: Add engine inertia power to driver model.
- (at_gear): Correct bug when no gears.
- (manual): Correct typo bug.
- (clutch_tc) :gh:`515`: Remove unused function.
- (torque_converter) :gh:`515`: Correct typo.
- (torque_converter) :gh:`515`: Add parameters for the m1000 curve.
- (clutch) :gh:`509`: Correct `clutch_acceleration_window` default
  value.
- (torque_converter) :gh:`515`: Add missing default.
- (engine): Correct typo `weigth` --> `weight`.
- (torque_converter) :gh:`515`: Introduce the m1000 curve.
- (vehicle) :gh:`509`: Split `traction_acceleration_limits` into
  `traction_deceleration_limit` and `traction_acceleration_limit`.
- (gear_box) :gh:`509`: Split the identification of first and last
  gear_box_ratios.
- (torque_converter) :gh:`509`: Correct bug in `next` method.
- (driver) :gh:`509`: Correct WLTP cycle velocity prediction.
- (at_gear) :gh:`509`: Revert correction of `correct_gear_full_load`
  method.
- (at_gear): Avoid invalid calibration of `GSMColdHot` model.
- (core): Correct `_run_variations` function.
- (at_gear) :gh:`509`: Correct `correct_gear_full_load` method.
- (at_gear) :gh:`509`: Correct `_upgrade_gsm` function.
- (schema) :gh:`509`: Correct limits of `wheel_drive_load_fraction`.
- (driver) :gh:`509`: Correct maximum distance.
- (co2_emission) :gh:`509`: Set zero when nan in
  `calculate_phases_co2_emissions`.
- (physical) :gh:`509`: Add wildcard to `path_velocities`,
  `path_distances`, and `path_elevations`.
- (physical) :gh:`509`: Add wildcard to `path_velocities`,
  `path_distances`, and `path_elevations`.
- (template) :gh:`503`: Correct documentation for dice parameters.
- (start_stop) :gh:`512`: Consider `start_stop_activation_time` in the
  S/S calibration.
- (electrics) :gh:`509`: Postpone use of `times` vector in
  `ElectricModel` formulas.
- (final_drive) :gh:`509`: Correct `FinalDriveModel` formulas.
- (vehicle) :gh:`509`: Correct `VehicleModel` formulas.
- (gear_box, engine, electrics) :gh:`509`: Correct bugs on prediction
  models.
- (gear_box) :gh:`509`: Correct delta time.
- (cli): Correct opening of web interface in windows.
- (write): Correct variable name of ta writing function.
- (load): Correct schema for models.
- (plan): Strip id plan.
- (cli): Add `--encryption-keys-passwords` option to read TA files.
- (cli): Add test file for `conf` cmd.
- (physical) :gh:`506`: Use basic types in default to dump and load
  easily.
- (load) :gh:`506`: Correct message when folder path do not exist.
- (plan) :gh:`506`: Correct inputs extraction when dice is not
  installed.
- (co2mpas) :gh:`506`: Avoid to save empty summary.
- (co2mpas) :gh:`506`: Error in mkdir and demos folder.
- (cli): Correct x- and y- label default.
- (co2mpas) :gh:`506`: Add initialization of pandalone filters.
- (co2mpas) :gh:`506`: Error in mkdir and demos folder.
- (doc) :gh:`506`: Broken link.
- (write) :gh:`506`: `makedirs` if output folder does not exist.
- (co2mpas) :gh:`506`: Correct behaviour of simulation plan.
- (co2mpas) :gh:`506`: Correct behaviour of input_domains.
- Update copyright.
- (sim:physical): Avoid domain warnings.
- (sim:demos): Add dice parameter incomplete.
- (sim:input): Add dice parameter incomplete.

``v3.0.0``, 29-Jan-2019: "VOLO" Release
---------------------------------------

|co2mpas| 3.0.X becomes official on February 1st, 2019.

- There will be an overlapping period with the previous official |co2mpas| version
  **2.0.0** of 2 weeks (until February 15th).

- This release incorporates the amendments of the Regulation (EU) 2017/1153,
  `2018/2043 <https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32018R2043&from=EN)>`_
  of 18 December 2018 to the Type Approval procedure along with few fixes on the
  software.

- The engineering-model is 100% the same with the
  `2.1.0, 30-Nov-2018: "DADO" Release <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2sim-v2.1.0>`_
  and the version-bump (2.X.X --> 3.X.X) is just a facilitation for the users,
  to recognize which release is suitable for the amended Correlation Regulations.

- The Type Approval mode (_TA_) of this release is **incompatible** with all
  previous Input File versions. The _Batch_ mode, for engineering purposes,
  remains compatible.

- the _TA_ mode of this release generates a single "_.zip_" output that contains
  all files used and generated by |co2mpas|.

- This release is comprised of 4 python packages:
  `co2sim <https://pypi.org/project/co2sim/3.0.0/>`_, `co2dice <https://pypi.org/project/co2dice/3.0.0/>`_,
  `co2gui <https://pypi.org/project/co2gui/3.0.0/>`_ and `co2mpas <https://pypi.org/project/co2mpas/3.0.0/>`_.

Installation
~~~~~~~~~~~~
This release will not be distributed as an **AllInOne** (AIO) package. It is
based on the `2.0.0, 31-Aug-2018: "Unleash" Release <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2mpas-r2.0.0>`_,
launched on 1 September 2018. There are two options of installation:

  1. Install it in your current working `AIO-v2.0.0 <https://github.com/JRCSTU/co2mpas/releases/tag/co2mpas-r2.0.0>`_.
  2. **Preferably** in a clean `AIO-v2.0.0 <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2mpas-r2.0.0>`_,
     to have the possibility to use the old |co2mpas|-v2.0.0 + DICE2 for the
     two-week overlapping period;

- To install:
   ```console
   pip uninstall co2sim co2dice co2gui co2mpas -y
   pip install co2mpas
   ```

.. note::
   If you want to install this specific version at a later date, after more
   releases have happened, use this command:
   ```console
   pip install co2mpas==3.0.0
   ```

Important Changes since `2.1.0` release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model
~~~~~
No model changes.

IO Data
~~~~~~~
- Input-file version from 3.0.0 --> **3.0.1**.
  - It hosts few modifications after interactions with users.
  - The input file contained in this release cannot run in older co2mpas
    releases in the _TA_ mode.

DICE
~~~~
- The old DICE2 is deprecated, and must not be used after 15th of February,
- it is replaced by the centralized DICE3 server. There will be a new procedure
  to configure the keys to _sign_ and _encrypt_ the data.

Demo Files
~~~~~~~~~~
- The input-file changed, and we have prepared new demo files to help the users
  adjust. Since we do not distribute an **AllInOne** package, you may download the new files:
   - from the console:
   ```console
   co2mpas demo --download
   ```

   - From this `link <https://github.com/JRCSTU/allinone/tree/master/Archive/Apps/.co2mpas-demos>`_


``v2.0.0``, 31 Aug 2018: "Unleash"
----------------------------------
Changes since 1.7.4.post0:

BREAKING:
~~~~~~~~~
1. The ``pip`` utility contained in the old AIO is outdated (9.0.1) and
   cannot correctly install the transitive dependencies of new ``co2mpas``, even
   for development purposes.  Please upgrade your ``pip`` before following the
   installation or upgrade instructions for developers (e.g. in :term:`AIO`
   use ``../Apps/WinPython/scripts/upgrade_pip.bat``).

2. The ``vehicle_family_id`` format has changed (but old format is still
   supported)::

       OLD: FT-TA-WMI-yyyy-nnnn
       NEW: FT-nnnnnnnnnnnnnnn-WMI-x

3. The co2mpas python package has been splitted (see :gh:`408`), and is now
   served by 4 python packages listed below.  In practice this means that you
   can still receive bug-fixes and new features for the DICE or the GUI, while
   keeping the simulation-model intact.

   1. ``co2sim``: the simulator, for standalone/engineering work. Now all
      IO-libraries and graph-drawing are optional, specified the ``io`` &
      ``plot`` "extras". If you need just the simulator to experiment, you need
      this command to install/upgrade it with::

          pip install co2sim[io,plot] -U

   2. ``co2dice``: the backend & commands for :abbr:`DICE (Distributed Impromptu
      Co2mpas Evaluation)`.

   3. ``co2gui``: the GUI.

   4. ``co2mpas``: installs all of the above, and ``[io,plot]`` extras.


   The relationships between the sub-projects are depicted below::

       co2sim[io,plot]
         |    |
         |  co2dice
         |  /  \
        co2gui  WebStamper
          |
       co2mpas

   .. Note::
     ``co2sim`` on startup checks if the old ``co2mpas-v1.x`` is still
     installed, and aborts In that case, uninstall all projects and re-install
     them, to be on the safe side, with this commands::

         pip uninstall -y co2sim co2dice co2gui co2mpas
         pip install co2sim co2dice co2gui co2mpas -U


Model:
~~~~~~

- feat(co2_emissions): Add ``engine_n_cylinders`` as input value and a TA
  parameter.

- feat(ta): New TA output file.

  Running CO2MPAS in TA mode, will produce an extra file containing the DICE
  report. This file will be used in the feature version of DICE.

- feat(core): Improve calibration performances 60%.

- feat(manual): Add a manual prediction model according GTR.

- feat(gearbox): Add utility to design gearbox ratios if they cannot be
  identified based on ``maximum_velocity`` and ``maximum_vehicle_laden_mass``.

  This is not affecting the TA mode.

- fix(co2mpas_template.xlsx): The parameter "Vehicle Family ID" changes to
  "Interpolation Family ID".

- fix(co2mpas_template.xlsx): Meta data.

  Add additional sheets for meta data.
  As for September 2018,
  the user can voluntarily add data related to the all WLTP tests held for
  a specific Interpolation Family ID.
  Since this addition is optional, the cells are colored orange.

- fix(vehicle): Default ``n_dyno_axes`` as function of
  ``n_wheel_drive`` for wltp (4wd-->2d, 2wd-->1d).

  If nothing is specified, default values now are:
  ``n_dyno_axes = 1``
  ``n_wheel_drive = 2``

  If only ``n_wheel_drive`` is selected, then the default for
  ``n_dyno_axes`` is calculated as function of ``n_wheel_drive`` for wltp
  (4wd-->2d, 2wd-->1d)

  If only n_dyno_axes is selected, then the default for
  ``n_wheel_drive`` is always 2.

- fix(vva): Remove ``_check_vva``.

  ``engine_has_variable_valve_actuation = True`` and
  ``ignition_type = 'compression'`` is permitted.

- fix(ki_factor): Rename ``ki_factor`` to ``ki_multiplicative`` and add
  ``ki_additive value``.

- fix(start_stop): Disable ``start_stop_activation_time`` when
  ``has_start_stop == True``.

- fix(co2_emission): Disable ``define_idle_fuel_consumption_model`` when
  `idle_fuel_consumption` is not given.

- fix(ta): Disable function `define_idle_fuel_consumption_model`
  and `default_start_stop_activation_time`.

- fix(electrics): Improve calculation of state of charges.

- fix(at): Correct ``correct_gear_full_load`` method using the best gear
  instead the minimum when there is not sufficient power.


IO Data
~~~~~~~
- BREAK: Bumped input-file version from ``2.2.8 --> 2.3.0``.  And improved
  file-version comparison (:term:`Semantic Versioning`)

- CHANGE: Changed :term:`vehicle_family_id` format, but old format is still
  supported (:gh:`473`)::

        OLD: FT-TA-WMI-yyyy-nnnn
        NEW: FT-nnnnnnnnnnnnnnn-WMI-x

- feat: Input-template provide separate H/L fields for both *ki multiplicative*
  and *Ki additive* parameters.

- drop: remove deprecated  ``co2mpas gui`` sub-command - ``co2gui`` top-level
  command is the norm since January 2017.


Dice
~~~~
- FEAT: Added a new **"Stamp" button** on the GUI, stamping with *WebStamper*
  in the background in one step; internally it invokes the new ``dicer`` command
  (see below)(:gh:`378`).

- FEAT: Added the simplified top-level sub-command ``co2dice dicer`` which
  executes *a sequencer of commands* to dice new **or existing** project
  through *WebStamper*, in a single step.::

      co2dice dicer -i co2mpas_demo-1.xlsx -o O/20180812_213917-co2mpas_demo-1.xlsx

  Specifically when the project exists, e.g. when clicking again the *GUI-button,
  it compares the given files *bit-by-bit* with the ones present already in the
  project, and proceeds *only when there are no differences.

  Otherwise (or on network error), falling back to cli commands is needed,
  similar to what is done with abnormal cases such as ``--recertify``,
  over-writing files, etc.

- All dice-commands and *WebStamper* now also work with files, since *Dices*
  can potentially be MBs in size; **Copy + Paste** becomes problematic in these
  cases.

- Added low-level ``co2dice tstamp wstamp`` cli-command that Stamps a
  pre-generated :term:`Dice` through *WebStamper*.


- FEAT: The commands ``co2dice dicer|init|append|report|recv|parse`` and
  ``co2dice tstamp wstamp``, support one or more ``--write-file <path>/-W``
  options, to and every time they run,  they can *append* or *overwrite* into
  all given ``<path>`` these 3 items as they are generated/received:

    1. :term:`Dice report`;
    2. :term:`Stamp`  (or any errors received from :term:`WebStamper`;
    3. :term:`Decision`.

  By default, one ``<path>`` is always ``~/.co2dice/reports.txt``, so this
  becomes the de-facto "keeper" of all reports exchanged (to mitigate a *known
  limitation* about not being able to retrieve old *stamps*).
  The location of the *reports.txt* file is configurable with

    - ``c.ReportsKeeper.default_reports_fpath`` configuration property, and/or
    - :envvar:`CO2DICE_REPORTS_FPATH` (the env-var takes precedence).

- feat: command ``co2dice project report <report-index>`` can retrieve older
  reports (not just the latest one).  Negative indexes count from the end, and
  need a trick to use them::

       co2dice project report -- -2

  There is still no higher-level command to retrieveing *Stamps*
  (an old *known limitation*); internal git commands can do this.

- drop: deprecate all email-stamper commands; few new enhancements were applied
  on them.

- feat(:gh:`466`, :gh:`467`, io, dice):
  Add ``--with-inputs`` on ``co2dice project init|append|report|dicer`` commands
  that override flag in user-data `.xlsx` file, and attached all inputs
  encrypted in dice.

- feat: add 2 sub-commands in `report` standalone command::

      co2dice report extract  # that's the old `co2dice report`
      co2dice report unlock   # unlocks encrypted inputs in dice/stamps

- feat(dice): all dice commands accept ``--quiet/-q`` option that
  along with ``--verbose/-v`` they control the eventual logging-level.

  It is actually possible to give multiple `-q` / `-v` in the command line,
  and the verbose level is an algebraic additions of all of them, starting
  from *INFO* level.

  BUT if any -v is given, the `Spec.verbosed` trait-parameter is set to true.
  (see :gh:`476`, :gh:`479`).

- doc: small fixes on help-text of project commands.

- feat(dice): prepare the new-dice functionality of ``tar``\ing everything
  (see :gh:`480`).

  The new ``flag.encrypt_inputs`` in input-xlsx file, configured
  by :envvar:`ENCRYPTION_KEYS_PATH`, works for dice-2 but not yet respected
  by the old-dice commands;
  must revive :git:`4de77ea1e`.

- refact: renamed various internal classes and modules for clarity.


Various
~~~~~~~
- FIX: Support `pip >= 10+` (see :ghp:`26`).
- break: changed cmd-line scripts entry-points; if you install from sources,
  remember to run first: :code:`pip install -e {co2mpas-dir}`
- Pinned versions of dependencies affecting the accuracy of the calculations,
  to achieve stronger reproducibility; these dependent libraries are shiped
  with AIO (see :gh:`427`).
- Accurate versioning of project with :term:`polyvers`.
- feat: provide a *docker* script, ensuring correct *numpy-base+MKL* installed
  in *conda* requirements.
- WebStamp: split-off `v1.9.0a1` as separate sub-project in sources.


Known Limitations
~~~~~~~~~~~~~~~~~
- Reproducibility of results has been greatly enhanced, with quasi-identical
  results in different platforms (*linux/Windows*).
- DICE:
  - Fixed known limitation of `1.7.3` (:gh:`448`) of importing stamps from an
    older duplicate dice.
  - It is not possible to ``-recertify`` from ``nedc`` state
    (when mored files have been appended after stamping).
  - There is still no high level command to view Stamps (see low-level command
    in the old known limitation item).
    But :term:`stamp`\s received are now save in :file:`~/.co2dice/reports.txt`
    (along with :term:`dice`\s and :term:`decision`\s).
  - The decision-number generated still never includes the numbers 10, 20, …90.
  - All previous known limitations regarding :term:`mail-stamper` still apply.
    But these commands are now *deprecated*.


Intermediate releases for ``2.0.x``:
------------------------------------
.. Note::
  - Releases with ``r`` prefix signify version published in *PyPi*.
  - Releases with ``v`` prefix signify internal milestones.


``co2mpas-r2.0.0.post0``, 1 Sep 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
doc: Just to fix site and *PyPi* landing page.


``r2.0.0``, 31 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~
- fix: hide excess warnings.


``co2sim/co2gui: v2.0.0rc3``, ``co2dice/webstamper: v2.0.0rc1``, 30 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- FIX: Print remote-errors when WebStamper rejects a Dice.
- fix: WebStamper had regressed and were reacting violently with http-error=500
  ("server-failure") even on client mistakes;  now they became http-error=400.
- fix: eliminate minor deprecation warning about XGBoost(seed=) keyword.


``v2.0.0rc2`` for ``co2sim`` & ``co2gui``, 28 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- FIX: add data (xlsx-files & icons) to `co2sim` & `co2gui` wheels.
- ``v2.0.0rc1`` tried but didn't deliver due to missing package-data folders.


``v2.0.0rc0``, 24 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~
- DROP: make ``co2deps`` pinning-versions project disappear into the void,
  from where it came from, last milestone.

  Adding a moribund co2-project into PyPi (until `pip bug pypa/pip#3878
  <https://github.com/pypa/pip#3878>`_ gets fixed) is a waste of effort.

- ENH: extracted ``plot`` extras from ``co2sim`` dependencies.
  Significant work on all project dependencies (:gh:`408`, :gh:`427` & :gh:`463`).

  Coupled with the new ``wltp-0.1.0a3`` & ``pandalone-0.2.4.post1`` releases,
  now it is possible to use co2mpas-simulator with narrowed-down dependencies
  (see docker-image size reduction, above).

- REFACT: separated DICE from SIM subprojects until really necessary
  (e.g. when extracting data from appended files).  Some code-repetition needed,
  started moving utilities from ``__main__.py`` into own util-modules, at least
  for `co2dice`.

- ENH: update alpine-GCC in *docker* with recent instructions,and eventually
  used the debian image, which ends up the same size with less fuzz.
  Docker-image  `co2sim` wheel is now created *outside of docker* with
  its proper version-id of visible; paths updated, scripts enhanced,
  files documented.

- ENH: `setup.py` does not prevent from running in old Python versions
  (e.g to build *wheels* in Py-2, also in :gh:`408`).

- feat: dice-report encryption supports multiple recipients.
- feat: gui re-reads configurations on each DICE-button click.
- chore: add *GNU Makefiles* for rudimentary support to clean, build and
  maintain the new sub-projectrs.


``v2.0.0b0``, 20 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
- BREAK: SPLIT CO2MPAS(:gh:`408`) and moved packages in :file:`.{sub-dir}/src/`:

   1. ``co2sim[io]``: :file:`{root}/pCO2SIM`
   2. ``co2dice``: :file:`{root}/pCO2DICE`
   3. ``co2gui``: :file:`{root}/pCO2GUI`
   4. ``co2deps``: :file:`{root}/pCO2DEPS`
   5. ``co2mpas[pindeps]``: :file:`{root}`
   - ``WebStamper``: :file:`{root}/pWebStamper`

  - Also extracted ``io`` extras from ``co2sim`` dependencies.

- enh: use *GNU Makefile* for developers to manage sub-projects.
- enh: Dice-button reloads configurations when clicked (e.g. to read
  ``WstampSpec.recpients`` parameter if modified by the user-on-the-spot).
- enh: dice log-messages denote reports with line-numberss (not char-nums).



Intermediate releases for ``1.9.x``:
------------------------------------

``v1.9.2rc1``, 16 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~
- FIX: GUI mechanincs for logs and jobs.
- fix: finalized behavior for button-states.
- enh: possible to mute email-stamper deprecations with ``EmailStamperWarning.mute``.
- enh: RELAX I/O file-pairing rule for ``dicer`` cmd, any 2 io-files is now ok.


``v1.9.2rc0``, 14 Aug 2018 (BROKEN GUI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ENH: Add logging-timestamps in ``~/.co2dice/reports.txt`` maintained by
  the :class:`ReportsKeeper`(renamed from ``FileWritingMixin``) which now supports
  writing to multiple files through the tested *logging* library.

- enh: make location of the `reports.txt` file configurable with:
    - ``c.ReportsKeeper.default_reports_fpath`` property and
    - :envvar:`CO2DICE_REPORTS_FPATH` (env-var takes precedence).

- REFACT: move DicerCMD (& DicerSpec) in their own files and render them
  top-level sub-commands.

  Also renamed modules:

    - ``baseapp --> cmdlets`` not to confuse with ``base`` module.
    - ``dice --> cli`` not to confuse with ``dicer`` module and
    the too-overloaded :term;`dice`.

- enh: replace old output-clipping machinery in ``tstamp recv`` with
  shrink-slice.

- enh: teach GUI to also use HTTP-sessions (like ``dicer`` command does).

- GUI-state behavior was still not mature.


``r1.9.1b1``, 13 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
- FIX: ``project dicer`` command and GUI new *Dice-button* were failing to compare
  correctly existing files in project with new ones.

  Enhanced error-reporting of the button.

- doc: Update DICE-changes since previous major release.
- doc: Add glossary terms for links from new data in the excel input-file .
- doc: updated the dice changes for the forthcoming major-release, above
- dev: add "scafolding" to facilitate developing dice-button.


``v1.9.1b0``, 10 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
- FEAT: Finished implementing the GUI "Stamp" button
  (it appends also new-dice *tar*, see :gh:`378`).
  - Retrofitted `project dice` command into a new "DICER" class, working as
    *a sequencer of commands* to dice new **or existing** projects
    through *WebStamper* only.

    Specifically now it compares the given files with the ones already in the project.
    Manual intervention is still needed in abnormal cases
    (``--recertify``, over-writing files, etc).
  - Added  WebAPI + `co2dice tstamp wstamp` cli-commands to check stamps
    and connectivity to WebStamper.
  - Renamed cmd ``project dice --> dicer`` not to overload the *dice* word; it is
    a *sequencer* after all.

- feat: rename ``-W=~/co2dice.reports.txt --> ~/.co2dice/reports.txt`` to reuse dice folder.
- drop: removed `co2dice project tstamp` command, deprecated since 5-may-2017.
- enh: `project dicer` cmd uses HTTP-sessions when talking to WebStamper, but
  not the GUI button yet.
- fix: ``-W--write-fpath`` works more reliably, and by defaults it writes into
  renamed :file:`~/.co2dice/reports.txt`.


``v1.9.1a2``, 10 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
Fixes and features for the GUI *Stamp-button* and supporting ``project dice`` command.

- FEAT: ``co2dice project dicer|init|append|report|recv|parse`` and
  the ``co2dice tstamp wstamp`` commands, they have by default
  ``--write-file=~/.co2dice/reports.txt`` file, so every time they run,
  they *APPENDED* into this file these 3 items:

    1. :term:`Dice report`;
    2. :term:`Stamp`  (or any errors received from :term:`WebStamper`;
    3. :term:`Decision`.

- doc: deprecate all email-stamper commands; few new enhancements were applied
  on them.
- drop: remove deprecated  ``co2mpas gui`` cmd - `co2gui` is the norm since Jan 2017.
- doc: small fixes on help-text of project commands.
- refact: extract dice-cmd functionality into its own Spec class.
- sources: move ``tkui.py`` into it's own package. (needs re-install from sources).
- WIP: Add GUI "Stamp" button that appends also new-dice *tar* (see :gh:`378`).


``v1.9.1a1``, 10 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
Implement the new ``project dice`` command.

- Work started since `v1.9.1a0: 8 Aug 2018`.
- FEAT: NEW WEB-API CMDS:
  - ``co2dice project dicer``: Dice a new project in one action through WebStamper.
  - ``tstamp wstamp``: Stamp pre-generated Dice through WebStamper.
- feat: ``co2dice project report`` command can retrieve older reports.
  (not just the latest).  For *Stamps*, internal git commands are still needed.
- WIP: Add GUI "Stamp" button.


``r1.9.0b2``, 7 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~
Version in *PyPi* deemed OK for release.  Mostly doc-changes since `b1`.


``v1.9.0b1``, 2 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~
More changes at input-data, new-dice code and small model changes.
Not released in *PyPi*.

- feat(dice): teach the options ``--write-fpath/-W`` and ``--shrink`` to the commands::

      co2dice project (init|append|report|parse|trecv)

  so they can write directly results (i.e. report) in local files, and avoid
  printing big output to the console (see :gh:`466`).

  *WebStamper* also works now with files, since files can potentially be Mbs
  in size.

- feat(dice): teach dice commands ``--quiet/-q`` option that along with ``--verbose/-v``
  they control logging-level.

  It is actually possible to give multiple `-q` / `-v` in the command line,
  and the verbose level is an algebraic additions of all of them, starting
  from *INFO* level.

  BUT if any -v is given, the `Spec.verbosed` trait-parameter is set to true.
  (see :gh:`476`, :gh:`479`).

- feat(dice): prepare the new-dice functionality of taring everything
  (see :gh:`480`).

  Add ``flag.encrypt_inputs`` in input-xlsx file, configured
  by :envvar:`ENCRYPTION_KEYS_PATH`, but not yet respected by the dice commands;
  must revive :git:`4de77ea1e`.

- feat(WebStamper): Support Upload dice-reports from local-files & Download
  Stamp to local-files.

- fix(dice): fix redirection/piping of commands.

- fix(site): Update to latest `schedula-2.3.x` to fix site-generation
  (see :gh:`476`, :git:`e534168b`).

- enh(doc): Update all copyright notices to "2018".
- refact(sources): start using ``__main__.py`` also for dice, but without
  putting too much code in it, just for :pep:`366` relative-imports to work.


``r1.9.0b0``, 31 Jul 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
1st release with new-dice functionality.


``v1.9.0a2``, 11 Jul 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
- WebStamp: split-off `v1.9.0a1` as separate sub-project in sources.

IO Data:
^^^^^^^^
- IO: Input-template provide separate H/L fields for both *ki multiplicative* and
  *Ki additive* parameters.


``v1.9.0a1``, 5 Jul 2018
~~~~~~~~~~~~~~~~~~~~~~~~
Bumped *minor* number to signify that the :term:`VF_ID` and input-file version
have changed forward-incompatibly.  Very roughly tested (see :gh:`472`).
(`v1.9.0a0` was a checkpoint after `VF_ID` preliminary changes).

- CHANGE: Changed :term:`vehicle_family_id` format, but old format is still
  supported (:gh:`473`)::

        OLD: FT-TA-WMI-yyyy-nnnn
        NEW: FT-nnnnnnnnnnnnnnn-WMI-x

- BREAK: Bumped input-file version from ``2.2.8 --> 2.3.0``.  And improved
  file-version comparison (:term:`Semantic Versioning`)

- fix: completed transition to *polyversion* monorepo scheme.

- docker: ensure correct *numpy-base+MKL* installed in *conda* requirements.

Model:
^^^^^^
- FIX: Gear-model does not dance (:gh:`427`).
- fix: remove some pandas warnings


Intermediate releases for ``1.8.x``:
------------------------------------

``v1.8.1a2``, 12 Jun 2018
~~~~~~~~~~~~~~~~~~~~~~~~~
Tagged as ``co2mpas_v1.8.1a0`` just to switch *polyversion* repo-scheme,
from `mono-project --> monorepo` (switch will complete in next tag).

- feat(:gh:`466`, :gh:`467`, io, dice):
  Add ``--with-inputs`` on ``report`` commands that override flag in
  user-data `.xlsx` file, and attached all inputs encrypted in dice.

- Add 2 sub-commands in `report` standalone command::

      co2dice report extract  # that's the old `co2dice report`
      co2dice report unlock   # unlocks encrypted inputs in dice/stamps

- testing :gh:`375`:
  - dice: need *pytest* to run its TCs.
  - dice: cannot run all tests together, only one module by one.  All pass


``v1.8.0a1``, 7 Jun 2018
~~~~~~~~~~~~~~~~~~~~~~~~
- FIX dice, did not start due to `polyversion` not being engraved.
- The :envvar:`CO2MPARE_ENABLED` fails with::

      ERROR:co2mpas_main:Invalid value '1' for env-var[CO2MPARE_ENABLED]!
        Should be one of (0 f false n no off 1 t true y yes on).

``v1.8.0a0``, 6 Jun 2018
~~~~~~~~~~~~~~~~~~~~~~~~
PINNED REQUIRED VERSIONS, served with AIO-1.8.1a1


``v1.8.0.dev1``, 29 May 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- chore:(build, :gh:`408`, :git:`0761ba9d6`):
  Start versioning project with `polyvers` tool, as *mono-project*.
- feat(data, :gh:`???`):
  Implemented *co2mparable* generation for ex-post reproducibility studies.

``v1.8.0.dev0``, 22 May 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Included in 1st AIO-UpgradePack (see :gh:`463`).

- chore(build, :git:`e90680fae`):
  removed `setup_requires`;  must have
  these packages installed before attempting to install in "develop mode"::

      pip, setuptools setuptools-git >= 0.3, wheel, polyvers

- feat(deps): Add `xgboost` native-lib dependency, for speed.

Pre-``v1.8.0.dev0``, 15 Nov 2017
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- feat(model): Add utility to design gearbox ratios if they cannot be identified
  based on `maximum_velocity` and `maximum_vehicle_laden_mass`. This is not
  affecting the TA mode.
- feat(model): Add function to calculate the `vehicle_mass` from `curb mass`,
  `cargo_mass`, `curb_mass`, `fuel_mass`, `passenger_mass`, and `n_passengers`.
  This is not affecting the TA mode.
- Dice & WebStamper updates...


Intermediate releases for ``1.7.x``:
------------------------------------

``v1.7.4.post3``, 10 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Settled dependencies for :command:`pip` and :command:`conda` environments.


``v1.7.4.post2``, 8 Aug 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Fixed regression by "piping to stdout" of previous broken release `1.7.1.post1`.
- Pinned dependencies needed for downgrading from `v1.9.x`.

  Transitive dependencies are now served from 2 places:

  - :file:`setup.py`:  contains bounded dependency versions to ensure proper
    functioning, but not reproducibility.

    These bounded versions apply when installing from *PyPi* with command
    ``pip instal co2mpas==1.7.4.post2``; then :command:`pip` will install
    dependencies with as few as possible transitive re-installations.

  - :file:`requirements/exe.pip` & :file:`requirements/install_conda_reqs.sh`:
    contain the *pinned* versions of all calculation-important dependent libraries
    (see :gh:`463`).

    You need to get the sources (e.g. git-clone the repo) to access this file,
    and then run the command ``pip install -r <git-repo>/requirements/exe.pip``.

``v1.7.4.post1``, 3 Aug 2018 (BROKEN!)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Backport fixes to facilitate comparisons with forthcoming release 1.9+.

- Support `pip >= 10+` (fixes :ghp:`26`).
- Fix conflicting `dill` requirement.
- Fix piping dice-commands to stdout.


v1.7.4.post0, 11 Dec 2017
~~~~~~~~~~~~~~~~~~~~~~~~~
Never released in *PyPi*, just for fixes for WebStamper and the site for "Toketos".

- feat(wstamp): cache last sender+recipient in cookies.


v1.7.4, 15 Nov 2017: "Toketos"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- feat(dice, :gh:`447`): Allow skipping ``tsend -n`` command to facilitate
  :term:`WebStamper`, and transition from ``tagged`` --> ``sample`` / ``nosample``.

- fix(co2p, :gh:`448`): `tparse` checks stamp is on last-tag (unless forced).
  Was a "Known limitation" of previous versions.


v1.7.3.post0, 16 Oct 2017
~~~~~~~~~~~~~~~~~~~~~~~~~
- feat(co2p): The new option ``--recertify`` to ``co2dice project append`` allows to extend
  certification files for some vehile-family with new ones

  .. Note::
     The old declaration-files are ALWAYS retained in the history of "re-certified"
     projects.  You may control whether they old files will be also visible in the
     new Dice-report or not.

     For the new dice-report to contain ALL files (and in in alphabetical-order),
     use *different* file names - otherwise, the old-files will be overwritten.
     In the later case, the old files will be visible only to those having access
     to the whole project, such as the TAAs after receiving the project's exported
     archive.

- fix(co2p): ``co2dice project`` commands were raising NPE exception when iterating
  existing dice tags, e.g. ``co2dice project export .`` to export only the current
  project raised::

      AttributeError: 'NoneType' object has no attribute 'startswith'

- fix(tstamp): ``co2dice tstamp`` were raising NPE exceptions when ``-force`` used on
  invalid signatures.

Known Limitations
^^^^^^^^^^^^^^^^^
co2dice(:gh:`448`): if more than one dice-report is generated for a project,
it is still possible to parse anyone tstamp on the project - no check against
the hash-1 performed.  So practically in this case, the history of the project
is corrupted.



v1.7.3, 16 August 2017: "T-REA" Release
---------------------------------------
- Dice & model fine-tuning.
- Includes changes also from **RETRACTED** ``v1.6.1.post0``, 13 July 2017,
  "T-bone" release.

The Dice:
~~~~~~~~~
- feat(config): stop accepting test-key (``'CBBB52FF'``); you would receive this
  error message::

      After July 27 2017 you cannot use test-key for official runs!

      Generate a new key, and remember to re-encrypt your passwords with it.
      If you still want to run an experiment, add `--GpgSpec.allow_test_key=True`
      command-line option.

  You have to modify your configurations and set ``GpgSpec.master_key`` to your
  newly-generated key, and **re-encrypt your passowords in persist file.**

- feat(config): dice commands would complain if config-file(s) missing; remember to
  transfer your configurations from your old AIO (with all changes needed).

- feat(AIO): prepare for installing AIO in *multi-user/shared* environments;
  the important environment variable is ``HOME`` (read ``[AIO]/.co2mpad_env.bat``
  file and run ``co2dice config paths`` command).  Enhanced ``Cmd.config_paths``
  parameter to properly work with *persistent* JSON file even if a list of
  "overlayed" files/folders is given.

- feat(config): enhance ``co2dice config (desc | show | paths)`` commands
  to provide help-text and configured values for specific classes & params
  and all interesting variables affecting configurations.
  (alternatives to the much  coarser ``--help`` and ``--help-all`` options).

- Tstamping & networking:

  - feat(:gh:`382`): enhance handling of email encodings on send/recv:

    - add configurations choices for *Content-Transfer-Enconding* when sending
      non-ASCII emails or working with Outlook (usually `'=0A=0D=0E'` chars
      scattered in the email); read help on those parameters, with this command::

          co2dice config desc transfer_enc  quote_printable

    - add ``TstampSender.scramble_tag`` & ``TstampReceiver.un_quote_printable``
      options for dealing with non-ASCII dice-reports.

  - ``(t)recv`` cmds: add ``--subject``, ``--on`` and ``--wait-criteria`` options for
    search criteria on the ``tstamp recv`` and ``project trecv`` subcmds;
  - ``(t)recv`` cmds: renamed ``email_criteria-->rfc-criteria``, enhancing their
    syntax help;
  - ``(t)parse`` can guess if a "naked" dice-reports tags is given
    (specify ``--tag`` to be explicit).
  - ``(t)recv`` cmd: added ``--page`` option to download a "slice" of from the server.
  - improve ``(t)parse`` command's ``dice`` printout to include project/issuer/dates.
  - ``(t)recv``: BCC-addresses were treated as CCs; ``--raw`` STDOUT was corrupted;
    emails received
  - feat(report): print out the key used to sign dice-report.

- Projects:

  - feat(project): store tstamp-email verbatim, and sign 2nd HASH report.
  - refact(git): compatible-bump of dice-report format-version: ``1.0.0-->1.0.1``.
  - feat(log): possible to modify selectively logging output with
    ``~/logconf.yaml`` file;  generally improve error handling and logging of
    commands.
  - ``co2dice project export``:

    - fix(:ghp:`18`): fix command not to include dices from all projects.
    - feat(:gh:`423`, :gh:`435`): add ``--out`` option to set the out-fpath
      of the archive, and the ``--erase-afterwards`` to facilitate starting a
      project.

      .. Note::
        Do not (ab)use ``project export --erase-afterwards`` on diced projects.


  - ``co2dice project open``: auto-deduce project to open if only one exists.
  - ``co2dice project backup``: add ``--erase-afterwards`` option.

Known Limitations
^^^^^^^^^^^^^^^^^
  - Microsoft Outlook Servers are known to corrupt the dice-emails; depending
    on the version and the configurations, most of the times they can be fixed.
    If not, as a last resort, another email-account may be used.

    A permanent solution to the problem is will be provided when the
    the *Exchange Web Services (EWS)* protocol is implemented in *co2mpas*.

  - On *Yahoo* servers, the ``TstampReceiver.subject_prefix`` param must not
    contain any brackets (``[]``).  The are included by default, so you have to
    modify that in your configs.

  - Using GMail accounts to send Dice may not(!) receive the reply-back "Proof of
    Posting" reply (or it may delay up to days).  Please perform tests to discover that,
    and use another email-provided if that's the case.

    Additionally, Google's security provisions for some countries may be too
    strict to allow SMTP/IMAP access.  In all cases, you need to enable allow
    `less secure apps <https://support.google.com/accounts/answer/6010255>`_ to
    access your account.

  - Some combinations of outbound & inbound accounts for dice reports and timsestamps
    may not work due to `DMARC restrictions <https://en.wikipedia.org/wiki/DMARC>`_.
    JRC will offer more alternative "paths" for running Dices.  All major providers
    (Google, Yahoo, Microsoft) will not allow your dice-report to be stamped and forwarded
    to ``TstampSender.stamp_recipients`` other than the Comission; you may (or may not)
    receive "bounce" emails explaining that.

  - There is no high level command to view the stamp for some project;
    Assuming your project is in ``sample`` or ``nosample`` state, use this cmd::

        cat %HOME%/.co2dice/repo/tstamp.txt

- The decision-number generated never includes the numbers 10, 20, ...90.
  This does not change the odds for ``SAMPLE``/``NOSAMPLE`` but it does affect
  the odds for double-testing *Low* vs *High* vehicles (4 vs 5).


Datasync
~~~~~~~~
- :gh:`390`: Datasync was producing 0 values in the first and/or in the last
  cells. This has been fixed extending the given signal with the first and last
  values.
- :gh:`424`: remove buggy interpolation methods.


Model-changes
~~~~~~~~~~~~~
- :git:`d21b665`, :git:`5f8f58b`, :git:`33538be`: Speedup the model avoiding
  useless identifications during the prediction phase.

Vehicle model
^^^^^^^^^^^^^
- :git:`d90c697`: Add road loads calculation from vehicle and tyre category.
- :git:`952f16b`: Update the `rolling_resistance_coeff` according to table A4/1
  of EU legislation not world wide.
- :git:`952f16b`: Add function to calculate `aerodynamic_drag_coefficient` from
  vehicle_body.

Thermal model
^^^^^^^^^^^^^
- :gh:`169`: Add a filter to remove invalid temperature derivatives (i.e.,
  `abs(DT) >= 0.7`) during the cold phase.

Clutch model
^^^^^^^^^^^^
- :gh:`330`: Some extra RPM (peaks) has been verified before the engine's stops.
  This problem has been resolved filtering out `clutch_delta > 0` when `acc < 0`
  and adding a `features selection` in the calibration of the model.

Engine model
^^^^^^^^^^^^
- :git:`4c07751`: The `auxiliaries_torque_losses` are function of
  `engine_capacity`.

CO2 model
^^^^^^^^^
- :gh:`350`: Complete fuel default characteristics (LHV, Carbon Content, and
  Density).
- :git:`2e890f0`: Fix of the bug in `tau_function` when a hot cycle is given.
- :gh:`399`: Implement a fuzzy rescaling function to improve the
  stability of the model when rounding the WLTP bag values.
- :gh:`401`: Set co2_params limits to avoid unfeasible results.
- :gh:`402`: Rewrite of `calibrate_co2_params` function.
- :gh:`391`, :gh:`403`: Use the `identified_co2_params` as initial guess of the
  `calibrate_co2_params`. Update co2 optimizer enabling all steps in the
  identification and disabling the first two steps in the calibration. Optimize
  the parameters that define the gearbox, torque, and power losses.


IO & Data:
~~~~~~~~~
- fix(xlsx, :gh:`426`): excel validation formulas on input-template & demos did
  not accept *vehicle-family-id* with single-digit TA-ids.
- :gh:`314`, gh:`410`: MOVED MOST DEMO-FILES to AIO archive - 2 files are left.
  Updated ``co2mpas demo`` command to use them if found; add ``--download``
  option to get the very latest from Internet.
- main: rename logging option ``--quite`` --> ``--quiet``.
- :gh:`380`: Add cycle scores to output template.
- :gh:`391`: Add model scores to summary file.
- :gh:`399`: Report `co2_rescaling_scores` to output and summary files.
- :gh:`407`: Disable input-file caching by default (renamed option
  ``--override-cache --> use-cache``.

Known Limitations
^^^^^^^^^^^^^^^^^
- The ``co2mpas modelgraph`` command cannot plot flow-diagrams if Internet
  Explorer (IE) is the default browser.


GUI
~~~
- feat: ``co2gui`` command  does not block, and stores logs in temporary-file.
  It launches this file in a text-editor in case of failures.
- feat: remember position and size between launches (stored in *persistent* JSON
  file).


AIO
~~~
- Detect 32bit Windows early, and notify user with an error-popup.
- Possible to extract archive into path with SPACES (not recommended though).
- Switched from Cygwin-->MSYS2 for the POSIX layer, for better support in
  Windows paths, and `pacman` update manager.
  Size increased from ~350MB --> ~530MB.

  - feat(install):  reimplement cygwin's `mkshortcut.exe` in VBScript.
  - fix(git): use `cygpath.exe` to convert Windows paths and respect
    mount-points (see `GitPython#639
    <https://github.com/gitpython-developers/GitPython/pull/639>`_).

- Use ``[AIO]`` to signify the ALLINONE base-folder in the documentation; use it
  in co2mpas to suppress excessive development warnings.


.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS