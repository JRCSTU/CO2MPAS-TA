Changelog
=========

v4.3.1 (2022-02-03)
-------------------
Fix
~~~
- (schema): Correct maximum velocity range schema.

v4.3.0 (2022-01-13)
-------------------
Feat
~~~~
- (bin): Add bin folder to publish the repo.
- (doc): Update copyright.
- (doc): Update documentation.
- (input): Update input template.
- (core): Improve speed performances.
- (load): Calculate default `is_plugin` from `input_type`.
- (core): Improve speed performances.
- (physical): Enable, simplify, and refactor some function.
- (physical): Add `fuel_consumptions_liters` and
  `fuel_consumptions_liters_value` calculations.
- (physical): Add useful outputs.
- (wheels): Add `euro` tyre code.
- (physical): Remove minor warnings.
- (alternator): Add calibration for `alternator_charging_currents`.
- (schema): Make ravel optional.
- (dice): Add flag as inputs.

Fix
~~~
- (core): Correct some deprecation warning.
- (core): Correct some deprecation warning.
- (load): Correct error message formatting error.
- (doc): deleted all reference to TAA.
- (faq): Glossary corrected.

v4.2.0 (2021-10-01)
-------------------
Feat
~~~~
- (template): Update input template according to JET.
- (core): Update copyright.
- (core): Update copyright.
- (excel): Enable input dictionary merging.
- (schema): Add `has_capped_velocity` and `maximum_velocity_range`
  parameters.
- (models): Speedup model creation.
- (core): Update model according to cars-data database.
- (excel): Drop precondition stage.
- (excel): Add `WLTP-M` cycle.
- (schema): Add `maximum_velocity`, `capped_velocity`, and
  `vehicle_mass_running_order` params.
- (excel): Add parser for `matrix` sheets.
- (core): Modify input file according to new DICE4.
- (engine): Implement temperature time-shift.
- (thermal): Add `after_treatment_warm_up_phases` to calibrate the
  thermal model.
- (fc): Replace tau function with `after_treatment_warm_up_phases`.
- (load): Use `xlref` instead of `pandalone`.
- (load): Improve schema parser performances.
- (hybrid): Add feature to plot the `FuelMapModel` and replace
  `matplotlib` with `plotly`.
- (vehicle): Filter out close elevation points.
- (vehicle): Add formula to identify vehicle road loads and vehicle
  mass.
- (clutch_tc): Add formulas to calculate `wheel_powers` from
  `gear_box_powers_out`.
- (clutch_tc): Add formulas to calculate `gear_box_powers_in` from
  `clutch_tc_powers`.
- (engine): Add formulas to calculate `clutch_tc_powers` from
  `engine_powers_out`.
- (docker): Add script to extract the exe distribution.
- (gear_box): Add alternative `engine_thermostat_temperature`
  identification when missing `gear_box_powers_out`.
- (gear_box): Add alternative `gears` identification when missing
  `motive_powers`.
- (utils): Add `pairwise` function.
- (report): Add all vehicle summary info to augmented summary.
- (co2): Add scalar values to outputs for phases_co2_emissions.
- (report): Remove `numpy` warnings when comparing outputs vs targets.
- (summary): Add all comparisons in the augmented summary.
- (model): Add default to configure WLTP selection data.
- (selector): Add default to use all wltp input data as target in
  calibration.
- (selector): Add default to modify selector strategy.

Fix
~~~
- (excel): Update dice parameter parser.
- (schema): Remove unused functions.
- (excel): Add function to calculate vehicle mass from test mass.
- (excel): Remove `xgboost` warning.
- (schema): Correct schema error on error formatting.
- (excel): Remove `openpyxl` and `xgboost` warning.
- (excel): Ensure correct excel parsing of matrix tables.
- (load): Ensure excel reading specifying the engine.
- (cli): Correct typo.
- (excel): Rename `.wet.ta` file extension to `.jet.ta`.
- (excel): Ensure remote reading using xlref.
- (rtd): Correct documentation rendering in `rtd`.
- (excel): Update parser according to new `xlref`.
- (schema): Add pax into validation schema.
- (load): Add new file extension `.wet.ta`.
- (load): Correct parser for gear box ratios.
- (engine): Correct wrong link to calculate `full_load_speeds`.
- (engine): Disable `idle_model_detector` in case of hybrids.
- (hybrid): Add simple fc calibration model for hybrids to bypass
  `after_treatment_warm_up_phases`.
- (excel): Correct Ref class.
- (demos): Correct missing data in simplan.
- (co2): Correct wrong function name.
- (gear_box): Correct missing formula.
- (physical): Remove warnings.
- (final_drive) :gh:`571`: Correct calculation of final drive powers.
- (batteries) :gh:`570`: Implement constant model (i.e. r0 = 0).
- (templates): Remove unused hidden dice report.
- (cli): Correct logging level.
- (hybrid): Correct calculation of engine power losses when speed is
  zero.
- (gear_box): Correct gears identification for hybrid.
- (templates) :gh:`567`: Correct typo in NEDC-L cell.
- (gear_box): Correct bug when `motive_powers is None`.
- (ta): Correct early closure of input file.
- (gear_box): Restructure loss model and correct thermal calculation.
- (gear_box): Improve performance of
  `calculate_gear_box_efficiencies_torques_temperatures` function.
- (docker): Updater pyinstaller version to 3.6.
- (docker): Correct requirements.
- (plot): Remove page caching from dsp plot.
- (fc): Correct calculation order for extended phases.
- (docker): Correct build script.
- (setup): Remove wtlp limitation dependency that brakes the setup.
- (physical): Remove syntax error warning.
- (write): Use `get_node` instead `search_node_description`.
- (cli) :gh:`564`: Correct bug of `co2mpas sync template` cli.
- (core): Avoid numpy when import just co2mpas.
- (schema): Improve float parser.
- (doc): Add glossary links for `Time Series` and `General Terms`.
- (doc): Add iframe with interactive model graph.
- (doc): Add missing sub-model doc.
- (doc): Correct `extract_calibrated_model` link.
- (faq): FAQ link corrected.
- (faq): Updated where to download.
- (doc): Description of `has_periodically_regenerating_systems`
  according to 2017/1151.
- (doc) :gh:`563`: Change to name, surname of the team members.
- (docs): executable name.

Other
~~~~~
- Update copyright.

v4.1.10 (2019-11-07)
--------------------
Fix
~~~
- (optimization) :gh:`561`: Use float32 for fmin error function.

v4.1.9 (2019-11-04)
-------------------
Fix
~~~
- (excel): Correct parser for all-l.
- (schema): Correct error message for input file version.
- (template): Correct wrong cell reference.
- (setup): Fixed link setup.


v4.1.8 (2019-10-24): **Wine** Release
-------------------------------------
|co2mpas| project has been split into multiple repositories (:gh:`506`). The
`current <https://github.com/JRCSTU/CO2MPAS-TA>`_ repository contains just
|co2mpas| model. The other functionalities are installed as extra (i.e.,
`DICE <https://github.com/JRCSTU/DICE>`_,
`GUI <https://github.com/JRCSTU/co2wui>`_,
`sync <https://github.com/vinci1it2000/syncing>`_).


Important changes:
~~~~~~~~~~~~~~~~~~
The main changes made in this release regards:

GUI
^^^
A new graphical user interface (`GUI <https://github.com/JRCSTU/co2wui>`_)
has replaced the previous one.

Documentation
^^^^^^^^^^^^^
All documentation has been reviewed and updated (:gh:`533`, :gh:`540`). There
are two new sections: FAQ, and Contributing to |co2mpas|. The documentation is
now stored in Read the Docs (see the `site <https://co2mpas.readthedocs.io>`_).

I/O Data & Demo
^^^^^^^^^^^^^^^
The input excel file has been updated to version 3.1.0. (:gh:`544`), as per the
2019 amendments to Regulations (EU) 2017/1152 and 2017/1153.

The demo files have been reviewed and now four files are available
(:gh:`544`, :gh:`538`):

    1. *co2mpas_conventional.xlsx*: conventional vehicle,
    2. *co2mpas_simplan.xlsx*: sample simulation plan,
    3. *co2mpas_hybrid.xlsx*: hybrid parallel vehicle.
    4. *co2mpas_plugin.xlsx*: hybrid plugin vehicle.

Model
^^^^^
- Implemented Hybrids Electric Model for parallel, planetary, and serial
  architectures (:gh:`516`, :gh:`536`, :gh:`540`, :gh:`541`). It consists of
  nine electric motors (i.e., P0, P1, P2 planetary, P2, P3 front, P3 rear,
  P4 front, P4 rear, and starter), one DC/DC converter, and two batteries
  (i.e., service and drive batteries).
- Improved the stability of the thermal model (:gh:`458`, :gh:`498`, :gh:`516`),
  the gearbox identification (:gh:`551`) and the alternator model.
- Corrected the calibration of the Start/Stop model (:gh:`512`).
- Updated the torque converter model according to VDI253 standard (:gh:`515`).
- Refined the cylinder deactivation model (:gh:`517`).
- Implemented parser for PAX tyre code (:gh:`507`).
- Added formulas to calculate the corrected |co2| emissions according to WLTP
  and NEDC regulations (:gh:`539`).

Known Limitations
~~~~~~~~~~~~~~~~~
1. Certain programs (for example Skype) could be pre-empting (or reserving)
   some tcp/ip ports and therefore could conflict with |co2mpas| graphical
   interface that tries to launch a web server on a port in the higher range
   (> 10000).
2. Certain antivirus (for example Avast) could include python in the list of
   malicious software; however, this is not to be considered harmful. If this
   happens the antivirus should be disabled when running |co2mpas|, or a special
   exclusion should be granted to the |co2mpas| executable.
3. If |co2mpas| is installed in Windows 7 without ServicePack-1, you will get an
   error like the following::

        Error loading Python DLL 'C:\Users\admin\AppData\Local\Temp\_MEI60402\python36.dll'.
        LoadLibrary: The specified procedure could not be found.
        Error loading Python DLL 'C:\Users\admin\AppData\Local\Temp\_MEI59722\python36.dll'.
        LoadLibrary: The specified procedure could not be found.
        Delete file: C:\apps\co2mpas\pkgs\env.txt
        Output folder: C:\apps\co2mpas\conda-meta
        Extract: history
        Creating CO2MPAS menus...
        Error loading Python DLL 'C:\Users\admin\AppData\Local\Temp\_MEI51722\python36.dll'.
        LoadLibrary: The specified procedure could not be found.
        Execute: "C:\apps\co2mpas\pythonw.exe" -E -s "C:\apps\co2mpas\Lib\_nsis.py" mkdirs
        Running post install...
        Execute: "C:\apps\co2mpas\pythonw.exe" -E -s "C:\apps\co2mpas\Lib\_nsis.py" post_install
        Created uninstaller: C:\apps\co2mpas\Uninstall-CO2MPAS.exe
        Completed

4. If you use Internet Explorer version 9 or earlier, you might experience some
   problems (i.e., impossible to choose the input file for the synchronisation,
   etc..).

v3.0.0 (2019-01-29): "VOLO" Release
-----------------------------------

|co2mpas| 3.0.X becomes official on February 1st, 2019.

- There will be an overlapping period with the previous official |co2mpas| version
  **2.0.0** of 2 weeks (until February 15th).
- This release incorporates the amendments of the Regulation (EU) 2017/1153,
  `2018/2043 <https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32018R2043&from=EN)>`_
  of 18 December 2018 to the Type Approval procedure along with few fixes on the
  software.
- The engineering-model is 100% the same with the
  `2.1.0, 30-Nov-2018: "DADO" Release <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2sim-v2.1.0>`_
  and the version-bump (2.X.X --> 3.X.X) is just facilitation for the users,
  to recognize which release is suitable for the amended Correlation Regulations.
- The Type Approval mode (_TA_) of this release is **incompatible** with all
  previous Input File versions. The _Batch_ mode, for engineering purposes,
  remains compatible.
- the _TA_ mode of this release generates a single "_.zip_" output that contains
  all files used and generated by |co2mpas|.
- This release is comprised of 4 python packages:
  `co2sim <https://pypi.org/project/co2sim/3.0.0/>`_,
  `co2dice <https://pypi.org/project/co2dice/3.0.0/>`_,
  `co2gui <https://pypi.org/project/co2gui/3.0.0/>`_, and
  `co2mpas <https://pypi.org/project/co2mpas/3.0.0/>`_.

Installation
~~~~~~~~~~~~
This release will not be distributed as an **AllInOne** (AIO) package. It is
based on the `2.0.0, 31-Aug-2018: "Unleash" Release
<https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2mpas-r2.0.0>`_, launched
on 1 September 2018. There are two options for installation:

  1. Install it in your current working `AIO-v2.0.0`_.
  2. **Preferably** in a clean `AIO-v2.0.0`_,
     to have the possibility to use the old |co2mpas|-v2.0.0 + DICE2 for the
     two-week overlapping period;

.. _AIO-v2.0.0: https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2mpas-r2.0.0

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

Important Changes
~~~~~~~~~~~~~~~~~

Model
^^^^^
No model changes.

IO Data
^^^^^^^
- Input-file version from 3.0.0 --> **3.0.1**.
  - It hosts a few modifications after interactions with users.
  - The input file contained in this release cannot run in older |co2mpas|
  releases in the _TA_ mode.

DICE
^^^^
- The old DICE2 is deprecated, and must not be used after the 15th of February,
- it is replaced by the centralized DICE3 server. There will be a new procedure
  to configure the keys to _sign_ and _encrypt_ the data.

Demo Files
^^^^^^^^^^
- The input-file changed, and we have prepared new demo files to help the users
  adjust. Since we do not distribute an **AllInOne** package, you may download
  the new files:

  - from the console:
     ```console
     co2mpas demo --download
     ```

  - From this `link <https://github.com/JRCSTU/allinone/tree/master/Archive/Apps/.co2mpas-demos>`_


v2.0.0 (2018-08-31): "Unleash" Release
--------------------------------------
Changes since 1.7.4.post0:

Breaking:
~~~~~~~~~
1. The ``pip`` utility contained in the old AIO is outdated (9.0.1) and
   cannot correctly install the transitive dependencies of new ``|co2mpas|``, even
   for development purposes.  Please upgrade your ``pip`` before following the
   installation or upgrade instructions for developers.

2. The ``vehicle_family_id`` format has changed (but old format is still
   supported)::

       OLD: FT-TA-WMI-yyyy-nnnn
       NEW: FT-nnnnnnnnnnnnnnn-WMI-x

3. The |co2mpas| python package has been splitted (see :gh:`408`), and is now
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

Model
^^^^^
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
^^^^^^^
- BREAK: Bumped input-file version from ``2.2.8 --> 2.3.0``.  And improved
  file-version comparison

- CHANGE: Changed :term:`vehicle_family_id` format, but old format is still
  supported (:gh:`473`)::

        OLD: FT-TA-WMI-yyyy-nnnn
        NEW: FT-nnnnnnnnnnnnnnn-WMI-x

- feat: Input-template provide separate H/L fields for both *ki multiplicative*
  and *Ki additive* parameters.

- drop: remove deprecated  ``co2mpas gui`` sub-command - ``co2gui`` top-level
  command is the norm since January 2017.

Dice
^^^^
- FEAT: Added a new **"Stamp" button** on the GUI, stamping with *WebStamper*
  in the background in one step; internally it invokes the new ``dicer`` command
  (see below)(:gh:`378`).
- FEAT: Added the simplified top-level sub-command ``co2dice dicer`` which
  executes *a sequencer of commands* to dice new **or existing** project
  through *WebStamper*, in a single step.::

   co2dice dicer -i co2mpas_demo-1.xlsx -o O/20180812_213917-co2mpas_demo-1.xlsx

  Specifically when the project exists, e.g. when clicking again the
  *GUI-button*, it compares the given files *bit-by-bit* with the ones present
  already in the project, and proceeds *only when there are no differences*.
  Otherwise (or on network error), falling back to cli commands is needed,
  similar to what is done with abnormal cases such as ``--recertify``,
  over-writing files, etc.
- All dice-commands and *WebStamper* now also work with files, since *Dices*
  can potentially be MBs in size; **Copy + Paste** becomes problematic in these
  cases.
- Added low-level ``co2dice tstamp wstamp`` cli-command that Stamps a
  pre-generated Dice through *WebStamper*.
- FEAT: The commands ``co2dice dicer|init|append|report|recv|parse`` and
  ``co2dice tstamp wstamp``, support one or more ``--write-file <path>/-W``
  options, to and every time they run,  they can *append* or *overwrite* into
  all given ``<path>`` these 3 items as they are generated/received:

    1. Dice report;
    2. Stamp (or any errors received from WebStamper);
    3. Decision.

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
^^^^^^^
- FIX: Support `pip >= 10+` (see :ghp:`26`).
- break: changed cmd-line scripts entry-points; if you install from sources,
  remember to run first: :code:`pip install -e {co2mpas-dir}`
- Pinned versions of dependencies affecting the accuracy of the calculations,
  to achieve stronger reproducibility; these dependent libraries are shiped
  with AIO (see :gh:`427`).
- Accurate versioning of project with polyvers.
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
  - It is not possible to ``-recertify`` from ``nedc`` state (when mored files
    have been appended after stamping).
  - There is still no high level command to view Stamps (see low-level command
    in the old known limitation item).
    But stamp\s received are now save in :file:`~/.co2dice/reports.txt`
    (along with dice\s and decision\s).
  - The decision-number generated still never includes the numbers 10, 20, …90.
  - All previous known limitations regarding mail-stamper still apply.
    But these commands are now *deprecated*.

Intermediate releases for ``2.0.x``:
------------------------------------
.. Note::
  - Releases with ``r`` prefix signify version published in *PyPi*.
  - Releases with ``v`` prefix signify internal milestones.

``|co2mpas|-r2.0.0.post0``, 1 Sep 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  now it is possible to use |co2mpas|-simulator with narrowed-down dependencies
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
   6. ``WebStamper``: :file:`{root}/pWebStamper`

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
  *a sequencer of commands* to dice new **or existing** projects through
  *WebStamper* only.
  Specifically now it compares the given files with the ones already in the project.
  Manual intervention is still needed in abnormal cases (``--recertify``,
  over-writing files, etc).
- Added  WebAPI + `co2dice tstamp wstamp` cli-commands to check stamps and
    connectivity to WebStamper.
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

    1. Dice report;
    2. Stamp  (or any errors received from the WebStamper);
    3. Decision.
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

IO Data
^^^^^^^
- IO: Input-template provide separate H/L fields for both *ki multiplicative* and
  *Ki additive* parameters.

``v1.9.0a1``, 5 Jul 2018
~~~~~~~~~~~~~~~~~~~~~~~~
Bumped *minor* number to signify that the VF_ID and input-file version
have changed forward-incompatibly.  Very roughly tested (see :gh:`472`).
(`v1.9.0a0` was a checkpoint after `VF_ID` preliminary changes).

- CHANGE: Changed :term:`vehicle_family_id` format, but old format is still
  supported (:gh:`473`)::

        OLD: FT-TA-WMI-yyyy-nnnn
        NEW: FT-nnnnnnnnnnnnnnn-WMI-x

- BREAK: Bumped input-file version from ``2.2.8 --> 2.3.0``.  And improved
  file-version comparison (Semantic Versioning)
- fix: completed transition to *polyversion* monorepo scheme.
- docker: ensure correct *numpy-base+MKL* installed in *conda* requirements.

Model
^^^^^
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
  WebStamper, and transition from ``tagged`` --> ``sample`` / ``nosample``.

- fix(co2p, :gh:`448`): `tparse` checks stamp is on last-tag (unless forced).
  Was a "Known limitation" of previous versions.

v1.7.3.post0, 16 Oct 2017
~~~~~~~~~~~~~~~~~~~~~~~~~
- feat(co2p): The new option ``--recertify`` to ``co2dice project append`` allows
  to extend certification files for some vehile-family with new ones

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

DICE
~~~~
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
    the *Exchange Web Services (EWS)* protocol is implemented in *|co2mpas|*.
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
~~~~~~~~~~
- fix(xlsx, :gh:`426`): excel validation formulas on input-template & demos did
  not accept *vehicle-family-id* with single-digit TA-ids.
- :gh:`314`, gh:`410`: MOVED MOST DEMO-FILES to AIO archive - 2 files are left.
  Updated ``|co2mpas| demo`` command to use them if found; add ``--download``
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
  in |co2mpas| to suppress excessive development warnings.


.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |co2| replace:: CO\ :sub:`2`