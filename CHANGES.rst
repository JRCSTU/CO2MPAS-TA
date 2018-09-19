###############
CO2MPAS Changes
###############
.. contents::
.. _changes:


``v2.0.0``, 31 Aug 2018: "Unleash"
==================================
Changes since 1.7.4.post0:

BREAKING:
---------
1. The ``pip`` utility contained in the old AIO is outdated (9.0.1) and
   cannot correctly install the transitive dependencies of new ``co2mpas``, even for
   development purposes.  Please upgrade your ``pip`` before following the installation
   or upgrade instructions (e.g. in :term:`AIO` use ``../Apps/WinPython/scripts/upgrade_pip.bat``).

2. The ``vehicle_family_id`` format has changed (but old format is still supported)::

       OLD: FT-TA-WMI-yyyy-nnnn
       NEW: FT-nnnnnnnnnnnnnnn-WMI-x

3. The co2mpas python package has been splitted (see :gh:`408`), and is now served
   by 4 python packages listed below.  In practice this means that you can still receive
   bug-fixes and new features for the DICE or the GUI, while keeping the simulation-model
   intact.

   1. ``co2sim``: the simulator, for standalone/engineering work. Now all IO-libraries
      and graph-drawing are optional, specified the ``io`` & ``plot`` "extras".
      If you need just the simulator to experiment, you need this command
      to install/upgrade it with::

          pip install co2sim[io,plot] -U

   2. ``co2dice``: the backend & commands for :abbr:`DICE (Distributed Impromptu Co2mpas Evaluation)`.

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
     ``co2sim`` on startup checks if the old ``co2mpas-v1.x`` is still installed,
     and aborts In that case, uninstall all projects and re-install them,
     to be on the safe side, with this commands::

         pip uninstall -y co2sim co2dice co2gui co2mpas
         pip install co2sim co2dice co2gui co2mpas -U


Model:
------

- feat(co2_emissions): Add ``engine_n_cylinders`` as input value and a TA parameter.

- feat(ta): New TA output file.

  Running CO2MPAS in TA mode, will produce an extra file containing the DICE report.
  This file will be used in the feature version of DICE.

- feat(core): Improve calibration performances 60%.

- feat(manual): Add a manual prediction model according GTR.

- feat(gearbox): Add utility to design gearbox ratios if they cannot be identified
  based on ``maximum_velocity`` and ``maximum_vehicle_laden_mass``.

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

  ``engine_has_variable_valve_actuation = True`` and ``ignition_type = 'compression'``
  is permitted.

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
-------
- BREAK: Bumped input-file version from ``2.2.8 --> 2.3.0``.  And improved
  file-version comparison (:term:`Semantic Versioning`)

- CHANGE: Changed :term:`vehicle_family_id` format, but old format is still
  supported (:gh:`473`)::

        OLD: FT-TA-WMI-yyyy-nnnn
        NEW: FT-nnnnnnnnnnnnnnn-WMI-x

- feat: Input-template provide separate H/L fields for both *ki multiplicative* and
  *Ki additive* parameters.

- drop: remove deprecated  ``co2mpas gui`` sub-command - ``co2gui`` top-level command
  is the norm since January 2017.


Dice
----
- FEAT: Added a new **"Stamp" button** on the GUI, stamping with *WebStamper*
  in the background in one step; internally it invokes the new ``dicer`` command
  (see below)(:gh:`378`).

- FEAT: Added the simplified top-level sub-command ``co2dice dicer`` which
  executes *a sequencer of commands* to dice new **or existing** project
  through *WebStamper*, in a single step.::

      co2dice dicer -i co2mpas_demo-1.xlsx -o O/20180812_213917-co2mpas_demo-1.xlsx

  Specifically when the project exists, e.g. when clicking again the *GUI-button,
  it compares the given files *bit-by-bit* with the ones present already in the project,
  and proceeds *only when there are no differences.

  Otherwise (or on network error), falling back to cli commands is needed,
  similar to what is done with abnormal cases such as ``--recertify``,
  over-writing files, etc.

- All dice-commands and *WebStamper* now also work with files, since *Dices*
  can potentially be MBs in size; **Copy + Paste** becomes problematic in these cases.

- Added low-level ``co2dice tstamp wstamp`` cli-command that Stamps a pre-generated
  :term:`Dice` through *WebStamper*.


- FEAT: The commands ``co2dice dicer|init|append|report|recv|parse`` and
  ``co2dice tstamp wstamp``, support one or more ``--write-file <path>/-W`` options,
  to and every time they run,  they can *append* or *overwrite* into all given ``<path>``
  these 3 items as they are generated/received:

    1. :term:`Dice report`;
    2. :term:`Stamp`  (or any errors received from :term:`WebStamper`;
    3. :term:`Decision`.

  By default, one ``<path>`` is always ``~/.co2dice/reports.txt``, so this becomes
  the de-facto "keeper" of all reports exchanged (to mitigate a *known limitation*
  about not being able to retrieve old *stamps*).
  The location of the *reports.txt* file is configurable with

    - ``c.ReportsKeeper.default_reports_fpath`` configuration property, and/or
    - :envvar:`CO2DICE_REPORTS_FPATH` (the env-var takes precedence).

- feat: command ``co2dice project report <report-index>`` can retrieve older reports
  (not just the latest one).  Negative indexes count from the end, and
  need a trick to use them::

       co2dice project report -- -2

  There is still no higher-level command to retrieveing *Stamps*
  (an old *known limitation*); internal git commands can do this.

- drop: deprecate all email-stamper commands; few new enhancements were applied
  on them.

- feat(:gh:`466`, :gh:`467`, io, dice):
  Add ``--with-inputs`` on ``co2dice project init|append|report|dicer`` commands
  that override flag in user-data `.xlsx` file, and attached all inputs encrypted in dice.
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
-------
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
-----------------
- Reproducibility of results has been greatly enhanced, with quasi-identical results
  in different platforms (*linux/Windows*).
- DICE:
  - Fixed known limitation of `1.7.3` (:gh:`448`) of importing stamps from an older
    duplicate dice.
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




``v1.7.4.post3``, 10 Aug 2018
=============================
Settled dependencies for :command:`pip` and :command:`conda` environments.


``v1.7.4.post2``, 8 Aug 2018
============================
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
--------------------------------------
Backport fixes to facilitate comparisons with forthcoming release 1.9+.

- Support `pip >= 10+` (fixes :ghp:`26`).
- Fix conflicting `dill` requirement.
- Fix piping dice-commands to stdout.


v1.7.4.post0, 11 Dec 2017
=========================
Never released in *PyPi*, just for fixes for WebStamper and the site for "Toketos".

- feat(wstamp): cache last sender+recipient in cookies.


v1.7.4, 15 Nov 2017: "Toketos"
==============================
- feat(dice, :gh:`447`): Allow skipping ``tsend -n`` command to facilitate
  :term:`WebStamper`, and transition from ``tagged`` --> ``sample`` / ``nosample``.

- fix(co2p, :gh:`448`): `tparse` checks stamp is on last-tag (unless forced).
  Was a "Known limitation" of previous versions.


v1.7.3.post0, 16 Oct 2017
=========================
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
-----------------
co2dice(:gh:`448`): if more than one dice-report is generated for a project,
it is still possible to parse anyone tstamp on the project - no check against
the hash-1 performed.  So practically in this case, the history of the project
is corrupted.



v1.7.3, 16 August 2017: "T-REA" Release
=======================================
- Dice & model fine-tuning.
- Includes changes also from **RETRACTED** ``v1.6.1.post0``, 13 July 2017,
  "T-bone" release.

The Dice:
---------
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
~~~~~~~~~~~~~~~~~
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
--------
- :gh:`390`: Datasync was producing 0 values in the first and/or in the last
  cells. This has been fixed extending the given signal with the first and last
  values.
- :gh:`424`: remove buggy interpolation methods.


Model-changes
-------------
- :git:`d21b665`, :git:`5f8f58b`, :git:`33538be`: Speedup the model avoiding
  useless identifications during the prediction phase.

Vehicle model
~~~~~~~~~~~~~
- :git:`d90c697`: Add road loads calculation from vehicle and tyre category.
- :git:`952f16b`: Update the `rolling_resistance_coeff` according to table A4/1
  of EU legislation not world wide.
- :git:`952f16b`: Add function to calculate `aerodynamic_drag_coefficient` from
  vehicle_body.

Thermal model
~~~~~~~~~~~~~
- :gh:`169`: Add a filter to remove invalid temperature derivatives (i.e.,
  `abs(DT) >= 0.7`) during the cold phase.

Clutch model
~~~~~~~~~~~~
- :gh:`330`: Some extra RPM (peaks) has been verified before the engine's stops.
  This problem has been resolved filtering out `clutch_delta > 0` when `acc < 0`
  and adding a `features selection` in the calibration of the model.

Engine model
~~~~~~~~~~~~
- :git:`4c07751`: The `auxiliaries_torque_losses` are function of
  `engine_capacity`.

CO2 model
~~~~~~~~~
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
----------
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
~~~~~~~~~~~~~~~~~
- The ``co2mpas modelgraph`` command cannot plot flow-diagrams if Internet
  Explorer (IE) is the default browser.


GUI
---
- feat: ``co2gui`` command  does not block, and stores logs in temporary-file.
  It launches this file in a text-editor in case of failures.
- feat: remember position and size between launches (stored in *persistent* JSON
  file).


AIO:
----
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



v1.5.7.b3, 14 May 2017: Dice networking features for Ispra Workshop
===================================================================
Pre-released just a new `co2mpas` python package - not a new *AIO*.

The Dice:
---------
- fix(crypto, :gh:`382`): GPG-signing failed with non ASCII encodings, so had to
  leave gpg-encoding as is (`'Latin-1'`) for STDIN/OUT streams to work in
  foreign locales; fix crash when tstamp-sig did not pass (crash apparent only
  with ``-fd`` options).
- fix(report, :gh:`370`): was always accepting dice-reports, even if TA-flags
  were "engineering".

- refact(tstamp): rename configuration params (old names issue deprecation
  warnings)::

    x_recipients           --> tstamp_recipients
    timestamping_addresses --> tstamper_address           ## Not a list anymore!
    TstampReceiver.subject --> TstampSpec.subject_prefix  ## Also used by `recv` cmd.

- feat: renamed command: ``project tstamp -- > project tsend``.
  Now there is symmetricity between ``co2dice tstamp`` and ``co2dice project``
  cmds::

    tstamp send <--> project tsend
    tstamp recv <--> project recv

- feat: new commands:

  - ``tstamp recv``: Fetch tstamps from IMAP server and derive *decisions*
    OK/SAMPLE flags.
  - ``tstamp mailbox``: Lists mailboxes in IMAP server.
  - ``project trecv``: Fetch tstamps from IMAP server, derive *decisions*
    OK/SAMPLE flags and store them (or compare with existing).
  - ``config desc``: Describe config-params searched by ``'<class>.<param>'``
    (case-insensitive).

- feat(tstamp, :gh:`368`): Support *STARTTLS*, enhance ``DiceSpec.ssl`` config
  param::

      Bool/enumeration for what encryption to use when connecting to SMTP/IMAP
      servers:
      - 'SSL/TLS':  Connect only through TLS/SSL, fail if server supports it
                    (usual ports SMTP:465 IMAP:993).
      - 'STARTTLS': Connect plain & upgrade to TLS/SSL later, fail if server
                    supports it (usual ports SMTP:587 IMAP:143).
      - True:       enforce most secure encryption, based on server port above;
                    If port is `None`, identical to 'SSL/TLS'.
      - False:      Do not use any encryption;  better use `skip_auth` param,
                    not to reveal credentials in plain-text.

- feat(tstamp, :gh:`384`): support SOCKSv4/v5 for tunneling SMTP/IMAP through
  firewalls.
- feat(tstamp): Add ``tstamp recv`` and ``project trecv`` commands that
  connect to *IMAP* server, search for tstamp emails, parse them and
  derive the *decisions OK/SAMPLE* flags.  Can work also in "daemon" mode,
  waiting for new emails to arrive.
- feat(tstamp, :gh:`394`): Unify the initial project sub-cmds ``init``,
  ``append`` and ``report``, so now it's possible to run all three of them::

      co2dice project init --inp co2mpas_input.xlsx --out co2mpas_results.xlsx --report

  The ``project append`` supports also  the new ``--report`` option.
- feat(tstamp): ``tstamp login`` can check *SMTP*/*IMAP* server connection
  selectively.

Projects:
~~~~~~~~~
- fix(:gh:`371`): `export` cmd produces an archive with local branches without
  all dice-report tags.
- deprecate ``--reset-git-settings``, now resetting by default (inverted
  functionality possible with ``--preserved list``).

- fix(main, logconf.yml): crash `logging.config` is a module, not a  module
  attribute, (apparent only with``--logconf``).
- fix(io.schema, :gh:`379`): could not handle user-given bag-phases column.
- feat(tkui, :gh:`357`): harmonize GUI-logs colors with AIO-console's, add
  `Copy` popup-meny item.
- fix(baseapp): fix various logic flaws & minor bugs when autoencrypting
  ciphered config traits.
- chore(dep): vendorize  *traitlets* lib.
  add *PySocks* private dep.

Docs:
-----
- Add "Requirements" in installation section.



v1.5.5, file-ver: 2.2.6, 10-February 2017: "Stamp" release
==========================================================
.. image:: https://cloud.githubusercontent.com/assets/501585/20363048/
   09b0c724-ac3e-11e6-81b4-bc49d12e6aa1.png
   :align: center
   :width: 480

This |co2mpas| release contains few model changes; software updates;
and the `random sampling (DICE) command-line application
<https://co2mpas.io/glossary.html#term-dice-report>`_.

Results validated against real vehicles, are described in the
`validation report
<http://jrcstu.github.io/co2mpas/v1.5.x/validation_real_cases.html>`_; together
with the classic validation report for simulated `manual transmission vehicles
<http://jrcstu.github.io/co2mpas/v1.5.x/validation_manual_cases.html>`_
and `automatic transmission vehicles
<http://jrcstu.github.io/co2mpas/v1.5.x/validation_automatic_cases.html>`_.

The DICE
--------
The new command-line tool ``co2dice`` reads |co2mpas| input and output files,
packs them together, send their :term:`Hash-ID` in a request to a time-stamp
server, and decodes the response to a random number of (1/100 cases) to arrive
to these cases:
- **SAMPLE**, meaning "do sample, and double-test in NEDC",  or
- **OK**, meaning *no-sample*.

For its usage tkuidelines, visit the
`Wiki <https://github.com/JRCSTU/CO2MPAS-TA/wiki/CO2MPAS-user-tkuidelines>`.


Model-changes
-------------
- :gh:`325`: An additional check has been set for the input file to
  prevent |co2mpas| run when the input file states `has_torque_converter = True`
  and `gear_box_type = manual`.
- :gh:`264`: |co2mpas| glossary has been completely revised and it has migrated
  to the main `webpage <https://co2mpas.io/glossary.html>`_
  following *ReStructured Text* format.

Electric model
~~~~~~~~~~~~~~
- :gh:`281`, :gh:`329`:
  Improved prediction of the *electric model* of |co2mpas|, by setting a
  `balance SOC threshold` when the alternator is always on.


Clutch model
~~~~~~~~~~~~
- :gh:`330`: The *clutch model* has been updated to be fed with the
  `Torque converter model`.

- :gh:`330`: The *clutch model* prediction has been enhanced during gearshifts
  by remove `clutch phases` when
  ``(gears == 0……) | (velocities <= stop_velocity)``.


Final drive
~~~~~~~~~~~
- :gh:`342`: Enable an option to use more than one ``final_drive_ratios`` for
  vehicles equipped with dual/variable clutch.

IO
--
- :gh:`341`: Input template & demo files include now the ``vehicle_family_id``
  as a set of concatenated codes that are required to run the model in Type
  Approval mode.
- :gh:`356`: enhancements of the output and dice reports have been made.
- The *demo-files* are starting to move gradually from within |co2mpas| to the
  site.

GUI
---
- :gh:`359`: Don't keep files that do not exist in the output list after
  simulation.
- GUI launches with ``co2tkui`` command (not with ``co2mpas gui``).

Software and Build chores(build, site, etc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Only on ``CONSOLE`` command left - use ``[Ctrl+F2]`` to open bash console tab.
- Launch commands use ``.vbs`` scripts to avoid an empty console window.
- Reduced the length of the AIO archive name::

        co2mpas_ALLINONE-64bit-v1.5.0.b0  --> co2mpas_AIO-v1.5.0

- Enhanced plotting of the *plot_workflow* for faster navigation on |co2mpas|
  model.
- The Dispatcher library has been moved to a separate package (*schedula*).

- Enhanced **desktop GUI** to launch |co2mpas| to perform the random sampling
  for TA in addition to launch simulations (engineering and type approval
  modes), synchronize time series, generate templates and demo-files.

- UPGRADES from CORPORATE ENVIRONMENTS is not supported any more.

- Dependencies: +schedula, +formulas, -keyring


Known Limitations
-----------------

1. *DICE* is considered to be in an *early alpha stage* of development, and not
   all bugs have been ironed out.
2. Concerning the *threat model* for the *DICE*, it  is relying "roughly" on
   following premises:

   a) A single cryptographic key will be shared among all TS personnel,
      not to hinder usability at this early stage.
   b) There are no measures to ensure the trust of the procedure BEFORE the
      time-stamping. The TS personnel running *DICE*, and its PC are to be
      trusted for non-tampering;
   c) The (owner of the) time-stamp service is assumed not to collude with the
      OEMs (or if doubts are raised, more elaborate measures can be *gradually*
      introduced).
   d) The *DICE* does not strive to be tamper-resistant but rather
      tamper-evident.
   e) The denial-of-service is not considered at this stage;  but given a
      choice between blocking the Type Approval, and compromising IT-security,
      at the moment we choose the later - according to the above premise,
      humans interventions are acceptable, as long as they are recorded in the
      :term:`Hash DB` keeping a detectable non-reputable trace.

3. *DICE* needs an email server that is capable to send *cleat-text* emails
   through. Having an account-password & hostname of an SMTP server will
   suffice - most *web-email* clients might spoil the encoding of the message
   (i.e. *Web Outlook* is known to cause problems, *GMail* work OK if set to
   ``plain-text``).

4. Not all *DICE* operations have been implemented yet - in particular, you
   have to use a regular Git client to extract files from it ([1], [2], [3]).
   Take care not to modify the a project after it has been diced!

5. There is no *expiration timeout* enforced yet on the tstamp-requests - in
   the case that *a request is lost, or it takes arbitrary long time to return
   back*,  the TS may *force* another tstamp-request. At this early stage,
   human witnesses will reconcile which should be the authoritative
   tstamp-response, should they eventually arrive both. For this decision, the
   *Hash DB* records are to be relied.

6. The last part of DICE, re-importing projects archives and/or dice-reports
   into TAA registry has not yet been implemented completely (i.e. not working
   at all or not validating if hash-ids have changed).

7. There are currently 4 cmd-line tools:  ``co2mpas``, ``co2gui``, ``co2dice``
   & ``datasync``. It is expected that in a next release they will be united
   under a single ``co2`` cmd.

8. Regarding the "|co2mpas| model, all limitations from previous *"Rally"*
   release still apply.

- [1] https://desktop.github.com/
- [3] https://www.atlassian.com/software/sourcetree
- [2] https://www.gitkraken.com/


v1.4.1, file-ver: 2.2.5, 17-November 2016: "Rally" release
==========================================================
.. image:: https://cloud.githubusercontent.com/assets/501585/20363048/
   09b0c724-ac3e-11e6-81b4-bc49d12e6aa1.png
   :align: center
   :width: 480

This |co2mpas| release contains both key model and software updates; additional
capabilities have been added for the user, namely:

- the **Declaration mode:** template & demo files now contain just the minimum
  inputs required to run under *Type Approval (TA)* command;
- a **desktop GUI** to launch |co2mpas| and perform selected tasks (i.e.
  *simulate*, *datasync* time-series for a specific cycle, *generate
  templates*);
- several **model changes**:

  - improved handling of real-measurement data-series - results validated
    against real vehicles, are described in the `this release's validation
    report <http://jrcstu.github.io/co2mpas/v1.4.x/validation_real_cases.html>`_
    ;

  - support of a series of **technologies**, some marked as "untested" due to
    the lack of sufficient experimental data for their validation:

    +----------------------------------------+-----------+-----------+
    |                                        | petrol    | diesel    |
    +========================================+===========+===========+
    |      *Variable Valve Actuation (VVA):* |     X     |           |
    +----------------------------------------+-----------+-----------+
    |                           *Lean Burn:* |     X     |           |
    +----------------------------------------+-----------+-----------+
    |               *Cylinder Deactivation:* | untested  | untested  |
    +----------------------------------------+-----------+-----------+
    |     *Exhaust Gas Recirculation (EGR):* | untested  |     X     |
    +----------------------------------------+-----------+-----------+
    | *Selective Catalytic Reduction (SCR):* |           | untested  |
    +----------------------------------------+-----------+-----------+
    |          *Gearbox Thermal Management:* | untested  | untested  |
    +----------------------------------------+-----------+-----------+

- *enhancements and diagrams for the result files*, very few,
  *backward-compatible changes in the Input files*;
- the project's sources are now *"practically" open* in *GitHub*, so
  many of *the serving URLs have changed:*

  - sources are now served from *github*: https://github.com/JRCSTU/CO2MPAS-TA
  - a **Wiki** hosting `*simple guidelines*
    <https://github.com/JRCSTU/CO2MPAS-TA/wiki/CO2MPAS-user-guidelines>`_
    on how to download, install, and run the |co2mpas| software;
  - the `*Issues-tracker* <https://github.com/JRCSTU/CO2MPAS-TA/issues>`_ for
    collecting feedback,
  - installation files distributed from `*Github-Releases page*
    <https://github.com/JRCSTU/CO2MPAS-TA/releases>`_ (the
    https://files.co2mpas.io/ url has been deprecated).

The study of this release's results are contained in these 3 reports:
`manual <http://jrcstu.github.io/co2mpas/v1.4.x/validation_manual_cases.html>`_,
`automatic
<http://jrcstu.github.io/co2mpas/v1.4.x/validation_automatic_cases.html>`_,
and `real <http://jrcstu.github.io/co2mpas/v1.4.x/validation_real_cases.html>`_
cars, respectively.

.. Note::
   Actually *v1.4.1* is NOT published in *PyPi* due to corrupted ``.whl``
   archive. *v1.4.2* has been published in its place, and *v1.4.3* in the site.


Model-changes
-------------
- :gh:`250`, :gh:`276`:
  Implementation of the type approval command, defining declaration and
  engineering data.

- :gh:`228`:
  Add an option to bash cmd ``-D, --override`` to vary the data model from the
  cmd instead modifying the input file. Moreover with the new option
  ``--modelconf`` also the constant parameters can be modified.

  The cmd options ``--out-template=<xlsx-file>``,  ``--plot-workflow``,
  ``--only-summary``, and ``--engineering-mode=<n>`` have been transformed as
  internal flags that can be input from the input file or from the cmd
  (e.g., ``-D flag.xxx``).

  Add special plan id ``run_base``. If it is false, the base model is just
  parsed but not evaluated.

- :gh:`251`:
  The model-selector can enabled or disabled (default). Moreover, model-selector
  preferences can be defined in order to select arbitrary calibration models
  for each predictions.


Wheels model
~~~~~~~~~~~~
- :gh:`272` (:git:`b52bb51`, :git:`8b9ee77`): Select the tyre code with the
  minimum difference but with :math:`r_wheels > r_dynamic`. Update the default
  `tyre_dynamic_rolling_coefficient`  from :math:`0.975 --> 3.05 / 3.14`.


Electrics model
~~~~~~~~~~~~~~~
- :gh:`259`, :gh:`268` (:git:`7855e1f`, :git:`0d647ad`, :git:`9ab380b`):
  Add ``initial_state_of_charge`` in the input file of physical model and remove
  the preconditioning sheet. Use the ``initial_state_of_charge`` just to
  calibrate the model on WLTP and not to predict. The prediction is done
  selecting ``initial_state_of_charge`` according to cycle_type:
  + WLTP: 90,
  + NEDC: 99.

- :gh:`281`: Various improvements on the electric model:

  + Identification of charging statuses. This correct the model calibration.
  + Correct min and max charging SOC when a plateau (balance point) is fount.
  + Correct ``electric_loads`` when :math:`|off load| > |on load|`, choosing
    that with the minimum mean absolute error.


Vehicle model
~~~~~~~~~~~~~
- :git:`b6318e2`, :git:`c218b53`, :git:`991df88`:
  Add new data node ``angle_slopes``. This allows a prediction with variable
  slope, while before was constant value for all the simulation. The average
  slope (``av_slope``) is calculated per each phase and it is added to the
  output.
- :gh:`255`: Force velocities to math:`be >= -1 km`.


Engine model
~~~~~~~~~~~~
- :gh:`210` (:git:`5438d49`,:git:`7630832`): Improve identification of
  ``idle_engine_speed_median`` and ``identify_idle_engine_speed_std``, using the
  `DBSCAN` algorithm. Correct the identification of ``idle_engine_speed_std``
  and set maximum limit (:math:`0.3 * idle_engine_speed_median`).
- :gh:`265` (:git:`8da5eb4`): Add ``identify_engine_max_speed`` function to get
  the maximum engine speed from the T1 map speed vector.
- :gh:`202` (:git:`5792ae7`): Add a function to calculate hot idling fuel
  consumption based on co2mpas solution.
- :gh:`283` (:git:`70bd182`): Calculation of engine mass with respect to
  ``ignition_type`` and ``engine_max_power``.


Gearbox model
~~~~~~~~~~~~~
- :gh:`255` (:git:`32e6923`): Add warning log when gear-shift profile is
  generated from WLTP pkg.
- :gh:`288` (:git:`11f5ad5`): Link the ``gear_box_efficiency_constants`` to the
  parameter ``has_torque_converter``.
- :gh:`299`: Implement the gearbox thermal management (not validated, not enough
  data).


CO2 model
~~~~~~~~~
- :git:`370ca2c`: Fix of a minor bug on the calibration status when cycle is
  purely cold.
- :gh:`205`, :gh:`207`: Calibrate ``co2_params`` using co2 emission identified
  in the third step.
- :gh:`301`: Implement the exhaust gas recirculation and selective catalytic
  reduction technologies (EGR for petrol and SCR for diesel not validated, not
  enough data).
- :gh:`295`: Implement the lean burn technology. (partially validated on
  synthetic data)
- :gh:`285`: Implement the cylinder deactivation strategy.(not validated, not
  enough data)
- :gh:`287`: Implement the variable valve activation strategy.
- :gh:`259` (:git:`119fa28`): Implement ki factor correction for vehicle with
  periodically regenerating systems. Now the model predicts the declared CO2
  value.
- :gh:`271` (:git:`0972723`): Add a check for idle fuel consumption different
  than 0 in the input.


Cycle model
~~~~~~~~~~~
- :git:`444087b`: Add new data node ``max_time``. This allows to replicate the
  theoretical velocity profile when :math:`max_time > theoretical time`.
- :gh:`279` (:git:`8880d9d`,:git:`93b78db`): Add input vector variable
  ``bag_phases`` to extract the integration times for bags phases. Move
  ``select_phases_integration_times`` from ``co2_emissions`` to ``cycle``.


Clutch model
~~~~~~~~~~~~
- :gh:`256` (:git:`0e9bc3e`): FIX waring ``'No inliers found by ransac.py'``,
  implementing SafeRANSACRegressor.
- :gh:`288`,`251` (:git:`93c4212`): Use `has_torque_converter` to set the torque
  converter.

IO
~~
- :gh:`259` (:git:`beecf14`): Update the new input template 2.2.5.
- :gh:`278`: Implement a default output template file.
- :gh:`249` (:git:`12384c9`): Sort outputs according to workflow distance.
- :gh:`254` (:git:`08eac81`): FIX check for input file version.
- :gh:`251` (:git:`893f8aa`, :git:`f5a75b2`, :git:`c52886f`): Update outputs
  with new model-selector. Add default selector. Use a separate flag to enable
  the selector: ``use_selector`` configuration in case of declaration mode.
- :gh:`278` (:git:`0da7c72`, :git:`35134f1`): Add info table into summary sheet.
  Add named reference for each value inside a table.


Naming conventions
~~~~~~~~~~~~~~~~~~
- :git:`b8ce65f`: : If cycle is not given the defaults are ``nedc-h``,
  ``nedc-l``, ``wltp-h`` and ``wltp-l``.

Build Chores(build, site, etc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :gh:`189`: Open public GitHub repo; clone old releases.
- Use `ReadTheDocs <https://co2mpas-ta.readthedocs.io/>`_ for automated building
  of project-site, SSL-proxied by https://co2mpas.io.
- Depracated
- Allow to run only under *Python-3.5*, set trove-classifiers accordingly.
- Dependencies: +toolz, +Pillow, +openpyxl, +python-gnupg, +gitpython +keyring,
  +transitions, -easygui, -cachetool, -cycler.
  - Changes of URLs, opensourcing repository.

Known Limitations
-----------------

1. **Model sensitivity**: The sensitivity of CO2MPAS to moderately differing
   input time-series has been tested and found within expected ranges when
   *a single measured WLTP cycle is given as input* on each run - if both
   WLTP H & L cycles are given, even small changes in those paired time-series
   may force the `model-selector
   <http://co2mpas.io/explanation.html#model-selection>`_
   to choose different combinations of calibrated model, thus arriving in
   significantly different fuel-consumption figures between the runs.
2. **Theoretical WLTP**: The theoretical WLTP cycles produced MUST NOT
   be used for declaration - the profiles, both for Velocities and GearShifts
   are not up-to-date with the GTR.
   Specifically, these profiles are generated by the `python WLTP project
   <wltp.io>`_ which it still produces *GTR phase-1a* profiles.


v1.3.1, file-ver: 2.2.1, 18-Jul 2016: "Qube" release
====================================================
.. image:: https://cloud.githubusercontent.com/assets/501585/18394783/
   f392a136-76bb-11e6-9d6c-fe2ab6bad8e2.png
   :align: center
   :width: 480

This release contains both key model and software changes; additional
capabilities have been added for the user, namely:

- the prediction (by default) of *WLTP* cycle with the theoretical velocity
  and gear shifting profiles (do not use it for *declaration* purposes, read
  "Known Limitations" for this release, below);
- predict in a single run both *High/Low NEDC* cycles from *WLTP* ones;
- the ``datasync`` command supports more interpolation methods and templates
  for the typical need to synchronize dyno/OBD data;
- the new template file follows the regulation for the "declaration mode"
  (among others, tire-codes);

while several model changes improved the handling of real-measurement
data-series.

The study of this release's results are contained in these 3 reports:
`manual <http://jrcstu.github.io/co2mpas/v1.3.x/validation_manual_cases.html>`__,
`automatic
<http://jrcstu.github.io/co2mpas/v1.3.x/validation_automatic_cases.html>`__,
and `real <http://jrcstu.github.io/co2mpas/v1.3.x/validation_real_cases.html>`__
cars, respectively.


Model-changes
-------------
- :gh:`100`: Now co2mpas can predict bot *NEDC H/L* cycles.
  If just one NEDC is needed, the user can fill the fields of the relative NEDC
  and leave others blank.
- :gh:`225` (:git:`178d9f5`): Implement the WLTP pkg within CO2MPAS for
  calculating theoretical velocities and gear shifting.
  Now co2mpas is predicting by default the *WLTP* cycle with the theoretical
  velocity and gear shifting profiles. If velocity and/or gear shifting profiles
  are not respecting the profiles declared by the manufacturer, the correct
  theoretical profiles can be provided (as in the previous version) using the
  ``prediction.WLTP`` sheet.


Thermal model
~~~~~~~~~~~~~
- :gh:`242`: Update of the thermal model and the thermostat temperature
  identification. This is needed to fix some instabilities of the model, when
  the data provided has not a conventional behaviour. The changes applied to the
  model are the followings:

  1. Filter outliers in thermal model calibration.
  2. Select major features thermal model calibration.
  3. Use ``final_drive_powers_in`` as input of the thermal model instead the
     ``gear_box_powers_in``.
  4. Update the ``identify_engine_thermostat_temperature`` using a simplified
     thermal model.


Engine model
~~~~~~~~~~~~
- :git:`bfbbb75`: Add ``auxiliaries_power_loss`` calculation node for engine
  power losses due to engine auxiliaries ``[kW]``. By default, no auxiliaries
  assumed (0 kW).
- :git:`0816e64`: Add functions to calculate the ``max_available_engine_powers``
  and the ``missing_powers``. The latest tells if the vehicle has sufficient
  power to drive the cycle.
- :git:`71baf52`: Add inverse function to calculate engine nominal power
  ``[kW]`` from ``engine_max_torque`` and ``engine_max_speed_at_max_power``.


Vehicle model
~~~~~~~~~~~~~
- :git:`1a700b6`: Add function to treat ``obd_velocities`` and produce the
  ``velocities``. This function uses a Kalman Filter in order to smooth the
  noise in the OBD velocities ``[km/h]``, and it takes a considerable time to
  run (~5min is not uncommon, depending on the sampling frequency).
- :git:`8ded622`: FIX acceleration when adjacent velocities are zero. This error
  was due to the interpolation function that does not like discontinuities.


Electrics model
~~~~~~~~~~~~~~~
- :git:`f17a7bc`, :git:`70fbef3`, :git:`e7e3198`: Enhance calibration and
  identification of the alternator model. A new model has been added to model
  the initialization of the alternator. This is used for the first seconds of
  the alternator's operation. It corresponds to a new alternator ``status: 3``.
- :gh:`213`: Link alternator nominal power to max allowable energy recuperation.
  The amount of energy recuperated should not exceed the maximum alternator
  power provided by the user or calculated by the model.
- :git:`5d8e644`: In order to link the *start stop model* with the
  *electric model*, the latest uses as input the ``gear_box_powers`` instead
  of the ``clutch_tc_powers``.


Clutch /Torque-converter/AT models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :git:`48a836e`: FIX identification of the clutch and torque converter delta
  speeds. This has corrected the calculation of the power that flows to the
  engine.


Wheels model
~~~~~~~~~~~~
- :git:`73b3eff`: FIX function to identify the ``r_dynamic`` from
  ``velocity_speed_ratios``, ``gear_box_ratios``, and ``final_drive_ratio``.

- :gh:`229`: Add functions to calculate/identify the ``r_dynamic`` from
  ``tyre_code``. A new parameter ``tyre_dynamic_rolling_coefficient`` has been
  introduced to calculate the ``r_dynamic`` from the ``r_wheels``. This new
  calibrated coefficient belong to the ``engine_speed_model``.


Start/Stop model
~~~~~~~~~~~~~~~~
- :git:`4362cca`, :git:`b8db380`, :git:`5d8e644`: Improve identification and
  performance of *start stop model*:

  + Add a ``DefaultStartStopModel`` if this cannot be identified from the data.
  + Impose that during a vehicle stop (i.e., `vel == 0`) the engine cannot be
    switched on and off (just on).
  + Move start/stop functions in a separate module.
  + Add two nodes ``use_basic_start_stop`` and ``is_hybrid``.
  + Differentiate the start stop model behavior: basic and complex models. The
    basic start stop model is function of velocity and acceleration. While, the
    complex model is function of velocity, acceleration, temperature, and
    battery state of charge. If ``use_basic_start_stop`` is not defined, the
    basic model is used as default, except when the vehicle is hybrid.


CO2 model
~~~~~~~~~
- :gh:`210`: The definition of the fuel cut off boundary has been modified.
  Now `idle_cutoff=idle * 1.1`

- :gh:`230`: Add a function to calculate ``fuel_carbon_content`` from
  ``fuel_carbon_content_percentage``.

- :git:`fef1cc5`, :git:`fef1cc5`, :git:`94469c7`: minor reorganizations of
  the model


Engine cold start model
~~~~~~~~~~~~~~~~~~~~~~~
- :gh:`244`: Update cold start RPM model. Now there is a single model that is a
  three linear model function of the temperature and three coefficients that are
  calibrated.


Datasync
--------
- :gh:`231`: The synchronization done by technical services is not as precise as
  expected for CO2MPAS. Thus, the tool provides custom template according to the
  cycle to be synchronized.
- :gh:`232`: Add more interpolation methods that the user can use for the
  signals' resampling.


IO
--
- :gh:`198`, :gh:`237`, :gh:`215`: Support `simulation plan
  <https://co2mpas.io/usage.html#simulation-plan>`_  in input files.


Input
~~~~~
- :gh:`214`: Check the initial temperature provided by the user with that of the
  OBD time series. If the difference is greater than 0.5C a message is raised to
  the user and simulation does not take place. This can be disabled with adding
  to cmd ``--soft-validation``.
- :gh:`240`: Update the comments of the parameters in the input template.
- :gh:`240`: Add ``ignition_type`` node and rename ``eco_mode`` with
  ``fuel_saving_at_strategy``. New fuel_types: ``LPG``, ``NG``, ``ethanol``, and
  ``biodiesel``.


Output
~~~~~~
- :git:`2024df7`: Update chart format as scatter type.
- :gh:`248`: FIX **Delta Calculation** following the regulation.
  ``delta = co2_wltp - co2_nedc``.
- :git:`26f994c`: Replace ``comparison`` sheet with ``summary`` sheet.
- :gh:`246`, :git:`368caca`: Remove fuel consumption in l/100km from the
  outputs.
- :gh:`197`: Remove ``--charts`` flag. Now the output excel-file always
  contains charts by default.


ALLINONE
--------
- Upgraded WinPython from ``3.4.1`` --> ``3.5.2``.
- Include VS-redistributable & GPG4Win installable archives.
- Add *node.js* and have *npm* & *bower* installed, so that
  the *declarativewidgets* extension for *jupyter notebook* works ok.
  (not used yet by any of the ipython files in co2mpas).


Known Limitations
-----------------

1. **Model sensitivity**: The sensitivity of CO2MPAS to moderately differing
   input time-series has been tested and found within expected ranges when
   *a single measured WLTP cycle is given as input* on each run - if both
   WLTP H & L cycles are given, even small changes in those paired time-series
   may force the `model-selector
   <http://co2mpas.io/explanation.html#model-selection>`_
   to choose different combinations of calibrated model, thus arriving in
   significantly different fuel-consumption figures between the runs.
2. **Theoretical WLTP**: The theoretical WLTP cycles produced MUST NOT
   be used for declaration - the profiles, both for Velocities and GearShifts
   are not up-to-date with the GTR.
   Specifically, these profiles are generated by the `python WLTP project
   <wltp.io>`_ which it still produces *GTR phase-1a* profiles.


v1.2.5, file-ver: 2.2, 25-May 2016: "Panino/Sandwich" release ("PS")
====================================================================
.. image:: https://cloud.githubusercontent.com/assets/501585/15218135/
   a1bd7c0-185e-11e6-9180-3aacf4b37d7b.png
   :align: center
   :width: 480

3nd POST-Panino release.
It contains a bug fix in for creating directories.

It is not accompanied by an ALLINONE archive.


v1.2.4, file-ver: 2.2, 12-May 2016: retracted release
=====================================================
2nd POST-Panino release.
It contains the minor filtering fixes from ``1.2.3`` EXCEPT from
the thermal changes, so as to maintain the model behavior of ``1.2.2``.

It is not accompanied by an ALLINONE archive.


v1.2.3, file-ver: 2.2, 11-May 2016: retracted release
=====================================================
1st POST-Panino release, retracted due to unwanted thermal model changes,
and not accompanied by a ALLINONE archive.

- Thermal model calibration is done filtering out ``dT/dt`` outliers,
- the validation of currents' signs has been relaxed, accepting small errors
  in the inputs, and
- Minor fixes in ``calculate_extended_integration_times`` function, used for
  hot-cycles.


v1.2.2, file-ver: 2.2, 19-Apr 2016: "Panino" release
====================================================
.. image:: https://cloud.githubusercontent.com/assets/501585/14559450/
   20a56554-0309-11e6-9c4d-22fc72e3d934.png
   :align: center
   :width: 480

This release contains both key model and software changes; additional
capabilities have been added for the user, namely,

- the capability to accept a **theoretical WLTP** cycle and predict its
  difference from the predicted NEDC (:gh:`186`, :gh:`211`),
- the synchronization ``datasync`` command tool (:gh:`144`, :gh:`218`), and
- improve and explain the `naming-conventions
  <http://co2mpas.io/explanation.html#excel-input-data-naming-conventions>`_
  used in the model and in the input/output excel files (:gh:`215`);

while other changes improve the quality of model runs, namely,

- the introduction of schema to check input values(:gh:`60`, :gh:`80`),
- several model changes improving the handling of real-measurement data-series,
  and
- several crucial engineering fixes and enhancements on the model-calculations,
  including fixes based on  LAT's assessment of the "O'Snow" release.

The study of this release's results are contained in `these 3 report files
<https://jrcstu.github.io/co2mpas/>`_ for *manual*,  *automatic* and *real*
cars, respectively.


Model-changes
-------------
- :gh:`6`: Confirmed that *co2mpas* results are  reproducible in various setups
  (py2.4, py2.5, with fairly recent combinations of numpy/scipy libraries);
  results are still expected to differ between 32bit-64bit platforms.

Engine model
~~~~~~~~~~~~
- :gh:`110`: Add a function to identify *on_idle*
  as ``engine_speeds_out > MIN_ENGINE_SPEED`` and ``gears = 0``,
  or ``engine_speeds_out > MIN_ENGINE_SPEED`` and ``velocities <= VEL_EPS``.
  When engine is idling, power flowing towards the engine is disengaged, and
  thus engine power is greater than or equal to zero. This correction is applied
  only for cars not equiped with Torque Converter.
- :git:`7340700`: Remove limits from the first step ``co2_params`` optimization.
- :gh:`195`: Enable calibration of ``co2_params`` with vectorial inputs in
  addition to bag values (in order of priority):

    - ``fuel_consumptions``,
    - ``co2_emissions``,
    - ``co2_normalization_references`` (e.g. engine loads)

  When either ``fuel_consumptions`` or ``co2_emissions`` are available, a direct
  calibration of the co2_emissions model is performed. When those are not
  available, the optimization takes place using the reference normalization
  signal - if available - to redefine the initial solution and then optimize
  based on the bag values.
- :git:`346963a`: Add ``tau_function`` and make thermal exponent (parameter *t*)
  a function of temperature.
- :git:`9d7dd77`: Remove parameter *trg* from the optimization, keep temperature
  target as defined by the identification phase.
- :git:`079642e`: Use
  ``scipy.interpolate.InterpolatedUnivariateSpline.derivative`` for the
  calculation of ``accelerations``.
- :git:`31f8ccc`: Fix prediction of unreliable rpm taking max gear and idle into
  account.
- :gh:`169`: Add derivative function for conditioning the temperature signal
  (resolves resolution issues).
- :gh:`153`: Add ``correct_start_stop_with_gears`` function and flag; default
  value ``True`` for manuals and ``False`` for automatics. The functions
  *forces* the engine to start when gear goes from zero to one, independent of
  the status of the clutch.
- :gh:`47`: Exclude first seconds when the engine is off before performing the
  temperature model calibration.

Electrics model
~~~~~~~~~~~~~~~
- :gh:`200`: Fix identification of ``alternator_status_threshold`` and
  ``charging_statuses`` for cars with no break energy-recuperation-system(BERS).
  Engine start windows and positive alternator currents are now excluded from
  the calibration.
- :gh:`192`: Add ``alternator_current_threshold`` in the identification of the
  ``charging_statuses``.
- :gh:`149`: Fix identification of the charging status at the beginning of the
  cycle.
- :gh:`149`, :gh:`157`: Fix identification of minimum and maximum state of
  charge.
- :gh:`149`: Add previous state of charge to the alternator current model
  calibration. Use GradientBoostingRegressor instead of DecisionTreeRegressor,
  due to over-fitting of the later.

Clutch /Torque-converter/AT models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :gh:`179`: Add lock up mode in the torque converter module.
- :gh:`161`: Apply ``correct_gear_shifts`` function before clearing the
  fluctuations on the ``AT_gear`` model.


IO
--
- :gh:`215`: improve and explain the `naming-conventions
  <http://co2mpas.io/explanation.html#excel-input-data-naming-conventions>`_
  used in the model and in the input/output excel files;
  on model parameters internally and on model parameters used on the
  Input/Output excel files.

Input
~~~~~
- :gh:`186`, :gh:`211`: Add a ``theoretical_WLTP`` sheet on the inputs. If
  inputs are provided, calculate the additional theoretical cycles on the
  prediction and add the results on the outputs.
- :gh:`60`, :gh:`80`: Add schema to validate shape/type/bounds/etc of input
  data. As an example, the sign of the electric currents is now validated before
  running the model. The user can add the flag ``--soft-validation`` to skip
  this validation.
- :git:`113b09b`: Fix pinning of ``co2_params``, add capability to fix
  parameters outside predefined limits.
- :gh:`104`: Add ``eco_mode`` flag. Apply ``correct_gear`` function when
  ``eco_mode = True``.
- :gh:`143`: Use electrics from the preconditioning cycle to calculate initial
  state of charge for the WLTP. Default initial state of charge is set equal to
  99%.

Output
~~~~~~
- :gh:`198`: Add calculation of *willans factors* for each phase.
- :gh:`164`: Add fuel consumption ``[l/100km]``, total and per subphase, in the
  output file.
- :gh:`173`: Fix metrics and error messages on the calibration of the clutch
  model (specifically related to calibration failures when data are not of
  adequate quality).
- :gh:`180`: Remove calibration outputs from the charts. Target signals are not
  presented if not provided by the user.
- :gh:`158`: Add ``apply_f0_correction`` function and report ``correct_f0`` in
  the summary, when the flag for the preconditioning correction is *True* in the
  input.
- :gh:`168`: Add flag/error message when input data are missing and/or vectors
  have not the same length or contain empty cells.
- :gh:`154`: Add ``calculate_optimal_efficiency`` function. The function returns
  the engine piston speeds and bmep for the calibrated co2 params, when the
  efficiency is maximum.
- :gh:`155`: Add *simple willans factors* calculation on the physical model and
  on the outputs, along with average positive power, average speed when power is
  positive, and average fuel consumption.
- :gh:`160`: Add process bar to the console when running batch simulations.
- :gh:`163`: Add sample logconf-file with all loggers; ``pandalone.xleash.io``
  logger silenced bye default.


Jupyter notebooks
-----------------
- :gh:`171`: Fix ``simVehicle.ipynb`` notebook of *O'snow*.

Cmd-line
--------
- :gh:`60`, :gh:`80`: Add flag ``--soft-validation`` to skip schema validation
  of the inputs.
- :gh:`144`, :gh:`145`, :gh:`148`, :gh:`29`, :gh:`218`: Add ``datasync``
  command. It performs re-sampling and shifting of the provided signals read
  from excel-tables. Foreseen application is to resync dyno times/velocities
  with OBD ones as reference.
- :gh:`152`: Add ``--overwrite-cache`` flag.
- : Add ``sa`` command, allowing to perform Sensitivity Analysis
  runs on fuel parameters.
- :gh:`140`, :gh:`162`, :gh:`198`, :git:`99530cb`: Add ``sa`` command that
  builds and run batches with slightly modified values on each run, useful for
  sensitivity-analysis; not fully documented yet.
- :git:`284a7df`: Add output folder option for the model graphs.

Internals
---------
- :gh:`135`: Merge physical calibration and prediction models in a unique
  physical model.
- :gh:`134`: Probable fix for generating dispatcher docs under *Cygwin*.
- :git:`e562551`, :git:`3fcd6ce`: *Dispatcher*: Boost and fix *SubDispatchPipe*,
  fix ``check wait_in`` for sub-dispatcher nodes.
- :gh:`131`: ``test_sub_modules.py`` deleted. Not actually used and difficult
  in the maintenance. To be re-drafted when will be of use.

Documentation
-------------
- improve and explain the `naming-conventions
  <http://co2mpas.io/explanation.html#excel-input-data-naming-conventions>`_
  used in the model and in the input/output excel files (:gh:`215`);

Known Limitations
-----------------
- *Model sensitivity*: The sensitivity of CO2MPAS to moderately differing input
  time-series has been tested and found within expected ranges when
  *a single measured WLTP cycle is given as input* on each run - if both
  WLTP H & L cycles are given, even small changes in those paired time-series
  may force the `model-selector
  <http://co2mpas.io/explanation.html#model-selection>`_
  to choose different combinations of calibrated model, thus arriving in
  significantly different fuel-consumption figures between the runs.


v1.1.1.fix2, file-ver: 2.1, 09-March 2016: "O'Udo" 2nd release
==============================================================
2nd POSTFIX release.

- electrics, :gh:`143`: Add default value ``initial_state_of_charge := 99``.
- clutch, :gh:`173`: FIX calibration failures with a `No inliers found` by
  `ransac.py` error.


v1.1.1.fix1, file-ver: 2.1, 03-March 2016: "O'Udo" 1st release
==============================================================
1st POSTFIX release.

- :gh:`169`, :gh:`169`: modified theta-filtering for real-data.
- :gh:`171`: update forgotten ``simVehicle.ipynb`` notebook to run ok.


v1.1.1, file-ver: 2.1, 09-Feb 2016: "O'snow" release
====================================================
.. image:: https://cloud.githubusercontent.com/assets/13638851/12930853/
   f2a79350-cf7a-11e5-9a0f-5fa6fc9aa1a4.png
   :align: center
   :width: 480

This release contains mostly model changes; some internal restructurings have
not affected the final user.

Several crucial bugs and enhancements have been been implemented based on
assessments performed by LAT.  A concise study of this release's results
and a high-level description of the model changes is contained in this `JRC-LAT
presentation <http://files.co2mpas.io/CO2MPAS-1.1.1/
JRC_LAT_CO2MPAS_Osnow-validation_n_changelog.pptx>`_.


Model-changes
-------------
Engine model
~~~~~~~~~~~~
- Fix extrapolation in ``engine.get_full_load()``, keeping constant the boundary
  values.
- Update ``engine.get_engine_motoring_curve_default()``. The default motoring
  curve is now determined from the engine's friction losses parameters.
- Add engine speed cut-off limits.
- :gh:`104`: Apply *derivative* scikit-function for smoothing
  real data to acceleration & temperature.
- :gh:`82`, :gh:`50`: Add (partial) engine-inertia & auxiliaries torque/power
  losses.
- Optimizer:

  - :git:`84cc3ae8`: Fix ``co2_emission.calibrate_model_params()`` results
    selection.
  - :gh:`58`: Change error functions: *mean-abs-error* is used instead of
    *mean-squared-error*.
  - :gh:`56`: Cold/hot parts distinction based on the first occurrence of *trg*;
    *trg* not optimized.
  - :gh:`25`: Simplify calibration method for hot part of the cycle,
    imposing ``t=0``.

Temperature model
~~~~~~~~~~~~~~~~~
- :gh:`118`, :gh:`53`: Possible to run hot start cycles & fixed
  temperature cycles.
- :gh:`94`: Fix bug in
  ``co2_emission.calculate_normalized_engine_coolant_temperatures()``, that
  returned *0* when ``target_Theta > max-Theta`` in NEDC.
- :gh:`79`: Enhance temperature model: the calibration does not take into
  account the first 10secs and the points where ``Delta-Theta = 0``.
- :gh:`55`: Add an additional temperature model, ``f(previous_T, S, P, A)``;
  chose the one which gives the best results.

Gearbox model
~~~~~~~~~~~~~
- :gh:`49`: Fix bug in the estimation of the gear box efficiency for negative
  power, leading to an overestimation of the gear box temperature. (still open)
- :gh:`45`: ATs: Fix bug in the *GSPV matrix* leading to vertical up-shifting
  lines.

S/S model
~~~~~~~~~
- :gh:`85`: Correct internal gear-shifting profiles according to legislation.
- :gh:`81`: MTs: correct S/S model output -start engine- when ``gear > 0``.
- :gh:`75`, :git:`3def98f3`: Fix gear-identification for
  initial time-steps for real-data; add warning message if WLTP does not
  respect input S/S activation time.

Electrics model
~~~~~~~~~~~~~~~
- :gh:`78`, :gh:`46`: Fix bug in
  ``electrics.calibrate_alternator_current_model()`` for real cars, fix fitting
  error when alternator is always off.
- :gh:`17`: Add new alternator status model, bypassing the DT when
  ``battery_SOC_balance`` is given, ``has_energy_recuperation`` equals to one,
  but BERS is not identified in WLTP.

Clutch/Torque-converter models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :gh:`83`: Add a second clutch model, equals to no-clutch, when clutch model
  fails.
- :gh:`16`: Add torque converter.

Vehicle model
~~~~~~~~~~~~~
- :gh:`76`: Remove first 30 seconds for the engine speed model
  selection.
- :git:`e8cabe10`, :git:`016e7060`: Rework model-selection code.


IO
--

Inputs:
~~~~~~~
- :gh:`62`: New compulsory fields in input data::

      velocity_speed_ratios
      co2_params
      gear_box_ratios
      full_load_speeds
      full_load_torques
      full_load_powers

- Add `fuel_carbon_content` input values for each cycle.
- Correct units in `initial_SOC_NEDC`.
- Replace `Battery SOC [%]` time series with ``target state_of_charges``.
- :gh:`61`, :gh:`119`: Add dyno type and driveline type (2WD, 4WD) for each
  cycle. Those are used to specify inertia coefficients and drivetrain
  efficiency (default efficiency for `final_drive_efficiency` changed to 0.98).
  (still open)
- :gh:`44`: Correct `battery_SOC_balance` and `battery_SOC_window` as
  not *compulsory*.
- :gh:`25`: Add option of 'freezing' the optimization parameters.

Outputs:
~~~~~~~~
- :gh:`96`: Produce a single excel with all infos in multiple sheets.
- :gh:`20`: Produce html report with basic interactive graphs (unfinished).
- :git:`5064efd3`: Add charts in excel output.
- :gh:`120`, :gh:`123`: Use excel named-ranges for all columns -
  possible to use specific xl-file as output template, utilizing those
  named-ranges.
- :git:`a03c6805`: Add `status_start_stop_activation_time` to cycle results.
- :git:`f8b85d98`: Add comparison between WLTP prediction vs WLTP inputs &
  WLTP calibration.
- :gh:`102`: Write errors/warnings in the output.(still open)
- :gh:`101`: Add target UDC and target EUDC to the summary.
- :gh:`97`, :gh:`114`, :gh:`64`: Add packages and CO2MPAS versions,
  functions run info, and models' scores to the *proc_info* sheet.(still open)
- :gh:`93`, :gh:`52`: Add success/fail flags related to the optimization steps
  for each cycle, and global success/fail flags on the summary.


Cmd-line (running CO2MPAS)
--------------------------

- Normalize `main()` syntax (see ``co2mpas --help``):

  - Always require a subcommand (tip: try ``co2mpas batch <input-file-1>...``).
  - Drop the ``-I`` option, support multiple input files & folders as simple
    positional arguments in the command-line - ``-O`` now defaults to
    current-folder!
  - Report and halt if no input-files found.
  - GUI dialog-boxes kick-in only if invoked with the  ``--gui`` option.
    Added new dialog-box for cmd-line options (total GUIs 3 in number).
  - Autocomomplete cmd-line with ``[Tab]`` both for `cmd.exe` and *bash*
    (consoles pre-configured in ALLINONE).
  - Support logging-configuration with a file.
  - Other minor options renames and improvements.

- :git:`5e91993c`: Add option to skip saving WLTP-prediction.
- :gh:`88`: Raise warning (console & summary-file) if incompatible ``VERSION``
  detected in input-file.
- :gh:`102`: Remove UI pop-up boxes when running - users have to check
  the *scores* tables in the result xl-file.
- :gh:`91`: Disallow installation and/or execution under ``python < 3.4``.
- :git:`5e91993c`: Add option to skip saving WLTP-prediction.
- :gh:`130`: Possible to plot workflow int the output folder with
  ``--plot-workflow`` option.


Documentation
-------------

- :gh:`136`: Add section explaining the CO2MPAS selector model.
- Comprehensive JRC-LAT presentation for validation and high-level summary
  of model changes  (mentioned above).
- New section on how to setup autocompletion for *bash* and *clink* on
  `cmd.exe`.
- Link to the "fatty" (~40Mb) `tutorial input xl-file
  <http://files.co2mpas.io/CO2MPAS-1.1.1/co2mpas_tutorial_1_1_0.xls>`_.


Internals
---------

- *dispatcher*: Functionality, performance, documentation and debugging
  enhancements for the central module that is executing model-nodes.
- :git:`1a6a901f6c`: Implemented new architecture for IO files.
- :gh:`103`: Problem with simulation time resolved (caused by new IO).
- :gh:`94`, :gh:`99`: Fixed error related to ``argmax()`` function.
- :gh:`25`: Retrofit optimizer code to use *lmfit* library to provide for
  easily playing with parameters and optimization-methods.
- :gh:`107`: Add *Seatbelt-TC* reporting sources of discrepancies, to
  investigate repeatability(:gh:`7`) and reproducibility(:gh:`6`) problems.
- :gh:`63`: Add TCs for the core models. (still open)



v1.1.0-dev1, 18-Dec-2015: "Natale" internal JRC version
=======================================================
Distributed before Christmas and included assessments from LAT.
Model changes reported in "O'snow" release, above.


v1.0.5, 11-Dec 2015: "No more console" release, no model changes
================================================================
.. image:: https://cloud.githubusercontent.com/assets/501585/11741701/
   2680714-a003-11e5-9ae6-c58a343f1a3f.png
   :align: center
   :width: 480

- main: Failback to GUI when demo/template/ipynb folder not specified in
  cmdline (prepare for Window's start-menu shortcuts).
- Install from official PyPi repo (simply type ``pip install co2mpas``).
- Add logo.

- ALLINONE:

  - FIX "empty" folder-selection lists bug.
  - Renamed ``cmd-console.bat`` --> ``CONSOLE.bat``.
  - By default store app's process STDOUT/STDERR into logs-files.
  - Add ``INSTALL.bat`` script that creates menu-entries for most common
    CO2MPAS task into *window StartMenu*.
  - Known Issue: Folder-selection dialogs still might appear
    beneath current window sometimes.



v1.0.4, 9-Nov 2015: 3rd public release, mostly model changes
============================================================
Model-changes in comparison to v1.0.1:

- Vehicle/Engine/Gearbox/Transmission:

  - :gh:`13`: If no `r_dynamic` given, attempt to identify it from ``G/V/N``
    ratios.
  - :gh:`14`: Added clutch model for correcting RPMs. Power/losses still
    pending.
  - :gh:`9`: Start-Stop: new model based on the given
    `start_stop_activation_time`, failing back to previous model if not
    provided. It allows engine stops after the 'start_stop_activation_time'.
  - :gh:`21`: Set default value of `k5` equal to `max_gear` to resolve high rpm
    at EUDC deceleration.
  - :gh:`18`: FIX bug in `calculate_engine_start_current` function (zero
    division).

- Alternator:

  - :gh:`13`: Predict alternator/battery currents if not privded.
  - :gh:`17`: Impose `no_BERS` option when ``has_energy_recuperation == False``.

- A/T:

  - :gh:`28`: Change selection criteria for A/T model
    (``accuracy_score-->mean_abs_error``); not tested due to lack of data.
  - :gh:`34`: Update *gspv* approach (cloud interpolation -> vertical limit).
  - :gh:`35`: Add *eco mode* (MVL) in the A/T model for velocity plateau.
    It selects the highest possible gear.
  - Add option to the input file in order to use a specific A/T model (
    ``specific_gear_shifting=A/T model name``).

- Thermal:

  - :gh:`33`, :gh:`19`: More improvements when fitting of the thermal model.

- Input files:

  - Input-files specify their own version number (currently at `2`).
  - :gh:`9`: Enabled Start-Stop activation time cell.
  - :gh:`25`, :gh:`38`: Add separate sheet for overriding engine's
    fuel-consumption and thermal fitting parameters (trg, t)
    (currently ALL or NONE have to be specified).
  - Added Engine load (%) signal from OBD as input vector.
    Currently not used but will improve significantly the accuracy of the
    cold start model and the execution speed of the program.
    JRC is working on a micro-phases like approach based on this signal.
  - Gears vector not necessary anymore. However providing gears vector
    improves the results for A/Ts and may also lead to better accuracies
    in M/Ts in case the RPM or gear ratios values are not of good quality.
    JRC is still analyzing the issue.

- Output & Summary files:

  - :gh:`23`: Add units and descriptions into output files as a 2nd header-line.
  - :gh:`36`, :gh:`37`: Add comparison-metrics into the summary (target vs
    output). New cmd-line option ``--only-summary`` to skip saving
    vehicle-files.

- Miscellaneous:

  - Fixes for when input is 10 Hz.
  - :gh:`20`: Possible to plot workflows of nested models
    (see Ipython-notebook).
  - Cache input-files in pickles, and read with up-to-date check.
  - Speedup workflow dispatcher internals.


v1.0.3, 13-Oct 2015, CWG release
================================
Still no model-changes in comparison to v1.0.1; released just to distribute
the *all-in-one* archive, provide better instructions, and demonstrate ipython
UI.

- Note that the CO2MPAS contained in the ALLINONE archive is ``1.0.3b0``,
  which does not affect the results or the UI in any way.


v1.0.2, 6-Oct 2015: "Renata" release, unpublished
=================================================
No model-changes, beta-testing "all-in-one" archive for *Windows* distributed
to selected active users only:

- Distributed directly from newly-established project-home on http://co2mpas.io/
  instead of emailing docs/sources/executable (to deal with blocked emails and
  corporate proxies)
- Prepare a pre-populated folder with WinPython + CO2MPAS + Consoles
  for Windows 64bit & 32bit (ALLINONE).
- ALLINONE actually contains ``co2mpas`` command versioned
  as ``1.0.2b3``.
- Add **ipython** notebook for running a single vehicle from the browser
  (see respective Usage-section in the documents) but fails!
- docs:
    - Update Usage instructions based on *all-in-one* archive.
    - Tip for installing behind corporate proxies (thanks to Michael Gratzke),
       and provide link to ``pandalone`` dependency.
    - Docs distributed actually from `v1.0.2-hotfix.0` describing
      also IPython instructions, which, as noted above, fails.

Breaking Changes
----------------
- Rename ``co2mpas`` subcommand: ``examples --> demo``.
- Rename internal package, et all ``compas --> co2mpas``.
- Log timestamps when printing messages.


v1.0.1, 1-Oct 2015: 2nd release
===============================
- Comprehensive modeling with multiple alternative routes depending on
  available data.
- Tested against a sample of 1800 artificially generated vehicles (simulations).
- The model is currently optimized to calculate directly the NEDC CO2 emissions.

Known Limitations
-----------------

#. When data from both WLTP H & L cycles are provided, the model results in
   average NEDC error of ~0.3gCO2/km +- 5.5g/km (stdev) over the 1800 cases
   available to the JRC. Currently no significant systematic errors are observed
   for UDC and EUDC cycles.  No apparent correlations to specific engine or
   vehicle characteristics have been observed in the present release.
   Additional effort is necessary in order to improve the stability of the tool
   and reduce the standard deviation of the error.
#. It has been observed that CO2MPAS tends to underestimate the power
   requirements due to accelerations in WLTP.
   More feedback is needed from real test cases.
#. The current gearbox thermal model overestimates the warm up rate of the
   gearbox.
   The bug is identified and will be fixed in future versions.
#. Simulation runs may under certain circumstances produce different families
   of solutions for the same inputs
   (i.e. for the CO2 it is in the max range of 0.5 g/km).
   The bug is identified and will be fixed in future versions.
#. The calculations are sensitive to the input data provided, and in particular
   the time-series. Time series should originate from measurements/simulations
   that correspond to specific tests from which the input data were derived.
   Mixing time series from different vehicles, tests or cycles may produce
   results that lay outside the expected error band.
#. Heavily quantized velocity time-series may affect the accuracy of the
   results.
#. Ill-formatted input data may NOT produce warnings.
   Should you find a case where a warning should have been raised, we kindly
   ask you to communicate the finding to the developers.
#. Misspelled input-data which are not compulsory, are SILENTLY ignored, and
   the calculations proceed with alternative routes or default-values.
   Check that all your input-data are also contained in the output data
   (calibration files).
#. The A/T module has NOT been tested by the JRC due to the lack of respective
   test-data.
#. The A/T module should be further optimized with respect to the gear-shifting
   method applied for the simulations. An additional error of 0.5-1.5g/km  in
   the NEDC prediction is expected under the current configuration based
   on previous indications.
#. The model lacks a torque-converter / clutch module. JRC requested additional
   feedback on the necessity of such modules.
#. The electric systems module has not been tested with real test data.
   Cruise time series result in quantized squared-shaped signals which are,
   in general, different from analog currents recorded in real tests.
   More test cases are necessary.
#. Currently the electric system module requires input regarding both
   alternator current and battery current in  order to operate. Battery current
   vector can be set to zero but this may reduce the accuracy of the tool.
#. The preconditioning cycle and the respective functions has not been tested
   due to lack of corresponding data.


v0, Aug 2015: 1st unofficial release
====================================
Bugs reported from v0 with their status up to date:

#. 1s before acceleration "press clutch" not applied in WLTP:
   **not fixed**, lacking clutch module, problem not clear in Cruise time
   series, under investigation
#. Strange engine speed increase before and after standstill:
   **partly corrected**, lack of clutch, need further feedback on issue
#. Upshifting seems to be too early, also observed in WLTP, probably
   gearshift point is not "in the middle" of shifting:
   **not fixed**, will be revisited in future versions after comparing with
   cruise results
#. RPM peaks after stop don't match the real ones:
   **pending**, cannot correct based on Cruise inputs
#. Although temperature profile is simulated quite good, the consumption between
   urban and extra-urban part of NEDC is completely wrong:
   **problem partly fixed**, further optimization in UDC CO2 prediction
   will be attempted for future versions.
#. Delta-RCB is not simulated correctly due to a too high recuperation energy
   and wrong application down to standstill:
   **fixed**, the present release has a completely new module for
   calculating electric systems. Battery currents are necessary.
#. Output of more signals for analysis would be necessary:
   **fixed**, additional signals are added to the output file.
   Additional signals could be made available if necessary (which ones?)
#. Check whether a mechanical load (pumps, alternator and climate offset losses)
   as torque-input at the crankshaft is applied:
   **pending**, mechanical loads to be reviewed in future versions after more
   feedback is received.
#. Missing chassis dyno setting for warm-up delta correction:
   **unclear** how this should be treated (as a correction inside the tool or
   as a correction in the input data)
#. SOC Simulation: the simulation without the SOC input is much too optimistic
   in terms of recuperation / providing the SOC signals does not work as
   intended with the current version:
   **fixed**, please review new module for electrics.
#. The gearshift module 0.5.5 miscalculates gearshifts:
   **partially fixed**, the module is now included in CO2MPAS v1 but due to lack
   in test cases has not been further optimized.
#. Overestimation of engine-power in comparison to measurements:
   **indeterminate**, in fact this problem is vehicle specific. In the
   test-cases provided to the JRC both higher and lower power demands are
   experienced. Small deviations are expected to have a limited effect on the
   final calculation. What remains open is the amount of power demand over WLTP
   transient phases which so far appears to be systematically underestimated in
   the test cases available to the JRC.
#. Overestimation of fuel-consumption during cold start:
   **partially fixed**, cold start over UDC has been improved since V0.
#. CO2MPAS has a pronounced fuel cut-off resulting in zero fuel consumption
   during over-runs:
   **fixed**, indeed there was a bug in the cut-off operation associated to
   the amount of power flowing back to the engine while braking.
   A limiting function is now applied. Residual fuel consumption is foreseen
   for relatively low negative engine power demands (engine power> -2kW)
#. A 5 second start-stop anticipation should not occur in the case of A/T
   vehicles: **fixed**.


.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
