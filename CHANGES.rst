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
   or upgrade instructions for developers (e.g. in :term:`AIO`
   use ``../Apps/WinPython/scripts/upgrade_pip.bat``).

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


Older Changes
=============
.. _changes-end:
See also the `historic changes of "unofficial" releases <doc/changes-unofficial.html>`_.
