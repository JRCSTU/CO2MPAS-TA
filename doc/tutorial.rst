
The sections below constitute a "reference" for |co2mpas| - a **tutorial**
is maintained in the *wiki* for this project at:
https://github.com/JRCSTU/CO2MPAS-TA/wiki/CO2MPAS-user-guidelines

|co2mpas| GUI
-------------
From *"Rally"* release, |co2mpas| can be launched through a *Graphical User Interface (GUI)*.
Its core functionality is provided from within the GUI.
Just ensure that the latest version of |co2mpas| is properly installed, and
that its version is the latest released, by checking the "About" menu,
as shown in the animation, below:

.. image:: _static/Co2mpasALLINONE-About.gif
   :scale: 75%
   :alt: Check Co2mpas-ALLINONE Version
   :align: center


Alternatively, open the CONSOLE and type the following command:

.. code-block:: console

    ## Check co2mpas version.
    $ co2mpas -V
    co2mpas-|version|


|co2mpas| command syntax
------------------------
To get the syntax of the |co2mpas| console-command, open a console where
you have installed |co2mpas| (see :ref:`co2mpas-install` above) and type::

    ## co2mpas help.
    $ co2mpas --help

    Predict NEDC CO2 emissions from WLTP.

    :Home:         http://co2mpas.io/
    :Copyright:    2015-2019 European Commission, JRC <https://ec.europa.eu/jrc/>
    :License:       EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>

    Use the `batch` sub-command to simulate a vehicle contained in an excel-file.


    USAGE:
      co2mpas ta          [-f] [-v] [-O=<output-folder>] [<input-path>]...
      co2mpas batch       [-v | -q | --logconf=<conf-file>] [-f]
                          [--use-cache] [-O=<output-folder>]
                          [--modelconf=<yaml-file>]
                          [-D=<key=value>]... [<input-path>]...
      co2mpas demo        [-v | -q | --logconf=<conf-file>] [-f]
                          [<output-folder>] [--download]
      co2mpas template    [-v | -q | --logconf=<conf-file>] [-f]
                          [<excel-file-path> ...]
      co2mpas ipynb       [-v | -q | --logconf=<conf-file>] [-f] [<output-folder>]
      co2mpas modelgraph  [-v | -q | --logconf=<conf-file>] [-O=<output-folder>]
                          [--modelconf=<yaml-file>]
                          (--list | [--graph-depth=<levels>] [<models> ...])
      co2mpas modelconf   [-v | -q | --logconf=<conf-file>] [-f]
                          [--modelconf=<yaml-file>] [-O=<output-folder>]
      co2mpas gui         [-v | -q | --logconf=<conf-file>]
      co2mpas             [-v | -q | --logconf=<conf-file>] (--version | -V)
      co2mpas             --help

    Syntax tip:
      The brackets `[ ]`, parens `( )`, pipes `|` and ellipsis `...` signify
      "optional", "required", "mutually exclusive", and "repeating elements";
      for more syntax-help see: http://docopt.org/


    OPTIONS:
      <input-path>                Input xlsx-file or folder. Assumes current-dir if missing.
      -O=<output-folder>          Output folder or file [default: .].
      --download                  Download latest demo files from ALLINONE GitHub project.
      <excel-file-path>           Output file [default: co2mpas_template.xlsx].
      --modelconf=<yaml-file>     Path to a YAMmodel-configuration YAML file.
      --use-cache                 Use the cached input file.
      --override, -D=<key=value>  Input data overrides (e.g., `-D fuel_type=diesel`,
                                  `-D prediction.nedc_h.vehicle_mass=1000`).
      -l, --list                  List available models.
      --graph-depth=<levels>      An integer to Limit the levels of sub-models plotted.
      -f, --force                 Overwrite output/template/demo excel-file(s).


    Model flags (-D flag.xxx, example -D flag.engineering_mode=True):
     engineering_mode=<bool>     Use all data and not only the declaration data.
     soft_validation=<bool>      Relax some Input-data validations, to facilitate experimentation.
     use_selector=<bool>         Select internally the best model to predict both NEDC H/L cycles.
     only_summary=<bool>         Do not save vehicle outputs, just the summary.
     plot_workflow=<bool>        Open workflow-plot in browser, after run finished.
     output_template=<xlsx-file> Clone the given excel-file and appends results into
                                 it. By default, results are appended into an empty
                                 excel-file. Use `output_template=-` to use
                                 input-file as template.

    Miscellaneous:
      -h, --help                  Show this help message and exit.
      -V, --version               Print version of the program, with --verbose
                                  list release-date and installation details.
      -v, --verbose               Print more verbosely messages - overridden by --logconf.
      -q, --quiet                 Print less verbosely messages (warnings) - overridden by --logconf.
      --logconf=<conf-file>       Path to a logging-configuration file, according to:
                                    https://docs.python.org/3/library/logging.config.html#configuration-file-format
                                  If the file-extension is '.yaml' or '.yml', it reads a dict-schema from YAML:
                                    https://docs.python.org/3/library/logging.config.html#logging-config-dictschema


    SUB-COMMANDS:
        gui             Launches co2mpas GUI (DEPRECATED: Use `co2gui` command).
        ta              Simulate vehicle in type approval mode for all <input-path>
                        excel-files & folder. If no <input-path> given, reads all
                        excel-files from current-dir. It reads just the declaration
                        inputs, if it finds some extra input will raise a warning
                        and will not produce any result.
                        Read this for explanations of the param names:
                          http://co2mpas.io/explanation.html#excel-input-data-naming-conventions
        batch           Simulate vehicle in scientific mode for all <input-path>
                        excel-files & folder. If no <input-path> given, reads all
                        excel-files from current-dir. By default reads just the
                        declaration inputs and skip the extra inputs. Thus, it will
                        produce always a result. To read all inputs the flag
                        `engineering_mode` have to be set to True.
                        Read this for explanations of the param names:
                          http://co2mpas.io/explanation.html#excel-input-data-naming-conventions
        demo            Generate demo input-files for co2mpas inside <output-folder>.
        template        Generate "empty" input-file for the `batch` cmd as <excel-file-path>.
        ipynb           Generate IPython notebooks inside <output-folder>; view them with cmd:
                          jupyter --notebook-dir=<output-folder>
        modelgraph      List or plot available models. If no model(s) specified, all assumed.
        modelconf       Save a copy of all model defaults in yaml format.


    EXAMPLES::

        # Don't enter lines starting with `#`.

        # View full version specs:
        co2mpas -vV

        # Create an empty vehicle-file inside `input` folder:
        co2mpas  template  input/vehicle_1.xlsx

        # Create work folders and then fill `input` with sample-vehicles:
        md input output
        co2mpas  demo  input

        # View a specific submodel on your browser:
        co2mpas  modelgraph  co2mpas.model.physical.wheels.wheels

        # Run co2mpas with batch cmd plotting the workflow:
        co2mpas  batch  input  -O output  -D flag.plot_workflow=True

        # Run co2mpas with ta cmd:
        co2mpas  batch  input/co2mpas_demo-0.xlsx  -O output

        # or launch the co2mpas GUI:
        co2gui

        # View all model defaults in yaml format:
        co2mpas modelconf -O output


Input template
--------------
The sub-commands ``batch`` (Run) and ``ta`` (Run TA) accept either a single
**input-excel-file** or a folder with multiple input-files for each vehicle.
You can download an *empty* input excel-file from the GUI:

.. image:: _static/Co2mpasALLINONE-Template.gif
   :scale: 75%
   :alt: Generate |co2mpas| input template
   :align: center

Or you can create an empty vehicle template-file (e.g., ``vehicle_1.xlsx``)
inside the *input-folder* with the ``template`` sub-command::

        $ co2mpas template input/vehicle_1.xlsx -f
        Creating TEMPLATE INPUT file 'input/vehicle_1.xlsx'...

The generated file contains descriptions to help you populate it with vehicle
data. For items where an array of values is required (e.g. gear-box ratios) you
may reference different parts of the spreadsheet following the syntax of the
`"xlref" mini-language <https://pandalone.readthedocs.org/en/latest/reference.html#module-pandalone.xleash>`_.

.. tip::
   For an explanation of the naming of the fields, read the :ref:`excel-model`
   section

Demo files
----------
The simulator contains demo-files that are a nice starting point to try out.
You can generate those *demo* vehicles from the GUI:

.. image:: _static/Co2mpasALLINONE-Demo.gif
   :scale: 75%
   :alt: Generate |co2mpas| demo files
   :align: center

Or you can create the demo files inside the *input-folder* with the ``demo``
sub-command::

    $ co2mpas demo input -f
    17:57:43       : INFO:co2mpas_main:Creating INPUT-DEMO file 't\co2mpas_demo-1.xlsx'...
    17:57:43       : INFO:co2mpas_main:Creating INPUT-DEMO file 't\co2mpas_simplan.xlsx'...
    17:57:43       : INFO:co2mpas_main:Run generated demo-files with command:
        co2mpas batch t

    You may find more demos inside `CO2MPAS/Demos` folder of your ALLINONE.


Demo description
~~~~~~~~~~~~~~~~
The generated demos above, along with those inside the ``CO2MPAS/Demos`` AIO-folder
have the following characteristics:

======= === ==== ==== === ==== ==== ==== ==== ========== ========
  id    AT  WLTPcalib S/S BERS NEDCtarg  plan NEDC-error metadata
------- --- --------- --- ---- --------- ---- ---------- --------
             H    L             H    L
======= === ==== ==== === ==== ==== ==== ==== ========== ========
   0         X                  X                            X
   1     X        X                  X                       X
   2         X        X   X     X
   3         X        X         X
   4     X        X       X          X
   5         X            X     X
   6     X   X        X         X             4.0 (> 4%)
   7     X   X        X   X     X             -5.65
   8         X    X             X    X
   9     X   X        X   X     X
simplan      X                  X         X
======= === ==== ==== === ==== ==== ==== ==== ========== ========


Synchronizing time-series
-------------------------
The model might fail in case your time-series signals are time-shifted and/or
with different sampling rates. Even if the run succeeds, the results will not
be accurate enough, because the data are not synchronized with the theoretical
cycle.

As an aid tool, you may use the ``datasync`` tool to "synchronize" and
"resample" your data, which have been acquired from different sources.

.. image:: _static/Co2mpasALLINONE-Datasync.gif
   :scale: 75%
   :alt: datasync tool
   :align: center

To get the syntax of the ``datasync`` console-command, open a console where
you have installed |co2mpas| and type::

    > datasync --help
    Shift and resample excel-tables; see https://co2mpas.io/usage.html#synchronizing-time-series

    Usage:
      datasync template [-f] [--cycle <cycle>] <excel-file-path>...
      datasync          [-v | -q | --logconf=<conf-file>] [--force | -f]
                        [--interp <method>] [--no-clone] [--prefix-cols]
                        [-O <output>] <x-label> <y-label> <ref-table>
                        [<sync-table> ...] [-i=<label=interp> ...]
      datasync          [-v | -q | --logconf=<conf-file>] (--version | -V)
      datasync          (--interp-methods | -l)
      datasync          --help

    Options:
      <x-label>              Column-name of the common x-axis (e.g. 'times') to be
                             re-sampled if needed.
      <y-label>              Column-name of y-axis cross-correlated between all
                             <sync-table> and <ref-table>.
      <ref-table>            The reference table, in *xl-ref* notation (usually
                             given as `file#sheet!`); synced columns will be
                             appended into this table.
                             The captured table must contain <x_label> & <y_label>
                             as column labels.
                             If hash(`#`) symbol missing, assumed as file-path and
                             the table is read from its 1st sheet .
      <sync-table>           Sheets to be synced in relation to <ref-table>, also in
                             *xl-ref* notation.
                             All tables must contain <x_label> & <y_label> as column
                             labels.
                             Each xlref may omit file or sheet-name parts; in that
                             case, those from the previous xlref(s) are reused.
                             If hash(`#`) symbol missing, assumed as sheet-name.
                             If none given, all non-empty sheets of <ref-table> are
                             synced against the 1st one.
      -O=<output>            Output folder or file path to write the results
                             [default: .]:

                             - Non-existent path: taken as the new file-path; fails
                               if intermediate folders do not exist, unless --force.
                             - Existent file: file-path to overwrite if --force,
                               fails otherwise.
                             - Existent folder: writes a new file
                               `<ref-file>.sync<.ext>` in that folder; --force
                               required if that file exists.

      -f, --force            Overwrite excel-file(s) and create any missing
                             intermediate folders.
      --prefix-cols          Prefix all synced column names with their source
                             sheet-names. By default, only clashing column-names are
                             prefixed.
      --no-clone             Do not clone excel-sheets contained in <ref-table>
                             workbook into output.
      --interp=<method>      Interpolation method used in the resampling for all
                             signals [default: linear]:
                             'linear', 'nearest', 'zero', 'slinear', 'quadratic',
                             'cubic' are passed to `scipy.interpolate.interp1d`.
                             'spline' and 'polynomial' require also to specify an
                             order (int), e.g. `--interp=spline3`.
                             'pchip' and 'akima' are wrappers around the scipy
                             interpolation methods of similar names.
                             'integral' is respecting the signal integral.

      -i=<label=interp>      Interpolation method used in the resampling for a
                             signal with a specific label
                             (e.g., `-i alternator_currents=integral`).
      -l, --interp-methods   List of all interpolation methods that can be used in
                             the resampling.
      --cycle=<cycle>        If set (e.g., --cycle=nedc.manual), the <ref-table> is
                             populated with the theoretical velocity profile.
                             Options: 'nedc.manual', 'nedc.automatic',
                             'wltp.class1', 'wltp.class2', 'wltp.class3a', and
                             'wltp.class3b'.

      <excel-file-path>      Output file.

    Miscellaneous:
      -h, --help             Show this help message and exit.
      -V, --version          Print version of the program, with --verbose
                             list release-date and installation details.
      -v, --verbose          Print more verbosely messages - overridden by --logconf.
      -q, --quiet            Print less verbosely messages (warnings) - overridden by --logconf.
      --logconf=<conf-file>  Path to a logging-configuration file, according to:
                               https://docs.python.org/3/library/logging.config.html#configuration-file-format
                             If the file-extension is '.yaml' or '.yml', it reads a dict-schema from YAML:
                               https://docs.python.org/3/library/logging.config.html#logging-config-dictschema

    * For xl-refs see: https://pandalone.readthedocs.org/en/latest/reference.html#module-pandalone.xleash

    SUB-COMMANDS:
        template             Generate "empty" input-file for the `datasync` cmd as
                             <excel-file-path>.


    Examples::

        ## Read the full contents from all `wbook.xlsx` sheets as tables and
        ## sync their columns using the table from the 1st sheet as reference:
        datasync times velocities folder/Book.xlsx

        ## Sync `Sheet1` using `Sheet3` as reference:
        datasync times velocities wbook.xlsx#Sheet3!  Sheet1!

        ## The same as above but with integers used to index excel-sheets.
        ## NOTE that sheet-indices are zero based!
        datasync times velocities wbook.xlsx#2! 0

        ## Complex Xlr-ref example:
        ## Read the table in sheet2 of wbook-2 starting at D5 cell
        ## or more Down 'n Right if that was empty, till Down n Right,
        ## and sync this based on 1st sheet of wbook-1:
        datasync times velocities wbook-1.xlsx  wbook-2.xlsx#0!D5(DR):..(DR)

        ## Typical usage for CO2MPAS velocity time-series from Dyno and OBD
        ## (the ref sheet contains the theoretical velocity profile):
        datasync template --cycle wltp.class3b template.xlsx
        datasync -O ./output times velocities template.xlsx#ref! dyno obd -i alternator_currents=integral -i battery_currents=integral

Datasync input template
~~~~~~~~~~~~~~~~~~~~~~~
The sub-command ``datasync`` accepts a single **input-excel-file**.
You can download an *empty* input excel-file from the GUI or you can use the
``template`` sub-command:

.. image:: _static/Co2mpasALLINONE-Datasync_Template.gif
   :scale: 75%
   :alt: datasync template
   :align: center

Or you can create an empty datasync template-file (e.g., ``datasync.xlsx``)
inside the *sync-folder* with the ``template`` sub-command::

    $ datasync template sync/datasync.xlsx --cycle wltp.class3b -f
    2016-11-14 17:14:00,919: INFO:__main__:Creating INPUT-TEMPLATE file 'sync/datasync.xlsx'...

All sheets must share 2 common columns ``times`` and ``velocities`` (for
datasync cmd are ``<x-label>`` and ``<y-label>``). These describe the reference
signal that is used to synchronize the data.

The ``ref`` sheet (``<ref-table>``) is considered to contain the "theoretical"
profile, while other sheets (``dyno`` and ``obd``, i.e. ``<sync-table>`` for
datasync cmd) contains the data to synchronize and resample.

Run datasync
~~~~~~~~~~~~
Fill the dyno and obd sheet with the raw data. Then, you can synchronize the
data, using the GUI as follows:

.. image:: _static/Co2mpasALLINONE-Datasync_Run.gif
   :scale: 75%
   :alt: datasync
   :align: center

Or you can synchronize the data with the ``datasync`` command::

    datasync times velocities template.xlsx#ref! dyno obd -i alternator_currents=integral -i battery_currents=integral

.. note::
   The synchronized signals are added to the reference sheet (e.g., ``ref``).

   - *synchronization* is based on the *fourier transform*;
   - *resampling* is performed with a specific interpolation method.

   All tables are read from excel-sheets using the `xl-ref syntax
   <https://pandalone.readthedocs.org/en/latest/reference.html#module-pandalone.xleash>`_.


Run batch
---------
The default sub-command (``batch``) accepts either a single **input-excel-file**
or a folder with multiple input-files for each vehicle, and generates a
**summary-excel-file** aggregating the major result-values from these vehicles,
and (optionally) multiple **output-excel-files** for each vehicle run.

To run all demo-files (note, it might take considerable time), you can use the
GUI as follows:

.. image:: _static/Co2mpasALLINONE-Batch_Run.gif
   :scale: 75%
   :alt: |co2mpas| batch
   :align: center

.. note:: the file ``co2mpas_simplan.xlsx`` has the ``flag.engineering_mode``
   set to ``True``, because it contains a "simulation-plan" with non declaration
   data.

Or you can run |co2mpas| with the ``batch`` sub-command::

   $ co2mpas batch input -O output
   2016-11-15 17:00:31,286: INFO:co2mpas_main:Processing ['../input'] --> '../output'...
     0%|          | 0/11 [00:00<?, ?it/s]: Processing ../input\co2mpas_demo-0.xlsx
   ...
   ...
   Done! [527.420557 sec]

.. Note::
  For demonstration purposes, some some of the actual models will fail;
  check the *summary file*.

Run Type-Approval (``ta``) command
----------------------------------
The Type Approval command simulates the NEDC fuel consumption and CO2 emission
of the given vehicle using just the required `declaration inputs
<https://github.com/JRCSTU/CO2MPAS-TA/wiki/TA_compulsory_inputs>`_ (marked as
compulsory inputs in input file version >= 2.2.5) and produces an NEDC
prediction. If |co2mpas| finds some extra input it will raise a warning and it
will not produce any result. The type approval command is the |co2mpas| running
mode that is fully aligned to the WLTP-NEDC correlation `Regulation
<http://ec.europa.eu/transparency/regcomitology/index.cfm?do=search.documentdeta
il&gYsYfQyLRa3DqHm8YKXObaxj0Is1LmebRoBfg8saKszVqHZGdIwy2rS97ztb5t8b>`_.


The sub-command ``ta`` accepts either a single **input-excel-file** or a folder
with multiple input-files for each vehicle, and generates a
**summary-excel-file** aggregating the major result-values from these vehicles,
and multiple **output-excel-files** for each vehicle run.

.. note::
   The user can insert just the input files and the output folder.

To run the type approval command you can use the GUI as follows:

.. image:: _static/Co2mpasALLINONE-TA_Run.gif
   :scale: 75%
   :alt: |co2mpas| ta
   :align: center

Or you can run |co2mpas| with the ``ta`` sub-command::

   $ co2mpas ta input -O output
   2016-11-15 17:00:31,286: INFO:co2mpas_main:Processing ['../input'] --> '../output'...
     0%|          | 0/1 [00:00<?, ?it/s]: Processing ../input\co2mpas_demo-0.xlsx
   ...
   ...
   Done! [51.6874 sec]

Output files
------------
The output-files produced on each run are the following:

- One file per vehicle, named as ``<timestamp>-<inp-fname>.xls``:
  This file contains all inputs and calculation results for each vehicle
  contained in the batch-run: scalar-parameters and time series for target,
  calibration and prediction phases, for all cycles.
  In addition, the file contains all the specific submodel-functions that
  generated the results, a comparison summary, and information on the python
  libraries installed on the system (for investigating reproducibility issues).

- A Summary-file named as ``<timestamp>-summary.xls``:
  Major |CO2| emissions values, optimized |CO2| parameters values and
  success/fail flags of |co2mpas| submodels for all vehicles in the batch-run.


Custom output xl-files as templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You may have defined customized xl-files for summarizing time-series and
scalar parameters. To have |co2mpas| fill those "output-template" files with
its results, execute it with the ``-D flag.output_template=file/path.xlsx``
option.

To create/modify one output-template yourself, do the following:

1. Open a typical |co2mpas| output-file for some vehicle.

2. Add one or more sheets and specify/referring |co2mpas| result-data using
   `named-ranges <https://www.google.it/search?q=excel+named-ranges>`_.

   .. Warning::
      Do not use simple/absolute excel references (e.g. ``=B2``).
      Use excel functions (indirect, lookup, offset, etc.) and array-functions
      together with string references to the named ranges
      (e.g. ``=indirect("output.prediction.nedc_h.pa!_co2_emission_value")``).

3. (Optional) Delete the old sheets and save your file.

4. Use that file together with the ``-D flag.output_template=file/path.xlsx``
   argument.


Simulation plan
---------------
It is possible to launch |co2mpas| once, and have it run the model multiple
times, with variations on the input-data, all contained in a single
(or more) input file(s).

The data for **base model** are contained in the regular sheets, and any
variations are provided in additional sheets which names starting with
the ``plan.`` prefix.
These sheets must contain a table where each row is a single simulation,
while the columns names are the parameters that the user want to vary.
The columns of these tables can contain the following special names:

- **id**: Identifies the variation id.
- **base**: this is a file path of a |co2mpas| excel input, this model will be
  used as new base vehicle.
- **run_base**: this is a boolean. If true the base model results are computed
  and stored, otherwise the data are just loaded.

You can use the GUI as follows:

.. image:: _static/Co2mpasALLINONE-Plan_Run.gif
   :scale: 75%
   :alt: |co2mpas| batch simulation plan
   :align: center

.. note:: the file ``co2mpas_simplan.xlsx`` has the ``flag.engineering_mode``
   set to ``True``, because it contains a "simulation-plan" with non declaration
   data.

Or you can run |co2mpas| with the ``batch`` sub-command::

   $ co2mpas batch input/co2mpas_simplan.xlsx -O output
   2016-11-15 17:00:31,286: INFO:co2mpas_main:Processing ['../input/co2mpas_simplan.xlsx'] --> '../output'...
     0%|          | 0/4 [00:00<?, ?it/s]: Processing ../input\co2mpas_simplan.xlsx
   ...
   ...
   Done! [180.4692 sec]



Debugging and investigating results
-----------------------------------

- Make sure that you have installed `graphviz`, and when running the simulation,
  append also the ``-D flag.plot_workflow=True`` option.

  .. code-block:: console

        $ co2mpas batch bad-file.xlsx -D flag.plot_workflow=True

  A browser tab will open at the end with the nodes processed.

- Use the ``modelgraph`` sub-command to plot the offending model (or just
  out of curiosity).  For instance:

  .. code-block:: console

        $ co2mpas modelgraph co2mpas.model.physical.wheels.wheels

  .. module:: co2mpas

  .. dispatcher:: d
     :alt: Flow-diagram Wheel-to-Engine speed ratio calculations.
     :height: 240
     :width: 320

     >>> import co2mpas
     >>> d = co2mpas.model.physical.wheels.wheels()

- Inspect the functions mentioned in the workflow and models and search them
  in `CO2MPAS documentation <http://co2mpas.io/>`_ ensuring you are
  visiting the documents for the actual version you are using.


