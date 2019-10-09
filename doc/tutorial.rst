#########
Tutorials
#########
This section explains the functionalities of |co2mpas| GUI through some video
tutorials:

- `Inputs`_

  - `Get input template`_
  - `Synchronize time-series`_
  - `Download demo files`_
- `Run`_

  - `Type approval mode`_
  - `Engineering mode`_
  - `Simulation plan`_
- `Output results`_

  - `Output files`_

    - `Custom output xl-files as templates`_
  - `Debugging and investigating results`_

Inputs
======
This section shows the utilities to generate and populate the |co2mpas| input
file.

Get input template
------------------
Check the video to see how to download an empty input excel-file. The generated
file contains the instructions on how to fill the required inputs. For more
information use the command ``co2mpas template -h``.

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/download_input_template.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

Synchronize time-series
-----------------------
If you have time-series not well synchronized and/or with different sampling
rates, the model might fail. As aid tool, you may use the ``syncing`` tool to
"synchronize" and "re-sample" your data. To use the tool you should execute the
following steps:

- Generate and download an *empty* input excel-file (see the video).
  For more information use the command ``co2mpas syncing template -h``.

  .. raw:: html

      <p>
        <video width="100%" height="%100" controls playsinline preload="metadata">
          <source src="_static/video/download_datasync_template.mp4" type="video/mp4">
        Your browser does not support the video tag.
        </video>
      </p>

  .. note::
     All sheets must contain values for columns ``times`` and ``velocities``,
     because they are the reference signals used to synchronize the data with
     the theoretical velocity profile.

- Run data synchronization, see the video.
  For more information use the command ``co2mpas syncing sync -h``.

  .. raw:: html

      <p>
        <video width="100%" height="%100" controls playsinline preload="metadata">
          <source src="_static/video/run_datasync.mp4" type="video/mp4">
        Your browser does not support the video tag.
        </video>
      </p>

.. note::
   The synchronized signals are saved in to the ``synced`` sheet.


Download demo files
-------------------
|co2mpas| contains 3 demo-files that can be used as starting point to try out:

1. *co2mpas_conventional.xlsx*: conventional vehicle,
2. *co2mpas_simplan.xlsx*: sample simulation plan,
3. *co2mpas_hybrid.xlsx*: hybrid parallel vehicle.

Check the video to see how to download them. For more information use the
command ``co2mpas demo -h``.

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/name.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

Run
===
This section shows the three ways to run |co2mpas|. For more information use the
command ``co2mpas run -h``.

Type approval mode
------------------
The Type Approval command simulates the NEDC fuel consumption and CO2 emission
of the given vehicle using just the required declaration inputs and produces an
NEDC prediction. If |co2mpas| finds some extra input or there is some missing it
will raise a warning and it will not produce any result. The type approval
command is fully aligned to the WLTP-NEDC correlation `Regulation
<https://eur-lex.europa.eu/legal-content/it/TXT/?uri=CELEX%3A32017R1151>`_.

To successfully run |co2mpas| in type approval mode, see the following steps:

.. _upload_file:

1. Upload your file/s (multiple file are accepted):

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/run_simulation_TA_1.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

2. Switch TA mode ON and click run:

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/run_simulation_TA_2.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

.. _download_results:

3. Get the results:

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/run_simulation_TA_3.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

Engineering mode
----------------
This section explain how to run |co2mpas| in engineering mode:

1. Upload excel file/s (see :ref:`previous step <upload_file>`),
2. Switch TA mode ON and click run:

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/run_simulation_2.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

.. note:: 5 advanced options are available: **use only declaration mode**,
    **hard validation**, **enable selector**, **only summary**, and
    **use custom configuration file**. Flag the box to activate them.

    .. image:: _static/advanced_options.png
       :width: 100%
       :alt: |co2mpas| advanced options
       :align: center

3. Get the results  (see :ref:`previous step <download_results>`).

Simulation plan
---------------
It is possible to launch |co2mpas| once, and have it run the model multiple
times, with variations on the input-data, all contained in a single
(or more) input file(s).

The data for **base model** are contained in the regular sheets, and any
variations are provided in additional sheets which names starting with
the ``plan.`` prefix.
These sheets must contain a table where each row is a single simulation,
while the columns names are the parameters that the user wishes to vary.
The columns of these tables can contain the following special names:

- **id**: Identifies the variation id.
- **base**: It is the file path of a |co2mpas| excel input. The data are used as
  new base vehicle.
- **run_base**: If TRUE [default] the base model results are computed
  and stored, otherwise the data are just loaded.

To run the simulation plan you can follow the `steps in previous section
<Engineering mode>`_.

.. note::
    The simulation plan cannot run in type-approval mode.

Output results
==============
This section shows the three ways to run |co2mpas|. For more information use the
command ``co2mpas run -h``.

Output files
------------
The output-files produced every run are the following:

- One zip folder per vehicle, named as ``<timestamp>-<ip-name>. co2mpas.zip``.
  This folder contains 4 files:

1. co2mpas.hash    (.txt file)

2. co2mpas.input   (.xlsx file)

3. co2mpas.output  (.xlsx file)

4. co2mpas.ta      (.TA file)

  **co2mpas.output** presents the results of |co2mpas| calculations:
  scalar-parameters and time series for target, calibration and prediction
  phases, for all cycles.


- A Summary-file named as ``<timestamp>-summary.xlsx``:
  Major |CO2| emissions values, optimized |CO2| parameters values and
  success/fail flags of |co2mpas| submodels for all vehicles run.


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

  .. dispatcher:: dsp
     :alt: Flow-diagram Wheel-to-Engine speed ratio calculations.
     :height: 240
     :width: 320

     >>> from co2mpas.core.model.physical.wheels import dsp
     >>> dsp = dsp.register(memo={})



.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
