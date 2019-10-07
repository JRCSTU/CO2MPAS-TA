#####
Usage
#####
This section explains |co2mpas|' functionalities through video tutorials.

.. contents::

Inputs
======

Get Input Template
------------------
You can download an empty input excel-file from the GUI. Check the video to
see how:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/download_input_template.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>


.. admonition:: Command Line Interface

   You can create the template using the command ``co2mpas template`` in the
   console as follows::

       $ co2mpas template ./template.xlsx -TT input
       INFO:co2mpas:CO2MPAS input template written into (./template.xlsx).
       INFO:co2mpas:Done! [0.01 sec]

   For more information about the command ``co2mpas template``, you can type
   the command ``co2mpas template -h``.


The generated file contains descriptions to help you populate it with vehicle
data. For more details on the fields, read the :ref:`glossary` section.

.. image:: _static/input_template.png
   :scale: 40%
   :alt: input template
   :align: center

Download demo files
-------------------
Co2mpas contains 3 demo-files that can be used as starting point to try out:

1. *co2mpas_conventional* ---> for conventional vehicles

2. *co2mpas_simplan* ---> as simulation plan

3. *co2mpas_hybrid*  ---> for hybrid vehicles

You can download them via the GUI in a folder called *co2mpas-demo*. Check the
video to see how:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/name.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>


.. admonition:: Command Line Interface

    Or you can use the console to create the demo files inside the *co2mpas_demo*
    folder with the ``demo`` sub-command::

    $ co2mpas demo ./co2mpas_demo/


Synchronize time-series
-----------------------
If you have time-series signals not synchronized and/or with different sampling
rates, the model might fail. As aid tool, you may use the ``syncing`` tool to
"synchronize" and "resample" your data.

.. admonition:: Command Line Interface

    To get the syntax of the ``syncing`` console-command, open the console and
    type::

    $ syncing --help

Download datasync input template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Check the video to see how to download an *empty* input excel-file:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/download_datasync_template.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

.. admonition:: Command Line Interface

    or you can use the ``template`` sub-command::

    $ syncing template ./datasync.xlsx

Run datasync
~~~~~~~~~~~~
The *ref* sheet (<ref-table>) should contain the “theoretical” profile, while
the other two sheets (*dyno* and *obd*, i.e. <sync-table> for datasync cmd)
consist of the data to synchronize and resample.
**All sheets must contain values for columns ``times`` and ``velocities`` ,**
because these two parameters are the reference signals used to synchronize the
data.

Fill the dyno and obd sheet with the test data. Then, check the video to see how
to synchronize:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_datasync.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

.. admonition:: Command Line Interface
    Or you can use the console to synchronize the data::

    #TODO

.. note::
   The synchronized signals are added to the reference sheet (e.g., ``ref``).

   - *synchronization* is based on the *fourier transform*;
   - *resampling* is performed with a specific interpolation method.

   All tables are read from excel-sheets using the `xl-ref syntax
   <https://pandalone.readthedocs.org/en/latest/reference.html#module-pandalone.xleash>`_.


Run
===

Run in Engineering mode
-----------------------
To successfully run |co2mpas| and download the final results, follow these 3
steps:

1. upload your file/s (multiple file are accepted)

2. press run

3. download archive

Check the video to see how to **upload a file**:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_simulation_1.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

Check the video to see how to **run the file**:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_simulation_2.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

Check the video to see how to **get your results**:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_simulation_3.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>


.. note:: 5 advanced options are available: *use only declaration mode*,
    *hard validation*, *enable selector*, *only summary*,
    *use custom configuration file*. Flag the box to activate them.

.. image:: _static/advanced_options.png
   :scale: 40%
   :alt: |co2mpas| advanced options
   :align: center

.. admonition:: Command Line Interface

    Or you can run |co2mpas| with the ``run`` sub-command::

    $ co2mpas run input -O output
        INFO:co2mpas_main:Processing ['../input'] --> '../output'...
         0%|          | 0/11 [00:00<?, ?it/s]: Processing ../input\co2mpas_demo-0.xlsx
        ...
        ...
        Done! [527.420557 sec]

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

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_simulation_3.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

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



Run in Type Approval mode
-------------------------
The Type Approval command simulates the NEDC fuel consumption and CO2 emission
of the given vehicle using just the required `declaration inputs
<https://github.com/JRCSTU/CO2MPAS-TA/wiki/TA_compulsory_inputs>`_  and produces
an NEDC prediction. If |co2mpas| finds some extra input it will raise a warning
and it will not produce any result. The type approval command is fully aligned
to the WLTP-NEDC correlation `Regulation
<https://eur-lex.europa.eu/legal-content/it/TXT/?uri=CELEX%3A32017R1151>`_.

To successfully run |co2mpas| TA and download the final results, follow these 4
steps:

1. upload your file/s (multiple file are accepted)

2. switch TA mode ON

3. press run

3. download archive

Check the video to see how to **upload a file**:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_simulation_TA_1.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>


Check the video to see how to **run in TA the file**:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_simulation_TA_2.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

Check the video to see how to **get your results**:

.. raw:: html

    <video width="100%" height="%100" controls>
      <source src="_static/run_simulation_TA_3.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

Or you can run |co2mpas| with the ``ta`` sub-command::

   $ co2mpas ta input -O output
   2016-11-15 17:00:31,286: INFO:co2mpas_main:Processing ['../input'] --> '../output'...
     0%|          | 0/1 [00:00<?, ?it/s]: Processing ../input\co2mpas_demo-0.xlsx
   ...
   ...
   Done! [51.6874 sec]


Output results
==============

Output file
------------
The output-files produced every run are the following:

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
