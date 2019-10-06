.. _start-quick:

.. image:: doc/_static/CO2MPAS_banner.png
   :width: 640

######################################################################
|co2mpas|: Vehicle simulator predicting NEDC |CO2| emissions from WLTP
######################################################################

:official:      | `3.0.X <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2mpas-v3.0.0>`_: from 01-Feb-2019 to 20-Oct-2019
                | `4.1.X <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2mpas-r4.1.2>`_: from 20-Oct-2019

:release:       4.1.0
:rel_date:      2019-10-06 21:00:00
:home:          http://co2mpas.io/
:repository:    https://github.com/JRCSTU/CO2MPAS-TA
:pypi-repo:     https://pypi.org/project/co2mpas/
:keywords:      CO2, fuel-consumption, WLTP, NEDC, vehicle, automotive,
                EU, JRC, IET, STU, correlation, back-translation, policy,
                monitoring, M1, N1, simulator, engineering, scientific
:developers:    .. include:: AUTHORS.rst
:copyright:     2015-2019 European Commission (`JRC <https://ec.europa.eu/jrc/>`_)
:license:       `EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>`_

.. _start-pypi:
.. _start-intro:

What is |co2mpas|?
==================
|co2mpas| is backward-looking longitudinal-dynamics |CO2| and fuel-consumption
simulator for light-duty M1 & N1 vehicles (cars and vans), specially crafted to
*estimate the CO2 emissions of vehicles undergoing NEDC* testing based on the
emissions produced *WLTP testing* during :term:`type-approval`, according to
the :term:`EU legislation`\s *1152/EUR/2017 and 1153/EUR/2017* (see `History`_
section, below).

It is an open-source project
(`EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>`_) developed for
Python-3.6+ using :term:`WinPython` & :term:`Anaconda` under Windows 7, Anaconda
under MacOS, and standard python environment & Anaconda under Linux. It runs
either as a *console command* or as a *desktop GUI application*, and it uses
Excel-files or pure python structures (dictionary and lists) for its input &
output data.

History
-------
The *European Commission* has introduced the *WLTP* as test procedure for the
type I test of the European type-approval of Light-duty vehicles as of
September 2017. Its introduction has required the adaptation of |CO2|
certification and monitoring procedures set by European regulations (443/2009,
510/2011, 1152/EUR/2017 and 1153/EUR/2017). European Commission’s *Joint
Research Centre* (JRC) has been assigned the development of this vehicle
simulator to facilitate this adaptation.

The European Regulation setting the conditions for using |co2mpas| can be
found in `the Comitology Register
<http://ec.europa.eu/transparency/regcomitology/index.cfm?do=search.documentdetail&dos_id=0&ds_id=45835&version=2>`_
after its adoption by the *Climate Change Committee* which took place on
June 23, 2016 and its 2nd vote for modifications, on April 27, 2017.

Installation
============
There are two installation procedures that can be follow.

Ordinary (for developers and/or researchers)
--------------------------------------------
To install it use (with root privileges):

.. code-block:: console

    $ pip install co2mpas

Or download the last git version and use (with root privileges):

.. code-block:: console

    $ python setup.py install


Install extras
^^^^^^^^^^^^^^
Some additional functionality is enabled installing the following extras:
- cli: enables the command line interface.
- sync: enables the time series synchronization tool (i.e., `syncing` previously
  named `datasync`).
- gui: enables the graphical user interface.
- plot: enables to plot the |co2mpas| model and the workflow of each run.
- io: enables to read/write excel files.
- dice: enables the Type Approval mode.
- driver: enables the driver model.

To install co2mpas and all extras, do:

.. code-block:: console

    $ pip install co2mpas[all]

Official (for type approval)
----------------------------
You may find usage Guidelines in the wiki:
https://github.com/JRCSTU/CO2MPAS-TA/wiki/CO2MPAS-user-guidelines

Requirements
^^^^^^^^^^^^
- These are the  minimum IT requirements for the Computer to run CO2MPAS & DICE:
- 64-bit Intel or AMD processor (x86_64, aka x64, aka AMD64);
- Microsoft Windows 7, or later;
- 4 GB RAM (more recommended);
- 2.4 GB hard disk storage for extracting the software, more space for the input/output files;
- Execution-rights to the installation folder (but no Admin-rights).
- An e-mail account to send & receive DICE e-mails;
- Unhindered HTTP/HTTPS  web-access (no firewall on ports 80, 443);
  or access through HTTP Proxy;
- (optional) Excel, to view & edit simulation’s input and output files;
- (optional) GitHub account to submit and resolve issues.


.. _end-quick:

Quick Start
===========

.. code-block:: console
    ## Create a template excel-file for inputs:
    $ co2mpas template vehicle_1.xlsx

    ###################################################
    ## Edit generated `./input/vehicle_1.xlsx` file. ##
    ###################################################

    ## Launch GUI, select the edited template as Input, and click `Run`:
    $ co2mpas gui

And the GUI pops up:

.. image:: _static/CO2MPAS_GUI.png
   :width: 640

Command-line alternatives:

.. code-block:: console


    ## To synchronize the Dyno and OBD data with the theoretical:
    $ datasync template --cycle wltp.class3b template.xlsx
    $ datasync -O ./output times velocities template.xlsx#ref! dyno obd -i alternator_currents=integral -i battery_currents=integral

    ## To generate demo-files in you current folder:
    $ co2mpas demo

    ## Run batch simulator on the first demo.
    $ co2mpas batch co2mpas_demo-0.xlsx

    #########################################################
    ## Inspect generated results in you current dicectory. ##
    #########################################################

    ## Run type approval command on your data.
    $ co2mpas ta vehicle_1.xlsx -O output

    ## Start using the DICE command-line tool:
    $ co2dice --help


Model
=====
Execution Model
---------------
The execution of |co2mpas| model for a single vehicle is a stepwise procedure
of 3 stages: ``precondition``, ``calibration``, and ``prediction``.
These are invoked repeatedly, and subsequently combined, for the various cycles,
as shown in the "active" flow-diagram of the execution, below:

.. module:: co2mpas

.. dispatcher:: dsp
   :opt: depth=-1
   :alt: Flow-diagram of the execution of various Stages and Cycles sub-models.
   :width: 640

   >>> from co2mpas.core.model import dsp
   >>> dsp = dsp.register(memo={})

.. Tip:: The models in the diagram are nested; explore by clicking on them.

1. **Precondition:** identifies the initial state of the vehicle by running
   a preconditioning *WLTP* cycle, before running the *WLTP-H* and *WLTP-L*
   cycles.
   The inputs are defined by the ``input.precondition.wltp_p`` node,
   while the outputs are stored in ``output.precondition.wltp_p``.

2. **Calibration:** the scope of the stage is to identify, calibrate and select
   (see next sections) the best physical models from the WLTP-H and WLTP-L
   inputs (``input.calibration.wltp_x``).
   If some of the inputs needed to calibrate the physical models are not
   provided (e.g. ``initial_state_of_charge``), the model will select the
   missing ones from precondition-stage's outputs
   (``output.precondition.wltp_p``).
   Note that all data provided in ``input.calibration.wltp_x`` overwrite those
   in ``output.precondition.wltp_p``.

3. **Prediction:** executed for the NEDC and as well as for the WLTP-H and
   WLTP-L cycles. All predictions use the ``calibrated_models``. The inputs to
   predict the cycles are defined by the user in ``input.prediction.xxx`` nodes.
   If some or all inputs for the prediction of WLTP-H and WLTP-L cycles are not
   provided, the model will select from ```output.calibration.wltp_x`` nodes a
   minimum set required to predict |CO2| emissions.

.. _excel-model:

Excel input: data naming conventions
------------------------------------
This section describes the data naming convention used in the |co2mpas| template
(``.xlsx`` file). In it, the names used as **sheet-names**, **parameter-names**
and **column-names** are "sensitive", in the sense that they construct a
*data-values tree* which is then fed into into the simulation model as input.
These names are split in "parts", as explained below with examples:

- **sheet-names** parts::

                  base.input.precondition.WLTP-H.ts
                  └┬─┘ └─┬─┘ └────┬─────┘ └─┬──┘ └┬┘
      scope────────┘     │        │         │     │
      usage──────────────┘        │         │     │
      stage───────────────────────┘         │     │
      cycle─────────────────────────────────┘     │
      sheet_type──────────────────────────────────┘


  First 4 parts above are optional, but at least one of them must be present on
  a **sheet-name**; those parts are then used as defaults for all
  **parameter-names** contained in that sheet. **type** is optional and specify
  the type of sheet.

- **parameter-names**/**columns-names** parts::

                     plan.target.prediction.vehicle_mass.WLTP-H
                     └┬─┘ └─┬─┘ └────┬────┘ └────┬─────┘ └──┬─┘
      scope(optional)─┘     │        │           │          │
      usage(optional)───────┘        │           │          │
      stage(optional)────────────────┘           │          │
      parameter──────────────────────────────────┘          │
      cycle(optional)───────────────────────────────────────┘

  OR with the last 2 parts reversed::

                    plan.target.prediction.WLTP-H.vehicle_mass
                                           └──┬─┘ └────┬─────┘
      cycle(optional)─────────────────────────┘        │
      parameter────────────────────────────────────────┘

.. note::
   - The dot(``.``) may be replaced by space.
   - The **usage** and **stage** parts may end with an ``s``, denoting plural,
     and are not case-insensitive, e.g. ``Inputs``.


Description of the name-parts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **scope:**

   - ``base`` [default]: values provided by the user as input to |co2mpas|.
   - ``plan``: values selected (see previous section) to calibrate the models
     and to predict the |CO2| emission.
   - ``flag``: values provided by the user as input to ``run_base`` and
     ``run_plan`` models.
   - ``meta``: values provided by the user as meta data of the vehicle test.

2. **usage:**

   - ``input`` [default]: values provided by the user as input to |co2mpas|.
   - ``data``: values selected (see previous section) to calibrate the models
     and to predict the |CO2| emission.
   - ``output``: |co2mpas| precondition, calibration, and prediction results.
   - ``target``: reference-values (**NOT USED IN CALIBRATION OR PREDICTION**) to
     be compared with the |co2mpas| results. This comparison is performed in the
     *report* sub-model by ``compare_outputs_vs_targets()`` function.
   - ``config``: values provided by the user that modify the ``model_selector``.

3. **stage:**

   - ``precondition`` [imposed when: ``wltp-p`` is specified as **cycle**]:
     data related to the precondition stage.
   - ``calibration`` [default]: data related to the calibration stage.
   - ``prediction`` [imposed when: ``nedc`` is specified as **cycle**]:
     data related to the prediction stage.
   - ``selector``: data related to the model selection stage.

4. **cycle:**

   - ``nedc-h``: data related to the *NEDC High* cycle.
   - ``nedc-l``: data related to the *NEDC Low* cycle.
   - ``wltp-h``: data related to the *WLTP High* cycle.
   - ``wltp-l``: data related to the *WLTP Low* cycle.
   - ``wltp-precon``: data related to the preconditioning *WLTP* cycle.
   - ``wltp-p``: is a shortcut of ``wltp-precon``.
   - ``nedc`` [default]: is a shortcut to set values for both ``nedc-h`` and
     ``nedc-l`` cycles.
   - ``wltp`` [default]: is a shortcut to set values for both ``wltp-h`` and
     ``wltp-l`` cycles.
   - ``all``: is a shortcut to set values for ``nedc``, ``wltp``,
     and ``wltp-p`` cycles.

5. **param:** any data node name (e.g. ``vehicle_mass``) used in the physical
   model.

6. **sheet_type:** there are three sheet types, which are parsed according to
   their contained data:

   - **pl** [parsed range is ``#A1:__``]: table of scalar and time-depended
     values used into the simulation plan as variation from the base model.
   - **pa** [parsed range is ``#B2:C_``]: scalar or not time-depended
     values (e.g. ``r_dynamic``, ``gear_box_ratios``, ``full_load_speeds``).
   - **ts** [parsed range is ``#A2:__``]: time-depended values (e.g.
     ``times``, ``velocities``, ``gears``). Columns without values are skipped.
     **COLUMNS MUST HAVE THE SAME LENGTH!**

   ..note:: If it is not defined, the default value follows these rules:
     When **scope** is ``plan``, the sheet is parsed as **pl**.
     If **scope** is ``base`` and **cycle** is missing in the **sheet-name**,
     the sheet is parsed as **pa**, otherwise it is parsed as **ts**.

Calibrated Physical Models
--------------------------
There are potentially eight models calibrated from input scalar-values and
time-series (see :doc:`reference`):

1. *AT_model*,
2. *electrics_model*,
3. *clutch_torque_converter_model*,
4. *co2_params*,
5. *after_treatment_model*,
6. *engine_coolant_temperature_model*,
7. *engine_speed_model*, and
8. *control_model*.

Each model is calibrated separately over *WLTP_H* and *WLTP_L*.
A model can contain one or several functions predicting different quantities.
For example, the electric_model contains the following functions/data:

- *alternator_current_model*,
- *alternator_status_model*,
- *electric_load*,
- *max_battery_charging_current*,
- *start_demand*.

These functions/data are calibrated/estimated based on the provided input
(in the particular case: *alternator current*, *battery current*, and
*initial SOC*) over both cycles, assuming that data for both WLTP_H and WLTP_L
are provided.

.. Note::
    The ``co2_params`` model has a third possible calibration configuration
    (so called `ALL`) using data from both WLTP_H and WLTP_L combined
    (when both are present).


Model selection
---------------

.. Note::
   Since *v1.4.1-Rally*, this part of the model remains disabled,
   unless the ``flag.use_selector`` is true.

For the type approval mode the selection is fixed. The criteria is to select the
models calibrated from *WLTP_H* to predict *WLTP_H* and *NEDC_H*; and
from *WLTP_L* to predict *WLTP_L* and *NEDC_L*.

While for the engineering mode the automatic selection can be enabled adding
`-D flag.use_selector=True` to the batch command.
Then to select which is the best calibration
(from *WLTP_H* or *WLTP_L* or *ALL*) to be used in the prediction phase, the
results of each stage are compared against the provided input data (used in the
calibration).
The calibrated models are THEN used to recalculate (predict) the inputs of the
*WLTP_H* and *WLTP_L* cycles. A **score** (weighted average of all computed
metrics) is attributed to each calibration of each model as a result of this
comparison.

.. Note::
    The overall score attributed to a specific calibration of a model is
    the average score achieved when compared against each one of the input
    cycles (*WLTP_H* and *WLTP_L*).

    For example, the score of `electric_model` calibrated based on *WLTP_H*
    when predicting *WLTP_H* is 20, and when predicting *WLTP_L* is 14.
    In this case the overall score of the the `electric_model` calibrated
    based on *WLTP_H* is 17. Assuming that the calibration of the same model
    over *WLTP_L* was 18 and 12 respectively, this would give an overall score
    of 15.

    In this case the second calibration (*WLTP_L*) would be chosen for
    predicting the NEDC.

In addition to the above, a success flag is defined according to
upper or lower limits of scores which have been defined empirically by the JRC.
If a model fails these limits, priority is then given to a model that succeeds,
even if it has achieved a worse score.

The following table describes the scores, targets, and metrics for each model:

.. image:: _static/CO2MPAS_model_score_targets_limits.png
   :width: 600 px
   :align: center

.. _end-intro:
.. _start-badges:

.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. |clink| replace:: *Clink*
.. _clink: http://mridgers.github.io/clink/
.. |EUPL| replace:: *EUPL*
.. _EUPL: https://joinup.ec.europa.eu/page/eupl-text-11-12

.. _end-badges:
