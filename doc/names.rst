#######################
Data naming conventions
#######################
This section describes the data naming convention used in |co2mpas| input
template to construct the *data-values tree*, i.e. the input of the simulation
model. There are two naming conventions in the excel file:

- the **sheet-name** name is composed of 5 parts, all optional, but at least
  one of the first 4 must be present::

                  base.input.precondition.WLTP-H.ts
                  └┬─┘ └─┬─┘ └────┬─────┘ └─┬──┘ └┬┘
      scope────────┘     │        │         │     │
      usage──────────────┘        │         │     │
      stage───────────────────────┘         │     │
      cycle─────────────────────────────────┘     │
      sheet_type(optional)────────────────────────┘

- the **data-name** consists of 5 parts, the ``parameter`` part is mandatory,
  and the last 2 parts can be reversed::

                     plan.target.prediction.vehicle_mass.WLTP-H
                     └┬─┘ └─┬─┘ └────┬────┘ └────┬─────┘ └──┬─┘
      scope(optional)─┘     │        │           │          │
      usage(optional)───────┘        │           │          │
      stage(optional)────────────────┘           │          │
      parameter──────────────────────────────────┘          │
      cycle(optional)───────────────────────────────────────┘

.. note::
   - The dot(``.``) may be replaced by space.
   - The **usage** and **stage** parts may end with an ``s``, denoting plural,
     and are not case-insensitive, e.g. ``Inputs`` sheet.


Description of the name-parts
=============================
The options of each name part are described in the following sections.

scope
-----

   - ``base`` [default]: input to |co2mpas| simulation model.
   - ``dice``: input for dice.
   - ``meta``: meta data of the vehicle test.
   - ``plan``: data used to run a simulation plan and overwrite the inputs of
     the simulation model.

usage
-----

   - ``input`` [default]: input to calibration and/or to prediction models.
   - ``data``: intermediate data between calibration and prediction models
     (see :ref: `model`).
   - ``output``: |co2mpas| precondition, calibration, and prediction results.
   - ``target``: reference-values (**NOT USED NEITHER IN CALIBRATION NOR IN**
     **PREDICTION**) to be compared with the |co2mpas| results.

stage
-----

   - ``calibration`` [default]: calibration data.
   - ``prediction`` [default, if **cycle** in ``nedc``]: prediction data.

cycle
-----

   - ``nedc-h``: *NEDC High* cycle data.
   - ``nedc-l``: *NEDC Low* cycle data.
   - ``wltp-h``: *WLTP High* cycle data.
   - ``wltp-l``: *WLTP Low* cycle data.
   - ``nedc`` [default]: shortcut to set values for ``nedc-h`` & ``nedc-l``.
   - ``wltp`` [default]: shortcut to set values for ``wltp-h`` & ``wltp-l``.
   - ``all-h``: shortcut to set values for ``nedc-h`` and ``wltp-h``.
   - ``all-l``: shortcut to set values for ``nedc-l`` and ``wltp-l``.
   - ``all``: shortcut to set values for ``nedc`` and ``wltp``.

param
-----
Any data-name (e.g. ``vehicle_mass``) used in the physical model.

sheet_type
----------
There are three sheet types, which are parsed according to their contained data:

   - **pl** [parsed range is ``#A1:__``]: table of scalar and time-depended
     values used into the simulation plan as a variation from the base model.
   - **pa** [parsed range is ``#B2:C_``]: scalar or not time-depended
     values (e.g. ``r_dynamic``, ``gear_box_ratios``, ``full_load_speeds``).
   - **ts** [parsed range is ``#A2:__``]: time-depended values (e.g.
     ``times``, ``velocities``, ``gears``). Columns without values are skipped.
     **COLUMNS MUST HAVE THE SAME LENGTH!**

.. note::
   If it is not defined, the default value follows these rules:
   When **scope** is ``plan``, the sheet is parsed as **pl**.
   If **scope** is ``base`` and **cycle** is missing in the **sheet-name**,
   the sheet is parsed as **pa**, otherwise, it is parsed as **ts**.

Simulation plan
===============
Variations to the base model must be inserted in the provided additional sheets,
characterized by the plan. prefix. These sheets consist of tables composed by
rows, each row is a single simulation, and columns, the parameters that the user
wishes to vary; the columns can contain the following special names:

- **id**: Identifies the variation id.
- **base**: It is the file path of a CO2MPAS excel input. The data are used as
  new base vehicle.
- **run_base**: If TRUE [default] the base model results are computed and stored,
  otherwise the data are just loaded.

.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
