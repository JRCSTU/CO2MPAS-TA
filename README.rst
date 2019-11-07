.. image:: doc/_static/image/banner.png
   :width: 100%

.. _start-info:

######################################################################
|co2mpas|: Vehicle simulator predicting NEDC |CO2| emissions from WLTP
######################################################################

:official:         | `3.0.X <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/co2mpas-v3.0.0>`_: from 01-Feb-2019 to 01-Jan-2020
                   | `4.1.X <https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/v4.1.10>`_: from 31-Oct-2019
:release:          4.1.10
:rel_date:         2019-11-07 12:00:00
:home:             http://co2mpas.readthedocs.io/
:repository:       https://github.com/JRCSTU/CO2MPAS-TA
:pypi-repo:        https://pypi.org/project/co2mpas/
:keywords:         |CO2|, fuel-consumption, WLTP, NEDC, vehicle, automotive,
                   EU, JRC, IET, STU, correlation, back-translation, policy,
                   monitoring, M1, N1, simulator, engineering, scientific
:mail box:         |CO2MPAS| Team: <JRC-CO2MPAS@ec.europa.eu>
:team:             .. include:: AUTHORS.rst
:copyright:        2015-2019 European Commission (`JRC <https://ec.europa.eu/jrc/>`_)
:license:          `EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>`_

.. _end-info:
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
Python-3.6+. It runs either as a *console command* or as a
*desktop GUI application*, and it uses Excel-files or pure python structures
(dictionary and lists) for its input & output data.

History
-------
The *European Commission* has introduced the *WLTP* as the test procedure for
the type I test of the European type-approval of Light-duty vehicles as of
September 2017. Its introduction has required the adaptation of |CO2|
certification and monitoring procedures set by European regulations (443/2009,
510/2011, 1152/EUR/2017 and 1153/EUR/2017). European Commission’s *Joint
Research Centre* (JRC) has been assigned the development of this vehicle
simulator to facilitate this adaptation.

The European Regulation setting the conditions for using |co2mpas| can be
found in `the Comitology Register
<http://ec.europa.eu/transparency/regcomitology/index.cfm?do=search.documentdetail&dos_id=0&ds_id=45835&version=2>`_
after its adoption by the *Climate Change Committee* which took place on
June 23, 2016, and its 2nd vote for modifications, in April 27, 2017.

.. _end-intro:
.. _start-install:

Installation
============
Two installation procedures can be followed:

- `Official (for type approval)`_
- `Ordinary (for developers and/or researchers)`_

Official (for type approval)
----------------------------
To install |co2mpas| you have to download the
`installer <https://github.com/JRCSTU/CO2MPAS-TA/releases/download/v4.1.10/installer.exe>`_
and then execute it (see the steps shown in the video below).

.. raw:: html

    <video width="100%" height="%100" controls playsinline preload="metadata">
      <source src="_static/video/install.mp4" type="video/mp4">
      <source src="doc/_static/video/install.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>

After the installation, two icons will appear on your desktop: one for the GUI
and one for the console.

    .. image:: _static/image/icons.png
       :width: 50%
       :alt: co2mpas desktop icons
       :align: center

.. admonition:: IT Requirements

   To run |co2mpas| for Type Approval purposes (i.e |co2mpas| official) some
   minimum IT requirements are mandatory:

   - 64-bit Intel or AMD processor (x86_64, aka x64, aka AMD64),
   - Microsoft Windows 7 SP1, or later,
   - 4 GB RAM (more recommended),
   - 4 GB hard disk storage for installing the software,
   - Execution-rights to the installation folder (but no Admin-rights),
   - (optional) Excel, to view & edit simulation’s input and output files,
   - Chrome/Firefox/Safari/Edge or IExplore 10+.

Ordinary (for developers and/or researchers)
--------------------------------------------
.. _start-install-dev:

To install |co2mpas| use (with root privileges):

.. code-block:: console

    $ pip install co2mpas

Or download the latest git version and use (with root privileges):

.. code-block:: console

    $ python setup.py install


Install extras
^^^^^^^^^^^^^^
Some additional functionality is enabled installing the following extras:

- ``cli``: enables the command line interface.
- ``sync``: enables the time series synchronization tool (i.e.,
  `syncing <https://github.com/vinci1it2000/syncing>`_ previously named
  ``datasync``).
- ``gui``: enables the graphical user interface.
- ``plot``: enables to plot the |co2mpas| model and the workflow of each run.
- ``io``: enables to read/write excel files.
- ``dice``: enables the Type Approval mode.
- ``driver``: enables the driver model (currently is not available).

To install co2mpas and all extras, do:

.. code-block:: console

    $ pip install co2mpas[all]

.. _end-install-dev:
.. _end-install:
.. _start-quick:

Quick Start
===========
The following steps are basic commands to get familiar with |co2mpas| procedural
workflow using the command line interface:

- `Run in engineering mode`_
- `Input file`_
- `Data synchronization`_
- `GUI start-up`_

Run in engineering mode
-----------------------
To run |co2mpas| in engineering mode with some sample data, you have to:

1. Generate some demo files inside the ``./input`` folder, to get familiar with
   the input data (for more info check
   the `link <_build/co2mpas/co2mpas.cli.html#co2mpas-demo>`__)::

    ## Generate the demo files and open a demo file.
    $ co2mpas demo ./input
    $ start ./input/co2mpas_conventional.xlsx

2. Run |co2mpas| in engineering mode and inspect the results in the ``./output``
   folder. The workflow is plotted on the browser (for more info check the
   `link <_build/co2mpas/co2mpas.cli.html#co2mpas-run>`__)::

    ## Run co2mpas and open the output folder.
    $ co2mpas run ./input/co2mpas_conventional.xlsx -O ./output -PL
    $ start ./output

.. image:: _static/image/output_workflow.png
   :width: 100%
   :alt: Output workflow
   :align: center

Input file
----------
To create an input file with your data, you have to:

1. Generate an empty input template file (i.e., ``vehicle.xlsx``) inside
   the ``./input`` folder::

    ## Generate template file.
    $ co2mpas template ./input/vehicle.xlsx -TT input

2. Follow the instructions provided in the excel file to fill the required
   inputs::

    ## Open the input template.
    $ start ./input/vehicle.xlsx

.. image:: _static/image/input_template.png
   :width: 100%
   :alt: Input template
   :align: center

Data synchronization
--------------------
To synchronize the `dyno` and `OBD` data with the theoretical cycle, you have
to:

1. Generate a `synchronization template` file ``wltp.xlsx``::

    ## Generate template file.
    $ co2mpas syncing template ./to_sync/wltp.xlsx -CT wltp -WC class3b -GB automatic

   .. note::
      With the command above, the file contains the theoretical ``WLTP``
      velocity profile for an ``automatic`` vehicle of ``class3b``. For more
      info type ``co2mpas syncing template -h`` or click the
      `link <_build/co2mpas/co2mpas.cli.html#co2mpas-syncing-template>`__
2. Fill the ``dyno`` and ``obd`` sheets with the relative data collected in the
   laboratory::

    ## Open the input template.
    $ start ./to_sync/wltp.xlsx

3. Synchronize the data with the theoretical velocity profile::

    $ co2mpas syncing sync ./to_sync/wltp.xlsx ./sync/wltp.sync.xlsx

4. Copy/Paste the synchronized data (``wltp.sync.xlsx``) contained in the
   ``synced`` sheet into the relative sheet of the input template::

    ## Open the synchronized data.
    $ start ./sync/wltp.sync.xlsx

GUI start-up
------------
To launch the |co2mpas| GUI from the console, you can use the following
command::

    ## Open the GUI of co2mpas.
    $ co2wui run

.. image:: _static/image/gui_start_up.png
   :width: 100%
   :alt: GUI start-up
   :align: center

.. _end-quick:
.. _start-sub:
.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. _end-sub:
