.. image:: doc/_static/CO2MPAS_banner.png
   :width: 100%

.. _start-quick:

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
- sync: enables the time series synchronization tool (i.e., ``syncing``
  previously named ``datasync``).
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


.. _end-intro:
.. _start-badges:

.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. |clink| replace:: *Clink*
.. _clink: http://mridgers.github.io/clink/
.. |EUPL| replace:: *EUPL*
.. _EUPL: https://joinup.ec.europa.eu/page/eupl-text-11-12

.. _end-badges:
