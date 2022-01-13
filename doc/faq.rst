###
FAQ
###

This page contains the most Frequently Asked Questions regarding |CO2MPAS|
model, regulation and inputs.

- `General`_

  - `Where can the user download the latest version of the CO2MPAS?`_
  - `Is CO2MPAS free and will it be in the future?`_
  - `What is CO2MPAS physical background and which formulas are applied?`_
  - `Where can the user find information on the status of the validation?`_
  - `Where can the user find CO2MPAS workshop material?`_
- `Model`_

  - `What is the Data synchronization tool and how does it work?`_
  - `What is the model selector?`_
  - `Is it possible to simulate other cycles than NEDC or WLTP? How about real on-road tests?`_
  - `Is the usage of internal / development signals allowed (if equivalence is shown)?`_
  - `What is the start-stop (S/S) activation time?`_
  - `How to insert a new drive_battery_technology for hybrid electric vehicles?`_

*For more questions, please visit:* 
https://github.com/JRCSTU/CO2MPAS-TA/wiki/FAQ

General
=======

Where can the user download the latest version of the |CO2MPAS|?
----------------------------------------------------------------
To download the |CO2MPAS| latest version click `here <https://co2mpas.readthedocs.io/en/stable>`_.
To be notified of every |CO2MPAS| release, **watch** our GitHub
`repository <https://github.com/JRCSTU/CO2MPAS-TA>`_.

Is |CO2MPAS| free and will it be in the future?
-----------------------------------------------
|CO2MPAS| is and will be free.
To maintain it under the current EUPL license, any modifications made to the
program, or a copy of it, must be licensed in the same way:
`EUPL <https://eupl.eu/>`_.

What is |CO2MPAS| physical background and which formulas are applied?
---------------------------------------------------------------------
|CO2MPAS| is a backward-looking longitudinal-dynamics |CO2| and
fuel-consumption simulator for light-duty M1 & N1 vehicles (cars and vans).
To check the formulas the user can visit the
`functions' description pages <https://co2mpas.readthedocs.io/en/stable/model.html#co2mpas-model>`_.

Where can the user find information on the status of the validation?
--------------------------------------------------------------------
Detailed validation reports are provided together with every major release of
|CO2MPAS| in the `validation chapter <http://jrcstu.github.io/co2mpas/>`_ of
the wiki.

Validation is performed as well by independent contractors (LAT) of DG CLIMA. 
Moreover, some stakeholders have conducted independent validation 
tests on real cars in collaboration with the JRC. The results of these tests
have been included in the above-mentioned reports as "real cars".

Where can the user find |CO2MPAS| workshop material?
----------------------------------------------------
Workshop material is always uploaded in the
`presentation chapter <https://github.com/JRCSTU/CO2MPAS-TA/wiki/Presentations-from-CO2MPAS-meetings>`_.

Model
=====

What is the Data synchronization tool and how does it work?
-----------------------------------------------------------
Synchronization of data from different sources is essential for robust results.
|CO2MPAS| `syncing` tool uses a common signal as a reference. 
To avoid time-aligned signals, we advise using the velocity present on the
dyno and the obd at the same time.
`syncing` tool will shift and re-sample the other signals according to the
reference signal provided.
The user is allowed to apply different ways of re-sampling the original signals. 
For more information, please see the instructions.  

What is the model selector?
---------------------------
|CO2MPAS| consists of several models. If the user provides both WLTP-H and
WLTP-L data, the same models will be calibrated twice, according to the data
provided by each configuration.
If the option *model selector* is switched on, |CO2MPAS| will use the model that
provides the best scores, no matter if the model was calibrated with another
cycle. For example, if the alternator model of the High configuration is better,
the same model will be used to predict the Low configuration as well.    

Is it possible to simulate other cycles than NEDC or WLTP? How about real on-road tests?
----------------------------------------------------------------------------------------
Yes, |CO2MPAS| can simulate other cycles, as well as on-road tests. The user can
simulate using several extra parameters beyond the official laboratory-measured
ones.
The user can input the velocity profile followed, road grade, extra auxiliaries
losses, extra passengers, different road loads, temperatures, etc.
The user will find an example file when downloading the
`demo <https://co2mpas.readthedocs.io/en/stable/tutorial.html#download-demo-files>`_
files.

Is the usage of internal / development signals allowed (if equivalence is shown)?
---------------------------------------------------------------------------------
OBD signals are regulated and the only one to be used.

What is the start-stop (S/S) activation time?
---------------------------------------------
S/S is the time elapsed from the beginning of the NEDC test to the first time
the Start-Stop system is enabled, expressed in seconds [s].

How to insert a new `drive_battery_technology` for hybrid electric vehicles?
----------------------------------------------------------------------------
The parameter already contains a preselection of
`drive batteries technologies <https://co2mpas.readthedocs.io/en/stable/glossary.html#term-drive-battery-technology>`_
as a drop-down menu. If you need to insert a different technology, you should
remove the "data validation rule" of the excel input file, insert the new data
and proceed with the co2mpas run.

.. |CO2MPAS| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. _DG CLIMA's note: https://ec.europa.eu/clima/sites/clima/files/transport/vehicles/cars/docs/correlation_implementation_information_en.pdf 

