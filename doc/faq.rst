######
F.A.Q.
######

This page contains the most Frequently Asked Questions regarding |CO2MPAS|
model, regulation and inputs.

- `General`_

  - `Is the use of CO2MPAS correlation tool mandatory for all light-duty vehicles?`_
  - `Where can the user download the latest version of the CO2MPAS?`_
  - `Is CO2MPAS free and will it be in the future?`_
  - `What is CO2MPAS physical background and which formulas are applied?`_
  - `Where can the user find information on the status of the validation?`_
  - `Where can the user find CO2MPAS workshop material?`_
- `Model`_

  - `What is the difference between Type Approval mode and engineering mode in CO2MPAS run?`_
  - `What is the Data synchronization tool and how does it work?`_
  - `What is the model selector?`_
  - `Is it possible to simulate other cycles than NEDC or WLTP? How about real on-road tests?`_
  - `Is the usage of internal / development signals allowed (if equivalence is shown)?`_
  - `What is the start-stop (S/S) activation time? What might happen if the user declares it wrong?`_
  - `How to insert a new drive_battery_technology for hybrid vehicles?`_
- `Regulation`_

  - `Where to find the correlation regulation?`_
  - `Should CO2MPAS be used for each car or each family ID?`_
  - `Is it possible to do a physical test, instead of accepting CO2MPAS results?`_
  - `What is DICE and who should use it?`_
  - `What are the verification and deviation factors, and when do they need to be recorded?`_

*For more questions, please visit:* 
https://github.com/JRCSTU/CO2MPAS-TA/wiki/F-A-Q

General
=======

Is the use of |CO2MPAS| correlation tool mandatory for all light-duty vehicles?
-------------------------------------------------------------------------------
No, the use of |CO2MPAS| is not compulsory for hybrids and plug-in hybrids,
double testing is allowed instead.
For all the other technologies, the use of the correlation tool is mandatory.
For more information and updates please see refer to point 3 
of `DG CLIMA's note`_. 

Where can the user download the latest version of the |CO2MPAS|?
----------------------------------------------------------------
Visit our GitHub repository
`CO2MPAS-TA <https://github.com/JRCSTU/CO2MPAS-TA/>`_ to download |CO2MPAS| .
Subscribe to  `issue #8 <https://github.com/JRCSTU/CO2MPAS-TA/issues/8>`_ 
to get notifications for the new |CO2MPAS| releases.
** You can find the documentation in the
`wiki <https://github.com/JRCSTU/CO2MPAS-TA/wiki/>`_. **

Is |CO2MPAS| free and will it be in the future?
-----------------------------------------------
|CO2MPAS| is and will be free.
To maintain it under the current EUPL license, any modifications made to the
program, or a copy of it, must be licensed in the same way:
`EUPL <https://eupl.eu/>`_.


What is |CO2MPAS| physical background and which formulas are applied?
---------------------------------------------------------------------
|CO2MPAS| is backward-looking longitudinal-dynamics |CO2| and
fuel-consumption simulator for light-duty M1 & N1 vehicles (cars and vans).
**To check the formulas the user can visit the functions' 
description pages: .....** 

Where can the user find information on the status of the validation?
--------------------------------------------------------------------
Detailed validation reports are provided together with every release of
|CO2MPAS| in the `validation chapter <http://jrcstu.github.io/co2mpas/>`_ of
the wiki.
Here the latest 3 reports: 

- `manual <http://jrcstu.github.io/co2mpas/v2.0.x/validation_manual_cases.html>`_;    
- `automatic <http://jrcstu.github.io/co2mpas/v2.0.x/validation_automatic_cases.html>`_;   
- and `real cars <http://jrcstu.github.io/co2mpas/v2.0.x/validation_real_cases.html>`_, 
  respectively;

Validation is performed as well by independent contractors (LAT) of DG CLIMA. 
Moreover, some stakeholders have conducted independent validation 
tests on real cars in collaboration with the JRC. The results of these tests
have been included in the above-mentioned reports as "real cars".

Where can the user find |CO2MPAS| workshop material?
----------------------------------------------------
Workshop material is always uploaded in
`presentation chapter <https://github.com/JRCSTU/CO2MPAS-TA/wiki/Presentations-from-CO2MPAS-meetings>`_.

Model
=====

What is the difference between Type Approval mode and engineering mode in |CO2MPAS| run?
----------------------------------------------------------------------------------------
In Type Approval mode |CO2MPAS| simulates the NEDC |CO2| emission 
of the given vehicle fully aligned to the WLTP-NEDC correlation Regulation. 
If |CO2MPAS| finds some extra input it will raise a warning and it will not 
produce any result. 
The same will happen in the case of missing inputs. 
The engineering mode provides the user with full control of the tool. 
By using the engineering mode, 
all the running flags become available to the user. 
Moreover, the user can provide additional inputs beyond the declaration ones 
and check their effect on the NEDC |CO2| prediction. 

What is the Data synchronization tool and how does it work?
-----------------------------------------------------------
Synchronization of data from different sources is essential for robust results.
|CO2MPAS| `syncing` tool uses a common signal as a reference. 
To avoid time-aligned signals, we advise using the velocity present on the
dyno and the obd at the same time.
`syncing` tool will shift and re-sample the other signals 
according to the reference signal provided. 
The user is allowed to apply different ways of re-sampling the original signals. 
For more information, please see the instructions.  

What is the model selector?
---------------------------
|CO2MPAS| consists of several models. If the user provides both WLTP-H and WLTP-L
data, the same models will be calibrated twice, according to the data provided
by each configuration.
If the option *model selector* is switched on, |CO2MPAS| will use the model that
provides the best scores, no matter if the model was calibrated with another
cycle. For example, if the alternator model of the High configuration is better,
the same model will be used to predict the Low configuration as well.    

Is it possible to simulate other cycles than NEDC or WLTP? How about real on-road tests?
----------------------------------------------------------------------------------------
Yes, |CO2MPAS| can simulate other cycles, as well as on-road tests. 
The user can simulate with several extra parameters beyond the 
official laboratory-measured ones. 
The user can input the velocity profile followed, road grade, 
extra auxiliaries losses, extra passengers, different road loads, temperatures, 
etc. 
The user will find an example file when downloading the demo files. 
**Also, please check the instructions.**        

Is the usage of internal / development signals allowed (if equivalence is shown)?
---------------------------------------------------------------------------------
OBD signals are regulated and the only one to be used.

What is the start-stop (S/S) activation time? What might happen if the user declares it wrong?
----------------------------------------------------------------------------------------------
S/S is the time elapsed from the beginning of the NEDC test to the first time
the Start-Stop system is enabled, expressed in seconds [s].
S/S is one of the variables that are subject to verification in case of random
testing. If during verification test (random test) S/S activation time declared
in |CO2MPAS| is lower than the span between the beginning of the NEDC test and
the first engine stop, that will result in Verification Factor equal to 1 and
this will have implications on the entire fleet of the specific vehicle
manufacturer.

How to insert a new `drive_battery_technology` for hybrid vehicles?
-------------------------------------------------------------------
For TA: please contact us at jrc-co2mpas@ec.europa.eu.

For non TA: the parameter already contains a preselection of
`drive batteries technologies <https://co2mpas.readthedocs.io/en/stable/glossary.html#drive_battery_technology>`_
as drop-down menu. If you need to insert a different technology, you should
remove the "data validation rule" of the excel input file, insert the new data
and proceed with the co2mpas run.

Regulation
==========

Where to find the correlation regulation?
-----------------------------------------
Below some useful links: 
 
- The correlation regulation for passenger vehicles
  `REGULATION (EU) 2017/1153, <https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02017R1153-20190201&from=EN>`_
- The correlation regulation for light commercial vehicles
  `REGULATION (EU) 2017/1152 <https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02017R1152-20190201&from=EN>`_
- Also, in this `document, <https://ec.europa.eu/clima/sites/clima/files/transport/vehicles/cars/docs/faq_wltp_correlation_en.pdf](https://ec.europa.eu/clima/sites/clima/files/transport/vehicles/cars/docs/faq_wltp_correlation_en.pdf>`_
  the reader will find some frequently asked question regarding the correlation
  procedure.

Should |CO2MPAS| be used for each car or each family ID?
--------------------------------------------------------
For each interpolation family ID. 
Vehicle-H and Vehicle-L are utilized to define the interpolation line of
Interpolation Family ID.

Is it possible to do a physical test, instead of accepting |CO2MPAS| results?   
-----------------------------------------------------------------------------
Yes, there are cases when |CO2MPAS|
does not need to be used and physical test shall be performed instead. 
**where are they described?**

What is DICE and who should use it?
-----------------------------------
DICE is the tool assigning a random number to each IDIF type approved. 
It is used only for type approving purposes, by designated users. 

What are the verification and deviation factors, and when do they need to be recorded?
--------------------------------------------------------------------------------------
These values need to be recorded when the random number is 90, or above,
and the |CO2MPAS| deviation is equal or higher than 4 percent.
For more details please refer to the correlation regulation.

.. |CO2MPAS| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. _DG CLIMA's note: https://ec.europa.eu/clima/sites/clima/files/transport/vehicles/cars/docs/correlation_implementation_information_en.pdf 

 
