##############
CO2MPAS F.A.Q.
##############
.. contents::


This page contains the most frequently asked questions about the |CO2MPAS| software.
It consists of questions related to the |CO2MPAS| model, 
input parameters, test procedures, and Regulation

*This page will be frequently updated.*
*For more questions that we frequently receive, please visit:* 
https://github.com/JRCSTU/CO2MPAS-TA/wiki/F-A-Q


General
=======


1. |CO2MPAS| tool users
------------------------
Q: 
Is it mandatory for manufacturers of passenger vehicles to use the |CO2MPAS| 
correlation tool in order to receive a type approval for a specific vehicle?   

A: 
Only in the case of hybrids and plug-in hybrids OEMs can skip the use of 
correlation tool and go directly with double testing (NEDC). 
For all other technologies, the use of the correlation tool mandatory.
For more information and updates please see read point 3 
of the `DG CLIMA's note`_. 


2. Physical background of |CO2MPAS|
------------------------------------
Q: 
What is the physical background and the formulas used in |CO2MPAS| tool? 

A:
|CO2MPAS| is backward-looking longitudinal-dynamics |CO2| and
fuel-consumption simulator for light-duty M1 & N1 vehicles (cars and vans).
All the formulas are visible to the users if you visit the respective functions' 
description pages.

3. |CO2MPAS| version - New Releases
-------------------------------------
Q:
Where is information on how to download the last version of the |CO2MPAS| tool exists? 
Where can I find the documentation and release notes?

A: 
For technical guidelines you may search into the 
`wiki <https://github.com/JRCSTU/CO2MPAS-TA/wiki/>`_. 
We strongly recommend you to register into our GitHub repository, 
`CO2MPAS-TA <https://github.com/JRCSTU/CO2MPAS-TA/>`_. 
There, you may `open an issue <https://github.com/JRCSTU/CO2MPAS-TA/issues/new>`_ 
if help is needed, 
and subscribe to the `issue #8 <https://github.com/JRCSTU/CO2MPAS-TA/issues/8>`_ 
in order to get notifications for the new |CO2MPAS| releases.

4. Validation of |CO2MPAS|
---------------------------
Q: 
Where can we find information on the status of the validation, 
the choices of respective test flexibilities, 
test mass and RL in both tests, and their estimated effect?

A: 
Detailed validation reports are provided together with every release of |CO2MPAS|
in the wiki in the `validation chapter <http://jrcstu.github.io/co2mpas/>`_. 
The latest reports are contained in these 3 files: 

- `manual <http://jrcstu.github.io/co2mpas/v2.0.x/validation_manual_cases.html>`_;    
- `automatic <http://jrcstu.github.io/co2mpas/v2.0.x/validation_automatic_cases.html>`_;   
- and `real cars <http://jrcstu.github.io/co2mpas/v2.0.x/validation_real_cases.html>`_, 
  respectively;

Validations are also performed by independent contractors (LAT) of DG CLIMA. 
Of course, other involved stakeholders are performing independently validation 
tests on real cars in close collaboration with the JRC. 
The results of which, are included in the above-mentioned reports as "real cars".

5. Work shop material
-----------------------
Q: 
Where can we find the material presented in a workshop for |CO2MPAS|?

A: 
Material related to a workshop is always uploaded in the wiki in the 
`presentation chapter <https://github.com/JRCSTU/CO2MPAS-TA/wiki/Presentations-from-CO2MPAS-meetings>`_

6. |CO2MPAS| License
----------------------
Q: 
Is |CO2MPAS| free to use and will it be in the future?

A: 
|CO2MPAS| is and will remain, free.
And we note that for keeping it free,
any modifications you or anybody else are making to the program, 
or a copy of it, they must be also licensed under the same EUPLicense.
In particular, the EUPL is NOT a "permissive" license [1], 
but a "copyleft" one [2].

.. [1] https://en.wikipedia.org/wiki/Permissive_software_licence
.. [2] https://en.wikipedia.org/wiki/Copyleft


Model
=====


7. Type Approval mode / Engineering mode
------------------------------------------
Q: 
What is difference between Type Approval mode and normal mode in |CO2MPAS| run?

A: 
In Type Approval mode |CO2MPAS| simulates the NEDC |CO2| emission 
of the given vehicle fully aligned to the WLTP-NEDC correlation Regulation. 
If |CO2MPAS| finds some extra input it will raise a warning and it will not 
produce any result. 
The same will happen in the case of missing inputs. 
The engineering mode provides the user full control of the tool. 
By using the engineering mode, 
all the running flags become available for the user. 
Also, the user can provide additional inputs beyond the declaration ones 
and check their effect on the NEDC |CO2| prediction. 
The |CO2MPAS| model run in engineering mode, 
is less strict than the type approval mode and therefore it will more likely 
produce a result/produce error logs that can help understanding why the 
program is not predicting the NEDC |CO2| emission.

8. Enable selector
--------------------
Q:
What is the model selector? 

A: 
|CO2MPAS| consist of several models. 
In case the user provides both WLTP-H and WLTP-L data, 
the same models will be calibrated twice, 
according to the data provided by each configuration. 
If the option for model selector is switched on, 
|CO2MPAS| will use the model that provides the best scores 
when compared to the input data, 
no matter if the model was calibrated with another cycle. 
For example, if the alternator model of the High configuration is better, 
the same model will be used to predict for the Low configuration as well.    

9. Simulation plan
--------------------
Q: 
Does |CO2MPAS| have the capacity to simulate other cycles or real on-road tests? 

A: 
Yes |CO2MPAS| is able to simulate on-road tests. 
The user can simulate with several extra parameters parameters beyond the 
official laboratory measured ones. 
The user can input the velocity profile followed, road grade, 
extra auxiliaries losses, extra passengers, different road loads, temperatures, 
etc. 
The user will find an example file when downloading the demo files. 
Also, please check the instructions.        

10. Signals for |CO2MPAS| run
--------------------------------
Q: 
Is usage of internal / development signals allowed (if equivalence is shown)?

A: 
OBD signals are regulated and are the ones to be used.

11. Data synchronization
-------------------------
Q: 
How does the Data synchronization tool works, and what its use? 

A: 
Synchronization of data from different sources 
is very essential for robust results. 
|CO2MPAS| `syncing` tool uses a common signal as a reference. 
We advise is to use the velocity which is present on the dyno and the obd at the same time. 
In this way, you don't need time-aligned signals.
`syncing` tool will shift and re-sample the other signals 
according to reference signal provided. 
The user is able to use different ways of re-sampling the original signals. 
For more information, please see the instructions.     

12. Start-Stop activation
---------------------------
Q: 
How to fill the input parameter for the start-stop activation time? 

A: 
The Start-Stop (SS) activation time declared when the SS system is enabled 
in order to operate the next time the conditions for engine deactivation are met.
One should be extremely careful about the value declared in this field as 
it is one of the variables that are subjected to verification in case of random testing.
If during TA the vehicle is subjected to double testing,
the authority overlooking the test will control that the SS activation time 
declared in |CO2MPAS| is not lower than the time elapsed 
from the start of the NEDC test when the engine stops for the first time. 
In case the engine stops for the first time at t_test > t_declared, 
there might be severe implications on the entire fleet 
of the specific vehicle manufacturer.


Regulation
==========


13. Correlation Regulation 
-----------------------------
Q: 
Where to find the correlation regulation?

A: 
Below some useful links: 
 
- The correlation regulation for passenger vehicles **REGULATION (EU) 2017/1153**.
  `Here the consolidated version with latest updates on 21.12.2018 <https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02017R1153-20190201&from=EN>`_
- The correlation regulation for light commercial vehicles **REGULATION (EU) 2017/1152**. 
  `Here the consolidated version with latest updates on 21.12.2018 <https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02017R1152-20190201&from=EN>`_ 
- Also, in this `document <https://ec.europa.eu/clima/sites/clima/files/transport/vehicles/cars/docs/faq_wltp_correlation_en.pdf](https://ec.europa.eu/clima/sites/clima/files/transport/vehicles/cars/docs/faq_wltp_correlation_en.pdf>`_
  the reader will find some frequently asked question regarding the correlation procedure. 

14. Family interpolation
-------------------------- 
Q:    
Do we run the |CO2MPAS| software for each car, or each family?

A:  
|CO2MPAS| is only used at Type Approval. 
The OEM chooses the vehicle family and tests Vehicle-H and Vehicle-L. 
Then, uses them as input for |CO2MPAS| and gets as output the NEDC equivalents 
of vehicle L and H (L' and H'). 
L' and H' are used to define an interpolation line in all similar to the WLTP one, 
from which the NEDC |CO2| value for each NEDC Type Approval Value 
in the middle can be derived. 
This interpolation line is used to derive the value to put in the 
Certificate of Conformity.  

15. Physical test 
--------------------   
Q:       
In case that the manufacturer from the beginning wants to do a physical test, 
is there a way to over-pass |CO2MPAS|?   

A:    
Commission will describe cases/ technologies where |CO2MPAS| 
does not need to be used and physical test shall be performed instead. 
In all other cases, 
|CO2MPAS| must be used.

16. DICE and procedure
------------------------
Q:     
What is the DICE and how to proceed in different cases?   

A:
DICE is the tool for 10 percent random sampling of the Interpolation Families 
type approved with the use of |CO2MPAS|. 
This tool can be used only by users designated by the Member States. 
For cases that errors have occurred, 
the user either from a Type Approval Authority, 
or a Technical service needs to be advised by the `DG CLIMA's note`_. 
Except for much useful information, 
paragraph 5 describes the steps to be followed in different cases of errors. 
In any case, the user needs to inform DG CLIMA (EC-CO2-LDV-IMPLEMENTATION@ec.europa.eu) 
and JRC (JRC-CO2MPAS@ec.europa.eu) for all the details before any action is taken.

17. Deviation and verifications factors
-----------------------------------------
Q: 
What are the verification and deviation factors, and when do they need to be 
recorded? 

A: 
For information about deviation and the verification factor, the user can 
be advised by the correlation regulation. 
These values needs to be recorded in case that the random number received from the 
DICE is 90, or above, and the |CO2MPAS| deviation is equal to 4 percent or more. 
For details please refer to the regulation.  



.. _substs:

.. |CO2MPAS| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`
.. _DG CLIMA's note: https://ec.europa.eu/clima/sites/clima/files/transport/vehicles/cars/docs/correlation_implementation_information_en.pdf 

 
