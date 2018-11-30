==================================================================
co2DICE: Distributed Impromptu Co2mpas Evaluation
==================================================================

.. image:: https://img.shields.io/pypi/v/co2dice.svg
    :alt: Deployed in PyPi?
    :target: https://pypi.org/pypi/co2dice

.. image:: https://readthedocs.org/projects/co2mpas/badge/?version=latest
    :target: https://co2mpas.readthedocs.io/en/latest/?badge=latest
    :alt: Auto-generated documentation status

.. _coord-start:

:version:       2.0.0+60.g61a571ac
:updated:       2018-11-30T11:01:15.734196
:Documentation: http://co2mpas.io/
:repository:    https://github.com/JRCSTU/CO2MPAS-TA/
:pypi-repo:     https://pypi.org/project/co2dice/
:copyright:     2018 JRC.C4(STU), European Commission (`JRC <https://ec.europa.eu/jrc/>`_)
:license:       `EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>`_

All project's documentation hosted at https://co2mpas.io/


Development
===========
To run Test-Cases you may set these env-vars:

- :envvar:`WEBSTAMPER_CHECK_URL`
- :envvar:`WEBSTAMPER_STAMP_URL`
- :envvar:`STAMP_CHAIN_DIR`

or create a traits-configuration file on  ``~/co2dice_config.py`` by default,
or specify more files with :envvar:`CO2DICE_VAR_NAME`.

Also you need to  contain in oyur :envvar:`PATH`:
- the project installed (e.g. in "develop" mode) for the executable commands to work;
- GnuPG-2;
- Git (Git MinGW for Windows, either the official or the one in *MSYS2*);
- Python
