# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
co2DICE: Distributed Impromptu Co2mpas Evaluation.


This is an articulated application comprised of the following:

- A library implemented with :class:`cmdlets.Spec` instances performing
  the backend-tasks: prepare/sign/send/receive/decide/archive of
  *co2mpas* results.
- the ``co2dice`` hierarchical cmd-line tool,
  implemented with :class:`cmdlets.Cmd` instances.
- A GUI adaptor, based on the :mod:`tkinter` framework;

::
           .------------.
    ,-.    |     GUI    |----+
    `-'    *------------*    |   .--------------.
    /|\    .------------.    +---| Spec classes |-.
     |     |  co2dice   |-.  |   *--------------* |
    / \    |    CMDs    |----+     *--------------*
           *------------* |
             *------------*

The ``Spec`` and ``Cmd`` classes are build on top of the
`traitlets framework <http://traitlets.readthedocs.io/>`
to read and validate configuration parameters found in files
and/or cmd-line arguments (see :mod:`cmdlets`).

For usage examples read the "Random Sampling" section in the manual (http://co2mpas.io).
"""
from polyversion import polyversion, polytime

from ._vendor.traitlets import traitlets as trt

__copyright__ = "Copyright (C) 2015-2018 European Commission (JRC)"
__license__   = "EUPL 1.1+"             # noqa
__title__     = "co2dice"               # noqa
__summary__   = __doc__.splitlines()[0] # noqa
__uri__       = "https://co2mpas.io"    # noqa

#: Project's PEP 440 version from Git (or env[co2dice_VERSION])
#: FIXME: change co2dice's pname in polyversion() when co2dice graduates to own project.
__version__ = '2.0.0'
__updated__ = '2018-08-31T14:22:45.259533'
version = __version__

#: The :term:`Semantic Versioning` for Input/Output files.
__dice_report_version__ = '1.0.2'

class CmdException(trt.TraitError):
    pass
