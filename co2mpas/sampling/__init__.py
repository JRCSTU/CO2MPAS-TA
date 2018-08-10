# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
co2dice: Distributed Impromptu type-Certificate Evaluation of *co2mpas* results.


This is an articulated application comprised of the following:

- A library implemented with :class:`baseapp.Spec` instances performing
  the backend-tasks: prepare/sign/send/receive/decide/archive of
  *co2mpas* results.
- the ``co2dice`` hierarchical cmd-line tool,
  implemented with :class:`baseapp.Cmd` instances.
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
and/or cmd-line arguments (see :mod:`baseapp`).

For usage examples read the "Random Sampling" section in the manual (http://co2mpas.io).
"""
from polyversion import polyversion, polytime

from .._vendor import traitlets as trt

__copyright__ = "Copyright (C) 2015-2018 European Commission (JRC)"
__license__   = "EUPL 1.1+"             # noqa
__title__     = "co2dice"               # noqa
__summary__   = __doc__.splitlines()[0] # noqa
__uri__       = "https://co2mpas.io"    # noqa

#: Project's PEP 440 version from Git (or env[co2mpas_VERSION])
#: FIXME: change co2dice's pname in polyversion() when co2dice graduates to own project.
__version__ = polyversion(pname='co2mpas')
__updated__ = polytime(pname='co2mpas')
version = __version__


class CmdException(trt.TraitError):
    pass
