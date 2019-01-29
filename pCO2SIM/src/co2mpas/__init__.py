# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
"Vehicle simulator predicting NEDC CO2 emissions from WLTP time-series.

.. currentmodule:: co2mpas

.. autosummary::
    :nosignatures:
    :toctree: _build/co2mpas/

    ~model
    ~io
    ~batch
    ~datasync
    ~plot
    ~plan
    ~utils
    ~report
"""
from polyversion import polyversion, polytime

__copyright__ = "Copyright (C) 2015-2018 European Commission (JRC)"
__license__   = "EUPL 1.1+"                         # noqa
__title__     = "co2mpas"                           # noqa
__summary__   = __doc__.splitlines()[0]             # noqa
__uri__       = "https://co2mpas.io"                # noqa

#: The Project's PEP 440 and :term:`Semantic Versioning` is retrived
#: from from Git tags (or env[co2mpas_VERSION]).
__version__   = '3.0.0'
#: Release date.
__updated__   = '2019-01-29T17:26:46.273370'
version       = __version__                         # noqa

#: The :term:`Semantic Versioning` for Input/Output files.
__file_version__        = "3.0.1"                   # noqa


if __name__ == '__main__':
    """
    Print ``module.__<attr_name>__` for any cmd-line args like ``<attr-name>``

    separated by semicolons(';').
    """
    import sys

    my_module = sys.modules[__name__]
    if len(sys.argv) > 1:
        out = ';'.join(str(getattr(my_module, '__%s__' % a.replace('-', '_')))
                       for a in sys.argv[1:])
        sys.stdout.write(out)
    else:
        from pprint import pprint
        attrs = {k: v
                 for k, v in vars(my_module).items()
                 if k.startswith('__') and
                 k not in {'__name__', '__loader__', '__builtins__',
                           '__cached__', '__annotations__', '__spec__',
                           '__package__'}}
        pprint(attrs)
