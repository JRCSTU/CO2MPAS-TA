# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
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

__copyright__ = "Copyright (C) 2015-2017 European Commission (JRC)"
__license__   = "EUPL 1.1+"
__title__     = "co2mpas"
__summary__   = "Vehicle simulator predicting NEDC CO2 emissions from WLTP time-series."
__uri__       = "https://co2mpas.io"

#: Authoritative project's PEP 440 version.
#: Remember to manually update version in CHANGES.rst.
__version__   = '1.8.1a2'
version       = __version__
#: Release date.
__updated__ = '2018-06-12T19:31:13.781522'

#: Input/Output file's version.
__file_version__        = "2.2.7"

#: Compatible Input file's version.
__input_file_version__  = "2"

__dice_report_version__ = '1.0.2'


#: Define VehicleFamilyId (aka ProjectId) pattern here not to import the world on use.
#: Note: referenced by :data:`io.schema.vehicle_family_id_regex` and
#: :meth:`.sampling.tstamp.TstampReceiver.extract_dice_tag_name()`.
vehicle_family_id_pattern = r'(IP|RL|RM|PR)-(\d{2})-(\w{2,3})-(\d{4})-(\d{4})'


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
