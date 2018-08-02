#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL 1.2+ (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"`co2dice` launch-script setting up sys-path so relative imports work in ``main()`."
import sys


def main():
    """
    Cmd-line entrypoint to `co2dice` master command/

    - Invokes :func:`co2mpas.sampling.run()` with ``sys.argv[1:]``.
    - In order to set cmd-line arguments, invoke directly the function above.
    """
    req_ver = (3, 6)
    if sys.version_info < req_ver:
        raise NotImplementedError(
            "Sorry, Python >= %s is required, found: %s" %
            (req_ver, sys.version_info))

    from co2mpas.sampling import dice
    dice.run(argv=sys.argv[1:])


if __name__ == '__main__':
    ## Pep366 must always be the 1st thing to run.
    if not globals().get('__package__'):
        __package__ = 'co2mpas.sampling'  # noqa: A001 F841 @ReservedAssignment

    sys.exit(main())
