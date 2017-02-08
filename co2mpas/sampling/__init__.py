# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
co2dice: prepare/sign/send/receive/validate/archive Type Approval sampling emails of *co2mpas*.

This is an articulated application comprised of the following:

- A GUI application, based on the `kivy UI framework
  <https://kivy.org/>` or plain :mod:`tkinter`;
- a library performing the backend-tasks,
  implemented with :class:`baseapp.Spec` instances;
- the ``co2dice`` hierarchical cmd-line tool,
  implemented with :class:`baseapp.Cmd` instances.

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

The GUI part relies additionally on the *kivy* configuration scheme.

For usage examples read the "Random Sampling" section in the manual (http://co2mpas.io).
"""
from collections import namedtuple, defaultdict
import enum
import re
from typing import Text, Tuple

import traitlets as trt


class CmdException(trt.TraitError):
    pass


_file_arg_regex = re.compile('(inp|out)=(.+)', re.IGNORECASE)

all_io_kinds = tuple('inp out other'.split())


class PFiles(namedtuple('PFiles', all_io_kinds)):
    """
    Holder of project-files stored in the repository.

    :ivar inp: ``[fname1, ...]``
    :ivar out: ``[fname1, ...]``
    :ivar other: ``{index: fname}``
    """
    ## INFO: Defined here to avoid circular deps between report.py <-> project.py,
    #  because it is used in their function declarations.

    @staticmethod
    def _io_kinds_list(*io_kinds) -> Tuple[Text]:
        """
        :param io_kinds:
            if none specified, return all kinds,
            otherwise, validates and converts everything into a string.
        """
        if not io_kinds:
            io_kinds = all_io_kinds
        else:
            assert not (set(io_kinds) - set(all_io_kinds)), (
                "Invalid io-kind(s): ", set(io_kinds) - set(all_io_kinds))
        return tuple(set(io_kinds))

    def nfiles(self):
        return sum(len(f) for f in self._asdict().values())


#: Allow creation of PFiles with partial arguments.
PFiles.__new__.__defaults__ = ([], ) * len(all_io_kinds)
