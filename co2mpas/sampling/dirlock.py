# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from collections import defaultdict
import contextlib
import os
from threading import Condition
from typing import Text

import os.path as osp


DIR_LOCK_WAIT_SEC = 2
_dir_locks = defaultdict(Condition)  # type: Mapping[str, Condition]


@contextlib.contextmanager
def locked_on_dir(dpath: Text, lock_wait_sec=DIR_LOCK_WAIT_SEC):
    dirname = osp.dirname(dpath) or '.'
    assert osp.isdir(dirname), ("Missing parent-folder!", dirname)
    dlock = _dir_locks[dpath]  # type: Condition

    with dlock:
        try:
            while True:
                try:
                    os.mkdir(dpath)
                    break
                except FileExistsError:
                    dlock.wait(lock_wait_sec)

            yield
        finally:
            os.rmdir(dpath)
            dlock.notify()
