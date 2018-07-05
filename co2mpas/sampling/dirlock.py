# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import contextlib
import logging
import os
import time
from typing import Text

import os.path as osp



## Spin-loop delay before retrying `mkdir(dir)`.
DIR_LOCK_WAIT_SEC = 2
## Max time to abort wit-lock.
ABORT_SEC = 7

log = logging.getLogger(__name__)

@contextlib.contextmanager
def locked_on_dir(dpath: Text,
                  lock_wait_sec=DIR_LOCK_WAIT_SEC,
                  abort_sec=ABORT_SEC):
    dirname = osp.dirname(dpath) or '.'
    if not osp.isdir(dirname):
        raise ValueError("Parent-folder '%s' is missing!", dirname)

    start_t = time.clock()
    try:
        while True:
            try:
                os.mkdir(dpath)
                break
            except FileExistsError:
                elapsed_sec = (time.clock() - start_t)
                if elapsed_sec > abort_sec:
                    raise TimeoutError(dpath)
                log.debug("Dirlocking '%s' for %s sec...", dpath, lock_wait_sec)
                time.sleep(lock_wait_sec)

        yield
    finally:
        os.rmdir(dpath)
