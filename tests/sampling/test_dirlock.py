#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from co2mpas.__main__ import init_logging
from co2mpas.sampling import dirlock
import logging
import os
import tempfile
import time
import unittest

import ddt

import functools as fnt
import multiprocessing as mp
import os.path as osp
import subprocess as sbp
import textwrap as tw
import threading as th


init_logging(level=logging.DEBUG)

log = logging.getLogger(__name__)

mydir = osp.dirname(__file__)

duration = 2.


def lock_n_sleep(tdir, label):
    log.info('Started %s', label)
    with dirlock.locked_on_dir(tdir, 0.2):
        time.sleep(duration)

def cmd_task_factory(tdir, label):
    prog_path = osp.join(osp.dirname(tdir), 'p.py')
    with open(prog_path, 'wt') as f:
        f.write(tw.dedent("""
            import time
            from co2mpas.sampling import dirlock;

            with dirlock.locked_on_dir(%r, 0.2):
                time.sleep(%s)
        """ % (tdir, duration)))

    def task():
        log.info('Started %s.', label)
        p = sbp.run(['python', prog_path],
                    universal_newlines=True,
                    stdout=sbp.PIPE, stderr=sbp.PIPE)
        assert not p.returncode and not p.stdout and not p.stderr, (
            p.returncode, p.stdout, p.stderr)

    return task

def worker_factory(worker_type: "thread | proc", tdir, label):
    label = '%s.%s' % (worker_type, label)
    if worker_type == 'cmd':
        worker = th.Thread(target=cmd_task_factory(tdir, label),
                           daemon=True)
    elif worker_type == 'thread':
        worker = th.Thread(target=lock_n_sleep, args=(tdir, label),
                           daemon=True)
    elif worker_type == 'proc':
        worker = mp.Process(target=lock_n_sleep, args=(tdir, label),
                            daemon=True)
    else:
        assert False, worker_type

    return worker

@ddt.ddt
class TDirlock(unittest.TestCase):

    @ddt.data(
        ('thread', 1),
        ('thread', 2),
        ('thread', 5),

        ('proc'  , 1),
        ('proc'  , 2),
        ('proc'  , 5),

        ('cmd'  , 1),
        ('cmd'  , 2),
        ('cmd'  , 5),
    )
    def test_workers(self, case):
        worker_type, nprocs = case
        start_t = time.clock()
        with tempfile.TemporaryDirectory() as tdir:
            tdir = osp.join(tdir, 'L')
            workers = [worker_factory(worker_type, tdir, i) for i in range(nprocs)]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
        elapsed = time.clock() - start_t

        exp_duration = nprocs * duration
        assert abs(elapsed - exp_duration) <= (duration / 3)
