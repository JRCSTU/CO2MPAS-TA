#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from co2mpas.__main__ import init_logging
import glob
import logging
import os
import unittest
import os.path as osp
import yaml

mydir = os.path.dirname(__file__)

init_logging(level=logging.DEBUG)


class IO(unittest.TestCase):

    def test_read(self):


        import schedula as sh
        import co2mpas
        from co2mpas.batch import vehicle_processing_model
        dsp = vehicle_processing_model()

        files = osp.join(osp.dirname(co2mpas.__file__), 'demos', '*.xlsx')
        res = {
            osp.basename(f): dsp(
                {'input_file_name': f, 'overwrite_cache': True},
                outputs=['base_data', 'plan_data']
            ) for f in glob.glob(files)
        }
        res = {k for k, v in sh.stack_nested_keys(res, depth=6)}
        #with open(osp.join(osp.dirname(__file__), 'read.yaml'), 'w') as file:
        #    yaml.dump(res, file)

        with open(osp.join(osp.dirname(__file__), 'read.yaml')) as file:
            self.assertSetEqual(res, yaml.load(file))
