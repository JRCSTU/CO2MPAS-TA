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
import tempfile
import schedula as sh

mydir = os.path.dirname(__file__)

init_logging(level=logging.DEBUG)


class IO(unittest.TestCase):

    def test_read(self):
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
        # with open(osp.join(osp.dirname(__file__), 'read.yaml'), 'w') as file:
        #    yaml.dump(res, file)

        with open(osp.join(osp.dirname(__file__), 'read.yaml')) as file:
            self.assertSetEqual(res, yaml.load(file))

    def test_ta_output(self):
        import co2mpas
        from co2mpas.batch import vehicle_processing_model
        from co2mpas.io.ta import generate_keys, define_decrypt_function
        file = osp.join(
            osp.dirname(co2mpas.__file__), 'demos', 'co2mpas_demo-1.xlsx'
        )
        dsp = vehicle_processing_model()
        with tempfile.TemporaryDirectory() as d:
            passwords = ('p_secret', 'p_server')
            generate_keys(d, passwords=passwords)
            keys = glob.glob(osp.join(d, '*.co2mpas.keys'))
            self.assertSetEqual(
                {osp.basename(f)[:-13] for f in keys},
                {'dice', 'server', 'secret'}
            )

            sol = dsp(
                {'input_file_name': file, 'overwrite_cache': True,
                 'type_approval_mode': True, 'output_folder': d,
                 'variation':{
                     'flag.encryption_keys': osp.join(
                        d, 'dice.co2mpas.keys'
                    ),
                     'flag.sign_key': osp.join(
                        d, 'sign.co2mpas.key'
                    )
                 }},
                outputs=['base_data', 'plan_data']
            )
            sol['base_data']['only_summary'] = True

            res = dsp(sol)

            decrypt = define_decrypt_function(
                osp.join(d, 'secret.co2mpas.keys'), passwords=passwords
            )

            r = res.get_node('run_base', 'write_ta_output')[0]
            s = decrypt(res['solution']['output_ta_file'])
            self.maxDiff = None
            for k in ('ta_id', 'dice_report', 'data', 'meta'):
                it0 = sorted(sh.stack_nested_keys(r[k], key=(k,)))
                it1 = sorted(sh.stack_nested_keys(s[k], key=(k,)))
                self.assertEqual(set(i for i, j in it0), set(i for i, j in it1))
                for i, j in zip(it0, it1):
                    self.assertEqual(i, j)
