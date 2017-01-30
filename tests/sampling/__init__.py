#! python
# -*- coding: utf-8 -*-
#
# Copyright 2015-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os.path as osp

mydir = osp.dirname(__file__)
test_inp_fpath = osp.join(mydir, 'input.xlsx')
test_out_fpath = osp.join(mydir, 'output.xlsx')
test_vfid = 'RL-99-BM3-2017-0001'