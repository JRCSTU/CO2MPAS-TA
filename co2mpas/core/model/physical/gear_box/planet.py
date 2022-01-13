# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to bypass the gear box.
"""
import schedula as sh
from co2mpas.defaults import dfl
from .cvt import (
    default_correct_gear, identify_max_speed_velocity_ratio, predict_gears, CVT
)

dsp = sh.BlueDispatcher(
    name='empty model', description='Bypass for the gear box.'
)

dsp.add_func(default_correct_gear, outputs=['correct_gear'])
dsp.add_data('max_gear', 1)
dsp.add_func(predict_gears, outputs=['gears'])
dsp.add_data('stop_velocity', dfl.values.stop_velocity)
dsp.add_func(
    identify_max_speed_velocity_ratio, outputs=['max_speed_velocity_ratio']
)


@sh.add_function(dsp, outputs=['gear_shifting_model'])
def define_gear_shifting_model():
    """
    Return a fake gear shifting model.

    :return:
        Gear shifting model.
    :rtype: CVT
    """
    return CVT()


dsp.add_function(
    function_id='calculate_gear_box_speeds_in',
    function=sh.bypass,
    inputs=['gear_box_speeds_out'],
    outputs=['gear_box_speeds_in']
)
