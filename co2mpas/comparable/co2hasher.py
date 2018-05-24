# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from schedula import Dispatcher, add_args

import toolz.dicttoolz as dtz
from co2mpas.model.physical.gear_box import GearBoxLosses, GearBoxModel
from co2mpas.model.physical.engine.co2_emission import IdleFuelConsumptionModel, FMEP
from co2mpas.model.physical.wheels import WheelsModel
from co2mpas.model.physical.final_drive import FinalDriveModel
from . import ComparableHasher, _convert_partial


def _remove_timestamp_from_plan(item):
    notstamp = lambda k: k != 'timestamp'
    item = dtz.keyfilter(notstamp, item)
    item['flag'] = dtz.keyfilter(notstamp, item['flag'])

    return item


def _convert_interp_partial_in_fmep(item):
    item = vars(item).copy()
    item['fbc'] = _convert_partial(item['fbc'])
    return item


def _convert_fmep_in_idle(item):
    item = vars(item).copy()
    item['fmep_model'] = _convert_interp_partial_in_fmep(item['fmep_model'])
    return item


class Co2Hasher(ComparableHasher):
    #: Map
    #:    {<funame}: (<xarg1>, ...)}
    #: A `None` value exclude the whole function.
    @property
    def funs_to_exclude(self):
        return {
            #'get_cache_fpath': None,
            'default_start_time': None,
            'default_timestamp': None,
        }

    @property
    def funs_to_reset(self):
        return {
        }

    @property
    def args_to_exclude(self):
        return {
            '_',  # make it a set, even when items below missing
            'output_folder',
            'vehicle_name',
            'output_file_name',
            'timestamp',
            'start_time',
            'output_file_name',     # contains timestamp
            'excel',
            'name',

            'gear_filter',          # a function
            'tau_function',         # a function
            'k_factor_curve',       # a function
            # 'full_load_curve',      # an InterpolatedUnivariateSpline
        }

    @property
    def args_to_print(self):
        return {
            '_',
            'full_bmep_curve',
            'correct_gear',
            'error_function_on_emissions',
        }

    @property
    def args_to_convert(self):
        return {
            'base_data': _remove_timestamp_from_plan,
            'plan_data': _remove_timestamp_from_plan,
            #'correct_gear': _convert_obj,
            #'gear_box_loss_model': _convert_obj,
            'idle_fuel_consumption_model': _convert_fmep_in_idle,
            'fmep_model': _convert_interp_partial_in_fmep,
        }

    @property
    def objects_to_convert(self):
        return (
            Dispatcher, add_args,
            GearBoxLosses,
            GearBoxModel,
            IdleFuelConsumptionModel,
            FMEP,
            WheelsModel,
            FinalDriveModel,
        )
