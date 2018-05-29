# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Co2mpas-model specific conversions for co2mparable-hasher.

.. Tip::
    Read instructions on :mod:`hasher` explaining how to launch co2mpas
    in *debugging mode* and populate the structures below.
"""
from co2mpas.model.physical.clutch_tc.torque_converter import TorqueConverter
from co2mpas.model.physical.electrics import Alternator_status_model, \
    AlternatorCurrentModel, ElectricModel
from co2mpas.model.physical.engine import EngineModel
from co2mpas.model.physical.engine.co2_emission import IdleFuelConsumptionModel, FMEP
from co2mpas.model.physical.engine.start_stop import EngineStartStopModel, \
    StartStopModel
from co2mpas.model.physical.engine.thermal import EngineTemperatureModel, \
    ThermalModel
from co2mpas.model.physical.final_drive import FinalDriveModel
from co2mpas.model.physical.gear_box import GearBoxLosses, GearBoxModel
from co2mpas.model.physical.gear_box.at_gear import CorrectGear, GSMColdHot
from co2mpas.model.physical.wheels import WheelsModel

from schedula import Dispatcher, add_args
from schedula.utils.sol import Solution
from xgboost import XGBRegressor

import functools as fnt
import numpy as np
import toolz.dicttoolz as dtz

from .hasher import Hasher, _convert_partial, _convert_obj, _convert_dict


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


def _convert_correct_gear(cg):
    item = vars(cg).copy()
    #mvl = item['mvl']
    pipe = item['pipe']
    ppipe = item['prepare_pipe']
    del item['pipe'], item['prepare_pipe']
    return (*_convert_dict(item), *pipe, *ppipe)  # + \
    #_convert_obj(mvl)


_convert_wltp_hl = fnt.partial(dtz.keyfilter,
                               lambda k: k != 'data')  # `data` is a Solution


def _convert_rank(rank):
    "Rank: a List-ofList where [0][4?] == dict(k: Alternator_status_mode}"
    return [_convert_dict(i) if isinstance(i, dict) else i
            for i in rank[0]]


class Co2Hasher(Hasher):
    #: Map
    #:    {<funame}: (<xarg1>, ...)}
    #: A `None` value exclude the whole function.
    funcs_to_exclude = {
        #'get_cache_fpath': None,
        'default_start_time': None,
        'default_timestamp': None,
        'get_cache_fpath': None,
        'parse_excel_file': None,
        'cache_parsed_data': None,
        'get_template_file_name': None,
        'bypass': None,
        'combine_dicts': None,  # no value mutation
        ## Excel write
        'write_outputs': None,
        'write_to_excel': None,
        '_build_ref': None,
        '_build_id': None,
        '_col2index.*': None,
        '_index2col.*': None,
        ## Report: Actually they are good for a single vehicle.
        #'convert2df': None,  No, we need main-flags
        'parse_dsp_solution': None,
        'make_report': None,
        'get_report_output_data': None,
        'extract_summary': None,
    }

    funs_to_reset = {
    }

    args_to_skip = {
        '_',  # make it a set, even when items below missing
        'output_folder',
        'output_template',
        'vehicle_name',
        'input_file_name',
        'overwrite_cache',
        'output_file_name',
        'timestamp',
        'start_time',
        'output_file_name',     # contains timestamp
        'excel',
        'name',
        'output_data',          # all program data

        ## proc_infos
        'main_flags',
        'dfs',                  # model-names executed

        'gear_filter',          # a function
        'tau_function',         # a `lognorm` native func with 1 input
        'CO2MPAS_results',      # a map with 2 Solutions (h, l)
        'calibrated_models',    # a dict that gets updated.
        'prediction_inputs',    # a dict that gets updated.

        ## WIP: Bored to death...
        'rank',
        'models',
        'model',
        '_item_0'
    }

    args_to_convert = {
        'base_data': _remove_timestamp_from_plan,
        'plan_data': _remove_timestamp_from_plan,
        'correct_gear': _convert_obj,
        'gear_box_loss_model': _convert_obj,
        'idle_fuel_consumption_model': _convert_fmep_in_idle,
        'fmep_model': _convert_interp_partial_in_fmep,
        'correct_gear': _convert_correct_gear,
        'gears': np.asarray,
        'input/wltp_l': _convert_wltp_hl,
        'input/wltp_h': _convert_wltp_hl,
        'inputs<0>': _convert_dict,
        'inputs<1>': _convert_dict,
        'metrics': _convert_dict,  # dict of funcs
        'rank': _convert_rank,
        'data.prediction.models_nedc_h': _convert_dict,
        'data.prediction.models_nedc_l': _convert_dict,
        'data.prediction.models_wltp_h': _convert_dict,
        'data.prediction.models_wltp_l': _convert_dict,
        'data.prediction.wltp_h': _convert_dict,
        'data.prediction.wltp_l': _convert_dict,
        'models_nedc_h': _convert_dict,
        'models_nedc_l': _convert_dict,
        'models_wltp_h': _convert_dict,
        'models_wltp_l': _convert_dict,

        'output_data': _convert_dict,

        ## WIP: doesn't work
        'models': _convert_dict,
        'model': _convert_dict,
    }

    #: Converts them through the standard :func:`_convert_obj()`.
    classes_to_convert = (
        Dispatcher, add_args,
        GearBoxLosses,
        GearBoxModel,
        IdleFuelConsumptionModel,
        FMEP,
        WheelsModel,
        FinalDriveModel,
        EngineTemperatureModel,
        EngineStartStopModel,
        StartStopModel,
        CorrectGear,
        Alternator_status_model,
        EngineModel,
        TorqueConverter,            # XGBoost regressor
        ThermalModel,               # XGBoost regressor
        AlternatorCurrentModel,     # XGBoost regressor
        ElectricModel,
        GSMColdHot,
    )

    classes_to_skip = {
        Solution,
        XGBRegressor,
    }

    #: False: before hash, True: after, None: both
    args_to_print = {
        'main_data': False,         # KEEP THIS ALWAYS
        'input_file_name': False,   # KEEP THIS ALWAYS
        'timestamp': False,         # KEEP THIS ALWAYS
        'dfs': False,               # KEEP THIS ALWAYS (# model-names executed)

        #'calibrated_models': False,
        #'prediction_inputs': False,
        #'output_data': True,            # too big!

        # Why Gears models don't always match???
        'CMV': None,                    # !@dancing order!@
        'GSPV': None,
        'CMV_Cold_Hot': None,           # !@dancing order!@
        'GSPV_Cold_Hot': None,          # !@dancing order!@

        ## WIP: Bored to death...
        #'models': None,
        #'_item_0': None,

        ## Good-to-have
        'raw_data': None,
    }
