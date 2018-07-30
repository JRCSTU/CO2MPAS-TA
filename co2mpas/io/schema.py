# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides CO2MPAS schema parse/validator.
"""

from collections import Iterable, OrderedDict
import datetime
import functools
import logging
import pprint
import re

from schema import Schema, Use, And, Or, Optional, SchemaError

import numpy as np
import os.path as osp
import schedula as sh

from .. import vehicle_family_id_pattern

log = logging.getLogger(__name__)


def check_data_version(flag):
    from co2mpas import __file_version__
    exp_ver = __file_version__
    exp_vinfo = tuple(exp_ver.split('.'))
    ok = False
    try:
        got_ver = flag['input_version']
        got_vinfo = tuple(got_ver.split('.'))

        if got_vinfo[:2] == exp_vinfo[:2]:
            return True

        if got_vinfo[:1] != exp_vinfo[:1]:
            msg = ("Input-file version %s is incompatible with expected %s)."
                   "\n  More failures may happen.")
            return False

        if got_vinfo[:2] > exp_vinfo[:2]:
            msg = ("Input-file version %s comes from the (incompatible) future (> %s))."
                   "\n  More failures may happen.")
        else:  # got_vinfo[:2] < exp_vinfo[:2]:
            msg = ("Input-file version %s is old (< %s))."
                   "\n  You may need to update it, to use new fields.")
            ok = True
        log.warning(msg, got_ver, exp_ver)
    except KeyError:
        msg = "\n  Input file version not found. Please update your input " \
              "file with a version >= %s."
        log.error(msg, exp_ver)

    return ok


def _ta_mode(data):
    base, errors = _validate_base_with_schema(data.get('base', {}))

    if _log_errors_msg(errors):
        return False

    data['base'], _, diff = _extract_declaration_data(base, {})

    for k in ('plan', 'flag'):
        diff.update((k,) + tuple(j.split('.')) for j in data.get(k, {}))

    diff -= {('flag', 'input_version'),
             ('flag', 'vehicle_family_id'),
             }

    if diff:
        diff = ['.'.join(k) for k in sorted(diff)]
        log.info('Since CO2MPAS is launched in type approval mode the '
                 'following data cannot be used:\n %s\n'
                 'If you want to include these data use the cmd batch.',
                 ',\n'.join(diff))
        return False

    if not sh.are_in_nested_dicts(data, 'flag', 'vehicle_family_id') \
            or not data['flag']['vehicle_family_id']:
        log.info('Since CO2MPAS is launched in type approval mode the '
                 '`vehicle_family_id` is required!\n'
                 'If you want to run without it use the cmd batch.')
        return False
    return True


def _extract_declaration_data(inputs, errors):
    from . import validations

    diff = set()
    inputs = validations.select_declaration_data(inputs, diff)
    errors = validations.select_declaration_data(errors)
    return inputs, errors, diff


def _eng_mode_parser(
        engineering_mode, soft_validation, use_selector, inputs, errors):
    from . import validations

    if not engineering_mode:
        inputs, errors, diff = _extract_declaration_data(inputs, errors)
        if diff:
            diff = ['.'.join(k) for k in sorted(diff)]
            log.info('Since CO2MPAS is launched in declaration mode the '
                     'following data are not used:\n %s\n'
                     'If you want to include these data add to the batch cmd '
                     '-D flag.engineering_mode=True',
                     ',\n'.join(diff))

    if not use_selector:
        inputs = validations.overwrite_declaration_config_data(inputs)

    if not soft_validation:
        for k, v in sh.stack_nested_keys(inputs, depth=3):
            for c, msg in validations.hard_validation(v, *k):
                sh.get_nested_dicts(errors, *k)[c] = SchemaError([], [msg])

    return inputs, errors


def validate_plan(plan, engineering_mode, soft_validation, use_selector):
    if not engineering_mode:
        msg = 'Simulation plan cannot be executed without enabling the ' \
              'engineering mode!\n' \
              'If you want to execute it, add to the batch cmd ' \
              '-D flag.engineering_mode=True'
        log.warning(msg)
        return sh.NONE

    from . import excel
    read_schema = define_data_schema(read=True)
    flag_read_schema = define_flags_schema(read=True)
    validated_plan, errors, v_data = [], {}, read_schema.validate
    v_flag = flag_read_schema.validate
    for i, data in plan.iterrows():
        inputs, inp = {}, {}
        data.dropna(how='all', inplace=True)
        plan_id = 'plan id:{}'.format(i[0])
        for k, v in excel._parse_values(data, where='in plan'):
            if k[0] == 'base':
                d = sh.get_nested_dicts(inp, *k[1:-1])
                v = _add_validated_input(d, v_data, (plan_id,) + k, v, errors)
            elif k[0] == 'flag':
                v = _add_validated_input({}, v_flag, (plan_id,) + k, v, errors)

            if v is not sh.NONE:
                inputs[k] = v

        errors = _eng_mode_parser(
            engineering_mode, soft_validation, use_selector, inp, errors
        )[1]

        validated_plan.append((i, inputs))

    if _log_errors_msg(errors):
        return sh.NONE

    return validated_plan


def _validate_base_with_schema(data, depth=4):
    read_schema = define_data_schema(read=True)
    inputs, errors, validate = {}, {}, read_schema.validate
    for k, v in sorted(sh.stack_nested_keys(data, depth=depth)):
        d = sh.get_nested_dicts(inputs, *k[:-1])
        _add_validated_input(d, validate, k, v, errors)

    return inputs, errors


def validate_base(data, engineering_mode, soft_validation, use_selector):
    i, e = _validate_base_with_schema(data)

    i, e = _eng_mode_parser(
        engineering_mode, soft_validation, use_selector, i, e
    )

    if _log_errors_msg(e):
        return sh.NONE

    return {'.'.join(k): v for k, v in sh.stack_nested_keys(i, depth=3)}


def validate_meta(data, soft_validation):
    i, e = _validate_base_with_schema(data, depth=2)
    if not soft_validation:
        from . import validations
        for k, v in sorted(sh.stack_nested_keys(i, depth=1)):
            for c, msg in validations.hard_validation(v, 'meta'):
                sh.get_nested_dicts(e, *k)[c] = SchemaError([], [msg])

    if _log_errors_msg(e):
        return sh.NONE

    return i


def validate_flags(flags):
    read_schema = define_flags_schema(read=True)
    inputs, errors, validate = {}, {}, read_schema.validate
    for k, v in sorted(flags.items()):
        _add_validated_input(inputs, validate, ('flag', k), v, errors)
    if _log_errors_msg(errors):
        return sh.NONE
    return inputs


def _add_validated_input(data, validate, keys, value, errors):
    try:
        k, v = next(iter(validate({keys[-1]: value}).items()))
        if v is not sh.NONE:
            data[k] = v
            return v
    except SchemaError as ex:
        sh.get_nested_dicts(errors, *keys[:-1])[keys[-1]] = ex
    return sh.NONE


def _log_errors_msg(errors):
    if errors:
        msg = ['\nInput cannot be parsed, due to:']
        for k, v in sh.stack_nested_keys(errors, depth=4):
            msg.append('{} in {}: {}'.format(k[-1], '/'.join(k[:-1]), v))
        log.error('\n  '.join(msg))
        return True
    return False


class Empty(object):
    def __repr__(self):
        return '%s' % self.__class__.__name__

    @staticmethod
    def validate(data):
        if isinstance(data, str) and data == 'EMPTY':
            return sh.EMPTY

        try:
            empty = not (data or data == 0)
        except ValueError:
            empty = np.isnan(data).all()

        if empty:
            return sh.NONE
        else:
            raise SchemaError('%r is not empty' % data, None)


# noinspection PyUnusedLocal
def _function(error=None, read=True, **kwargs):
    def _check_function(f):
        assert callable(f)
        return f

    if read:
        error = error or 'should be a function!'
        return _eval(Use(_check_function), error=error)
    return And(_check_function, Use(lambda x: sh.NONE), error=error)


# noinspection PyUnusedLocal
def _string(error=None, **kwargs):
    error = error or 'should be a string!'
    return Use(str, error=error)


# noinspection PyUnusedLocal
def _select(types=(), error=None, **kwargs):
    error = error or 'should be one of {}!'.format(types)
    types = {k.lower(): k for k in types}
    return And(str, Use(lambda x: types[x.lower()]), error=error)


def _check_positive(x):
    return x >= 0


# noinspection PyUnusedLocal
def _positive(type=float, error=None, check=_check_positive, **kwargs):
    error = error or 'should be as {} and positive!'.format(type)
    return And(Use(type), check, error=error)


# noinspection PyUnusedLocal
def _limits(limits=(0, 100), error=None, **kwargs):
    error = error or 'should be {} <= x <= {}!'.format(*limits)

    def _check_limits(x):
        return limits[0] <= x <= limits[1]

    return And(Use(float), _check_limits, error=error)


# noinspection PyUnusedLocal
def _eval(s, error=None, **kwargs):
    error = error or 'cannot be eval!'

    def _eval(x):
        from lmfit import Parameters, Parameter
        from co2mpas.model.physical.clutch_tc.clutch import ClutchModel
        from co2mpas.model.physical.clutch_tc.torque_converter import \
            TorqueConverter
        from co2mpas.model.physical.engine.start_stop import StartStopModel
        from co2mpas.model.physical.engine.cold_start import ColdStartModel
        from co2mpas.model.physical.engine.thermal import ThermalModel
        from co2mpas.model.physical.gear_box.at_gear import CMV, MVL, GSPV
        return eval(x)

    return Or(And(str, Use(_eval), s), s, error=error)


# noinspection PyUnusedLocal
def _dict(format=None, error=None, read=True, pformat=pprint.pformat, **kwargs):
    format = And(dict, format or {int: float})
    error = error or 'should be a dict with this format {}!'.format(format)
    c = Use(lambda x: {k: v for k, v in dict(x).items() if v is not None})
    if read:
        return _eval(Or(Empty(), And(c, Or(Empty(), format))), error=error)
    else:
        return And(_dict(format=format, error=error), Use(pformat))


# noinspection PyUnusedLocal
def _ordict(format=None, error=None, read=True, **kwargs):
    import pprint

    format = format or {int: float}
    msg = 'should be a OrderedDict with this format {}!'
    error = error or msg.format(format)
    c = Use(OrderedDict)
    if read:
        return _eval(Or(Empty(), And(c, Or(Empty(), format))), error=error)
    else:
        return And(_dict(format=format, error=error), Use(pprint.pformat))


def _check_length(length):
    if not isinstance(length, Iterable):
        length = (length,)

    def check_length(data):
        ld = len(data)
        return any(ld == l for l in length)

    return check_length


# noinspection PyUnusedLocal
def _type(type=None, error=None, length=None, **kwargs):
    type = type or tuple

    if length is not None:
        error = error or 'should be as {} and ' \
                         'with a length of {}!'.format(type, length)
        return And(_type(type=type), _check_length(length), error=error)
    if not isinstance(type, (Use, Schema, And, Or)):
        type = Or(type, Use(type))
    error = error or 'should be as {}!'.format(type)
    return _eval(type, error=error)


# noinspection PyUnusedLocal
def _index_dict(error=None, **kwargs):
    error = error or 'cannot be parsed as {}!'.format({int: float})
    c = {int: Use(float)}
    s = And(dict, c)

    def f(x):
        return {k: v for k, v in enumerate(x, start=1)}

    return Or(s, And(_dict(), c), And(_type(), Use(f), c), error=error)


# noinspection PyUnusedLocal
def _np_array(dtype=None, error=None, read=True, **kwargs):
    dtype = dtype or float
    error = error or 'cannot be parsed as np.array dtype={}!'.format(dtype)
    if read:
        c = Use(lambda x: np.asarray(x, dtype=dtype))
        return Or(And(str, _eval(c)), c, And(_type(), c), Empty(), error=error)
    else:
        return And(_np_array(dtype=dtype), Use(lambda x: x.tolist()),
                   error=error)


def _check_np_array_positive(x):
    """

    :param x:
        Array.
    :type x: numpy.array
    :return:
    """
    # noinspection PyUnresolvedReferences
    return (x >= 0).all()


# noinspection PyUnusedLocal
def _np_array_positive(dtype=None, error=None, read=True,
                       check=_check_np_array_positive, **kwargs):
    dtype = dtype or float
    error = error or 'cannot be parsed because it should be an ' \
                     'np.array dtype={} and positive!'.format(dtype)
    if read:
        c = And(Use(lambda x: np.asarray(x, dtype=dtype)), check)
        return Or(And(str, _eval(c)), c, And(_type(), c), Empty(),
                  error=error)
    else:
        return And(_np_array_positive(dtype=dtype), Use(lambda x: x.tolist()),
                   error=error)


# noinspection PyUnusedLocal
def _cold_start_speed_model(error=None, **kwargs):
    from co2mpas.model.physical.engine.cold_start import ColdStartModel
    return _type(type=ColdStartModel, error=error)

# noinspection PyUnusedLocal
def _cmv(error=None, **kwargs):
    from co2mpas.model.physical.gear_box.at_gear import CMV
    return _type(type=CMV, error=error)


# noinspection PyUnusedLocal
def _mvl(error=None, **kwargs):
    from co2mpas.model.physical.gear_box.at_gear import MVL
    return _type(type=MVL, error=error)


# noinspection PyUnusedLocal
def _gspv(error=None, **kwargs):
    from co2mpas.model.physical.gear_box.at_gear import GSPV
    return _type(type=GSPV, error=error)


# noinspection PyUnusedLocal
def _gsch(error=None, **kwargs):
    from co2mpas.model.physical.gear_box.at_gear import GSMColdHot
    return _type(type=GSMColdHot, error=error)


# noinspection PyUnusedLocal
def _dtc(error=None, read=True, **kwargs):
    if read:
        from co2mpas.model.physical.gear_box.at_gear import DTGS
        return _type(type=DTGS, error=error)
    return And(_dtc(), Use(lambda x: sh.NONE), error=error)

# noinspection PyUnusedLocal
def _cvt(error=None, read=True, **kwargs):
    if read:
        from co2mpas.model.physical.gear_box.cvt import CVT
        return _type(type=CVT, error=error)
    return And(_dtc(), Use(lambda x: sh.NONE), error=error)


def _parameters2str(data):
    from lmfit import Parameters
    if isinstance(data, Parameters):
        return data.dumps(sort_keys=True)


def _str2parameters(data):
    if isinstance(data, str):
        from lmfit import Parameters
        p = Parameters()
        p.loads(data)
        return p
    return data


def _parameters(error=None, read=True):
    if read:
        from lmfit import Parameters
        return And(Use(_str2parameters), _type(type=Parameters, error=error))
    else:
        return And(_parameters(), Use(_parameters2str), error=error)


# noinspection PyUnusedLocal
def _compare_str(s, **kwargs):
    return And(Use(str.lower), s.lower(), Use(lambda x: s))


def _convert_str(old_str, new_str, **kwargs):
    return And(Use(str), Or(old_str, new_str), Use(lambda x: new_str))


def _tyre_code(error=None, **kwargs):
    error = error or 'invalid tyre code!'
    from ..model.physical.wheels import _re_tyre_code_iso, _re_tyre_code_numeric
    c = Or(_re_tyre_code_iso.match, _re_tyre_code_numeric.match)
    return And(str, c, error=error)


def _tyre_dimensions(error=None, **kwargs):
    error = error or 'invalid format for tyre dimensions!'
    from ..model.physical.wheels import _format_tyre_dimensions
    return And(_dict(format=dict), Use(_format_tyre_dimensions), error=error)


def check_phases_separated(x):
    """
    >>> bags = [
        [3, 2, 3, 2, 4, 4, 1, 1], # INVALID!
        [3, 2, 2, 2, 4, 4, 1, 4], # INVALID!
        [3, 3, 2, 2, 4, 4, 1, 1], # valid
        ['P1', 'P3', 'P3', 'P2'], # valid
        [False, False],           # valid
        [],                       # valid
    ]
    >>> [check_phases_separated(x) for x in bags]
    [False, False, True, True, True, True]
    """
    if not len(x):
        return True

    x = np.asarray(x)
    ## See http://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates
    deduped_count = 1 + (
                x[1:] != x[:-1]).sum()  # [3,3,3,1,1,3] --> len([3,1,3])

    return deduped_count == np.unique(x).size


def _bag_phases(error=None, read=True, **kwargs):
    er = 'Phases must be separated!'
    if read:
        return And(_np_array(read=read),
                   Schema(check_phases_separated, error=er), error=error)
    else:
        return And(_bag_phases(error, True), _np_array(read=False), error=error)


def _file(error=None, **kwargs):
    er = 'Must be a file!'
    return And(_string(), Schema(osp.isfile, error=er), error=error)


def _dir(error=None, **kwargs):
    er = 'Must be a directory!'
    return And(_string(), Schema(osp.isdir, error=er), error=error)


def is_sorted(iterable, key=lambda a, b: a <= b):
    return all(key(a, b) for a, b in sh.pairwise(iterable))


# noinspection PyUnresolvedReferences
@functools.lru_cache(None)
def define_data_schema(read=True):
    cssm = _cold_start_speed_model(read=read)
    cmv = _cmv(read=read)
    dtc = _dtc(read=read)
    cvt = _cvt(read=read)
    gspv = _gspv(read=read)
    gsch = _gsch(read=read)
    string = _string(read=read)
    positive = _positive(read=read)
    greater_than_zero = _positive(
        read=read, error='should be as <float> and greater than zero!',
        check=lambda x: x > 0
    )
    greater_than_one = _positive(
        read=read, error='should be as <float> and greater than one!',
        check=lambda x: x >= 1
    )
    positive_int = _positive(type=int, read=read)
    limits = _limits(read=read)
    index_dict = _index_dict(read=read)
    np_array = _np_array(read=read)
    np_array_sorted = _np_array_positive(
        read=read, error='cannot be parsed because it should be an '
                         'np.array dtype=<float> with ascending order!',
        check=is_sorted
    )
    np_array_greater_than_minus_one = _np_array_positive(
        read=read, error='cannot be parsed because it should be an '
                         'np.array dtype=<float> and all values >= -1!',
        check=lambda x: (x >= -1).all()
    )
    np_array_bool = _np_array(dtype=bool, read=read)
    np_array_int = _np_array(dtype=int, read=read)
    _bool = _type(type=bool, read=read)
    function = _function(read=read)
    tuplefloat2 = _type(
        type=And(Use(tuple), (_type(float),)),
        length=2,
        read=read
    )
    tuplefloat = _type(type=And(Use(tuple), (_type(float),)), read=read)
    dictstrdict = _dict(format={str: dict}, read=read)
    ordictstrdict = _ordict(format={str: dict}, read=read)
    parameters = _parameters(read=read)
    dictstrfloat = _dict(format={str: Use(float)}, read=read)
    dictarray = _dict(format={str: np_array}, read=read)
    tyre_code = _tyre_code(read=read)
    tyre_dimensions = _tyre_dimensions(read=read)

    schema = {
        _compare_str('CVT'): cvt,
        _compare_str('CMV'): cmv,
        _compare_str('CMV_Cold_Hot'): gsch,
        _compare_str('DTGS'): dtc,
        _compare_str('GSPV'): gspv,
        _compare_str('GSPV_Cold_Hot'): gsch,
        _compare_str('MVL'): _mvl(read=read),
        'engine_n_cylinders': positive_int,
        'lock_up_tc_limits': tuplefloat2,
        _convert_str(
            'ki_factor', 'ki_multiplicative'
        ): greater_than_one,
        'ki_additive': positive,
        'tyre_dimensions': tyre_dimensions,
        'tyre_code': tyre_code,
        'wltp_base_model': _dict(format=dict, read=read),
        'fuel_type': _select(types=(
            'gasoline', 'diesel', 'LPG', 'NG', 'ethanol', 'biodiesel',
            'methanol', 'propane'), read=read),
        'obd_fuel_type_code': positive_int,
        'vehicle_category': _select(types='ABCDEFSMJ', read=read),
        'vehicle_body': _select(types=(
            'cabriolet', 'sedan', 'hatchback', 'stationwagon', 'suv/crossover',
            'mpv', 'coupé', 'bus', 'bestelwagen', 'pick-up'
        ), read=read),
        'tyre_class': _select(types=('C1', 'C2', 'C3'), read=read),
        'tyre_category': _select(types='ABCDEFG', read=read),
        'engine_fuel_lower_heating_value': positive,
        'fuel_carbon_content': positive,
        'engine_capacity': positive,
        'engine_stroke': positive,
        'engine_max_power': positive,
        _convert_str(
            'engine_max_speed_at_max_power', 'engine_speed_at_max_power'
        ): positive,
        'engine_max_speed': positive,
        'engine_max_torque': positive,
        'idle_engine_speed_median': positive,
        'engine_idle_fuel_consumption': greater_than_zero,
        'final_drive_ratio': positive,
        'r_dynamic': positive,
        'wltp_class': _select(types=('class1', 'class2', 'class3a', 'class3b'),
                              read=read),
        'downscale_phases': tuplefloat,
        'gear_box_type': _select(types=('manual', 'automatic', 'cvt'),
                                 read=read),
        'ignition_type': _select(types=('positive', 'compression'), read=read),
        'start_stop_activation_time': positive,
        'alternator_nominal_voltage': positive,
        'battery_capacity': positive,
        'state_of_charge_balance': limits,
        'state_of_charge_balance_window': limits,
        'initial_state_of_charge ': limits,
        'idle_engine_speed_std': positive,
        'alternator_nominal_power': positive,
        'alternator_efficiency': _limits(limits=(0, 1), read=read),
        'time_cold_hot_transition': positive,
        'co2_params': dictstrfloat,
        'willans_factors': dictstrfloat,
        'phases_willans_factors': _type(
            type=And(Use(tuple), (dictstrfloat,)), read=read),
        'optimal_efficiency': dictarray,
        'velocity_speed_ratios': index_dict,
        'gear_box_ratios': index_dict,
        'final_drive_ratios': index_dict,
        'speed_velocity_ratios': index_dict,
        'full_load_speeds': np_array_sorted,
        'full_load_torques': np_array,
        'full_load_powers': np_array,

        'vehicle_mass': positive,
        'f0_uncorrected': positive,
        'f2': positive,
        'f0': positive,
        'correct_f0': _bool,

        'co2_emission_low': positive,
        'co2_emission_medium': positive,
        'co2_emission_high': positive,
        'co2_emission_extra_high': positive,

        _compare_str('co2_emission_UDC'): positive,
        _compare_str('co2_emission_EUDC'): positive,
        'co2_emission_value': positive,
        'declared_co2_emission_value': positive,
        'n_dyno_axes': positive_int,
        'n_wheel_drive': positive_int,

        'engine_is_turbo': _bool,
        'has_start_stop': _bool,
        'has_gear_box_thermal_management': _bool,
        'has_energy_recuperation': _bool,
        'use_basic_start_stop': _bool,
        'is_hybrid': _bool,
        'has_roof_box': _bool,
        'has_periodically_regenerating_systems': _bool,
        'engine_has_variable_valve_actuation': _bool,
        'has_thermal_management': _bool,
        'engine_has_direct_injection': _bool,
        'has_lean_burn': _bool,
        'engine_has_cylinder_deactivation': _bool,
        'has_exhausted_gas_recirculation': _bool,
        'has_particle_filter': _bool,
        'has_selective_catalytic_reduction': _bool,
        'has_nox_storage_catalyst': _bool,
        'has_torque_converter': _bool,
        'is_cycle_hot': _bool,
        'use_dt_gear_shifting': _bool,
        _convert_str('eco_mode', 'fuel_saving_at_strategy'): _bool,
        'correct_start_stop_with_gears': _bool,
        'enable_phases_willans': _bool,
        'enable_willans': _bool,

        'alternator_charging_currents': tuplefloat2,
        'alternator_current_model': function,
        'alternator_status_model': function,
        'clutch_model': function,
        'co2_emissions_model': function,
        'co2_error_function_on_emissions': function,
        'co2_error_function_on_phases': function,
        'cold_start_speed_model': cssm,
        'clutch_window': tuplefloat2,
        'co2_params_calibrated': parameters,
        'co2_params_initial_guess': parameters,
        'cycle_type': string,
        'cycle_name': string,
        'specific_gear_shifting': string,
        'calibration_status': _type(type=And(Use(list),
                                             [(bool, Or(parameters, None))]),
                                    length=4,
                                    read=read),
        'electric_load': tuplefloat2,
        'engine_thermostat_temperature_window': tuplefloat2,
        'engine_temperature_regression_model': function,
        'electrics_prediction_model': function,
        'engine_prediction_model': function,
        'gear_box_prediction_model': function,
        'final_drive_prediction_model': function,
        'wheels_prediction_model': function,
        'engine_type': string,
        'full_load_curve': function,
        'fmep_model': function,
        'gear_box_efficiency_constants': dictstrdict,
        'gear_box_efficiency_parameters_cold_hot': dictstrdict,
        'config': dictstrdict,
        'scores': dictstrdict,
        'param_selections': dictstrdict,
        'model_selections': dictstrdict,
        'score_by_model': dictstrdict,
        'at_scores': ordictstrdict,

        'fuel_density': positive,
        'idle_engine_speed': tuplefloat2,
        'k1': positive_int,
        'k2': positive_int,
        'k5': positive_int,
        'max_gear': positive_int,

        'road_loads': _type(type=And(Use(tuple), (_type(float),)),
                            length=3,
                            read=read),
        'start_stop_model': function,
        'gear_box_temperature_references': tuplefloat2,
        'torque_converter_model': function,
        'phases_co2_emissions': tuplefloat,
        'bag_phases': _bag_phases(read=read),
        'phases_integration_times':
            _type(type=And(Use(tuple), (And(Use(tuple), (_type(float),)),)),
                  read=read),
        'active_cylinder_ratios': tuplefloat,
        'extended_phases_co2_emissions': tuplefloat,
        'extended_phases_integration_times':
            _type(type=And(Use(tuple), (And(Use(tuple), (_type(float),)),)),
                  read=read),
        'extended_integration_times': tuplefloat,
        'phases_fuel_consumptions': tuplefloat,
        'co2_rescaling_scores': tuplefloat,
        'accelerations': np_array,
        'alternator_currents': np_array,
        'active_cylinders': np_array,
        'alternator_powers_demand': np_array,
        'alternator_statuses': np_array_int,
        'auxiliaries_power_losses': np_array,
        'auxiliaries_torque_loss': tuplefloat,
        'auxiliaries_torque_losses': np_array,
        'battery_currents': np_array,
        'clutch_tc_powers': np_array,
        'clutch_tc_speeds_delta': np_array,
        'co2_emissions': np_array,
        'cold_start_speeds_delta': np_array,
        'engine_coolant_temperatures': np_array,
        'engine_powers_out': np_array,
        'engine_speeds_out': np_array,
        'engine_speeds_out_hot': np_array,
        'engine_starts': np_array_bool,
        'co2_normalization_references': np_array,
        'final_drive_powers_in': np_array,
        'final_drive_speeds_in': np_array,
        'final_drive_torques_in': np_array,
        'fuel_consumptions': np_array,
        'gear_box_efficiencies': np_array,
        'gear_box_powers_in': np_array,
        'gear_box_speeds_in': np_array,
        'gear_box_temperatures': np_array,
        'gear_box_torque_losses': np_array,
        'gear_box_torques_in': np_array,
        'gear_shifts': np_array_bool,
        'gears': np_array_int,
        'identified_co2_emissions': np_array,
        'motive_powers': np_array,
        'on_engine': np_array_bool,
        'clutch_phases': np_array_bool,
        'on_idle': np_array_bool,
        'state_of_charges': np_array,
        'times': np_array_sorted,
        'velocities': np_array_greater_than_minus_one,
        _compare_str('obd_velocities'): np_array_greater_than_minus_one,
        'wheel_powers': np_array,
        'wheel_speeds': np_array,
        'wheel_torques': np_array,
    }

    schema = {Optional(k): Or(Empty(), v) for k, v in schema.items()}
    schema[Optional(str)] = Or(_type(type=float, read=read), np_array)

    if not read:
        def f(x):
            return x is sh.NONE

        schema = {k: And(v, Or(f, Use(str))) for k, v in schema.items()}

    return Schema(schema)


#: Aka "ProjectId", referenced also by :mod:`.sampling.project`.
_vehicle_family_id_regex = re.compile('(?x)^%s$' % vehicle_family_id_pattern)
invalid_vehicle_family_id_msg = (
    "Invalid VF_ID '%s'!"
    "\n  New format is 'IP-nnn-WMI-x', where nnn is (2, 15) chars "
    "of A-Z, 0-9, or underscore(_),"
    "\n  (old format 'FT-ta-WMI-yyyy-nnnn' is still acceptable)."
)


def vehicle_family_id(error=None, **kwargs):
    def m(s):
        if not _vehicle_family_id_regex.match(s):
            raise SchemaError(invalid_vehicle_family_id_msg % s)
        return True

    return And(_string(**kwargs), m, error=error)


@functools.lru_cache(None)
def define_flags_schema(read=True):
    string = _string(read=read)
    isfile = _file(read=read)
    isdir = _dir(read=read)
    _bool = _type(type=bool, read=read)
    _datetime = _type(type=datetime.datetime, read=read)

    schema = {
        _compare_str('hostname'): string,
        _compare_str('input_version'): string,
        _compare_str('vehicle_family_id'): vehicle_family_id(read=read),
        _compare_str('modelconf'): isfile,
        _compare_str('encryption_keys'): string,

        _compare_str('encrypt_inputs'): _bool,
        _compare_str('soft_validation'): _bool,
        _compare_str('use_selector'): _bool,
        _compare_str('engineering_mode'): _bool,
        _compare_str('run_plan'): _bool,
        _compare_str('run_base'): _bool,
        _compare_str('only_summary'): _bool,
        _compare_str('plot_workflow'): _bool,
        _compare_str('overwrite_cache'): _bool,
        _compare_str('type_approval_mode'): _bool,

        _compare_str('vehicle_name'): string,

        _compare_str('output_template'): isfile,
        _compare_str('output_file_name'): string,
        _compare_str('output_folder'): isdir,

        _compare_str('start_time'): _datetime,
        _compare_str('timestamp'): string,
    }

    schema = {Optional(k): Or(Empty(), v) for k, v in schema.items()}

    if not read:
        def f(x):
            return x is sh.NONE

        schema = {k: And(v, Or(f, Use(str))) for k, v in schema.items()}

    return Schema(schema)
