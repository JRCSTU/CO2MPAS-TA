#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It provides validation functions and the validation model `dsp`.

Sub-Modules:

.. currentmodule:: co2mpas.core.load.validate

.. autosummary::
    :nosignatures:
    :toctree: validate/

    hard
"""
import schema
import logging
import schedula as sh

log = logging.getLogger(__name__)
dsp = sh.BlueDispatcher(
    name='validate_data', description='Validates input data.'
)


def _add_validated_input(data, validate, keys, value, errors):
    try:
        k, v = next(iter(validate({keys[-1]: value}).items()))
        if v is not sh.NONE:
            data[k] = v
            return v
    except schema.SchemaError as ex:
        sh.get_nested_dicts(errors, *keys[:-1])[keys[-1]] = ex
    return sh.NONE


def _validate_base_with_schema(data, depth=4):
    from ..schema import define_data_schema
    inputs, errors, validate = {}, {}, define_data_schema().validate
    for k, v in sorted(sh.stack_nested_keys(data, depth=depth)):
        d = sh.get_nested_dicts(inputs, *k[:-1])
        _add_validated_input(d, validate, k, v, errors)

    return inputs, errors


def _log_errors_msg(errors):
    if errors:
        msg = ['\nInput cannot be parsed, due to:']
        for k, v in sh.stack_nested_keys(errors, depth=4):
            msg.append('{} in {}: {}'.format(k[-1], '/'.join(k[:-1]), v))
        log.error('\n  '.join(msg))
        return True
    return False


def _mode_parser(
        type_approval_mode, declaration_mode, hard_validation, inputs, errors):
    """
    Parse input data for the declaration mode.

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :param declaration_mode:
        Use only the declaration data.
    :type declaration_mode: bool

    :param hard_validation:
        Add extra data validations.
    :type hard_validation: bool

    :param inputs:
        Input data.
    :type inputs: dict

    :param errors:
        Errors container.
    :type errors: dict

    :return:
        Parsed input data and errors container.
    :rtype: dict, dict
    """

    if type_approval_mode or declaration_mode:
        from dice.co2mpas.declaration import declaration_validation
        inputs, errors = declaration_validation(
            type_approval_mode, inputs, errors
        )

    if hard_validation:
        from .hard import hard_validation as validation
        inputs, errors = validation(inputs, errors)

    return inputs, errors


_kw = dict(inputs_kwargs=True, inputs_defaults=True)


@sh.add_function(dsp, outputs=['validated_base'], **_kw)
def validate_base(
        base=None, declaration_mode=False, hard_validation=False,
        enable_selector=False, type_approval_mode=False):
    """
    Validate base data.

    :param base:
        Base data.
    :type base: dict

    :param declaration_mode:
        Use only the declaration data.
    :type declaration_mode: bool

    :param hard_validation:
        Add extra data validations.
    :type hard_validation: bool

    :param enable_selector:
        Enable the selection of the best model to predict both H/L cycles.
    :type enable_selector: bool

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :return:
        Validated base data.
    :rtype: dict
    """
    i, e = _validate_base_with_schema(base or {})

    i, e = _mode_parser(
        type_approval_mode, declaration_mode, hard_validation, i, e
    )

    if _log_errors_msg(e):
        return sh.NONE
    i['enable_selector'] = enable_selector
    return {'.'.join(k): v for k, v in sh.stack_nested_keys(i, depth=3)}


@sh.add_function(dsp, outputs=['validated_meta'], **_kw)
def validate_meta(meta=None, hard_validation=False):
    """
    Validate meta data.

    :param meta:
        Meta data.
    :type meta: dict

    :param hard_validation:
        Add extra data validations.
    :type hard_validation: bool

    :return:
        Validated meta data.
    :rtype: dict
    """
    i, e = _validate_base_with_schema(meta or {}, depth=2)
    if hard_validation:
        from .hard import _hard_validation
        for k, v in sorted(sh.stack_nested_keys(i, depth=1)):
            for c, msg in _hard_validation(v, 'meta'):
                sh.get_nested_dicts(e, *k)[c] = schema.SchemaError([], [msg])

    if _log_errors_msg(e):
        return sh.NONE

    return i


@sh.add_function(dsp, outputs=['validated_dice'], **_kw)
def validate_dice(dice=None):
    """
    Validate DICE data.

    :param dice:
        DICE data.
    :type dice: dict

    :return:
        Validated DICE data.
    :rtype: dict
    """
    from ..schema import define_dice_schema
    inputs, errors, validate = {}, {}, define_dice_schema().validate
    for k, v in sorted((dice or {}).items()):
        _add_validated_input(inputs, validate, ('dice', k), v, errors)

    if inputs.get('extension') and inputs.get('wltp_retest', '-') != '-':
        sh.get_nested_dicts(errors, 'dice')['extension'] = (
            "Invalid combination `dice.extension == True` and "
            "`dice.wltp_retest != '-'`. Please set `dice.extension = False` "
            "or set `dice.wltp_retest = '-'`!"
        )

    if _log_errors_msg(errors):
        return sh.NONE
    return inputs


@sh.add_function(dsp, outputs=['validated_flag'], **_kw)
def validate_flag(flag=None):
    """
    Validate flags data.

    :param flag:
        Flags data.
    :type flag: dict

    :return:
        Validated flags data.
    :rtype: dict
    """
    from ..schema import define_flags_schema
    inputs, errors, validate = {}, {}, define_flags_schema().validate
    for k, v in sorted((flag or {}).items()):
        _add_validated_input(inputs, validate, ('flag', k), v, errors)
    if _log_errors_msg(errors):
        return sh.NONE
    return inputs


@sh.add_function(dsp, outputs=['validated_plan'], **_kw)
def validate_plan(
        plan=None, declaration_mode=False, hard_validation=False,
        type_approval_mode=False):
    """
    Validate plan data.

    :param plan:
        Plan data.
    :type plan: dict

    :param declaration_mode:
        Use only the declaration data.
    :type declaration_mode: bool

    :param hard_validation:
        Add extra data validations.
    :type hard_validation: bool

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :return:
        Validated plan data.
    :rtype: dict
    """
    import pandas as pd
    if plan and declaration_mode:
        msg = 'Simulation plan cannot be executed in declaration mode!\n' \
              'If you want to execute it remove -DM or -TA from the cmd.'
        log.warning(msg)
        return []
    import os.path as osp
    from ..excel import _parse_values as parse_key
    from ..schema import define_data_schema as _schema

    validated_plan, e, keys = [], {}, {'id', 'base', 'run_base'}
    validate, add = _schema().validate, validated_plan.append

    for _, d in pd.DataFrame(plan or []).iterrows():
        d = dict(d.dropna(how='all'))
        d['base'] = osp.abspath(d['base'])
        i, data, p_id = {}, {}, 'plan id:{}'.format(d['id'])

        for k, v in parse_key(sh.selector(set(d) - keys, d), where='in plan'):
            k, inp = (p_id,) + k, sh.get_nested_dicts(i, *k[1:-1])
            v = _add_validated_input(inp, validate, k, v, e)
            if v is not sh.NONE:
                sh.get_nested_dicts(data, '.'.join(k[2:-1]))[k[-1]] = v

        e = _mode_parser(
            type_approval_mode, declaration_mode, hard_validation, i, e
        )[1]
        add(sh.combine_dicts({'data': data}, base=sh.selector(keys, d)))

    if _log_errors_msg(e):
        return sh.NONE

    return validated_plan


@sh.add_function(dsp, outputs=['verified'], **_kw)
def validation_status(
        validated_base, validated_flag, validated_meta, validated_dice,
        validated_plan, type_approval_mode=False):
    """
    Returns the validation status.

    :param validated_base:
        Validated base data.
    :type validated_base: dict

    :param validated_flag:
        Validated flags data.
    :type validated_flag: dict

    :param validated_meta:
        Validated meta data.
    :type validated_meta: dict

    :param validated_dice:
        Validated DICE data.
    :type validated_dice: dict

    :param validated_plan:
        Validated plan data.
    :type validated_plan: dict

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :return:
        Validation status.
    :rtype: bool
    """
    if type_approval_mode:
        from dice.co2mpas.verify import verify_data
        return verify_data({
            'base': validated_base,
            'flag': validated_flag,
            'meta': validated_meta,
            'dice': validated_dice,
            'plan': validated_plan
        })
    return True
