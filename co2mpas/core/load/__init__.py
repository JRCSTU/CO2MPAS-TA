# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions to read inputs from excel.
"""
import io
import logging
import functools
import schedula as sh
from .excel import parse_excel_file
from .validate import dsp as _validate

log = logging.getLogger(__name__)

dsp = sh.BlueDispatcher(
    name='load_inputs',
    description='Loads from files the inputs for the CO2MPAS model.'
)


@sh.add_function(dsp, outputs=['input_file'])
def open_input_file(input_file_name):
    """
    Open the input file.

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :return:
        Input file.
    :rtype: io.BytesIO
    """
    with open(input_file_name, 'rb') as file:
        return io.BytesIO(file.read())


# noinspection PyUnusedLocal
def check_file_format(input_file_name, *args, extensions=('.xlsx',)):
    """
    Check file format extension.

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :param extensions:
        Allowed extensions.
    :type extensions: tuple[str]

    :return:
        If the extension of the input file is within the allowed extensions.
    :rtype: bool
    """
    return input_file_name.lower().endswith(extensions)


def load_from_dill(input_file):
    """
    Load inputs from .dill file.

    :param input_file:
        Input file.
    :type input_file: io.BytesIO

    :return:
        Raw input data.
    :rtype: dict
    """
    import dill
    return dill.load(input_file)


dsp.add_function(
    function=sh.add_args(load_from_dill),
    inputs=['input_file_name', 'input_file'],
    outputs=['raw_data'],
    input_domain=functools.partial(check_file_format, extensions=('.dill',))
)

dsp.add_function(
    function=parse_excel_file,
    inputs=['input_file_name', 'input_file'],
    outputs=['raw_data'],
    input_domain=check_file_format
)


@functools.lru_cache(None)
def _load_ta_function():
    from dice.co2mpas import dsp as _dice
    func = sh.SubDispatchFunction(
        _dice.register(memo={}),
        inputs=['input_file_name', 'input_file'],
        outputs=['base', 'meta']
    )
    func.output_type = 'dict'
    return func


@sh.add_function(dsp, outputs=['raw_data'], input_domain=functools.partial(
    check_file_format, extensions=('.co2mpas.ta',)
))
def load_ta_file(input_file_name, input_file):
    """
    Load inputs from .co2mpas.ta file.

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :param input_file:
        Input file.
    :type input_file: io.BytesIO

    :return:
        Raw input data.
    :rtype: dict
    """
    return _load_ta_function()(input_file_name, input_file)


@sh.add_function(dsp, inputs_kwargs=True, outputs=['data'])
def merge_data(
        raw_data, cmd_flags=None, soft_validation=False, engineering_mode=False,
        type_approval_mode=False, encryption_keys=None, sign_key=None):
    """
    Merge raw data with model flags.

    :param raw_data:
        Raw input data.
    :type raw_data: dict

    :param cmd_flags:
        Command line options.
    :type cmd_flags: dict

    :param soft_validation:
        Relax some Input-data validations, to facilitate experimentation.
    :type soft_validation: bool

    :param engineering_mode:
        Use all data and not only the declaration data.
    :type engineering_mode: bool

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :param encryption_keys:
        Encryption keys for TA mode.
    :type encryption_keys: str

    :param sign_key:
        User signature key for TA mode.
    :type sign_key: str

    :return:
        Merged raw data.
    :rtype: dict
    """
    flag = {k: v for k, v in dict(
        soft_validation=soft_validation,
        engineering_mode=engineering_mode,
        type_approval_mode=type_approval_mode,
        encryption_keys=encryption_keys,
        sign_key=sign_key
    ).items() if v is not None}
    data = sh.combine_dicts(raw_data, flag)
    data['flag'] = sh.combine_dicts(data.get('flag', {}), cmd_flags or {}, flag)
    return data


dsp.add_function(
    function=sh.SubDispatch(_validate, outputs=[
        'validated_plan', 'validated_flag', 'validated_dice', 'validated_meta',
        'validated_base', 'verified'
    ], output_type='list'),
    inputs=['data'],
    outputs=['plan', 'flag', 'dice', 'meta', 'base', 'verified']
)
