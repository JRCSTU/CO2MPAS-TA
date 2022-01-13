# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to load data from a CO2MPAS input file.

Sub-Modules:

.. currentmodule:: co2mpas.core.load

.. autosummary::
    :nosignatures:
    :toctree: load/

    excel
    schema
    validate
"""
import io
import logging
import functools
import schedula as sh
from .excel import parse_excel_file
from .validate import dsp as _validate

try:
    from co2mpas_dice import dsp as _dice
except ImportError:
    _dice = None

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
def check_file_format(input_file_name, *args, ext=('.xlsx',)):
    """
    Check file format extension.

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :param ext:
        Allowed extensions.
    :type ext: tuple[str]

    :return:
        If the extension of the input file is within the allowed extensions.
    :rtype: bool
    """
    return input_file_name.lower().endswith(ext)


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
    input_domain=functools.partial(check_file_format, ext=('.dill',))
)

dsp.add_function(
    function=parse_excel_file,
    inputs=['input_file_name', 'input_file'],
    outputs=['raw_data'],
    input_domain=check_file_format
)

if _dice is not None:
    _out, _inp = ['base', 'meta', 'dice'], [
        'input_file_name', 'input_file', 'encryption_keys',
        'encryption_keys_passwords'
    ]
    # noinspection PyProtectedMember
    dsp.add_function(
        function=sh.Blueprint(sh.SubDispatchFunction(
            _dice, function_id='load_ta_file', inputs=_inp[1:], outputs=_out
        ))._set_cls(sh.add_args),
        description='Load inputs from .co2mpas.ta file.',
        inputs=_inp,
        outputs=['raw_data'],
        filters=[functools.partial(sh.map_list, [_out])],
        input_domain=functools.partial(check_file_format, ext=(
            '.co2mpas.ta', '.dice.ta', '.jet.ta'
        ))
    )

    _out, _inp = ['data', 'dice'], ['input_file_name', 'input_file']
    # noinspection PyProtectedMember
    dsp.add_function(
        function=sh.Blueprint(sh.SubDispatchFunction(
            _dice, inputs=_inp[1:], outputs=_out
        ))._set_cls(sh.add_args),
        function_id='load_co2mpas_file',
        description='Load inputs from .co2mpas file.',
        inputs=_inp,
        outputs=['raw_data'],
        filters=[functools.partial(sh.map_list, [{}, 'dice'])],
        input_domain=functools.partial(check_file_format, ext=('.co2mpas',))
    )
else:
    dsp.add_data(
        'encryption_keys_passwords', description='Encryption keys passwords.'
    )


@sh.add_function(dsp, inputs_kwargs=True, outputs=['data'])
def merge_data(
        raw_data, cmd_flags=None, hard_validation=False, declaration_mode=False,
        type_approval_mode=False, encryption_keys=None, sign_key=None,
        enable_selector=False):
    """
    Merge raw data with model flags.

    :param raw_data:
        Raw input data.
    :type raw_data: dict

    :param cmd_flags:
        Command line options.
    :type cmd_flags: dict

    :param hard_validation:
        Add extra data validations.
    :type hard_validation: bool

    :param declaration_mode:
        Use only the declaration data.
    :type declaration_mode: bool

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :param encryption_keys:
        Encryption keys for TA mode.
    :type encryption_keys: str

    :param sign_key:
        User signature key for TA mode.
    :type sign_key: str

    :param enable_selector:
        Enable the selection of the best model to predict both H/L cycles.
    :type enable_selector: bool

    :return:
        Merged raw data.
    :rtype: dict
    """
    flag = {k: v for k, v in dict(
        hard_validation=hard_validation,
        declaration_mode=declaration_mode,
        type_approval_mode=type_approval_mode,
        encryption_keys=encryption_keys,
        sign_key=sign_key,
        enable_selector=enable_selector
    ).items() if v is not None}
    data = sh.combine_dicts(raw_data, flag)
    data['flag'] = sh.combine_dicts(data.get('flag', {}), cmd_flags or {}, flag)
    return data


def check_validation(sol):
    """
    Check if the data are verified.

    :param sol:
        Validation solution.
    :type sol: schedula.Solution

    :return:
        Validated data
    :rtype: list[dict]
    """
    sol = sol.get('verified') and sol or {}
    keys = 'plan', 'flag', 'dice', 'meta', 'base'
    return [sol.get('validated_%s' % k, sh.NONE) for k in keys]


dsp.add_function(
    function=sh.SubDispatch(_validate),
    inputs=['data'],
    outputs=['plan', 'flag', 'dice', 'meta', 'base'],
    filters=[check_validation]
)
