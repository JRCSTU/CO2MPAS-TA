# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to processes a CO2MPAS input file.

Sub-Modules:

.. currentmodule:: co2mpas.core

.. autosummary::
    :nosignatures:
    :toctree: core/

    load
    model
    report
    write
"""
import logging
import schedula as sh
import os.path as osp
from co2mpas.utils import check_first_arg
from .load import dsp as _load
from .model import dsp as _model
from .report import dsp as _report
from .write import dsp as _write

log = logging.getLogger(__name__)

dsp = sh.BlueDispatcher(
    name='core',
    description='Processes a CO2MPAS input file.'
)
_cmd_flags = [
    'only_summary', 'hard_validation', 'declaration_mode', 'enable_selector',
    'type_approval_mode', 'encryption_keys', 'sign_key', 'output_template',
    'output_folder', 'encryption_keys_passwords', 'augmented_summary'
]


@sh.add_function(
    dsp, inputs_kwargs=True, inputs_defaults=True, outputs=_cmd_flags
)
def parse_cmd_flags(cmd_flags=None):
    """
    Parses the command line options.

    :param cmd_flags:
        Command line options.
    :type cmd_flags: dict

    :return:
        Default parameters of process model.
    :rtype: tuple
    """
    flags = sh.combine_dicts(cmd_flags or {}, base={
        'only_summary': False,
        'hard_validation': False,
        'declaration_mode': False,
        'enable_selector': False,
        'type_approval_mode': False,
        'encryption_keys': None,
        'sign_key': None,
        'output_template': sh.NONE,
        'encryption_keys_passwords': None,
        'output_folder': './outputs',
        'augmented_summary': False
    })
    flags['declaration_mode'] |= flags['type_approval_mode']
    flags['hard_validation'] |= flags['declaration_mode']
    if flags['declaration_mode'] and not flags['type_approval_mode'] and \
            flags['enable_selector']:
        log.info(
            'Since CO2MPAS is launched in declaration mode the option '
            '--enable-selector is not used.\n'
            'If you want to use it remove -DM from the cmd.'
        )
        flags['enable_selector'] = False
    return sh.selector(_cmd_flags, flags, output_type='list')


dsp.add_dispatcher(
    dsp=_load,
    inputs=(
        'input_file_name', 'hard_validation', 'declaration_mode', 'cmd_flags',
        'type_approval_mode', 'input_file', 'raw_data', 'encryption_keys',
        'sign_key', 'encryption_keys_passwords', 'enable_selector'
    ),
    outputs=('plan', 'flag', 'dice', 'meta', 'base', 'input_file'),
)

# noinspection PyProtectedMember
dsp.add_function(
    function=sh.SubDispatch(_model),
    inputs=['base'],
    outputs=['solution'],
    input_domain=check_first_arg
)


@sh.add_function(dsp, outputs=['output_data'])
def parse_solution(solution):
    """
    Parse the CO2MPAS model solution.

    :param solution:
        CO2MPAS model solution.
    :type solution: schedula.Solution

    :return:
        CO2MPAS outputs.
    :rtype: dict[dict]
    """
    from .model.selector import calibration_cycles
    res = {}
    for k, v in solution.items():
        k = k.split('.')
        sh.get_nested_dicts(res, *k[:-1])[k[-1]] = v
    for k, v in list(sh.stack_nested_keys(res, depth=3)):
        n, k = k[:-1], k[-1]
        if n == ('output', 'calibration') and k in calibration_cycles:
            v = sh.selector(('co2_emission_value',), v, allow_miss=True)
            if v:
                d = sh.get_nested_dicts(res, 'target', 'prediction')
                d[k] = sh.combine_dicts(v, d.get(k, {}))
    from co2mpas.defaults import dfl
    if dfl.functions.parse_solution.CALIBRATION_AS_TARGETS:
        pred = sh.get_nested_dicts(res, 'target', 'prediction')
        pred.update(sh.combine_nested_dicts(sh.get_nested_dicts(
            res, 'output', 'calibration'
        ), pred, depth=2))

    res['pipe'] = solution.pipe

    return res


dsp.add_dispatcher(
    dsp=_report,
    inputs=['output_data', 'augmented_summary'],
    outputs=['report', 'summary'],
)


def check_only_summary(kw):
    """
    Check if `only_summary` is true.

    :param kw:
        `write` dispatcher inputs.
    :type kw: dict

    :return:
        Is `only_summary` true?
    :rtype: bool
    """
    return not kw.get('only_summary', True)


@sh.add_function(dsp, outputs=['vehicle_name'])
def default_vehicle_name(input_file_name):
    """
    Returns the vehicle name.

    :param input_file_name:
        File path.
    :type input_file_name: str

    :return:
        Vehicle name.
    :rtype: str
    """
    return osp.splitext(osp.basename(input_file_name))[0]


dsp.add_dispatcher(
    dsp=_write,
    inputs=[
        'input_file_name', 'input_file', 'base', 'dice', 'meta', 'vehicle_name',
        'output_file_name', 'report', 'flag', 'type_approval_mode', 'timestamp',
        'encryption_keys', 'output_folder', 'sign_key', 'output_template',
        {'only_summary': sh.SINK}
    ],
    outputs=[
        'output_file', 'start_time', 'timestamp', 'output_file_name', sh.SINK
    ],
    input_domain=check_only_summary
)
