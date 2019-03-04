# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions and a model `dsp` to processes a vehicle from the file
path to the write of its outputs.
"""

import datetime
import logging
import schedula as sh
import os.path as osp
from co2mpas.io import schema
from co2mpas.utils import check_first_arg, ret_v
from .model import dsp as _model
from .report import dsp as _report
from co2mpas.io import dsp as _write_outputs
from co2mpas.io.ta import dsp as _write_ta_output
from co2mpas.conf import defaults

log = logging.getLogger(__name__)

dsp = sh.BlueDispatcher(
    name='run_base',
    description='Processes a vehicle from the file path to the write of its'
                ' outputs.'
)

dsp.add_data('soft_validation', False)

dsp.add_function(
    function=schema.validate_meta,
    inputs=['meta', 'soft_validation'],
    outputs=['validated_meta']
)

dsp.add_function(
    function=schema.validate_dice,
    inputs=['dice'],
    outputs=['validated_dice']
)

dsp.add_data('engineering_mode', False)
dsp.add_data('enable_selector', False)

dsp.add_function(
    function=sh.add_args(schema.validate_base),
    inputs=['run_base', 'data', 'engineering_mode', 'soft_validation',
            'enable_selector'],
    outputs=['validated_base'],
    input_domain=check_first_arg,
    weight=10
)


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


@sh.add_function(dsp, outputs=['start_time'])
def default_start_time():
    """
    Returns the default run start time.

    :return:
        Run start time.
    :rtype: datetime.datetime
    """
    return datetime.datetime.today()


@sh.add_function(dsp, outputs=['timestamp'])
def default_timestamp(start_time):
    """
    Returns the default timestamp.

    :param start_time:
        Run start time.
    :type start_time: datetime.datetime

    :return:
        Run timestamp.
    :rtype: str
    """
    return start_time.strftime('%Y%m%d_%H%M%S')


dsp.add_data('output_folder', '.')


@sh.add_function(dsp, outputs=['output_file_name'])
def default_output_file_name(
        output_folder, vehicle_name, timestamp, ext='xlsx'):
    """
    Returns the output file name.

    :param output_folder:
        Output folder.
    :type output_folder: str

    :param vehicle_name:
        Vehicle name.
    :type vehicle_name: str

    :param timestamp:
        Run timestamp.
    :type timestamp: str

    :param ext:
        File extension.
    :type ext: str

    :return:
        Output file name.
    :rtype: str

    """
    ofname = osp.join(output_folder, '%s-%s' % (timestamp, vehicle_name))

    return '%s.%s' % (ofname, ext)


dsp.add_data('only_summary', False)

dsp.add_function(
    function=sh.add_args(sh.SubDispatch(_model), 2),
    inputs=['validated_meta', 'validated_dice', 'validated_base'],
    outputs=['dsp_solution']
)


@sh.add_function(dsp, outputs=['output_data'])
def parse_dsp_solution(dsp_solution):
    """
    Parses the co2mpas model results.

    :param dsp_solution:
        Co2mpas model after dispatching.
    :type dsp_solution: schedula.Solution

    :return:
        Mapped outputs.
    :rtype: dict[dict]
    """

    res = {}
    for k, v in dsp_solution.items():
        sh.get_nested_dicts(res, *k.split('.'), default=ret_v(v))

    for k, v in list(sh.stack_nested_keys(res, depth=3)):
        n, k = k[:-1], k[-1]
        if n == ('output', 'calibration') and k in ('wltp_l', 'wltp_h'):
            v = sh.selector(('co2_emission_value',), v, allow_miss=True)
            if v:
                d = sh.get_nested_dicts(res, 'target', 'prediction')
                d[k] = sh.combine_dicts(v, d.get(k, {}))

    res['pipe'] = dsp_solution.pipe

    return res


dsp.add_function(
    function=_report,
    inputs=['output_data', 'vehicle_name'],
    outputs=['report', 'summary'],
)

dfl = defaults.io_constants_dfl
dsp.add_data('encryption_keys', dfl.ENCRYPTION_KEYS_PATH)
dsp.add_data('sign_key', dfl.SIGN_KEY_PATH)

dsp.add_function(
    function=sh.add_args(_write_ta_output),
    inputs=['type_approval_mode', 'encryption_keys', 'vehicle_family_id',
            'sign_key', 'start_time', 'timestamp', 'data', 'meta',
            'validated_dice', 'report', 'output_folder', 'output_file',
            'input_file'],
    outputs=['output_ta_file'],
    input_domain=check_first_arg
)


@sh.add_function(dsp, outputs=['template_file_name'])
def get_template_file_name(output_template, input_file_name):
    """
    Returns the template file name.

    :param output_template:
        Template output.
    :type output_template: str

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :return:
        Template file name.
    :rtype: str
    """
    if output_template == '-':
        return input_file_name
    return output_template


@sh.add_function(dsp, outputs=['output_template'], weight=10)
def default_output_template():
    """
    Returns the default template output.

    :return:
        Template output.
    :rtype: str
    """
    from pkg_resources import resource_filename
    return resource_filename('templates', 'co2mpas_output_template.xlsx')


dsp.add_function(
    function=sh.add_args(_write_outputs),
    inputs=['only_summary', 'output_file_name', 'template_file_name',
            'report', 'start_time', 'flag', 'type_approval_mode'],
    outputs=['output_file'],
    input_domain=lambda *args: not args[0]
)

SITES = set()


# noinspection PyUnusedLocal
def plot_model_workflow(output_file_name=None, vehicle_name='', **kw):
    """
    Defines the kwargs to plot the dsp workflow.

    :param output_file_name:
        File name where to plot the workflow.
    :type output_file_name: str

    :param vehicle_name:
        Vehicle name.
    :type vehicle_name: str

    :param kw:
        Additional kwargs.
    :type kw: dict

    :return:
        Kwargs to plot the dsp workflow.
    :rtype: dict
    """
    try:
        ofname = None
        if output_file_name:
            ofname = osp.splitext(output_file_name)[0]
        log.info("Plotting workflow of %s... into '%s'", vehicle_name, ofname)
        return {'directory': ofname, 'sites': SITES, 'index': True}
    except RuntimeError as ex:
        log.warning(ex, exc_info=1)
    return sh.NONE


dsp.add_function(
    function=sh.add_args(plot_model_workflow),
    inputs=['plot_workflow', 'output_file_name', 'vehicle_name'],
    outputs=[sh.PLOT],
    weight=30,
    input_domain=check_first_arg
)

dsp = sh.SubDispatch(dsp)
