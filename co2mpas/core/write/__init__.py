# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions to read/write inputs/outputs from/on excel.
"""
import logging
import functools
import os.path as osp
import schedula as sh
from .excel import write_to_excel
from .convert import convert2df
from co2mpas.utils import check_first_arg_false, check_first_arg

log = logging.getLogger(__name__)
dsp = sh.BlueDispatcher(
    name='write', description='Write the outputs of the CO2MPAS model.'
)


@sh.add_function(dsp, outputs=['start_time'])
def default_start_time():
    """
    Returns the default run start time.

    :return:
        Run start time.
    :rtype: datetime.datetime
    """
    import datetime
    return datetime.datetime.today()


dsp.add_func(
    convert2df, inputs_kwargs=True, inputs_defaults=True, outputs=['dfs']
)


@sh.add_function(dsp, outputs=['output_template'], weight=sh.inf(1, 0))
def default_output_template():
    """
    Returns the default template output.

    :return:
        Template output.
    :rtype: str
    """
    from pkg_resources import resource_filename as res_fn
    return res_fn('co2mpas', 'templates/co2mpas_output_template.xlsx')


dsp.add_func(write_to_excel, outputs=['excel_output'])

dsp.add_function(
    function=sh.add_args(sh.bypass),
    inputs=['type_approval_mode', 'excel_output'],
    outputs=['output_file'],
    input_domain=check_first_arg_false
)


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


dsp.add_function(
    function=sh.add_args(default_output_file_name),
    inputs=['type_approval_mode', 'output_folder', 'vehicle_name', 'timestamp'],
    outputs=['output_file_name'],
    input_domain=check_first_arg_false
)


@functools.lru_cache(None)
def _write_ta_function():
    from dice.co2mpas import dsp as _dice
    from co2mpas import __version__
    _dice = _dice.register(memo={})
    _dice.add_data('co2mpas_version', __version__)
    return sh.SubDispatchFunction(_dice, inputs=[
        'base', 'dice', 'meta', 'report', 'excel_output', 'input_file',
        'encryption_keys', 'sign_key', 'output_folder', 'start_time',
        'timestamp'
    ], outputs=['output_file_name', 'output_file'])


# noinspection PyUnusedLocal
@sh.add_function(
    dsp, outputs=['output_file_name', 'output_file'],
    input_domain=check_first_arg
)
def write_ta_output(
        type_approval_mode, base, dice, meta, report, excel_output, input_file,
        encryption_keys, sign_key, output_folder, start_time, timestamp):
    """
    Write ta output file.

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :param base:
        Base data.
    :type base: dict

    :param dice:
        DICE data.
    :type dice: dict

    :param meta:
        Meta data.
    :type meta: dict

    :param report:
        Vehicle output report.
    :type report: dict

    :param excel_output:
        Excel output file.
    :type excel_output: io.BytesIO

    :param input_file:
        Input file.
    :type input_file: io.BytesIO

    :param encryption_keys:
        Encryption keys for TA mode.
    :type encryption_keys: str

    :param sign_key:
        User signature key for TA mode.
    :type sign_key: str

    :param output_folder:
        Output folder.
    :type output_folder: str

    :param start_time:
        Run start time.
    :type start_time: datetime.datetime

    :param timestamp:
        Run timestamp.
    :type timestamp: str

    :return:
        Output file.
    :rtype: io.BytesIO
    """
    return _write_ta_function()(
        base, dice, meta, report, excel_output, input_file, encryption_keys,
        sign_key, output_folder, start_time, timestamp
    )


@sh.add_function(dsp)
def save_output_file(output_file, output_file_name):
    """
    Save output file.

    :param output_file_name:
        Output file name.
    :type output_file_name: str

    :param output_file:
        Output file.
    :type output_file: io.BytesIO
    """
    output_file.seek(0)
    with open(output_file_name, 'wb') as f:
        f.write(output_file.read())
    log.info('Written into (%s)...', output_file_name)
