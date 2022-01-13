# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to write CO2MPAS output data.

Sub-Modules:

.. currentmodule:: co2mpas.core.write

.. autosummary::
    :nosignatures:
    :toctree: write/

    convert
    excel
"""
import os
import logging
import os.path as osp
import schedula as sh
from .excel import write_to_excel
from .convert import convert2df
from ... import default_start_time, default_timestamp
from co2mpas._version import __version__
from co2mpas.utils import check_first_arg, check_first_arg_false

try:
    from co2mpas_dice import dsp as _dice
except ImportError:
    _dice = None

log = logging.getLogger(__name__)
dsp = sh.BlueDispatcher(
    name='write', description='Write the outputs of the CO2MPAS model.'
)

dsp.add_func(default_start_time, outputs=['start_time'])
dsp.add_func(convert2df, outputs=['dfs'])


@sh.add_function(dsp, outputs=['output_template'], weight=sh.inf(1, 0))
def default_output_template():
    """
    Returns the default template output.

    :return:
        Template output.
    :rtype: str
    """
    from pkg_resources import resource_filename
    return resource_filename('co2mpas', 'templates/output_template.xlsx')


dsp.add_func(write_to_excel, outputs=['excel_output'])

dsp.add_function(
    function=sh.add_args(sh.bypass),
    inputs=['type_approval_mode', 'excel_output'],
    outputs=['output_file'],
    input_domain=check_first_arg_false
)

dsp.add_func(default_timestamp, outputs=['timestamp'])


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
    :type ext: str | None

    :return:
        Output file name.
    :rtype: str

    """
    fp = osp.join(output_folder, '%s-%s' % (timestamp, vehicle_name))
    if ext is not None:
        fp = '%s.%s' % (fp, ext)
    return fp


dsp.add_function(
    function=sh.add_args(default_output_file_name),
    inputs=['type_approval_mode', 'output_folder', 'vehicle_name', 'timestamp'],
    outputs=['output_file_name'],
    input_domain=check_first_arg_false
)

if _dice is not None:
    dsp.add_data('co2mpas_version', __version__)
    _out, _inp = ['output_file_name', 'output_file'], [
        'excel_input', 'base', 'start_time', 'excel_output', 'output_folder',
        'report', 'encryption_keys', 'meta', 'sign_key', 'dice', 'timestamp',
        'co2mpas_version', 'flag'
    ]

    # noinspection PyProtectedMember,PyTypeChecker
    dsp.add_function(
        function=sh.Blueprint(
            sh.SubDispatchFunction(_dice, inputs=_inp, outputs=_out)
        )._set_cls(sh.add_args),
        function_id='write_ta_output',
        description='Write ta output file.',
        inputs=['type_approval_mode', 'input_file'] + _inp[1:],
        outputs=_out,
        input_domain=check_first_arg
    )
else:
    dsp.add_data('base', description='Base data.')
    dsp.add_data('excel_input', description='Excel input file.')
    dsp.add_data('encryption_keys', description='Encryption keys for TA mode.')
    dsp.add_data('meta', description='Meta data.')
    dsp.add_data('sign_key', description='User signature key for TA mode.')


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
    os.makedirs(osp.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, 'wb') as f:
        f.write(output_file.read())
    log.info('CO2MPAS output written into (%s).', output_file_name)
