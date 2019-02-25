# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions to process vehicle files.
"""

import datetime
import functools
import logging
import re
import threading
from tqdm import tqdm

import schedula as sh
import co2mpas.io.excel as excel
import co2mpas.io.schema as schema
import co2mpas.utils as co2_utl
import os.path as osp

log = logging.getLogger(__name__)

files_exclude_regex = re.compile(r'^\w')


def parse_dsp_solution(solution):
    """
    Parses the co2mpas model results.

    :param solution:
        Co2mpas model after dispatching.
    :type solution: schedula.Solution

    :return:
        Mapped outputs.
    :rtype: dict[dict]
    """

    res = {}
    for k, v in solution.items():
        sh.get_nested_dicts(res, *k.split('.'), default=co2_utl.ret_v(v))

    for k, v in list(sh.stack_nested_keys(res, depth=3)):
        n, k = k[:-1], k[-1]
        if n == ('output', 'calibration') and k in ('wltp_l', 'wltp_h'):
            v = sh.selector(('co2_emission_value',), v, allow_miss=True)
            if v:
                d = sh.get_nested_dicts(res, 'target', 'prediction')
                d[k] = sh.combine_dicts(v, d.get(k, {}))

    res['pipe'] = solution.pipe

    return res


def notify_result_listener(result_listener, res, out_fpath=None):
    """Utility func to send to the listener the output-file discovered from the results."""
    if result_listener:
        if not out_fpath:
            it = []
            for k in ('output_file_name', 'output_ta_file'):
                if sh.are_in_nested_dicts(res, 'solution', k) and \
                        osp.isfile(res['solution'][k]):
                    it.append(res['solution'][k])
        else:
            it = sh.stlp(out_fpath)
        try:
            for fpath in it:
                result_listener((fpath, res))
        except Exception as ex:
            try:
                keys = list(res)
            except Exception:
                keys = '<no keys>'
            log.warning(
                "Failed notifying result-listener due to: %s\n  result-keys: %s",
                ex, keys, exc_info=1)


def process_folder_files(input_files, output_folder,
                         result_listener=None, **kwds):
    """
    Process all xls-files in a folder with CO2MPAS-model and produces summary.

    :param list input_files:
        A list of input xl-files.
    :param str output_folder:
        Where to store the results; the exact output-filenames will be::

            <timestamp>-<input_filename>.xlsx

    :param result_listener:
        A callable that will receive a 2 tuple for each file as it is produced::

                (<filepath>, <contents>)
    :type result_listener: callable

    """

    summary, start_time = _process_folder_files(input_files, output_folder,
                                                result_listener=result_listener,
                                                **kwds)

    timestamp = start_time.strftime('%Y%m%d_%H%M%S')

    summary_xl_file = osp.join(output_folder, '%s-summary.xlsx' % timestamp)
    _save_summary(summary_xl_file, start_time, summary)

    time_elapsed = (datetime.datetime.today() - start_time).total_seconds()
    log.info('Done! [%s sec]', time_elapsed)

    notify_result_listener(result_listener, summary, summary_xl_file)

    _pause_for_sites_shutdown()


class _custom_tqdm(tqdm):

    def format_meter(self, n, *args, **kwargs):
        bar = tqdm.format_meter(n, *args, **kwargs)
        try:
            return '%s: Processing %s\n' % (bar, self.iterable[n])
        except IndexError:
            return bar


def _yield_folder_files_results(
        start_time, input_files, output_folder, overwrite_cache=False,
        model=None, variation=None, type_approval_mode=False, modelconf=None):
    model = model or vehicle_processing_model()
    kw = {
        'output_folder': output_folder,
        'overwrite_cache': overwrite_cache,
        'modelconf': modelconf,
        'timestamp': start_time.strftime('%Y%m%d_%H%M%S'),
        'variation': variation or {},
        'type_approval_mode': type_approval_mode
    }

    _process_vehicle = sh.SubDispatch(model)

    for fpath in _custom_tqdm(input_files, bar_format='{l_bar}{bar}{r_bar}'):
        yield _process_vehicle({'input_file_name': fpath}, kw)


def _process_folder_files(*args, result_listener=None, **kwargs):
    """
    Process all xls-files in a folder with CO2MPAS-model.

    :param list input_files:
        A list of input xl-files.

    :param output_folder:
        Output folder.
    :type output_folder: str

    :param plot_workflow:
        If to show the CO2MPAS model workflow.
    :type plot_workflow: bool, optional

    :param output_template:
        The xlsx-file to use as template and import existing sheets from.

        - If file already exists, a clone gets updated with new sheets.
        - If it is None, it copies and uses the input-file as template.
        - if it is `False`, it does not use any template and a fresh output
          xlsx-file is created.
    :type output_folder: None,False,str

    """
    start_time = datetime.datetime.today()

    summary, n = {}, ('solution', 'summary')
    for res in _yield_folder_files_results(start_time, *args, **kwargs):
        if sh.are_in_nested_dicts(res, *n):
            _add2summary(summary, sh.get_nested_dicts(res, *n))
            notify_result_listener(result_listener, res)

    return summary, start_time


SITES = set()
SITES_STOPPER = threading.Event()


def _pause_for_sites_shutdown():
    if SITES:
        import time
        try:
            while not SITES_STOPPER.is_set():
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            pass
        while SITES:
            SITES.pop().shutdown()


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


def default_start_time():
    """
    Returns the default run start time.

    :return:
        Run start time.
    :rtype: datetime.datetime
    """
    return datetime.datetime.today()


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


def default_vehicle_name(fpath):
    """
    Returns the vehicle name.

    :param fpath:
        File path.
    :type fpath: str

    :return:
        Vehicle name.
    :rtype: str
    """
    return osp.splitext(osp.basename(fpath))[0]


def default_output_file_name(output_folder, fname, timestamp, ext='xlsx'):
    """
    Returns the output file name.

    :param output_folder:
        Output folder.
    :type output_folder: str

    :param fname:
        File name.
    :type fname: str

    :param timestamp:
        Run timestamp.
    :type timestamp: str

    :return:
        Output file name.
    :rtype: str

    """
    ofname = osp.join(output_folder, '%s-%s' % (timestamp, fname))

    return '%s.%s' % (ofname, ext)


def _add2summary(total_summary, summary, base_keys=None):
    base_keys = base_keys or {}
    for k, v in sh.stack_nested_keys(summary, depth=3):
        d = sh.get_nested_dicts(total_summary, *k, default=list)
        if isinstance(v, list):
            for j in v:
                d.append(sh.combine_dicts(j, base_keys))
        else:
            d.append(sh.combine_dicts(v, base_keys))


def _get_contain(d, *keys, default=None):
    try:
        key = keys[-1]
        if keys[-1] not in d:
            key = next((k for k in d if key in k or k in key))

        return d[key]
    except (StopIteration, KeyError):
        if len(keys) <= 1:
            return default
        return _get_contain(d, *keys[:-1], default=default)


def _save_summary(fpath, start_time, summary):
    if summary:
        from co2mpas.io.excel import _df2excel
        from co2mpas.io import _dd2df, _sort_key, _co2mpas_info2df, _add_units
        from pandas import MultiIndex, ExcelWriter

        p_keys = ('cycle', 'stage', 'usage', 'param')

        df = _dd2df(
            summary, index=['vehicle_name'], depth=3,
            col_key=functools.partial(_sort_key, p_keys=p_keys)
        )
        df.columns = MultiIndex.from_tuples(_add_units(df.columns))

        writer = ExcelWriter(fpath, engine='xlsxwriter')

        _df2excel(writer, 'summary', df, named_ranges=())

        _df2excel(writer, 'proc_info', _co2mpas_info2df(start_time))

        writer.save()
        log.info('Written into xl-file(%s)...', fpath)


def get_template_file_name(template_output, input_file_name):
    """
    Returns the template file name.

    :param template_output:
        Template output.
    :type template_output: str

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :return:
        Template file name.
    :rtype: str
    """
    if template_output == '-':
        return input_file_name
    return template_output


def check_first_arg(first, *args):
    return bool(first)


def prepare_data(raw_data, variation, input_file_name, overwrite_cache,
                 output_folder, timestamp, type_approval_mode, modelconf,
                 input_file=None):
    """
    Prepare the data to be processed.

    :param raw_data:
        Raw data from the input file.
    :type raw_data: dict

    :param variation:
        Variations to be applied.
    :type variation: dict

    :param input_file_name:
        Input file name.
    :type input_file_name: str

    :param overwrite_cache:
        Overwrite saved cache?
    :type overwrite_cache: bool

    :param output_folder:
        Output folder.
    :type output_folder: str

    :param timestamp:
        Run timestamp.
    :type timestamp: str

    :param type_approval_mode:
        Is launched for TA?
    :type type_approval_mode: bool

    :param modelconf:
        Path of modelconf that has modified the defaults.
    :type modelconf: str

    :return:
        Prepared data.
    :rtype: dict
    """
    raw_data = raw_data.copy()
    import pandas as pd
    raw_data['plan'] = pd.DataFrame(**raw_data.get('plan', {}))
    has_plan = not raw_data['plan'].empty
    match = {
        'scope': 'plan' if has_plan else 'base',
    }
    r = {}
    from pandalone.xleash import SheetsFactory, lasso
    from co2mpas.io import check_xlasso
    import pandas as pd

    sheets_factory = SheetsFactory()

    for k, v in excel._parse_values(variation, match, "in variations"):
        if isinstance(v, str) and check_xlasso(v):
            v = lasso(v, sheets_factory, url_file=input_file_name)
        sh.get_nested_dicts(r, *k[:-1])[k[-1]] = v

    if 'plan' in r:
        if has_plan:
            plan = raw_data['plan'].copy()
            for k, v in sh.stack_nested_keys(r['plan'], depth=4):
                plan['.'.join(k)] = v
        else:
            gen = sh.stack_nested_keys(r['plan'], depth=4)
            plan = pd.DataFrame([{'.'.join(k): v for k, v in gen}])
            excel._add_index_plan(plan, input_file_name)

        r['plan'] = plan
        has_plan = True

    if 'base' in r:
        r['base'] = sh.combine_nested_dicts(
            raw_data.get('base', {}), r['base'], depth=4
        )

    if 'flag' in r:
        r['flag'] = sh.combine_nested_dicts(
            raw_data.get('flag', {}), r['flag'], depth=1
        )

    if 'dice' in r:
        r['dice'] = sh.combine_nested_dicts(
            raw_data.get('dice', {}), r['dice'], depth=1
        )

    if 'meta' in r:
        r['meta'] = sh.combine_nested_dicts(
            raw_data.get('meta', {}), r['meta'], depth=2
        )

    data = sh.combine_dicts(raw_data, r)

    if type_approval_mode:
        variation, has_plan = {}, False
        if not schema._ta_mode(data):
            return {}, pd.DataFrame([])

    flag = data.get('flag', {}).copy()

    if 'run_base' not in flag:
        flag['run_base'] = not has_plan

    if 'run_plan' not in flag:
        flag['run_plan'] = has_plan

    flag['type_approval_mode'] = type_approval_mode
    flag['output_folder'] = output_folder
    flag['overwrite_cache'] = overwrite_cache
    if modelconf:
        flag['modelconf'] = modelconf

    if timestamp is not None:
        flag['timestamp'] = timestamp

    flag = schema.validate_flags(flag)

    if flag is sh.NONE:
        return {}, pd.DataFrame([])

    schema.check_data_version(flag)

    res = {
        'flag': flag,
        'dice': data.get('dice', {}),
        'meta': data.get('meta', {}),
        'variation': variation,
        'input_file_name': input_file_name,
        'input_file': input_file
    }
    res = sh.combine_dicts(flag, res)
    base = sh.combine_dicts(res, {'data': data.get('base', {})})
    plan = sh.combine_dicts(res, {'data': data.get('plan',
                                                   pd.DataFrame([]))})

    return base, plan


def check_run_base(data):
    return not data.get('run_plan', False)


def check_run_plan(data):
    return data.get('run_plan', False)


def _get_co2mpas_output_template_fpath():
    import pkg_resources

    fname = 'co2mpas_output_template.xlsx'
    return pkg_resources.resource_filename(__name__, fname)  # @UndefinedVariable


def vehicle_processing_model():
    """
    Defines the vehicle-processing model.

    .. dispatcher:: d

        >>> d = vehicle_processing_model()

    :return:
        The vehicle-processing model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='CO2MPAS vehicle_processing_model',
        description='Processes a vehicle from the file path to the write of its'
                    ' outputs.'
    )

    from .io import load_inputs

    d.add_dispatcher(
        include_defaults=True,
        dsp=load_inputs(),
        inputs={
            'input_file_name': 'input_file_name',
            'overwrite_cache': 'overwrite_cache'
        },
        outputs={
            'raw_data': 'raw_data',
            'input_file': 'input_file',
            sh.SINK: sh.SINK
        }
    )

    d.add_data(
        data_id='variation',
        default_value={}
    )

    d.add_data(
        data_id='overwrite_cache',
        default_value=False
    )

    d.add_data(
        data_id='output_folder',
        default_value='.'
    )

    d.add_data(
        data_id='timestamp',
        default_value=None
    )

    d.add_data(
        data_id='type_approval_mode',
        default_value=False
    )

    d.add_data(
        data_id='modelconf',
        default_value=None
    )

    d.add_function(
        function=prepare_data,
        inputs=['raw_data', 'variation', 'input_file_name', 'overwrite_cache',
                'output_folder', 'timestamp', 'type_approval_mode',
                'modelconf', 'input_file'],
        outputs=['base_data', 'plan_data']
    )

    d.add_function(
        function=run_base(),
        inputs=['base_data'],
        outputs=['solution'],
        input_domain=check_run_base
    )

    d.add_function(
        function=run_plan(),
        inputs=['plan_data'],
        outputs=['solution'],
        input_domain=check_run_plan
    )

    return d


def run_base():
    """
    Defines the vehicle-processing model.

    .. dispatcher:: d

        >>> d = run_base()

    :return:
        The vehicle-processing model.
    :rtype: Dispatcher
    """

    d = sh.Dispatcher(
        name='run_base',
        description='Processes a vehicle from the file path to the write of its'
                    ' outputs.'
    )

    d.add_data(
        data_id='engineering_mode',
        default_value=False
    )

    d.add_data(
        data_id='output_folder',
        default_value='.'
    )

    d.add_data(
        data_id='use_selector',
        default_value=False
    )

    d.add_data(
        data_id='soft_validation',
        default_value=False
    )

    d.add_function(
        function=schema.validate_meta,
        inputs=['meta', 'soft_validation'],
        outputs=['validated_meta']
    )

    d.add_function(
        function=schema.validate_dice,
        inputs=['dice'],
        outputs=['validated_dice']
    )

    d.add_function(
        function=sh.add_args(schema.validate_base),
        inputs=['run_base', 'data', 'engineering_mode', 'soft_validation',
                'use_selector'],
        outputs=['validated_base'],
        input_domain=check_first_arg,
        weight=10
    )

    d.add_data(
        data_id='only_summary',
        default_value=False
    )

    d.add_function(
        function=default_vehicle_name,
        inputs=['input_file_name'],
        outputs=['vehicle_name']
    )

    d.add_function(
        function=default_start_time,
        outputs=['start_time']
    )

    d.add_function(
        function=default_timestamp,
        inputs=['start_time'],
        outputs=['timestamp']
    )

    d.add_function(
        function=default_output_file_name,
        inputs=['output_folder', 'vehicle_name', 'timestamp'],
        outputs=['output_file_name']
    )

    from .model import model
    d.add_function(
        function=sh.add_args(sh.SubDispatch(model()), 2),
        inputs=['validated_meta', 'validated_dice', 'validated_base'],
        outputs=['dsp_solution']
    )

    d.add_function(
        function=parse_dsp_solution,
        inputs=['dsp_solution'],
        outputs=['output_data']
    )

    from .report import report
    d.add_function(
        function=report(),
        inputs=['output_data', 'vehicle_name'],
        outputs=['report', 'summary'],
    )

    from .io.ta import write_ta_output
    from .conf import defaults
    dfl = defaults.io_constants_dfl
    d.add_data('encryption_keys', dfl.ENCRYPTION_KEYS_PATH)
    d.add_data('sign_key', dfl.SIGN_KEY_PATH)

    d.add_function(
        function=sh.add_args(write_ta_output()),
        inputs=['type_approval_mode', 'encryption_keys',
                'vehicle_family_id', 'sign_key', 'start_time', 'timestamp',
                'data', 'meta', 'validated_dice', 'report', 'output_folder',
                'output_file', 'input_file'],
        outputs=['output_ta_file'],
        input_domain=check_first_arg
    )

    d.add_function(
        function=get_template_file_name,
        inputs=['output_template', 'input_file_name'],
        outputs=['template_file_name']
    )

    d.add_data(
        data_id='output_template',
        default_value=_get_co2mpas_output_template_fpath(),
        initial_dist=10
    )

    from .io import write_outputs
    d.add_function(
        function=sh.add_args(write_outputs()),
        inputs=['only_summary', 'output_file_name', 'template_file_name',
                'report', 'start_time', 'flag', 'type_approval_mode'],
        outputs=['output_file'],
        input_domain=lambda *args: not args[0]
    )

    d.add_function(
        function=sh.add_args(plot_model_workflow),
        inputs=['plot_workflow', 'output_file_name', 'vehicle_name'],
        outputs=[sh.PLOT],
        weight=30,
        input_domain=check_first_arg
    )

    return sh.SubDispatch(d)


def run_plan():
    """
    Defines the plan model.

    .. dispatcher:: d

        >>> d = run_plan()

    :return:
        The plan model.
    :rtype: Dispatcher
    """

    d = sh.Dispatcher(
        name='run_plan',
        description='Processes a vehicle plan.'
    )

    d.add_data(
        data_id='engineering_mode',
        default_value=False
    )

    d.add_data(
        data_id='use_selector',
        default_value=False
    )

    d.add_data(
        data_id='soft_validation',
        default_value=False
    )

    d.add_function(
        function=sh.add_args(schema.validate_plan),
        inputs=['run_plan', 'data', 'engineering_mode', 'soft_validation',
                'use_selector'],
        outputs=['validated_plan'],
        input_domain=check_first_arg
    )

    d.add_function(
        function=default_start_time,
        outputs=['start_time']
    )

    d.add_function(
        function=default_timestamp,
        inputs=['start_time'],
        outputs=['timestamp']
    )

    from .plan import make_simulation_plan
    d.add_function(
        function=make_simulation_plan,
        inputs=['validated_plan', 'timestamp', 'variation', 'flag'],
        outputs=['summary']
    )

    return sh.SubDispatch(d)
