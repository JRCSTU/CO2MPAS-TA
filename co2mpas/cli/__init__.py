# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
r"""
Define CO2MPAS command line interface.

.. click:: co2mpas.cli:cli
   :prog: co2mpas
   :show-nested:

"""
import os
import click
import logging
import click_log
import schedula as sh
import os.path as osp
from co2mpas import dsp as _process
from co2mpas._version import __version__

try:
    from co2mpas.cli.sync import cli as _sync
except ImportError:
    _sync = None
try:  # TODO: to be changed to co2mpas_gui.
    from co2wui.cli import cli as _gui
except ImportError:
    _gui = None

log = logging.getLogger('co2mpas.cli')
CO2MPAS_HOME = os.environ.get('CO2MPAS_HOME', '.')

log_config = dict(format="%(asctime)-15s:%(levelname)5.5s:%(name)s:%(message)s")


class _Logger(logging.Logger):
    # noinspection PyMissingOrEmptyDocstring
    def setLevel(self, level):
        super(_Logger, self).setLevel(level)
        logging.basicConfig(level=level, **log_config)
        rlog = logging.getLogger()
        # because `basicConfig()` does not reconfig root-logger when re-invoked.
        rlog.level = level
        logging.captureWarnings(True)


logger = _Logger('cli')
click_log.basic_config(logger)


@click.group(
    'co2mpas', context_settings=dict(help_option_names=['-h', '--help'])
)
@click.version_option(__version__)
def cli():
    """
    CO2MPAS command line tool.
    """


@cli.command('template', short_help='Generates input/output template file.')
@click.argument(
    'output-file', default='template.xlsx', required=False,
    type=click.Path(writable=True)
)
@click.option(
    '-TT', '--template-type', type=click.Choice(['input', 'output']),
    help='Template file type.', default='input', show_default=True
)
@click_log.simple_verbosity_option(logger)
def template(**inputs):
    """
    Writes a CO2MPAS input/output template into OUTPUT_FILE.

    OUTPUT_FILE: File path `.xlsx`. [default: ./template.xlsx]
    """
    return _process(inputs, ['template', 'done'])


@cli.command('demo', short_help='Generates sample demo files.')
@click.argument(
    'output-folder', default='./inputs', required=False,
    type=click.Path(writable=True, file_okay=False)
)
@click_log.simple_verbosity_option(logger)
def demo(output_folder):
    """
    Writes a CO2MPAS demo files into OUTPUT_FOLDER.

    OUTPUT_FOLDER: Folder path. [default: ./inputs]
    """
    return _process({'output_folder': output_folder}, ['demo', 'done'])


@cli.command('conf', short_help='Generates CO2MPAS model-configuration file.')
@click.argument(
    'output-file', default='./conf.yaml', required=False,
    type=click.Path(writable=True)
)
@click.option(
    '-MC', '--model-conf', type=click.Path(exists=True),
    help='Model-configuration file path `.yaml`.'
)
@click_log.simple_verbosity_option(logger)
def conf(output_file, **kwargs):
    """
    Writes a CO2MPAS model-configuration file into OUTPUT_FILE.

    OUTPUT_FILE: File path `.yaml`. [default: ./conf.yaml]
    """
    inputs = {sh.START: kwargs, 'output_file': output_file}
    return _process(inputs, ['conf', 'done'])


@cli.command('plot', short_help='Plots the CO2MPAS model.')
@click.option(
    '-C', '--cache-folder', help='Folder to save temporary html files.',
    default='./cache_plot', type=click.Path(file_okay=False, writable=True),
    show_default=True
)
@click.option(
    '-H', '--host', help='Hostname to listen on.', default='127.0.0.1',
    type=str, show_default=True
)
@click.option(
    '-P', '--port', help='Port of the webserver.', default=5000, type=int,
    show_default=True
)
@click_log.simple_verbosity_option(logger)
def plot(cache_folder, host, port):
    """
    Plots the full CO2MPAS model into CACHE_FOLDER.
    """
    return _process(
        dict(plot_model=True, cache_folder=cache_folder, host=host, port=port),
        ['plot', 'done']
    )


@cli.command('run', short_help='Run CO2MPAS model.')
@click.argument('input-files', nargs=-1, type=click.Path(exists=True))
@click.option(
    '-O', '--output-folder', help='Output folder.', default='./outputs',
    type=click.Path(file_okay=False, writable=True), show_default=True
)
@click.option(
    '-EK', '--encryption-keys', help='Encryption keys for TA mode.',
    default=osp.join(CO2MPAS_HOME, 'DICE_KEYS/dice.co2mpas.keys'),
    type=click.Path(), show_default=True
)
@click.option(
    '-SK', '--sign-key', help='User signature key for TA mode.',
    default=osp.join(CO2MPAS_HOME, 'DICE_KEYS/sign.co2mpas.key'),
    type=click.Path(), show_default=True
)
@click.option(
    '-C', '--cache-folder', help='Folder to save temporary html files.',
    default='./cache_plot', type=click.Path(file_okay=False, writable=True),
    show_default=True
)
@click.option(
    '-H', '--host', help='Hostname to listen on.', default='127.0.0.1',
    type=str, show_default=True
)
@click.option(
    '-P', '--port', help='Port of the webserver.', default=5000, type=int,
    show_default=True
)
@click.option(
    '-OT', '--output-template', help='Template output.',
    type=click.Path(exists=True)
)
@click.option(
    '-MC', '--model-conf', type=click.Path(exists=True),
    help='Model-configuration file path `.yaml`.'
)
@click.option(
    '-OS', '--only-summary', is_flag=True,
    help='Do not save vehicle outputs, just the summary.'
)
@click.option(
    '-AS', '--augmented-summary', is_flag=True,
    help='More outputs to the summary.'
)
@click.option(
    '-HV', '--hard-validation', is_flag=True, help='Add extra data validations.'
)
@click.option(
    '-DM', '--declaration-mode', is_flag=True,
    help='Use only the declaration data.'
)
@click.option(
    '-ES', '--enable-selector', is_flag=True,
    help='Enable the selection of the best model to predict both H/L cycles.'
)
@click.option(
    '-TA', '--type-approval-mode', is_flag=True, help='Is launched for TA?'
)
@click.option(
    '-PL', '--plot-workflow', is_flag=True,
    help='Open workflow-plot in browser, after run finished.'
)
@click.option(
    '-KP', '--encryption-keys-passwords',
    help='Encryption keys passwords file for reading TA files.',
    default='./DICE_KEYS/secret.passwords', type=click.Path(),
    show_default=True
)
@click_log.simple_verbosity_option(logger)
def run(input_files, cache_folder, host, port, plot_workflow, **kwargs):
    """
    Run CO2MPAS for all files into INPUT_FILES.

    INPUT_FILES: List of input files and/or folders
                 (format: .xlsx, .dill, .co2mpas.ta, .co2mpas).
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    inputs = dict(
        plot_workflow=plot_workflow, host=host, port=port, cmd_flags=kwargs,
        input_files=input_files, cache_folder=cache_folder, **{sh.START: kwargs}
    )
    os.makedirs(inputs.get('output_folder') or '.', exist_ok=True)
    return _process(inputs, ['plot', 'done', 'run'])


if _sync is not None:
    cli.add_command(_sync, 'syncing')

if _gui is not None:
    cli.add_command(_gui, 'gui')

if __name__ == '__main__':
    cli()
