# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
r"""
Define the command line interface.

.. click:: co2mpas.cli:cli
   :prog: co2mpas
   :show-nested:

"""
import click
import logging
import click_log
import schedula as sh
from co2mpas import dsp as _process
from co2mpas._version import __version__

log = logging.getLogger('co2mpas.cli')


class _Logger(logging.Logger):
    # noinspection PyMissingOrEmptyDocstring
    def setLevel(self, level):
        super(_Logger, self).setLevel(level)
        frmt = "%(asctime)-15s:%(levelname)5.5s:%(name)s:%(message)s"
        logging.basicConfig(level=level, format=frmt)
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
@click_log.simple_verbosity_option(logger)
def cli():
    """
    CO2MPAS command line tool.
    """


@cli.command('template', short_help='Generates sample template file.')
@click.argument(
    'output-file', default='template.xlsx', required=False,
    type=click.Path(writable=True)
)
def template(output_file='template.xlsx'):
    """
    Writes a CO2MPAS input template OUTPUT_FILE.

    OUTPUT_FILE: CO2MPAS input template file (.xlsx). [default: ./template.xlsx]
    """
    return _process({'output_file': output_file}, ['template', 'done'])


@cli.command('demo', short_help='Generates sample demo file.')
@click.argument(
    'output-folder', default='./inputs', required=False,
    type=click.Path(writable=True, file_okay=False)
)
def demo(output_folder):
    """
    Writes a CO2MPAS input template OUTPUT_FILE.

    OUTPUT_FILE: CO2MPAS input template file (.xlsx). [default: ./template.xlsx]
    """
    return _process({'output_folder': output_folder}, ['demo', 'done'])


@cli.command('conf', short_help='Generates sample template file.')
@click.argument(
    'output-file', default='conf.yaml', required=False,
    type=click.Path(writable=True)
)
@click.option('-MC', '--model-conf', type=click.Path(exists=True))
def conf(output_file='conf.yaml', **kwargs):
    """
    Writes a CO2MPAS input template OUTPUT_FILE.

    OUTPUT_FILE: CO2MPAS input template file (.xlsx). [default: ./template.xlsx]
    """
    inputs = {sh.START: kwargs, 'output_file': output_file}
    return _process(inputs, ['conf', 'done'])


@cli.command('plot', short_help='Generates sample template file.')
@click.option(
    '-O', '--cache-folder', help='Folder to save temporary html files.',
    default='./cache_plot', type=click.Path(file_okay=False, writable=True)
)
def plot(cache_folder='./cache_plot'):
    """
    Plots the full CO2MPAS model.
    """
    return _process({'cache_folder': cache_folder}, ['plot', 'done'])


@cli.command('run', short_help='')
@click.argument('input-files', nargs=-1, type=click.Path(exists=True))
@click.option(
    '-O', '--output-folder', help='Output folder.', default='./outputs',
    type=click.Path(file_okay=False, writable=True)
)
@click.option(
    '-OT', '--output-template', help='Template output.',
    type=click.Path(exists=True)
)
@click.option(
    '-EK', '--encryption-keys', help='Encryption keys for TA mode.',
    default='./DICE_KEYS/dice.co2mpas.keys', type=click.Path()
)
@click.option('-MC', '--model-conf', type=click.Path(exists=True))
@click.option(
    '-SK', '--sign-key', help='User signature key for TA mode.',
    default='./DICE_KEYS/sign.co2mpas.key', type=click.Path()
)
@click.option('-OS', '--only-summary', is_flag=True)
@click.option('-SV', '--soft-validation', is_flag=True)
@click.option('-EM', '--engineering-mode', is_flag=True)
@click.option('-ES', '--enable-selector', is_flag=True)
@click.option('-TA', '--type-approval-mode', is_flag=True)
@click.option('-PL', '--plot-workflow', is_flag=True)
def run(input_files, **kwargs):
    """
    Synchronise and re-sample data-sets defined in INPUT_FILE and writes shifts
    and synchronized data into the OUTPUT_FILE.

    INPUT_FILE: Data-sets input file (format: .xlsx, .json).

    OUTPUT_FILE: output file (format: .xlsx, .json).
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    inputs = {'cmd_flags': kwargs, 'input_files': input_files, sh.START: kwargs}
    return _process(inputs, ['plot', 'done', 'run'])


if __name__ == '__main__':
    cli()
