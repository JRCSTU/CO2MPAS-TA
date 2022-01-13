# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
r"""
Define SYNCING the command line interface.

.. click:: co2mpas.cli.sync:cli
   :prog: co2mpas syncing
   :show-nested:

"""
import os
import click
import click_log
import os.path as osp
import schedula as sh
from syncing.cli import cli, logger

_sync_params = {p.name: p for p in cli.commands['sync'].params}
_sync_params['reference_name'].default = 'theoretical'
_sync_params['x_label'].default = ('times',)
_sync_params['y_label'].default = ('velocities',)


@cli.command('template', short_help='Generates sample template file.')
@click.argument(
    'output-file', default='datasync.xlsx', required=False,
    type=click.Path(writable=True)
)
@click.option(
    '-CT', '--cycle-type', default='wltp', show_default=True,
    type=click.Choice(['nedc', 'wltp']), help='Cycle type.'
)
@click.option(
    '-WC', '--wltp-class', default='class3b', show_default=True,
    type=click.Choice(['class1', 'class2', 'class3a', 'class3b']),
    help='WLTP vehicle class.'
)
@click.option(
    '-GB', '--gear-box-type', default='automatic', show_default=True,
    type=click.Choice(['manual', 'automatic']), help='Gear box type.'
)
@click_log.simple_verbosity_option(logger)
def template(output_file, cycle_type, gear_box_type, wltp_class):
    """
    Writes a sample template OUTPUT_FILE.

    OUTPUT_FILE: SYNCING input template file (.xlsx). [default: ./datasync.xlsx]
    """

    import pandas as pd
    from co2mpas.core.model.physical.cycle import dsp
    theoretical = sh.selector(['times', 'velocities'], dsp(inputs=dict(
        cycle_type=cycle_type.upper(), gear_box_type=gear_box_type,
        wltp_class=wltp_class, downscale_factor=0
    ), outputs=['times', 'velocities'], shrink=True))
    base = dict.fromkeys((
        'times', 'velocities', 'target gears', 'engine_speeds_out',
        'engine_coolant_temperatures', 'co2_normalization_references',
        'alternator_currents', 'battery_currents', 'target fuel_consumptions',
        'target co2_emissions', 'target engine_powers_out'
    ), [])
    data = dict(theoretical=theoretical, dyno=base, obd=base)
    os.makedirs(osp.dirname(output_file) or '.', exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        for k, v in data.items():
            pd.DataFrame(v).to_excel(writer, k, index=False)
    return data


if __name__ == '__main__':
    cli()
