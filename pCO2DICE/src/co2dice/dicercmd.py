#!/usr/bin/env python
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
New cmd to simplify stamping through WebStamper in one step.
"""
from . import cmdlets, base
from ._vendor import traitlets as trt


class DicerCmd(cmdlets.Cmd):
    """
    Dice a new (or existing) project in one action through WebStamper.

    SYNTAX
        %(cmd_chain)s [OPTIONS] (--inp <co2mpas-input>) ... (--out <co2mpas-output>) ...
                                [<any-other-file>] ...

    - The number of input-files given must match the number of output-file,
      and "paired" in the order they are given.
    - The project (VFID) is extracted from the given files - if it exists
      in projects db, it must be in `tagged` or earlier state.
    - The flag `--write-file=+~/.co2dice/reports.txt` is always implied, so
      every time this cmd runs, it *APPENDS* into the file above
      these 3 items, when generated:
        1. Dice
        2. Stamp (or any error received)
        3. Decision
    """

    examples = trt.Unicode("""
        - In the simplest case, just give input/out files:
              %(cmd_chain)s --inp input.xlsx --out output.xlsx
        - To specify more files (e.g. H/L in different runs):
              %(cmd_chain)s --inp input1.xlsx --out output1.xlsx \
                      --inp input2.xlsx --out output2.xlsx

          Tip: In Windows `cmd.exe` shell, the continuation charachter is `^`.
    """)

    inp = trt.List(
        trt.Unicode(),
        help="Specify co2mpas INPUT files; use this option one or more times."
    ).tag(config=True)

    out = trt.List(
        trt.Unicode(),
        help="Specify co2mpas OUTPUT files; use this option one or more times."
    ).tag(config=True)

    def __init__(self, **kwds):
        from toolz import dicttoolz as dtz
        from . import crypto, project, report, dicer, tstamp

        kwds = dtz.merge(kwds, {
            'conf_classes': [
                crypto.GitAuthSpec, crypto.StamperAuthSpec, crypto.EncrypterSpec,
                project.Project, project.ProjectsDB,
                report.ReporterSpec, dicer.DicerSpec, tstamp.WstampSpec],
            'cmd_aliases': dtz.merge(
                base.reports_keeper_alias_kwd, {
                    ('i', 'inp'): ('DicerCmd.inp', DicerCmd.inp.help),
                    ('o', 'out'): ('DicerCmd.out', DicerCmd.out.help),
                }
            ), 'cmd_flags': {
                'with-inputs': (
                    {
                        'ReporterSpec': {'include_input_in_dice': True},
                    }, report.ReporterSpec.include_input_in_dice
                    .help),  # @UndefinedVariable
                **base.shrink_flags_kwd,
            },
        })
        super().__init__(**kwds)

    def run(self, *args):
        from . import dicer

        dicerspec = dicer.DicerSpec(config=self.config)

        ## Parse cli-args.
        pfiles = base.PFiles(inp=self.inp, out=self.out, other=args)

        cstep = 0

        def progress_write_file(msg: str=None, step=1, nsteps=None):
            nonlocal cstep

            cstep += step
            progress = '(%s out of %s) %s' % (cstep, nsteps, msg)
            dicerspec.store_report(progress)

            self.log.info(progress)

        yield dicerspec.do_dice_in_one_step(pfiles, progress_write_file)
