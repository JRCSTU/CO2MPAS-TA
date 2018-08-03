#!/usr/bin/env pythonw
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""
Main CO2MPAS dice command to prepare/sign/send/receive/validate/archive ...

Type Approval sampling emails of *co2mpas*.
"""
from co2mpas._vendor import traitlets as trt
from . import (
    __version__, __updated__, __uri__, __copyright__, __license__,  # @UnusedImport
    baseapp, CmdException)
from .baseapp import (APPNAME, Cmd,
                      chain_cmds)  # @UnusedImport
from collections import OrderedDict
from typing import Sequence, Text, List, Tuple  # @UnusedImport
import logging
import sys

import os.path as osp
import pandalone.utils as pndlu


__title__ = APPNAME
__summary__ = pndlu.first_line(__doc__)

try:
    _mydir = osp.dirname(__file__)
except Exception:
    _mydir = '.'


class DiceSpec(baseapp.Spec):
    """Common parameters dice functionality."""

    user_name = trt.Unicode(
        help="""The Name & Surname of the default user invoking the app.  Must not be empty!"""
    ).tag(config=True)

    user_email = trt.Unicode(
        help="""The email address of the default user invoking the app. Must not be empty!"""
    ).tag(config=True)

    def __init__(self, **kwds):
        cls = type(self)
        self.register_validators(
            cls.user_name,
            cls._is_not_empty, cls._is_all_latin)
        self.register_validators(
            cls.user_email,
            cls._is_not_empty, cls._is_all_latin, cls._is_pure_email_address)
        super().__init__(**kwds)


###################
##    Commands   ##
###################

class Co2diceCmd(Cmd):
    """
    Prepare/sign/send/receive/validate & archive Type Approval sampling emails for *co2mpas*.

    This is the root command for co2mpas DICE; use its sub-commands
    to "run the dice" on the files of a co2mpas run.

    Note:
      Do not run concurrently multiple instances.
    """

    name = trt.Unicode(APPNAME)
    version = __version__
    examples = trt.Unicode("""\
    - Try the `project` sub-command::
          %(cmd_chain)s  project

    - To learn more about command-line options and configurations::
          %(cmd_chain)s  config

    - Read configurations also from a `GMail` folder present in current-dir
      and view what was loaded actually::
          %(cmd_chain)s  config paths --config-paths GMail --config-paths ~/.co2dice
          %(cmd_chain)s  config show --source file --config-paths GMail --config-paths ~/.co2dice
    """)

    subcommands = OrderedDict([
        ('project', ('co2mpas.sampling.project.ProjectCmd',
                     "Commands to administer the storage repo of TA *projects*.")),
        ('report', ('co2mpas.sampling.report.ReportCmd',
                    "Extract the report parameters from the co2mpas input/output files, "
                    "or from *current-project*.")),
        ('tstamp', ('co2mpas.sampling.tstamp.TstampCmd',
                    "Commands to manage the communications with the Timestamp server.")),
        ('config', ('co2mpas.sampling.cfgcmd.ConfigCmd',
                    "Commands to manage configuration-options loaded from filesystem.")),
        ('tsigner', ('co2mpas.sampling.tsigner.TsignerCmd',
                     "A command that time-stamps dice-reports.")),
    ])


####################################
## INFO: Add all CMDs here.
#
def all_cmds():
    from . import cfgcmd, project, report, tstamp, tsigner
    from co2mpas import tkui
    return (
        (
            baseapp.Cmd,
            Co2diceCmd,
            project.ProjectCmd,
            report.ReportCmd,
            tstamp.TstampCmd,
            tsigner.TsignerCmd,
            cfgcmd.ConfigCmd,
            tkui.Co2guiCmd,
        ) +
        cfgcmd.config_subcmds +
        project.all_subcmds +
        tstamp.all_subcmds +
        report.all_subcmds
    )


## INFO: Add all SPECs here.
#
def all_app_configurables() -> Tuple:
    from . import crypto, project, report, tstamp, tsigner
    ## TODO: specs maybe missing from all-config-classes.
    all_config_classes = all_cmds() + (
        baseapp.Spec,
        project.ProjectSpec, project.Project, project.ProjectsDB,
        crypto.VaultSpec, crypto.GitAuthSpec,
        crypto.StamperAuthSpec, crypto.EncrypterSpec,
        report.ReporterSpec,
        tstamp.TstampSender,
        tstamp.TstampReceiver,
        tsigner.TsignerService,
    )

    # ## TODO: Enable when `project TstampCmd` dropped, and `report` renamed.
    # #
    # all_names = [cls.__name__ for cls in all_config_classes]
    # assert len(set(all_names)) == len(all_names), (
    #     "Duplicate configurable names!", sorted(all_names))

    return all_config_classes

####################################


def run(argv=(), **app_init_kwds):
    """
    Handles some exceptions politely and returns the exit-code.

    :param argv:
        Cmd-line arguments, nothing assumed if nothing given.
    """
    from co2mpas import __main__ as cmain

    log = logging.getLogger(APPNAME)

    if sys.version_info < (3, 5):
        return cmain.exit_with_pride(
            "Sorry, Python >= 3.5 is required, found: %s" % sys.version_info,
            logger=log)

    import schema
    import transitions

    ## Decide log-level.
    #  NOTE that the use of any `--verbose` option,
    #  will override log-level, by setting :attr:`Spec.verbose` trait to True.
    #
    from co2mpas .utils import logconfutils as lcu
    log_level, argv = lcu.log_level_from_argv(
        argv,
        start_level=20,  # 10=DEBUG, 20=INFO, 30=WARNING, ...
        eliminate_verbose=False, eliminate_quiet=True)

    cmain.init_logging(level=log_level, color=True, not_using_numpy=True)
    log = logging.getLogger(APPNAME)

    try:
        ## NOTE: HACK to fail early on first AIO launch.
        Cmd.configs_required = True
        cmd = Co2diceCmd.make_cmd(argv, **app_init_kwds)
        return baseapp.pump_cmd(cmd.start()) and 0
    except (CmdException,
            trt.TraitError,
            transitions.MachineError,
            schema.SchemaError) as ex:
        log.debug('App exited due to: %r', ex, exc_info=1)
        ## Suppress stack-trace for "expected" errors but exit-code(1).
        return cmain.exit_with_pride(str(ex), logger=log)
    except Exception as ex:
        ## Log in DEBUG not to see exception x2, but log it anyway,
        #  in case log has been redirected to a file.
        log.debug('App failed due to: %r', ex, exc_info=1)
        ## Print stacktrace to stderr and exit-code(-1).
        return cmain.exit_with_pride(ex, logger=log)
