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
## TODO: rename to `cli`
from co2dice._vendor.traitlets import traitlets as trt
from collections import OrderedDict
from typing import Sequence, Text, List, Tuple  # @UnusedImport
import logging
import sys

import os.path as osp

from . import (
    __version__, __updated__, __uri__, __copyright__, __license__,  # @UnusedImport
    cmdlets, CmdException, utils)
from .cmdlets import (APPNAME, Cmd,
                      chain_cmds)  # @UnusedImport


__title__ = APPNAME
__summary__ = utils.first_line(__doc__)

try:
    _mydir = osp.dirname(__file__)
except Exception:
    _mydir = '.'


class DiceSpec(cmdlets.Spec):
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
        ('dicer', ('co2dice.dicercmd.DicerCmd',
                   "Dice a new (or existing) project in one action through WebStamper.")),
        ('project', ('co2dice.project.ProjectCmd',
                     "Commands to administer the storage repo of TA *projects*.")),
        ('report', ('co2dice.report.ReportCmd',
                    "Extract the report parameters from the co2mpas input/output files, "
                    "or from *current-project*.")),
        ('tstamp', ('co2dice.tstamp.TstampCmd',
                    "Commands to manage the communications with the Timestamp server.")),
        ('config', ('co2dice.cfgcmd.ConfigCmd',
                    "Commands to manage configuration-options loaded from filesystem.")),
        ('tsigner', ('co2dice.tsigner.TsignerCmd',
                     "A command that time-stamps dice-reports.")),
    ])


####################################
## INFO: Add all CMDs here.
#
def all_cmds():
    from . import cfgcmd, project, dicercmd, report, tstamp, tsigner
    #import co2gui
    return (
        (
            cmdlets.Cmd,
            Co2diceCmd,
            dicercmd.DicerCmd,
            project.ProjectCmd,
            report.ReportCmd,
            tstamp.TstampCmd,
            tsigner.TsignerCmd,
            cfgcmd.ConfigCmd,
            #co2gui.Co2guiCmd,  # FIXME: should dice-CfgCmd refer to gui??
        ) +
        cfgcmd.config_subcmds +
        project.all_subcmds +
        tstamp.all_subcmds +
        report.all_subcmds
    )


## INFO: Add all SPECs here.
#
def all_app_configurables() -> Tuple:
    from . import crypto, project, dicer, report, tstamp, tsigner
    ## TODO: specs maybe missing from all-config-classes.
    all_config_classes = all_cmds() + (
        cmdlets.Spec,
        dicer.DicerSpec,
        project.ProjectSpec, project.Project, project.ProjectsDB,
        crypto.VaultSpec, crypto.GitAuthSpec,
        crypto.StamperAuthSpec, crypto.EncrypterSpec,
        report.ReporterSpec,
        tstamp.TstampSender,
        tstamp.TstampReceiver,
        tstamp.WstampSpec,
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
    from .utils import launchutils as lnu

    log = logging.getLogger(APPNAME)

    if sys.version_info < (3, 5):
        return lnu.exit_with_pride(
            "Sorry, Python >= 3.5 is required, found: %s" % sys.version_info,
            logger=log)

    import schema
    import transitions

    ## Decide log-level.
    #  NOTE that the use of any `--verbose` option,
    #  will override log-level, by setting :attr:`Spec.verbose` trait to True.
    #
    from .utils import logconfutils as lcu
    log_level, argv = lcu.log_level_from_argv(
        argv,
        start_level=20,  # 10=DEBUG, 20=INFO, 30=WARNING, ...
        eliminate_verbose=False, eliminate_quiet=True)

    lcu.init_logging(level=log_level, color=True, not_using_numpy=True,
                     # Load  this file automatically if it exists in HOME and configure logging,
                     # unless overridden with --logconf.
                     default_logconf_file=osp.expanduser(osp.join('~', '.co2_logconf.yaml'))
                     )
    log = logging.getLogger(APPNAME)

    from requests import HTTPError

    try:
        ## NOTE: HACK to fail early on first AIO launch.
        Cmd.configs_required = True
        cmd = Co2diceCmd.make_cmd(argv, **app_init_kwds)
        return cmdlets.pump_cmd(cmd.start()) and 0
    except (CmdException,
            trt.TraitError,
            transitions.MachineError,
            schema.SchemaError) as ex:
        log.debug('App exited due to: %r', ex, exc_info=1)
        ## Suppress stack-trace for "expected" errors but exit-code(1).
        return lnu.exit_with_pride(str(ex), logger=log)
    except HTTPError as ex:
        log.debug('App failed due to: %r', ex, exc_info=1)
        return lnu.exit_with_pride(
            "%s\n  remote error: %s",
            ex, ex.response.text,
            logger=log)
    except Exception as ex:
        ## Log in DEBUG not to see exception x2, but log it anyway,
        #  in case log has been redirected to a file.
        log.debug('App failed due to: %r', ex, exc_info=1)
        ## Print stacktrace to stderr and exit-code(-1).
        return lnu.exit_with_pride(ex, logger=log)
