#!/usr/bin/env pythonw
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""co2dice: prepare/sign/send/receive/validate/archive Type Approval sampling emails of *co2mpas*."""

from co2mpas import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from co2mpas.sampling import baseapp, CmdException
from co2mpas.sampling.baseapp import (APPNAME, Cmd,
                                      chain_cmds)  # @UnusedImport
from collections import OrderedDict
import logging
import os
import re
import sys
from typing import Sequence, Text, List, Tuple  # @UnusedImport

import os.path as osp
import pandalone.utils as pndlu
from co2mpas._vendor import traitlets as trt


__title__ = APPNAME
__summary__ = pndlu.first_line(__doc__)


try:
    _mydir = osp.dirname(__file__)
except:
    _mydir = '.'


def get_home_dir():
    """Get the real path of the home directory"""
    homedir = osp.expanduser('~')
    # Next line will make things work even when /home/ is a symlink to
    # /usr/home as it is on FreeBSD, for example
    homedir = osp.realpath(homedir)
    return homedir


def app_config_dir():
    """Get the config directory for this platform and user.

    Returns CO2DICE_CONFIG_DIR if defined, else ~/.co2dice
    """

    env = os.environ
    home_dir = get_home_dir()

    if env.get('CO2DICE_CONFIG_DIR'):
        return env['CO2DICE_CONFIG_DIR']

    return osp.abspath(osp.join(home_dir, '.co2dice'))


class DiceSpec(baseapp.Spec):
    """Common parameters dice functionality."""

    user_name = trt.Unicode(
        help="""The Name & Surname of the default user invoking the app.  Must not be empty!"""
    ).tag(config=True)

    user_email = trt.Unicode(
        help="""The email address of the default user invoking the app. Must not be empty!"""
    ).tag(config=True)

    def __init__(self, **kwds):
        self._register_validator(type(self)._is_not_empty,
                                 ['user_name', 'user_email'])
        self._register_validator(type(self)._is_all_latin,
                                 ['user_name', 'user_email'])
        self._register_validator(type(self)._is_pure_email_address,
                                 ['user_email'])
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
                    "Extract the report parameters from the co2mpas input/output files, or from *current-project*.")),
        ('tstamp', ('co2mpas.sampling.tstamp.TstampCmd',
                    "Commands to manage the communications with the Timestamp server.")),
        ('config', ('co2mpas.sampling.cfgcmd.ConfigCmd',
                    "Commands to manage configuration-options loaded from filesystem.")),
    ])


####################################
## INFO: Add all CMDs here.
#
def all_cmds():
    from co2mpas.sampling import cfgcmd, project, report, tstamp
    return (
        (
            baseapp.Cmd,
            Co2diceCmd,
            project.ProjectCmd,
            report.ReportCmd,
            tstamp.TstampCmd,
            cfgcmd.ConfigCmd,
        ) +
        cfgcmd.config_subcmds +
        project.all_subcmds +
        tstamp.all_subcmds)


## INFO: Add all SPECs here.
#
def all_app_configurables() -> Tuple:
    from co2mpas.sampling import crypto, project, report, tstamp
    from co2mpas import tkui
    ## TODO: specs maybe missing from all-config-classes.
    all_config_classes = all_cmds() + (
        baseapp.Spec,
        project.ProjectSpec, project.Project, project.ProjectsDB,
        crypto.VaultSpec, crypto.GitAuthSpec, crypto.StamperAuthSpec,
        report.Report,
        tstamp.TstampSender,
        tstamp.TstampReceiver,
        tkui.Co2guiCmd,
    )

    # ## TODO: Enable when `project TstampCmd` dropped, and `report` renamed.
    # #
    # all_names = [cls.__name__ for cls in all_config_classes]
    # assert len(set(all_names)) == len(all_names), (
    #     "Duplicate configurable names!", sorted(all_names))

    return all_config_classes

####################################


def main(argv=None, **app_init_kwds):
    """
    Handles some exceptions politely and returns the exit-code.

    :param argv:
        If `None`, use :data:`sys.argv`; use ``[]`` to explicitly use no-args.
    """
    from co2mpas import __main__ as cmain

    log = logging.getLogger(APPNAME)

    if sys.version_info < (3, 5):
        return cmain.exit_with_pride(
            "Sorry, Python >= 3.5 is required, found: %s" % sys.version_info,
            logger=log)

    import transitions

    ## At these early stages, any log cmd-line option
    #  enable DEBUG logging ; later will be set by `baseapp` traits.
    log_level = logging.DEBUG if cmain.is_any_log_option(argv) else None

    cmain.init_logging(level=log_level, color=True, not_using_numpy=True)
    log = logging.getLogger(APPNAME)

    try:
        ## NOTE: HACK to fail early on first AIO launch.
        Cmd.configs_required = True
        cmd = Co2diceCmd.make_cmd(argv, **app_init_kwds)
        return baseapp.pump_cmd(cmd.start()) and 0
    except (CmdException, trt.TraitError, transitions.MachineError) as ex:
        log.debug('App exited due to: %r', ex, exc_info=1)
        ## Suppress stack-trace for "expected" errors but exit-code(1).
        return cmain.exit_with_pride(str(ex), logger=log)
    except Exception as ex:
        ## Log in DEBUG not to see exception x2, but log it anyway,
        #  in case log has been redirected to a file.
        log.debug('App failed due to: %r', ex, exc_info=1)
        ## Print stacktrace to stderr and exit-code(-1).
        return cmain.exit_with_pride(ex, logger=log)


if __name__ == '__main__':
    if __package__ is None:
        __package__ = "co2mpas.sampling"  # @ReservedAssignment

    sys.exit(main())  # Use sys.argv.

    ## DEBUG AID ARGS, remember to delete them once developed.
    #argv = ''.split()
    #argv = '--debug'.split()
    #argv = '--help'.split()
    #argv = '--help-all'.split()
    #argv = 'config write'.split()
    #argv = 'config paths'.split()
    #argv = '--debug --log-level=0 --Mail.port=6 --Mail.user="ggg" abc def'.split()
    #argv = 'project --help'.split()
    #argv = 'project ls--vlevel=3'.split()
    #argv = '--debug'.split()
    #argv = 'project status -v'.split()
    #argv = 'project --help-all'.split()
    #argv = 'project status --verbose --debug'.split()
    #argv = 'project status --Project.verbose=2 --debug'.split()
    #argv = 'project ls --Project.preserved_git_settings=.*'.split()
    #argv = '--Project.preserved_git_settings=.*'.split()
    #argv = 'project init P1 --force'.split()
    #argv = 'project append  out=tests/sampling/output.xlsx'.split()
    #argv = 'project append  inp=tests/sampling/input.xlsx'.split()
    #argv = 'project report -n'.split()
    #argv = 'project report --vfids --project'.split()
    #argv = 'project export FT-12-ABC-2016-0001 P1'.split()

    #argv = 'project ls.'.split()
    #argv = 'project init PROJ1 -f'.split()
    #argv = 'config paths'.split()

    #argv = 'tstamp send'.split()
    #argv = 'tstamp login'.split()
    #sys.exit(main(argv))

    #from traitlets.config import trtc.get_config

    #c = trtc.get_config()
    #c.Application.log_level=0
    #c.Spec.log_level='ERROR'
    #cmd = chain_cmds([Co2diceCmd, ProjectCmd, InitCmd], argv=['project_foo'])
    # sys.exit(baseapp.pump_cmd(cmd.start()) and 0)
