#!/usr/bin/env pythonw
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""co2dice: prepare/sign/send/receive/validate/archive Type Approval sampling emails of *co2mpas*."""

from co2mpas import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp, CmdException
from co2mpas.sampling.baseapp import (APPNAME, Cmd,
                                      chain_cmds)  # @UnusedImport
from collections import OrderedDict
import logging
import os
import re
import textwrap
from typing import Sequence, Text, List, Tuple  # @UnusedImport

import os.path as osp
import pandalone.utils as pndlu
import traitlets as trt


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

_list_response_regex = re.compile(r'\((?P<flags>.*?)\) "(?P<delimiter>.*)" (?P<name>.*)')


def _parse_list_response(line):
    flags, delimiter, mailbox_name = _list_response_regex.match(line).groups()
    mailbox_name = mailbox_name.strip('"')
    return (flags, delimiter, mailbox_name)


class DiceSpec(baseapp.Spec):
    """Common parameters dice functionality."""

    user_name = trt.Unicode(
        None, allow_none=False,
        help="""The Name & Surname of the default user invoking the app.  Must not be empty!"""
    ).tag(config=True)

    user_email = trt.Unicode(
        None, allow_none=False,
        help="""The email address of the default user invoking the app. Must not be empty!"""
    ).tag(config=True)

    @trt.validate('user_name', 'user_email')
    def _is_not_empty(self, proposal):
        value = proposal['value']
        if not value:
            raise trt.TraitError('%s.%s must not be empty!'
                                 % (proposal['owner'].name, proposal['trait'].name))
        return value


###################
##    Commands   ##
###################

class Co2dice(Cmd):
    """
    Prepare/sign/send/receive/validate & archive Type Approval sampling emails for *co2mpas*.

    This is the root command for co2mpas "dice"; use subcommands or preferably GUI to accomplish sampling.

    TIP:
      If you bump into blocking errors, please use the `co2dice project backup` command and
      send the generated archive-file back to "CO2MPAS-Team <co2mpas@jrc.ec.europa.eu>",
      for examination.

    NOTE:
      Do not run concurrently multiple instances!
    """

    name = trt.Unicode(APPNAME)
    version = __version__
    examples = trt.Unicode("""
        Try the `project` sub-command:
            %(cmd_chain)s  project

        To check where is your configuration path, use:
            %(cmd_chain)s  config  show

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
            Co2dice,
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
    return all_cmds() + (
        baseapp.Spec, project.ProjectsDB,  # TODO: specs maybe missing from all-config-classes.
        crypto.VaultSpec,
        report.Report,
        tstamp.TstampSender,
        tstamp.TstampReceiver,
    )
####################################


def main(argv=None, log_level=None, **app_init_kwds):
    """
    :param argv:
        If `None`, use :data:`sys.argv`; use ``[]`` to explicitly use no-args.
    """
    import transitions

    init_logging(level=log_level, color=True)
    log = logging.getLogger(__name__)

    try:
        cmd = Co2dice.make_cmd(argv, **app_init_kwds)
        baseapp.consume_cmd(cmd.start())
    except (CmdException, trt.TraitError, transitions.MachineError) as ex:
        ## Suppress stack-trace for "expected" errors.
        #
        #  Note: For *transitions*, better pip-install from:
        #    https://github.com/ankostis/transitions@master
        #  to facilitate debugging from log/ex messages, unless
        #  tyarkoni/transitions#179 & tyarkoni/transitions#180 merged.
        log.debug('App exited due to: %s', ex, exc_info=1)
        log.error(ex.args[0])
        exit(-1)
    except Exception as ex:
        ## Shell will see any exception x2, but we have to log it anyways,
        #  in case log has been redirected to a file.
        #
        log.error('Launch failed due to: %s', ex, exc_info=1)
        raise ex


if __name__ == '__main__':
    argv = None  # Uses sys.argv.
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
    # Invoked from IDEs, so enable debug-logging.
    main(argv, log_level=logging.DEBUG)
    #main()

    #from traitlets.config import trtc.get_config

    #c = trtc.get_config()
    #c.Application.log_level=0
    #c.Spec.log_level='ERROR'
    #cmd = chain_cmds([Co2dice, ProjectCmd, InitCmd], argv=['project_foo'])
    # baseapp.consume_cmd(cmd.start())
