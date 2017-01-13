#!/usr/bin/env pythonw
#
# Copyright 2014-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""
co2dice: prepare/sign/send/receive/validate/archive Type Approval sampling emails of *co2mpas*.

.. Warning::
    Do not run multiple instances!
"""

from co2mpas import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from co2mpas.__main__ import init_logging
from co2mpas.sampling import baseapp, CmdException
from co2mpas.sampling.baseapp import (APPNAME, Cmd,
                                      chain_cmds)  # @UnusedImport
import logging
import os
import re
import textwrap
import types
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

_default_cfg = textwrap.dedent("""
        ---
        dice:
            timestamping_address: post@stamper.itconsult.co.uk
            default_recipients: [co2mpas@jrc.ec.europa.eu,EC-CO2-LDV-IMPLEMENTATION@ec.europa.eu]
            #other_recipients:
            #sender:
        gpg:
            trusted_user_ids: [CO2MPAS JRC-master <co2mpas@jrc.ec.europa.eu>]
        """)

_opts_to_remove_before_cfg_write = ['default_recipients', 'timestamping_address']


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


###################
##    Commands   ##
###################

class MainCmd(Cmd):
    """
    co2dice: prepare/sign/send/receive/validate & archive Type Approval sampling emails for *co2mpas*.

    This is root command for co2mpas "dice"; use subcommands or preferably GUI to accomplish sampling.

    TIP:
      If you bump into blocking errors, please use the `co2dice project backup` command and
      send the generated archive-file back to "CO2MPAS-Team <co2mpas@jrc.ec.europa.eu>",
      for examination.

    NOTE:
      Do not run concurrently multiple instances!
    """

    name = trt.Unicode(APPNAME)
    version = __version__
    #examples = """TODO: Write cmd-line examples."""

    def __init__(self, **kwds):
        # Note: do not use `build_sub_cmds()` to avoid loading subcmd-tree.
        sub_cmds = {
            'project': ('co2mpas.sampling.project.ProjectCmd',
                        "Commands to administer the storage repo of TA *projects*."),
            'report': ('co2mpas.sampling.report.ReportCmd',
                       "Extract the report parameters from the co2mpas input/output files, or from *current-project*."),
            'tstamp': ('co2mpas.sampling.tstamp.TstampCmd',
                       "Commands to manage the communications with the Timestamp server."),
            'config': ('co2mpas.sampling.cfgcmd.ConfigCmd',
                       "Commands to manage configuration-options loaded from filesystem.")}
        with self.hold_trait_notifications():
            dkwds = {
                'name': APPNAME,
                'subcommands': sub_cmds,
            }
            dkwds.update(kwds)
            super().__init__(**dkwds)


####################################
## INFO: Add all CMDs here.
#
def all_cmds():
    from co2mpas.sampling import cfgcmd, project, report, tstamp
    return (
        (
            baseapp.Cmd,
            MainCmd,
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
    from co2mpas.sampling import project, report, tstamp
    return all_cmds() + (
        baseapp.Spec, project.ProjectsDB,  # TODO: specs are missing from all-config-classes.
        report.Report,
        tstamp.TstampSender,
        tstamp.TstampReceiver,
    )
####################################


def run_cmd(cmd: Cmd, argv: Sequence[Text]=None):
    """
    Executes a (possibly nested) command, and print its (possibly lazy) results to `stdout`.

    Remember to have logging setup properly before invoking this.

    :param cmd:
        Use :func:`make_app()`, or :func:`chain_cmds()` if you want to prepare
        a nested cmd instead.
    :param argv:
        If `None`, use :data:`sys.argv`; use ``[]`` to explicitely use no-args.
    :return:
        May yield, so check if a type:`GeneratorType`.
    """
    cmd.initialize(argv)
    res = cmd.start()
    if res is not None:
        if isinstance(res, types.GeneratorType):
            for i in res:
                print(i)
        elif isinstance(res, (tuple, list)):
            print(os.linesep.join(res))
        else:
            print(res)


def main(argv=None, log_level=None, **app_init_kwds):
    """
    :param argv:
        If `None`, use :data:`sys.argv`; use ``[]`` to explicitely use no-args.
    """
    init_logging(level=log_level)
    log = logging.getLogger(__name__)

    try:
        ##MainCmd.launch_instance(argv or None, **app_init_kwds) ## NO No, does not return `start()`!
        app = MainCmd.instance(**app_init_kwds)
        run_cmd(app, argv)
    except (CmdException, trt.TraitError) as ex:
        ## Suppress stack-trace for "expected" errors.
        log.debug('App exited due to: %s', ex, exc_info=1)
        exit(ex.args[0])
    except Exception as ex:
        ## Shell will see any exception x2, but we have to log it anyways,
        #  in case log has been redirected to a file.
        #
        log.error('Launch failed due to: %s', ex, exc_info=1)
        raise ex


if __name__ == '__main__':
    argv = None
    ## DEBUG AID ARGS, remember to delete them once developed.
    #argv = ''.split()
    #argv = '--debug'.split()
    #argv = '--help'.split()
    argv = '--help-all'.split()
    #argv = 'config init'.split()
    #argv = 'config init --help-all'.split()
    #argv = 'config init help'.split()
    #argv = '--debug --log-level=0 --Mail.port=6 --Mail.user="ggg" abc def'.split()
    #argv = 'project --help-all'.split()
    #argv = '--debug'.split()
    #argv = 'project list --help-all'.split()
    #argv = 'project --Project.reset_settings=True'.split()
    #argv = 'project --reset-git-settings'.split()
    #argv = 'project infos --help-all'.split()
    #argv = 'project infos'.split()
    #argv = 'project --help-all'.split()
    #argv = 'project examine --as-json --verbose --debug'.split()
    #argv = 'project examine --Project.verbose=2 --debug'.split()
    #argv = 'project list  --Project.reset_settings=True'.split()
    #argv = '--Project.reset_settings=True'.split()
    #argv = 'project list  --reset-git-settings'.split()
    #argv = 'project init one'.split()

    #argv = 'project current'.split()
    #argv = 'config paths'.split()

    #argv = 'tstamp send'.split()
    # Invoked from IDEs, so enable debug-logging.
    main(argv, log_level=logging.DEBUG)
    #main()

    #from traitlets.config import trtc.get_config

    #c = trtc.get_config()
    #c.Application.log_level=0
    #c.Spec.log_level='ERROR'
    #run_cmd(chain_cmds([MainCmd, ProjectCmd, ProjectCmd.InitCmd], argv=['project_foo']))
