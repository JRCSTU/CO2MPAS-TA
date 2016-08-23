#!/usr/b in/env python
#
# Copyright 2014-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""co2dice: prepare/sign/send/receive/validate/archive Type Approval sampling emails of *co2mpas*."""

import base64
from collections import defaultdict, MutableMapping
import locale
import copy
import errno
import configparser
from email.mime.text import MIMEText
import imaplib
import inspect
import subprocess
import pprint
from pandalone.pandata import resolve_path
from pandalone import utils as pndl_utils
import io
import logging
import os
import re
import tempfile
import shutil
import smtplib

from boltons.setutils import IndexedSet as iset, IndexedSet
import gnupg
import hiyapyco
import keyring
import textwrap
from traitlets.config import ConfigFileNotFound
from toolz import dicttoolz
from toolz import itertoolz as itz
import traitlets as trt
from traitlets.config import Application, Configurable, SingletonConfigurable, get_config, catch_config_error

import functools as ft
import pandas as pd
import itertools as itt
import os.path as osp

from co2mpas.__main__ import CmdException, init_logging
from co2mpas import __uri__  # @UnusedImport
from co2mpas._version import (__version__, __updated__, __file_version__,   # @UnusedImport
                              __input_file_version__, __copyright__, __license__)  # @UnusedImport
__title__ = 'co2dice'
__summary__   = __doc__.split('\n')[0]


log = logging.getLogger(__name__) #TODO: Replace with App logger.
_mydir = osp.dirname(__file__)
CONF_VAR_NAME = '%s_CONFIG_FILE' % __title__.upper()

_project = 'co2mpas'

_default_cfg = textwrap.dedent("""
        ---
        dice:
            timestamping_address: post@stamper.itconsult.co.uk
            default_recipients: [co2mpas@jrc.ec.europa.eu,EC-CO2-LDV-IMPLEMENTATION@ec.europa.eu]
            #other_recipients:
            #sender:
        SMTP:
            ssl: on
            #host:
            #user:
            #login:
            #kwds:
        IMAPv4:
            ssl: on
            #user:
            #host:
            #kwds:
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

## TODO: Move to other file
def read_config(projname, override_fpaths=(), default_fpaths=()):
    home_cfg = osp.normpath(osp.expanduser('~/.%s.cfg' % projname))
    cwd_cfg = osp.abspath(('%s.cfg' % projname))
    if not osp.isfile(home_cfg):
        log.info('Creating home config-file %r...', home_cfg)
        os.makedirs(osp.dirname(home_cfg), exist_ok=True)
        with io.open(home_cfg, 'wt') as fp:
            fp.write(_default_cfg)

    cfg_fpaths = list(override_fpaths) + [home_cfg, cwd_cfg] + list(default_fpaths)
    cfg = hiyapyco.load(cfg_fpaths,
            method=hiyapyco.METHOD_MERGE,
            interpolate=True, # (default: False)
            castinterpolated=True, # (default: False)
            usedefaultyamlloader=False, # (default: False)
            loglevel=logging.INFO,
            failonmissingfiles=False, #  (default: True)
            #loglevelmissingfiles
    )
    return cfg

def splitnest(dic, sep=r'\.', factory=None):
    """
    Splits keys by separator and *nests* them in a new mapping (from factory).

    :param dic:     The mapping to split and nest its keys.
    :param str sep: the split-separator regex
    :param factory: The factory-function for the nested mappings.
                    By default `None`, it uses the type of the input `dic`.
                    Note that when `None`, it might fail if the constructor
                    of `dic` type expects specific args, such as `defaultdict`.

    >>> RES == {'a': {'b': {'c': {'d': 44}}, 'c': 2}, 'b': 3}
    >>> dice.splitnest({'a.b': 1, 'a-c': 2, 'b': 3, 'a.b.c,d':44}, '[.,-]') == RES
    True

    >>> dice.splitnest({'abc': 1})
    {'abc': 1}

    >>> dice.splitnest({})
    {}
    """
    def set_subkeys(d, subkeys, v):
        """
        >>> d = {}
        >>> set_subkeys(d, ['a','b',], 3)
        >>> d
        {'a': {'b': 3}}

        >>> set_subkeys(d, ['a','b',], 4)
        >>> d
        {'a': {'b': 4}}

        >>> set_subkeys(d, ['a','b', 'c'], 5)
        >>> d
        {'a': {'b': {'c' : 5}}

        >>> set_subkeys(d, ['a'], 0)
        >>> d
        {'a': 0}
        """
        k, subkeys = subkeys[0], subkeys[1:]
        if subkeys:
            try:
                set_subkeys(d[k], subkeys, v)
            except:
                d[k] = factory()
                set_subkeys(d[k], subkeys, v)
        else:
            d[k] = v

    sep = re.compile(sep)
    if factory:
        ndic = factory()
    else:
        factory = type(dic)
        ndic = factory()
    for k, v in dic.items():
        subkeys = [sk for sk in sep.split(k) if sk]
        if subkeys:
            set_subkeys(ndic, subkeys, v)
    return ndic

_NOT_SET = object()
def dotget_storage_value(key, default=_NOT_SET):
    if '.' in key:
        path = key.split('.')
        key = path[0]
        path = key[1:]
    else:
        path = None

    if not key in store:
        value = default
    else:
        value = store[key]
        if path:
            if value is None:
                value = default ## Missing path replaed by default
            value = dice.splitnest(value).get(path, default)
    if value is _NOT_SET:
        raise KeyError('key')
    return value


def store_config(cfg, projname):
    for opt in _opts_to_remove_before_cfg_write:
        cfg.remove_option('dice', opt)
    fpath = osp.expanduser('~/.%s.cfg' % projname)
    with io.open(fpath, 'wt', encoding='utf-8') as fd:
        cfg.write(fd)

def store_secret(master_pswd, key, secret):
    """
    Uses Microsoft's DPPAPI to store the actual passwords.

    :param str master_pswd:     master-password given by the user
    """
    kr=keyring.get_keyring()
    kr.set_password('%s.%s' %(_project, master_pswd), key, secret)

def retrieve_secret(master_pswd, key):
    """
    Uses Microsoft's DPPAPI to store the actual passwords.

    :param str master_pswd:     master-password given by the user
    """
    kr=keyring.get_keyring()
    return kr.get_password('%s.%s' %(_project, master_pswd), key)

def py_where(program, path=None):
    ## From: http://stackoverflow.com/a/377028/548792
    winprog_exts = ('.bat', 'com', '.exe')
    def is_exec(fpath):
        return osp.isfile(fpath) and os.access(fpath, os.X_OK) and (
                os.name != 'nt' or fpath.lower()[-4:] in winprog_exts)

    progs = []
    if not path:
        path = os.environ["PATH"]
    for folder in path.split(ospsep):
        folder = folder.strip('"')
        if folder:
            exe_path = osp.join(folder, program)
            for f in [exe_path] + ['%s%s'%(exe_path, e) for e in winprog_exts]:
                if is_exec(f):
                    progs.append(f)
    return progs


def where(program):
    try:
        res = subprocess.check_output('where "%s"' % program,
                universal_newlines=True)
        return res and [s.strip()
                       for s in res.split('\n') if s.strip()]
    except subprocess.CalledProcessError:
        return []
    except:
        return py_where(program)


def which(program):
    res = where(program)
    return res[0] if res else None


def _describe_gpg(gpg):
    gpg_path = gpg.gpgbinary
    if not osp.isabs(gpg_path):
        gpg_path = shutil.which(gpg_path)

    ver = str(gpg.version)
    nprivkeys = len(gpg.list_keys(True))
    nallkeys = len(gpg.list_keys())
    return gpg_path, ver, nprivkeys, nallkeys


def collect_gpgs():
    inc_errors=1
    gpg_kws={}
    gpg_paths = IndexedSet(itt.chain.from_iterable(where(prog) for prog in ('gpg2', 'gpg')))
    gnupghome = osp.expanduser('~/.gnupg')
    gpg_avail = []
    for gpg_path in gpg_paths:
        try:
            gpg = gnupg.GPG(gpgbinary=gpg_path, **gpg_kws)
            row = _describe_gpg(gpg)
        except Exception as ex:
            #raise
            if inc_errors:
                row = (gpg_path, '%s: %s' % (type(ex).__name__, str(ex)), None, None)
            else:
                continue
        gpg_avail.append(row)

    cols= ['GnuPG path', 'Version', '#PRIV', '#TOTAL']
    gpg_avail = pd.DataFrame(gpg_avail, columns=cols)
    return gpg_avail


#gpg_avail = collect_gpgs()

def gpg_del_gened_key(gpg, fingerprint):
    log.debug('Deleting secret+pub: %s', fingerprint)
    d = gpg.delete_keys(fingerprint, secret=1)
    assert (d.status, d.stderr) == ('ok', ''), (
            "Failed DELETING pgp-secret: %s" % d.stderr)
    d=gpg.delete_keys(fingerprint)
    assert (d.status, d.stderr) == ('ok', ''), (
            "Failed DELETING pgp-secret: %s" % d.stderr)


def gpg_genkey(name_email, name_real, name_comment=None, key_type='RSA', key_length=2048, **kwds):
    """See https://www.gnupg.org/documentation/manuals/gnupg/Unattended-GPG-key-generation.html#Unattended-GPG-key-generation"""
    args = locals().copy()
    args.pop('kwds')
    return gpg.gen_key_input(**args)

def _has_repeatitive_prefix(word, limit, char=None):
    c = word[0]
    if not char or c == char:
        for i  in range(1, limit):
            if word[i] != c:
                break
        else:
            return True


def gpg_gen_interesting_keys(gpg, name_real, name_email, key_length,
        predicate, nkeys=1, runs=0):
    keys = []
    for i in itt.count(1):
        del_key = True
        key = gpg.gen_key(gpg.gen_key_input(key_length=key_length,
                name_real=name_real, name_email=name_email))
        try:
            log.debug('Created-%i: %s', i, key.fingerprint)
            if predicate(key.fingerprint):
                del_key = False
                keys.append(key.fingerprint)
                keyid = key.fingerprint[24:]
                log.info('FOUND-%i: %s-->%s', i, keyid, key.fingerprint)
                nkeys -= 1
                if nkeys == 0:
                    break
        finally:
            if del_key:
                gpg_del_gened_key(gpg, key.fingerprint)
    return keys

def _log_into_server(login_cb, login_cmd, prompt):
    """
    Connects a credential-source(`login_db`) to a consumer(`login_cmd`).

    :param login_cb:
            An object with 2 methods::

                ask_user_pswd(prompt) --> (user, pswd)  ## or `None` to abort.
                report_failure(obj)
    :param login_cmd:
            A function like::

                login_cmd(user, pswd) --> xyz  ## `xyz` might be the server.
    """
    for login_data in iter(lambda: login_cb.ask_user_pswd('%s? ' % prompt), None):
        user, pswd = login_data
        try:
            return login_cmd(user, pswd)
        except smtplib.SMTPAuthenticationError as ex:
            login_cb.report_failure('%r' % ex)
    else:
        raise CmdException("User abort logging into %r email-server." % prompt)


def send_timestamped_email(msg, sender, recipients, host,
        login_cb=None, ssl=False, **srv_kwds):
    x_recs = '\n'.join('X-Stamper-To: %s' % rec for rec in recipients)
    msg = "\n\n%s\n%s" % (x_recs, msg)

    mail = MIMEText(msg)
    mail['Subject'] = '[CO2MPAS-dice] test'
    mail['From'] = sender
    mail['To'] = ', '.join([_timestamping_address, sender])
    with (smtplib.SMTP_SSL(host, **srv_kwds)
            if ssl else smtplib.SMTP(host, **srv_kwds)) as srv:
        if login_cb:
            login_cmd = lambda user, pswd: srv.login(user, pswd)
            prompt = 'SMTP(%r)' % host
            _log_into_server(login_cb, login_cmd, prompt)

        srv.send_message(mail)
    return mail


_list_response_regex = re.compile(r'\((?P<flags>.*?)\) "(?P<delimiter>.*)" (?P<name>.*)')


def _parse_list_response(line):
    flags, delimiter, mailbox_name = _list_response_regex.match(line).groups()
    mailbox_name = mailbox_name.strip('"')
    return (flags, delimiter, mailbox_name)

# see https://pymotw.com/2/imaplib/ for IMAP example.
def receive_timestamped_email(host, login_cb, ssl=False, **srv_kwds):
    prompt = 'IMAP(%r)' % host

    def login_cmd(user, pswd):
        srv = (imaplib.IMAP4_SSL(host, **srv_kwds)
                if ssl else imaplib.IMAP4(host, **srv_kwds))
        repl = srv.login(user, pswd)
        """GMAIL-2FAuth: imaplib.error: b'[ALERT] Application-specific password required:
        https://support.google.com/accounts/answer/185833 (Failure)'"""
        log.debug("Sent %s-user/pswd, server replied: %s", prompt, repl)
        return srv

    srv = _log_into_server(login_cb, login_cmd, prompt)
    try:
        resp = srv.list()
        print(resp[0])
        return [srv.retr(i+1) for i, msg_id in zip(range(10), resp[1])]
    finally:
        resp = srv.logout()
        if resp:
            log.warning('While closing %s srv responded: %s', prompt, resp)

def git_read_bytes(obj):
    bytes_sink = io.BytesIO()
    obj.stream_data(bytes_sink)
    return bytes_sink.getvalue()


_PGP_SIGNATURE  = b'-----BEGIN PGP SIGNATURE-----'
_PGP_MESSAGE    = b'-----BEGIN PGP MESSAGE-----'

def split_detached_signed(tag: bytes) -> (bytes, bytes):
    """
    Look at GPG signed content (e.g. the message of a signed tag object),
    whose payload is followed by a detached signature on it, and
    split these two; do nothing if there is no signature following.

    :param tag:
            As fetched from ``git cat-file tag v1.2.1``.
    :return:
            A 2-tuple(sig, msg), None if no sig found.
    """
    nl = b'\n'
    lines = tag.split(nl)
    for i, l in enumerate(lines):
        if l.startswith(_PGP_SIGNATURE) or l.startswith(_PGP_MESSAGE):
            return nl.join(lines[i:]), nl.join(lines[:i]) + nl


class DiceGPG(gnupg.GPG):
    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)

    def verify_detached_armor(self, sig: str, data: str):
    #def verify_file(self, file, data_filename=None):
        """Verify `sig` on the `data`."""
        logger = gnupg.logger
        #with tempfile.NamedTemporaryFile(mode='wt+',
        #                encoding='latin-1') as sig_fp:
        #sig_fp.write(sig)
        #sig_fp.flush(); sig_fp.seek(0) ## paranoid seek(), Windows at least)
        #sig_fn = sig_fp.name
        sig_fn = osp.join(tempfile.gettempdir(), 'sig.sig')
        logger.debug('Wrote sig to temp file: %r', sig_fn)

        args = ['--verify', gnupg.no_quote(sig_fn), '-']
        result = self.result_map['verify'](self)
        data_stream = io.BytesIO(data.encode(self.encoding))
        self._handle_io(args, data_stream, result, binary=True)
        return result


    def verify_detached(self, sig: bytes, msg: bytes):
        with tempfile.NamedTemporaryFile('wb+', prefix='co2dice_') as sig_fp:
            with tempfile.NamedTemporaryFile('wb+', prefix='co2dice_') as msg_fp:
                sig_fp.write(sig)
                sig_fp.flush()
                sig_fp.seek(0) ## paranoid seek(), Windows at least)

                msg_fp.write(msg)
                msg_fp.flush();
                msg_fp.seek(0)

                sig_fn = gnupg.no_quote(sig_fp.name)
                msg_fn = gnupg.no_quote(msg_fp.name)
                args = ['--verify', sig_fn, msg_fn]
                result = self.result_map['verify'](self)
                p = self._open_subprocess(args)
                self._collect_output(p, result, stdin=p.stdin)
                return result

def __GPG__init__(self, my_gpg_key):
    gpg_prog = 'gpg2.exe'
    gpg2_path = which(prog)
    self.assertIsNotNone(gpg2_path)
    gpg=gnupg.GPG(r'C:\Program Files (x86)\GNU\GnuPG\gpg2.exe')
    self.my_gpg_key
    self._cfg = read_config('co2mpas')

#####
## TODO: Move to pandalone
_is_dir_regex = re.compile(r'[^/\\][/\\]$')

def normpath(path):
    """Like :func:`osp.normpath()`, but preserving last slash."""
    p = osp.normpath(path)
    if _is_dir_regex.search(path) and p[-1] != os.sep:
        p = p + osp.sep
    return p

def abspath(path):
    """Like :func:`osp.abspath()`, but preserving last slash."""
    p = osp.abspath(path)
    if _is_dir_regex.search(path) and p[-1] != os.sep:
        p = p + osp.sep
    return p

## TODO: Move path-util to pandalone.
def convpath(fpath, abs_path=True, exp_user=True, exp_vars=True):
    """Without any flags, just pass through :func:`osp.normpath`. """
    if exp_user:
        fpath = osp.expanduser(fpath)
    if exp_vars:
        ## Mask UNC '\\server\share$\path` from expansion.
        fpath = fpath.replace('$\\', '_UNC_PATH_HERE_')
        fpath = osp.expandvars(fpath)
        fpath = fpath.replace('_UNC_PATH_HERE_', '$\\')
    fpath = abspath(fpath) if abs_path else normpath(fpath)
    return fpath

def ensure_dir_exists(path, mode=0o755):
    """ensure that a directory exists

    If it doesn't exist, try to create it and protect against a race condition
    if another process is doing the same.

    The default permissions are 755, which differ from os.makedirs default of 777.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif not os.path.isdir(path):
        raise IOError("%r exists but is not a directory" % path)
#####

class Mail(SingletonConfigurable):
    decription = """Generic mail configuration parameters, both for SMTP(sending emails) & IMAP(receiving emails)."""

    host = trt.CUnicode('',
            help="""The SMTP/IMAP server, e.g. 'smtp.gmail.com'."""
            ).tag(config=True)

    port = trt.CInt(587,
            help="""The SMTP/IMAP server's port, usually 587/465 for SSL, 25 otherwise."""
            ).tag(config=True)

    ssl = trt.CBool(True,
            help="""Whether to talk TLS/SSL to the SMTP/IMAP server; configure `port` separately!"""
            ).tag(config=True)

    user = trt.CUnicode(None, allow_none=True,
            help="""The user to authenticate with the SMTP/IMAP server."""
            ).tag(config=True)

    kwds = trt.Dict(
            help="""Any key-value pairs passed to the SMTP/IMAP mail-client libraries."""
            ).tag(config=True)

class SMTP(Mail):
    login = trt.CaselessStrEnum('login simple'.split(), default_value=None, allow_none=True,
            help="""Which SMTP mechanism to use to authenticate: [ login | simple | <None> ]. """
             ).tag(config=True)


class MailApp(Mail):
    description = """FFFF"""
    def start(self):
        print('MAIL')


def default_config_fname():
    """The config-file's basename (no path or extension) to search when not explicetely specified."""
    return '%s_config' % __title__

def default_config_dir():
    """The folder of to user's config-file."""
    return convpath('~/.%s' % __title__)

def default_config_fpath():
    """The full path of to user's config-file, without extension."""
    return osp.join(default_config_dir(), default_config_fname())


base_aliases = {
    'log-level' : 'Application.log_level',
    'config' : 'DiceApp.config_files',
    ## 'generate-config' : 'DiceApp.generate_config', # Too much help.
}

base_flags = {
    'debug': ({'Application' : {'log_level' : logging.DEBUG}},
            "Set log level to logging.DEBUG (maximize logging output)."),
    'gen-config': ({'DiceApp': {'generate_config': True}},
                        """
                        Store config defaults into `{confdir}`.
                        Note: It OVERRIDES any pre-existing configuration file!
                        """.format(confdir=convpath('~/.%s_config.py' % __title__))),
}
class DiceApp(Application):
    name        = __title__
    description = __summary__
    version     = __version__
    #examples = """TODO: Write cmd-line examples."""

    config_files = trt.Unicode(None, allow_none=True,
            help="""
            Absolute/relative path(s) to config files to OVERRIDE default configs.
            Multiple paths are separated by '{pathsep}' in descending order.
            Any extensions are ignored, and '.json' or '.py' are searched (in this order).
            If the path specified resolves to a folder, the filename `{appname}_config.[json | py]` is appended;
            Any command-line values take precendance over the `{confvar}` envvar.
            Use `--generate-config` to produce a skeleton of the config-file.
            """.format(appname=__title__, confvar=CONF_VAR_NAME, pathsep=osp.pathsep)
            ).tag(config=True)

    generate_config = trt.Union((trt.Bool(False), trt.Unicode()),
            help="""
            Store config defaults into path specified or into `{confpath}` if True.
            Note: It OVERWRITES any pre-existing configuration file!
            """.format(confpath=convpath('~/.%s_config.py' % __title__))
            ).tag(config=True) ## aliased

    aliases = base_aliases
    flags = base_flags


    @property
    def user_config_fpaths(self):
        fpaths = []
        config_files = os.environ.get(CONF_VAR_NAME, self.config_files)
        if config_files:
            def _procfpath(p):
                p = convpath(p)
                if osp.isdir(p):
                    p = osp.join(p, default_config_fname())
                else:
                    p = osp.splitext(p)[0]
                return p

            fpaths = config_files.split(osp.pathsep)
            fpaths = [_procfpath(p) for p in fpaths]

        return fpaths

    @property
    def load_config_files(self):
        """Load default user-specified overrides config files.


        Config-files in descending orders:

        - user-overrides:
          - :envvar:`<APPNAME>_CONFIG_FILE`, or if not set,
          - :attr:`config_file`;

        - default config-files:
            - ~/.<appname>/<appname>_config.{json,py} and
            - <this-file's-folder>/<appname>_config.{json,py}.
        """
        paths = [default_config_fpath(), _mydir] # List is in descending priority order.
        self.load_config_file(default_config_fname(), path=paths)

        user_conf_fpaths = self.build_config_fpaths
        for fp in user_conf_fpaths[::-1]:
            cdir, cfname = osp.split(fp)
            self.load_config_file(cfname, path=cdir)

    def write_default_config(self):
        if self.generate_config:
            if isinstance(self.generate_config, bool):
                config_file = '%s.py' % default_config_fpath()
            else:
                config_file = convpath(self.generate_config)
            config_text = self.generate_config_file()

            op = 'Over-writting' if osp.isfile(config_file) else 'Writting'
            log.info('%s config-file %r...', op, config_file)
            ensure_dir_exists(os.path.dirname(config_file), 0o700)
            with io.open(config_file, mode='wt') as fp:
                fp.write(config_text)


    @property
    def _dispatching(self):
        """True if dispatching to another command, or running ourselves."""
        return bool(self.generate_config or self.subapp)

    def __init__(self,
                classes=None,
                #subcommands={'mail': (MailApp, 'Anything')},
                raise_config_file_errors=True,
                **kwds):
        if classes is None:
            classes = [Mail]#, SMTP]
        super().__init__(classes=classes, **kwds)

    @catch_config_error
    def initialize(self, argv=None):
        ## Ensure all singleton-configurables will receive configs.
        #
        for cl in self.classes:
            if type(self) != cl and issubclass(cl, SingletonConfigurable):
                cl.instance(parent=self)
                # Plain configurables must be configured manualy.

        ## Parse cl-args before file-configs
        #  to detect sub-commands and update any :attr:`config_file`,
        #  load file-confis, and re-apply cmd-line configs as overrides.
        #  (trick copied from `jupyter-core`)
        self.parse_command_line(argv)
        if self._dispatching:
            return # Avoid contaminations with user if generating-config.
        cl_config = copy.deepcopy(self.config)
        self.load_config_files()
        self.update_config(cl_config)

    def start(self):
        if self.generate_config:
            self.write_default_config()
            return  #raise NoStart()

        print(self.subapp, self.classes)
        print('AAA', self.extra_args)
        print('CC', self.classes)
        print('C', self.config)
        print(self.document_config_options())

        return super().start()


def main(*argv, **app_init_kwds):
    app_init_kwds['raise_config_file_errors'] = True
    #argv = '--debug --log-level=0 --Mail.port=6 --Mail.user="ggg" abc def'.split()
    #argv = '--Mail.user="ggg" abc def'.split()
    #argv = '--DiceApp.raise_config_file_errors=True'.split()
    argv = '--help-all'.split()
    #argv = '--gen-config'.split()
    DiceApp.launch_instance(argv, kwargs=app_init_kwds)

if __name__ == '__main__':
    init_logging(verbose=False)
    main()