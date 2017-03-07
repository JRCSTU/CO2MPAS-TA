#!/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""
A *traitlets*[#]_ framework for building hierarchical :class:`Cmd` line tools delegating to backend- :class:`Spec`.


## Examples:

To run a base command, use this code::

    cd = MainCmd.make_cmd(argv, **app_init_kwds)  ## `sys.argv` used if `argv` is `None`!
    cmd.start()

To run nested commands and print its output, use :func:`baseapp.chain_cmds()` like that::

    cmd = chain_cmds([MainCmd, Sub1Cmd, Sub2Cmd], argv)  ## `argv` without sub-cmds
    baseapp.consume_cmd(cmd.start())

Of course you can mix'n match.

## Configuration and Initialization guidelines for *Spec* and *Cmd* classes

0. The configuration of :class:`HasTraits` instance gets stored in its ``config`` attribute.
1. A :class:`HasTraits` instance receives its configuration from 3 sources, in this order:

  a. code specifying class-attributes or running on constructors;
  b. configuration files (*json* or ``.py`` files);
  c. command-line arguments.

2. Constructors must allow for properties to be overwritten on construction; any class-defaults
   must function as defaults for any constructor ``**kwds``.

3. Some utility code depends on trait-defaults (i.e. construction of help-messages),
   so for certain properties (e.g. description), it is preferable to set them
   as traits-with-defaults on class-attributes.

4. Listen `Good Bait <https://www.youtube.com/watch?v=CE4bl5rk5OQ>`_ after 1:43.

.. [#] http://traitlets.readthedocs.io/
"""
from collections import OrderedDict
import contextlib
import copy
import io
import logging
import os
from typing import Sequence, Text, Any, Tuple, List  # @UnusedImport

from boltons.setutils import IndexedSet as iset
from toolz import dicttoolz as dtz, itertoolz as itz

import itertools as itt
import os.path as osp
import pandalone.utils as pndlu
import traitlets as trt
import traitlets.config as trtc

from . import CmdException
from .. import (__version__, __updated__, __uri__, __copyright__, __license__,  # @UnusedImport
                utils)
from ..__main__ import init_logging

################################################
## INFO: Modify the following variables on a different application.
APPNAME = 'co2dice'  # TODO: Cannot use baseapp with different app-names.
CONFIG_VAR_NAME = '%s_CONFIG_PATH' % APPNAME.upper()
PERSIST_VAR_NAME = '%s_PERSIST_PATH' % APPNAME.upper()

try:
    _mydir = osp.dirname(__file__)
except:
    _mydir = '.'

## Dice better have ordered reports
#  so put it here to be invoked only once.
utils.setup_yaml_ordered()


def default_config_fname():
    """The config-file's basename (no path or extension) to search when not explicitly specified."""
    return '%s_config' % APPNAME


def default_config_dir():
    """The folder of to user's config-file."""
    return pndlu.convpath('~/.%s' % APPNAME)


def default_config_fpaths():
    """The full path of to user's config-file, without extension."""
    return [osp.join(default_config_dir(), default_config_fname()),
            osp.join(pndlu.convpath(_mydir), default_config_fname())]


def default_persist_fpath():
    """The full path of to user's persistent config-file, without extension."""
    return osp.join(default_config_dir(), '%s_persist' % APPNAME)
#
################################################


## TODO: Delete FuzzyEnum if ipython/traitlets#371 merged
class FuzzyEnum(trt.Enum):
    """An case-ignoring enum matching choices by unique prefixes/substrings."""

    case_sensitive = False
    #: If True, choices match anywhere in the string, otherwise match prefixes.
    substring_matching = False

    def __init__(self, values, default_value=trt.Undefined,
                 case_sensitive=False, substring_matching=False, **kwargs):
        self.case_sensitive = case_sensitive
        self.substring_matching = substring_matching
        values = [trt.cast_unicode_py2(value) for value in values]
        super().__init__(values, default_value=default_value, **kwargs)

    def validate(self, obj, value):
        if isinstance(value, str):
            value = trt.cast_unicode_py2(value)
        if not isinstance(value, trt.six.string_types):
            self.error(obj, value)

        conv_func = (lambda c: c) if self.case_sensitive else str.lower
        substring_matching = self.substring_matching
        match_func = ((lambda v, c: v in c)
                      if substring_matching
                      else (lambda v, c: c.startswith(v)))
        value = conv_func(value)
        choices = self.values
        matches = [match_func(value, conv_func(c)) for c in choices]
        if sum(matches) == 1:
            for v, m in zip(choices, matches):
                if m:
                    return v

        self.error(obj, value)


class PeristentMixin:
    """
    A *cmd* and *spec* mixin to support storing of *persistent* traits into external file.

    *Persistent traits (ptrais)* are those tagged with `config` + `persist` boolean metadata.

    This is the lifecycle of *persistent* traits (*ptraits*):

    1. On app-init, invoke :meth:`load_pconfig()` to read persist-params from disk
        and populate the global :attr:`_pconfig` (and attr:`_pconfig_orig`).

    2. Merge pconfig with any regular config-values (preferably overriding them).

    3. On each *cmd* or *spec* construction:
       B. Ensure :meth:`trtc.Configurable.update_config()` is invoked and then
       C. invoke :meth:`observe_ptraits()` to mirror any ptrait-changes
          on global :attr:`_pconfig`.

    4. On observed changes, indeed the global :attr:`_pconfig` gets updated.

    5. On app-exit, remember to invoke :meth:`store_pconfig()`
       to store any changes on disk.

    """

    #: The "global" CLASS-property that dynamically receives changes from
    #: *cmd* and *spec* traits with *peristent* traits.
    #: NOTE: not to be modified by mixed-in classes.
    _pconfig = trtc.Config()

    #: A "global" CLASS-property holding persistent-trait values as loaded,
    #: to be checked for inequality and decide if changes need storing.
    _pconfig_orig = trtc.Config()

    @classmethod
    def load_pconfig(cls, fpath: Text):
        """
        Load persistent config into global :attr:`_pconfig` & :attr:`_pconfig_orig`.

        :param cls:
            This mixin where updated ptrait-values are to be stored *globally*.
        :param fpath:
            abs/relative file-path, with/without ``'.json'`` extension
        :return:
            A tuple ``(fpath, Config())`` with the read persistent config parameters.
            Config might be `None` if file not found.
        :raise:
            Any exception while reading json and converting it
            into :class:`trtc.Config`, unless file does not exist.

        .. Note::
            It does not apply the loaded configs - you have to
            invoke :meth:`trtc.Configurable.update_config(cls._pconfig)`.

            Currently both methods invoked by the final *Cmd* on :meth:`Cmd.initialize()`.
            You have to invoke them for "stand-alone" *Specs* or pass
            an already merged :class:`trtc.Config` instance.
        """
        import json

        cfg = None
        fpath = pndlu.ensure_file_ext(fpath, '.json')
        try:
            with io.open(fpath, 'rt', encoding='utf-8') as finp:
                cfg = json.load(finp)
        except FileNotFoundError:
            pass
        else:
            cfg = trtc.Config(cfg)

            cls._pconfig = cfg
            cls._pconfig_orig = copy.deepcopy(cfg)

        return fpath, cfg

    @classmethod
    def store_pconfig(cls, fpath: Text):
        """
        Stores ptrait-values from the global :attr:`_pconfig`into `fpath` as JSON.

        :param cls:
            This mixin where updated ptrait-values are to be stored *globally*.
        """
        cfg = cls._pconfig
        if cfg and cfg != cls._pconfig_orig:
            import json

            fpath = pndlu.ensure_file_ext(fpath, '.json')
            with io.open(fpath, 'wt', encoding='utf-8') as fout:
                json.dump(cfg, fout, indent=2)

    def _ptrait_observed(self, change):
        """The observe-handler for *persistent* traits."""
        cls = type(self)
        cls_name = cls.__name__
        name = change['name']
        value = change['new']

        cls._pconfig[cls_name][name] = value

    def check_unconfig_ptraits(self):
        for name, tr in self.traits(persist=True).items():
            if tr.metadata.get('config') is not True:
                raise trt.TraitError("Persistent trait %r not tagged as 'config'!" % name)

    def observe_ptraits(self: trt.HasTraits):
        """
        Establishes observers for all *persistent* traits to update :attr:`persistent_config` class property.

        Invoke this after regular config-values have been installed.
        """
        self.check_unconfig_ptraits()
        ptraits = self.trait_names(config=True, persist=True)
        if ptraits:
            self.observe(self._ptrait_observed, ptraits)


class HasCiphersMixin:
    """Mixin for :class:`trtc.Configurable` that may have :class:`crypto.Cipher` traits"""

    def decipher(self, cipher_trait: Text):
        """
        Decrypts a cipher trait of some instance.

        :param obj:
            The instance holding the trait-values.
        :return:
            The unencrypted object, or None if trait-value was None.

        .. Tip::
            Invoke it on the class, not on the trait: ``ObjClass.ctrait.decrypt(obj)``.
        """
        from . import crypto

        assert isinstance(cipher_trait, str), "%s is not a trait-name!" % cipher_trait

        value = getattr(self, cipher_trait, None)
        if value is not None:
            cls_name = type(self).__name__
            pswdid = '%s.%s' % (cls_name, cipher_trait)
            if not crypto.is_pgp_encrypted(value):
                self.log.warning("Found non-encrypted param %r!", pswdid)
            else:
                vault = crypto.get_vault(self.config)
                vault.log.debug("Decrypting cipher-trait(%r)...", pswdid)
                value = vault.decryptobj(pswdid, value)

        return value


class TolerableSingletonMixin:
    """Like :class:`trtc.SingletonConfigurable` but with unrestricted instances by hierarchy. """
    @classmethod
    def instance(cls, *args, **kwargs):
        if cls._instance is None or type(cls._instance) != cls:
            cls._instance = cls(*args, **kwargs)

        return cls._instance


###################
##     Specs     ##
###################

class Spec(trtc.LoggingConfigurable, PeristentMixin, HasCiphersMixin):
    """Common properties for all configurables."""
    ## See module documentation for developer's guidelines.

    @trt.default('log')
    def _log(self):
        return logging.getLogger(type(self).__name__)

    # The log level for the application
    log_level = trt.Enum((0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'),
                         default_value=logging.WARN,
                         help="Set the log level by value or name.").tag(config=True)

    @trt.observe('log_level')
    def _change_level_on_my_logger(self, change):
        new = change['new']
        if isinstance(new, str):
            new = getattr(logging, new)
            self.log_level = new
        self.log.setLevel(new)

    verbose = trt.Union(
        (trt.Integer(0), trt.Bool(False)),
        ## INFO: Add verbose flag explanations here.
        help="""
        Make various sub-commands increase their verbosity (not to be confused with --debug):
        Can be a boolean or 0, 1(==True), 2, ....

        SubCommands using this flag:
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        `project list/status`
            List project with the "long" format, include infos about the repo (when 2).
        `project init/open/append/tstamp`
            Print committed-msg instead of try/false/proj-name.
        `config show`
            Print parameters for all intermediate classes.
          """).tag(config=True)

    ## TODO: Retrofitt to force-flags (with code for each specific permission).
    force = trt.Bool(
        False,
        ## INFO: Add force flag explanations here.
        help="""
        Force various sub-commands to perform their duties without complaints.

        SubCommands using this flag:
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        `project backup`
            Whether to overwrite existing archives or to create intermediate folders.
        `project ...`
            Proceed project's lifecycle-stages even though conditions are not fulfilled.
        `config write
            Overwrite config-file, even if it already exists.
        `tstamp send`
            Send content to timestamp even if its signature fails to verify.
        """).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observe_ptraits()


###################
##    Commands   ##
###################


def cmd_class_short_help(app_class):
    desc = app_class.description
    return pndlu.first_line(isinstance(desc, str) and desc or app_class.__doc__)


def class2cmd_name(cls):
    name = cls.__name__
    if name.lower().endswith('cmd') and len(name) > 3:
        name = name[:-3]
    return pndlu.camel_to_cmd_name(name)


def build_sub_cmds(*subapp_classes):
    """Builds an ordered-dictionary of ``cmd-name --> (cmd-class, help-msg)``. """

    return OrderedDict((class2cmd_name(sa), (sa, cmd_class_short_help(sa)))
                       for sa in subapp_classes)


class CfgFilesRegistry(contextlib.ContextDecorator):
    """
    Locate and account config-files (``.json|.py``).

    - Collects a Locate and (``.json|.py``) files present in the `path_list`, or
    - Invoke this for every "manually" visited config-file, successful or not.
    """

    #: A list of 2-tuples ``(folder, fname(s))`` with loaded config-files
    #: in descending order (1st overrides later).
    visited_files = []

    def __enter__(self):
        self.visited_files = []
        return self

    def __exit__(self, *exc):
        self.visited_files = self._consolidate(self.visited_files)
        return False

    @staticmethod
    def _consolidate(visited_files):
        """
        Reverse and remove multiple, empty records.

        Example::

            >>> _consolidate([
            ... ('a/b/', None),
            ... ('a/b/', 'F1'),
            ... ('a/b/', 'F2'),
            ... ('a/b/', None),
            ... ('c/c/', None),
            ... ('c/c/', None),
            ... ('d/',   'F1'),
            ... ('d/',   None),
            ... ('c/c/', 'FF')])
            [('a/b/',   ['F1', 'F2']),
             ('c/c/',   []),
             ('d/',     ['F1']),
             ('c/c/',   ['FF'])]
        """
        consolidated = []
        prev = None
        for b, f in visited_files:
            if not prev:            # loop start
                prev = (b, [])
            elif prev[0] != b:      # new dir
                consolidated.append(prev)
                prev = (b, [])
            if f:
                prev[1].append(f)
        if prev:
            consolidated.append(prev)

        return consolidated

    def file_visited(self, fpath, miss=False):
        """Invoke this for every visited config-file, successful or not."""
        base, fname = osp.split(fpath)
        self.visited_files.append((base, None if miss else fname))

    def collect_fpaths(self, path_list: List[Text]):
        """
        Collects all (``.json|.py``) files present in the `path_list`, (descending order).

        :param path_list:
            A list of paths (absolute, relative, dir or folders)
            each one possibly separated by `osp.pathsep`.
        :return:
            fully-normalized paths, with ext
        """
        new_paths = iset()

        def try_json_and_py(basepath):
            found_any = False
            for ext in ('.py', '.json'):
                f = pndlu.ensure_file_ext(basepath, ext)
                if f in new_paths:
                    continue

                if osp.isfile(f):
                    new_paths.add(f)
                    self.file_visited(f)
                    found_any = True
                else:
                    self.file_visited(f, True)

            return found_any

        def _derive_config_fpaths(path: Text) -> List[Text]:
            """Return multiple *existent* fpaths for each config-file path (folder/file)."""

            p = pndlu.convpath(path)
            if osp.isdir(p):
                try_json_and_py(osp.join(p, default_config_fname()))
            else:
                found = try_json_and_py(p)
                ## Do not strip ext if has matched WITH ext.
                if not found:
                    try_json_and_py(osp.splitext(p)[0])

        for cf1 in path_list:
            for cf2 in cf1.split(os.pathsep):
                _derive_config_fpaths(cf2)

        return list(new_paths)


class Cmd(TolerableSingletonMixin, trtc.Application, Spec):
    """Common machinery for all (sub-)commands. """
    ## INFO: Do not use it directly; inherit it.
    # See module documentation for developer's guidelines.

    @trt.default('name')
    def _name(self):
        name = class2cmd_name(type(self))
        return name

    config_paths = trt.List(
        trt.Unicode(),
        None, allow_none=True,
        help="""
        Absolute/relative path(s) to read "static" configurable parameters from.

        If false, and no `{confvar}` envvar is defined, defaults to:
            {default}
        Multiple values may be given, and each value may be a single or multiple paths
        separated by '{pathsep}'.  All paths collected are considered in descending order
        (1st one overrides the rest).
        For paths resolving to folders, the filename `{appname}_config.[json | py]` is appended;
        otherwise, any file-extensions are ignored, and '.py' and/or '.json' are loaded (in this order).


        Tip:
            Use `config write` sub-command to produce a skeleton of the config-file.

        Note:
            A value in configuration files are ignored!  Set this from command-line
            (or in code, before invoking :meth:`Cmd.initialize()`).
            Any command-line values take precedence over the `{confvar}` envvar.

        Examples:
            To read and apply, in descending order `~/my_conf`, `/tmp/conf.py`  `~/.co2dice.json`
            issue:
                <cmd> --config-paths=~/my_conf:/tmp/conf.py  --config-paths=~/.co2dice.json  ...



        """.format(appname=APPNAME, confvar=CONFIG_VAR_NAME,
                   default=default_config_fpaths(), pathsep=osp.pathsep)
    ).tag(config=True)

    persist_path = trt.Unicode(
        os.environ.get(PERSIST_VAR_NAME), allow_none=True,
        help="""
        Absolute/relative path to read/write persistent parameters on runtime, if `{confvar}` envvar is not defined.

        If false, and no `{confvar}` envvar is defined, defaults to:
            {default}
        If path resolves to a folder, the filename `{appname}_persist.json` is appended;
        otherwise, the file-extensions is assumed to be `.json`.
        Persistent-parameters override "static" ones.

        Tip:
            Use `{appname} config write` sub-command to produce a skeleton of the config-file.

        Note:
            A value in configuration files are ignored!  Set this from command-line
            (or in code, before invoking :meth:`Cmd.initialize()`).
            Any command-line values take precedence over the `{confvar}` envvar.
        """.format(appname=APPNAME, confvar=PERSIST_VAR_NAME,
                   default=default_persist_fpath())
    ).tag(config=True)

    encrypt = trt.Bool(
        False,
        help="""
        Encrypt ciphered-configs on app startup: True: all classes, False: current-cmd's only

        Sample cmd to encrypt any freshly edited persistent-configs:
               %s --encrypt
        """ % APPNAME
    ).tag(config=True)

    @trt.default('log')
    def _log(self):
        ## Use a regular logger.
        return logging.getLogger(type(self).__name__)

    _cfgfiles_registry = CfgFilesRegistry()

    @property
    def loaded_config_files(self):
        return self._cfgfiles_registry.visited_files

    def _collect_static_fpaths(self):
        """Return fully-normalized paths, with ext."""
        env_paths = os.environ.get(CONFIG_VAR_NAME)
        env_paths = env_paths and [env_paths]
        config_paths = (self.config_paths or
                        env_paths or
                        default_config_fpaths())

        return self._cfgfiles_registry.collect_fpaths(config_paths)

    def _read_config_from_json_or_py(self, cfpath: Text):
        """
        :param cfpath:
            The absolute config-file path with either ``.py`` or ``.json`` ext.
        """
        log = self.log
        loaders = {
            '.py': trtc.PyFileConfigLoader,
            '.json': trtc.JSONFileConfigLoader,
        }
        ext = osp.splitext(cfpath)[1]
        loader = loaders.get(str.lower(ext))
        assert loader, cfpath  # Must exist.

        config = None
        try:
            config = loader(cfpath, path=None, log=log).load_config()
        except trtc.ConfigFileNotFound:
            ## Config-file deleted between collecting its name and reading it.
            pass
        except Exception as ex:
            if self.raise_config_file_errors:
                raise
            log.error("Failed loading config-file '%s' due to: %s",
                      cfpath, ex, exc_info=True)
        else:
            log.debug("Loaded config-file: %s", cfpath)

        return config

    def _read_config_from_static_files(self, config_paths):
        """
        :param config_paths:
            full normalized paths (descending order, 1st overrides the rest)
        """
        new_config = trtc.Config()
        ## Registry to detect collisions.
        loaded = {}  # type: Dict[Text, Config]

        for cfpath in config_paths[::-1]:
            config = self._read_config_from_json_or_py(cfpath)
            if config:
                for filename, earlier_config in loaded.items():
                    collisions = earlier_config.collisions(config)
                    if collisions:
                        import json
                        self.log.warning(
                            "Collisions detected in %s and %s config files."
                            " %s has higher priority: %s",
                            filename, cfpath, cfpath,
                            json.dumps(collisions, indent=2)
                        )
                loaded[cfpath] = config

                new_config.merge(config)

        return new_config

    @property
    def _resolved_persist_file(self):
        """Returns the 1st value in config, env, or default-value (might contain ``.json``)."""
        persist_path = (self.persist_path or
                        default_persist_fpath())
        persist_path = pndlu.convpath(persist_path)
        return persist_path

    def _read_config_from_persist_file(self):
        persist_path = self._resolved_persist_file
        try:
            persist_path, config = self.load_pconfig(persist_path)
        except Exception as ex:
            self._cfgfiles_registry.file_visited(persist_path)
            if self.raise_config_file_errors:
                raise
            self.log.error("Failed loading persist-file '%s' due to: %s",
                           persist_path, ex, exc_info=True)
        else:
            self._cfgfiles_registry.file_visited(persist_path, miss=not bool(config))
            self.log.debug("%s persist-file: %s",
                           'Loaded' if config else 'Missing',
                           persist_path)
            return config

    def load_configurables_from_files(self) -> Tuple[trtc.Config, trtc.Config]:
        """
        Load :attr:`config_paths`, :attr:`persist_path` and maintain :attr:`config_registry`.

        :return:
            A 2 tuple ``(static_config, persist_config)``, where the 2nd might be `None`.

        Configuration files are read and merged in descending orders:

        1. Persistent parameters from ``.json`` files, either in:
           - :envvar:`<APPNAME>_PERSIST_PATH`, or if not set,
           - :attr:`persist_path` (see its default-value);

        2. Static parameters from ``.json`` and/or ``.py`` files, either in:
           - :envvar:`<APPNAME>_CONFIG_PATHS`, or if not set,
           - :attr:`config_paths` (see its default-value).

        Code adapted from :meth:`load_config_file` & :meth:`Application._load_config_files`.
        """
        with self._cfgfiles_registry:
            persist_config = self._read_config_from_persist_file()
            static_paths = self._collect_static_fpaths()
            static_config = self._read_config_from_static_files(static_paths)

        return static_config, persist_config

    def write_default_config(self, config_file=None, force=False):
        if not config_file:
            config_file = default_config_fpaths()[0]
        else:
            config_file = pndlu.convpath(config_file)
            if osp.isdir(config_file):
                config_file = osp.join(config_file, default_config_fname())
        config_file = pndlu.ensure_file_ext(config_file, '.py')

        is_overwrite = osp.isfile(config_file)
        if is_overwrite:
            if not force:
                raise CmdException("Config-file '%s' already exists!"
                                   "\n  Specify `--force` to overwrite." % config_file)
            else:
                import shutil
                from datetime import datetime

                now = datetime.now().strftime('%Y%m%d-%H%M%S%Z')
                backup_name = '%s-%s.py' % (osp.splitext(config_file)[0], now)
                shutil.move(config_file, backup_name)

                op_msg = ", old file renamed --> '%s'" % backup_name
        else:
            op_msg = ""

        self.log.info("Writting config-file '%s'%s...", config_file, op_msg)
        pndlu.ensure_dir_exists(os.path.dirname(config_file), 0o700)
        config_text = self.generate_config_file()
        with io.open(config_file, mode='wt') as fp:
            fp.write(config_text)

    def print_subcommands(self):
        from ipython_genutils.text import indent, wrap_paragraphs, dedent

        """Print the subcommand part of the help."""
        ## Overridden, to print "default" sub-cmd.
        if not self.subcommands:
            return

        lines = ["Subcommands"]
        lines.append('=' * len(lines[0]))
        for p in wrap_paragraphs(self.subcommand_description.format(
                app=self.name)):
            lines.append(p)
        lines.append('')
        for subc, (cls, hlp) in self.subcommands.items():
            lines.append(subc)

            if hlp:
                lines.append(indent(dedent(hlp.strip())))
        lines.append('')
        print(os.linesep.join(lines))

    def _my_text_interpolations(self):
        cmd_chain = reversed(self.my_cmd_chain())
        return {'app_cmd': APPNAME,
                'cmd_chain': ' '.join(c.name for c in cmd_chain)}

    def print_description(self):
        ## Overridden for interpolating app-name.
        txt = self.description or self.__doc__
        txt %= self._my_text_interpolations()
        for p in trtc.wrap_paragraphs(txt):
            print(p)
            print()

    def print_examples(self):
        ## Overridden for interpolating app-name.
        txt = self.examples
        if txt:
            txt = txt.strip() % self._my_text_interpolations()
            print("Examples")
            print("--------")
            print()
            print(trtc.indent(trtc.dedent(txt)))
            print()

    ## Needed because some sub-cmd name clash and
    #  *argparse* screams about conflicts.
    #  See https://github.com/ipython/traitlets/pull/360
    def _create_loader(self, argv, aliases, flags, classes):
        return trtc.KVArgParseConfigLoader(
            argv, aliases, flags, classes=classes,
            log=self.log, conflict_handler='resolve')

    conf_classes = trt.List(
        trt.Type(trtc.Configurable), default_value=[],
        help="""
        Any *configurables* found in this prop up the cmd-chain are merged,
        along with any subcommands, into :attr:`classes`.
        """)

    config_show = trt.Bool(
        False,
        help="""Enable it to print the configurations before launching any command."""
    ).tag(config=True)

    cmd_aliases = trt.Dict(
        {},
        help="Any *flags* found in this prop up the cmd-chain are merged into :attr:`aliases`. """)

    cmd_flags = trt.Dict(
        {},
        help="Any *flags* found in this prop up the cmd-chain are merged into :attr:`flags`. """)

    def my_cmd_chain(self):
        """Return the chain of cmd-classes starting from my self or subapp."""
        cmd_chain = []
        pcl = self.subapp if self.subapp else self
        while pcl:
            cmd_chain.append(pcl)
            pcl = pcl.parent

        return cmd_chain

    @trt.observe('parent', 'conf_classes', 'cmd_aliases', 'cmd_flags', 'subapp')
    def _inherit_parent_cmd(self, change):
        """ Inherit config-related stuff from up the cmd-chain. """
        if self.parent:
            ## Collect parents, ordered like that:
            #    subapp, self, parent1, ...
            #
            cmd_chain = self.my_cmd_chain()

            ## Collect separately and merge  SPECs separately,
            #  to prepend them before SPECs at the end.
            #
            conf_classes = list(itz.concat(cmd.conf_classes for cmd in cmd_chain))

            ## Merge aliases/flags reversed.
            #
            cmd_aliases = dtz.merge(cmd.cmd_aliases for cmd in cmd_chain[::-1])
            cmd_flags = dtz.merge(cmd.cmd_flags for cmd in cmd_chain[::-1])
        else:
            ## We are root.

            cmd_chain = [self]
            conf_classes = list(self.conf_classes)
            cmd_aliases = self.cmd_aliases
            cmd_flags = self.cmd_flags

        cmd_classes = [type(cmd) for cmd in cmd_chain]
        self.classes = list(iset(cmd_classes + conf_classes))
        self.aliases.update(cmd_aliases)
        self.flags.update(cmd_flags)

    @trt.observe('log_level')
    def _init_logging(self, change):
        log_level = change['new']
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level)

        init_logging(level=log_level)

    def __init__(self, **kwds):
        from . import crypto

        cls = type(self)
        dkwds = {
            ## Traits defaults are always applied...??
            #
            'name': class2cmd_name(cls),

            ## Set some nice defaults for root-CMDs.
            #
            'cmd_aliases': {
                'config-paths': (
                    'Cmd.config_paths',
                    pndlu.first_line(Cmd.config_paths.help)),
                'persist-path': (
                    'Cmd.persist_path',
                    pndlu.first_line(Cmd.persist_path.help)),
                'vlevel': (
                    'Spec.verbose',
                    pndlu.first_line(Spec.verbose.help)),
            },
            'conf_classes': [crypto.VaultSpec],
            'cmd_flags': {
                ('d', 'debug'): (
                    {
                        'Application': {'log_level': 0},
                        'Spec': {'log_level': 0},
                        'Cmd': {
                            'raise_config_file_errors': True,
                            'config_show': True,
                        },
                    },
                    """
                    Log more (POSSIBLY PASSWORDS!) infos & fail early.

                    Not to be confused with `--verbose`.
                    """
                ),
                ('v', 'verbose'): (
                    {
                        'Spec': {'verbose': True},
                    },
                    pndlu.first_line(Spec.verbose.help)
                ),
                ('f', 'force'): (
                    {
                        'Spec': {'force': True},
                    },
                    pndlu.first_line(Spec.force.help)
                ),
                'config-show': (
                    {
                        'Cmd': {'config_show': True},
                    },
                    pndlu.first_line(Cmd.config_show.help),
                ),
                'encrypt': (
                    {
                        'Cmd': {'encrypt': True},
                    },
                    pndlu.first_line(Cmd.encrypt.help)
                )
            },
        }
        if cls.__doc__ and not isinstance(cls.description, str):
            dkwds['description'] = cls.__doc__
        dkwds.update(kwds)
        super().__init__(**dkwds)

    def all_app_configurables(self):
        """
        Return any configurable-class, to validate/report them on app startup.

        :return:
            an ordered-set of all app configurables (*cmds* + *specs*)

        It has to be a Cmd-method so that it can include this cmd's classes,
        even if the cmd has been created temporarilly (e.g. for some TC).
        """
        ## INFO: Circular-dep instead of abstract-method, so need not override in TCs.
        from . import dice
        return iset(itt.chain(dice.all_app_configurables(), self.classes))

    def _make_vault_from_configs(self, static_config, persist_config):
        from . import crypto

        vault_config = copy.deepcopy(static_config)
        if persist_config:
            vault_config.merge(persist_config)
        vault_config.merge(self.cli_config)
        vault = crypto.get_vault(vault_config)

        return vault

    def _validate_cipher_traits_against_config_files(self, static_config, persist_config):
        """
        Check plaintext :class:`crypto.Cipher` config-values and encrypt them if *persistent*, scream if *static*.

        TODO: UNTESTABLE :-( to speed-up app start-app, run in 2 "passes" (the 2nd pass is optional):

        - Pass-1 checks configs only for current Cmd's :attr:`classes`, and
          if any iregularities are detected (un-encrypted persistent or static ciphers),
          then laucnh ...
        - Pass-2 to search :meth:`all_app_configurables()`.
        """
        from . import crypto

        class NextPass(Exception):  # used to break the loop.
            pass

        ## Input records: since `persist_config` might be None,
        #  conditional create them
        #
        configs = list(
            c for c in zip(
                (static_config, persist_config),
                (False, True),
                ('static', 'persist'))
            if c[0]
        )
        vault = None  # lazily created
        ## Outputs
        #
        ntraits_encrypted = 0       # Counts encrypt-operations of *persist* traits.
        static_screams = iset()     # Collect non-encrypted *static* traits.

        def scan_config(config_classes, know_all_classes):
            """:return: true meaning full-scan is needed."""
            nonlocal vault, ntraits_encrypted

            rerun = False
            for config, encrypt_plain, config_source in configs:
                for clsname, traits in config.items():
                    cls = config_classes.get(clsname)
                    if not cls:
                        if know_all_classes:  # Only scream when full check.
                            self.log.warning(
                                "Unknown class `%s` in *%s* file-configs while encrypting values.",
                                clsname, config_source)
                        continue

                    for tname, tvalue in traits.items():
                        ctraits = cls.class_traits(config=True)
                        ctrait = ctraits.get(tname)
                        if not isinstance(ctrait, crypto.Cipher):
                            continue

                        ## Scream on static, encrypt on persistent.
                        #
                        if crypto.is_pgp_encrypted(tvalue):
                            continue

                        ## Irregularities have been found!
                        rerun = True

                        key = '%s.%s' % (clsname, tname)
                        if encrypt_plain:
                            self.log.info("Auto-encrypting cipher-trait(%r)...", key)
                            if not vault:
                                vault = self._make_vault_from_configs(static_config, persist_config)

                            config[clsname][tname] = vault.encryptobj(key, tvalue)
                            ntraits_encrypted += 1
                        else:
                            static_screams.add(key)

            return rerun

        try:
            # If --encrypt, go directly to pass-2.
            passes = (True, ) if self.encrypt else (False, True)

            ## Loop for the 2-passes trick.
            #
            for full_check in passes:
                scan_classes = (self.all_app_configurables()
                                if full_check
                                else self.classes)
                config_classes = {c.__name__: c
                                  for c in
                                  self._classes_with_config_traits(scan_classes)}
                rerun = scan_config(config_classes, full_check)
                if not rerun:
                    break
        finally:
            ## Ensure any encrypted traits are saved.
            #
            persist_path = self._resolved_persist_file
            if ntraits_encrypted:
                self.log.info("Updating persistent config %r with %d auto-encrypted values...",
                              persist_path, ntraits_encrypted)
                self.store_pconfig(persist_path)

        if static_screams:
            msg = "Found %d non-encrypted params in static-configs: %s" % (
                len(static_screams), static_screams)
            if self.raise_config_file_errors:
                raise trt.TraitError(msg)
            else:
                self.log.error(msg)

    def _is_dispatching(self):
        """True if dispatching to another command."""
        return isinstance(self.subapp, trtc.Application)  # subapp == trait | subcmd | None

    @trtc.catch_config_error
    def initialize(self, argv=None):
        """
        Invoked after __init__() by `make_cmd()` to apply configs and build subapps.

        :param argv:
            If undefined, they are replaced with ``sys.argv[1:]``!

        It parses cl-args before file-configs, to detect sub-commands
        and update any :attr:`config_paths`, then it reads all file-configs, and
        then re-apply cmd-line configs as overrides (trick copied from `jupyter-core`).

        It also validates config-values for :class:`crypto.Cipher` traits.
        """
        self.parse_command_line(argv)
        if self._is_dispatching():
            ## Only the final child reads file-configs.
            #  Also avoid contaminations with user if generating-config.
            return

        static_config, persist_config = self.load_configurables_from_files()
        self._validate_cipher_traits_against_config_files(static_config, persist_config)

        if persist_config:
            static_config.merge(persist_config)
        static_config.merge(self.cli_config)

        self.update_config(static_config)
        self.observe_ptraits()

    def start(self):
        """Dispatches into sub-cmds (if any), and then delegates to :meth:`run().

        If overriden, better invoke :func:`super()`, but even better
        to override :meth:``run()`.
        """
        if self.config_show:
            from pprint import pformat
            self.log.info('Running cmd %r with config: \n  %s',
                          self.name, pformat(self.config))

        if self.subapp is None:
            return self.run(*self.extra_args)

        return self.subapp.start()

    def run(self, *args):
        """Leaf sub-commands must inherit this instead of :meth:`start()` without invoking :func:`super()`.

        :param args:
            Invoked by :meth:`start()` with :attr:`extra_args`.

        By default, screams about using sub-cmds, or about doing nothing!
        """
        import ipython_genutils.text as tw

        if self.subcommands:
            cmd_line = ' '.join(cl.name
                                for cl in reversed(self.my_cmd_chain()))
            examples = str.strip(tw.dedent(self.examples.strip()))
            msg = tw.dedent(
                """Specify one of the sub-commands:
                    %s
                or type:
                    %s --help
                """) % (', '.join(self.subcommands.keys()), cmd_line)
            if examples:
                msg = "%s\nExamples\n--------\n%s\n" % (msg, examples)
                msg = msg.strip() % self._my_text_interpolations()
            raise CmdException(msg)
        assert False, "Override run() method in cmd subclasses."

    @classmethod
    def make_cmd(cls, argv=None, **kwargs):
        """
        Instanciate, initialize and return application.

        :param argv:
            Like :meth:`initialize()`, if undefined, replaced with ``sys.argv[1:]``.

        - Tip: Apply :func:`consume_cmd()` on return values to process generators of :meth:`run()`.
        - This functions is the 1st half of :meth:`launch_instance()` which
          invokes and discards :meth:`start()` results.
        """
        ## Overriden just to return `start()`.
        cmd = cls.instance(**kwargs)
        cmd.initialize(argv)

        return cmd


## Disable logging-format configs, because their observer
#    works on on loger's handlers, which might be null.
Cmd.log_format.tag(config=False)
Cmd.log_datefmt.tag(config=False)

## So that dynamic-default rules apply.
#
Cmd.description.default_value = None
Cmd.name.default_value = None

## Expose `raise_config_file_errors` instead of relying only on
#  :envvar:`TRAITLETS_APPLICATION_RAISE_CONFIG_FILE_ERROR`.
trtc.Application.raise_config_file_errors.tag(config=True)
Cmd.raise_config_file_errors.help = 'Whether failing to load config files should prevent startup.'


def chain_cmds(app_classes: Sequence[type(trtc.Application)],
               argv: Sequence[Text]=None,
               **root_kwds):
    """
    Instantiate(optionally) and run a list of ``[cmd, subcmd, ...]``, linking each one as child of its predecessor.

    :param app_classes:
        A list of cmd-classes: ``[root, sub1, sub2, app]``
        Note: you have to "know" the correct nesting-order of the commands ;-)
    :param argv:
        cmdline args passed to the root (1st) cmd only.
        Make sure they do not contain any sub-cmds.
        Like :meth:`initialize()`, if undefined, replaced with ``sys.argv[1:]``.
    :return:
        The root(1st) cmd to invoke :meth:`Aplication.start()`
        and possibly apply the :func:`consume_cmd()` on its results.

    - Normally `argv` contain any sub-commands, and it is enough to invoke
      ``initialize(argv)`` on the root cmd.  This function shortcuts
      arg-parsing for subcmds with explict cmd-chaining in code.
    - This functions is the 1st half of :meth:`Cmd.launch_instance()`.
    """
    if not app_classes:
        raise ValueError("No cmds to chained passed in!")

    app_classes = list(app_classes)
    root = app = None
    for app_cl in app_classes:
        if not isinstance(app_cl, type(trtc.Application)):
                    raise ValueError("Expected an Application-class instance, got %r!" % app_cl)
        if not root:
            ## The 1st cmd is always orphan, and gets returned.
            root = app = app_cl(**root_kwds)
        else:
            app.subapp = app = app_cl(parent=app)
        app.initialize(argv)

    app_classes[0]._instance = app

    return root


def consume_cmd(result):
    """
    Print a (possibly lazy) results to `stdout`.

    :param cmd:
        Whatever is retuened by a :meth:`Cmd.start()`/`Cmd.run()` methods.
    :return:
        Does not return anything!  Just prints stuff.

    - Remember to have logging setup properly before invoking this.
    - This the 2nd half of the replacement for :meth:`Application.launch_instance()`.
    """
    import types

    if result is not None:
        if isinstance(result, types.GeneratorType):
            for i in result:
                print(i)
        elif isinstance(result, (tuple, list)):
            print(os.linesep.join(result))
        else:
            print(result)
