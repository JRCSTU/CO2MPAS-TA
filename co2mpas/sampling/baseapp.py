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
    sys.exit(baseapp.pump_cmd(cmd.start()) and 0)

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
from collections import OrderedDict, defaultdict
import contextlib
import copy
import io
import logging
import os
from typing import Sequence, Text, Any, Tuple, List  # @UnusedImport

from boltons.setutils import IndexedSet as iset
from toolz import dicttoolz as dtz, itertoolz as itz
import yaml

import itertools as itt
import os.path as osp
import pandalone.utils as pndlu

from . import CmdException
from .. import (__version__, __updated__, __uri__, __copyright__, __license__)  # @UnusedImport
from ..__main__ import init_logging
from .._vendor import traitlets as trt
from .._vendor.traitlets import config as trtc


################################################
## INFO: Modify the following variables on a different application.
APPNAME = 'co2dice'  # TODO: Cannot use baseapp with different app-names.
CONFIG_VAR_NAME = '%s_CONFIG_PATHS' % APPNAME.upper()
PERSIST_VAR_NAME = '%s_PERSIST_PATH' % APPNAME.upper()

try:
    _mydir = osp.dirname(__file__)
except Exception:
    _mydir = '.'


def default_config_fname():
    """The config-file's basename (no path or extension) to search when not explicitly specified."""
    return '%s_config.py' % APPNAME


def default_config_dir():
    """The folder of to user's config-file."""
    return pndlu.convpath('~/.%s' % APPNAME)


def default_config_fpaths():
    """The full path of to user's config-file, without extension."""
    return [osp.join(default_config_dir(), default_config_fname())]


def default_persist_fpath(dirname=None):
    """The full path of to user's persistent config-file, without extension."""
    return osp.join(dirname or default_config_dir(), '%s_persist.json' % APPNAME)

#
################################################


def as_list(value):
    if not isinstance(value, list):
        value = [value]
    return value


def get_class_logger(cls):
    """Mimic log-hierarchies also for traitlet classes."""
    return logging.getLogger('%s.%s' % (cls.__module__, cls.__name__))


##############################
## Maintain ordered YAML
#  from http://stackoverflow.com/a/21912744
#
_MAPTAG = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def _construct_ordered_dict(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


def _ordered_dict_representer(dumper, data):
    return dumper.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        data.items())


def yaml_load(stream, Loader=yaml.SafeLoader):
    class OrderedLoader(Loader):
        pass

    OrderedLoader.add_constructor(_MAPTAG, _construct_ordered_dict)
    return yaml.load(stream, OrderedLoader)


def yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    OrderedDumper.add_representer(OrderedDict, _ordered_dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def setup_yaml_ordered():
    """
    Invoke it once it to enable app-wide ordered yaml.

    From http://stackoverflow.com/a/8661021 """

    yaml.add_representer(OrderedDict, _ordered_dict_representer)
    yaml.add_representer(defaultdict, _ordered_dict_representer)
    yaml.add_representer(tuple, yaml.SafeDumper.represent_list)
    yaml.add_constructor(_MAPTAG, _construct_ordered_dict)


## Dice better have ordered reports
#  so put it here to be invoked only once.
setup_yaml_ordered()
#
##############################


class PathList(trt.List):
    """split unicode strings on `os.pathsep` to form a the list of paths"""
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, trait=trt.Unicode(), **kwargs)

    def validate(self, obj, value):
        """break all elements also into `os.pathsep` segments"""
        value = super().validate(obj, value)
        value = [cf2
                 for cf1 in value
                 for cf2 in cf1.split(os.pathsep)]
        return value

    def from_string(self, s):
        if s:
            s = s.split(osp.pathsep)
        return s


class PeristentMixin(metaclass=trt.MetaHasTraits):
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
    def store_pconfig(cls, fpath: Text, log=None):
        """
        Stores ptrait-values from the global :attr:`_pconfig`into `fpath` as JSON.

        :param cls:
            This mixin where updated ptrait-values are to be stored *globally*.
        """
        cfg = cls._pconfig
        if cfg and cfg != cls._pconfig_orig:
            import json

            fpath = pndlu.ensure_file_ext(fpath, '.json')

            if log:
                if osp.exists(fpath):
                    action = 'Updat'
                    logmeth = log.debug
                else:
                    action = 'Creat'
                    logmeth = log.info

                logmeth("%sing persistent configs %r...", action, fpath)
            os.makedirs(osp.dirname(fpath), exist_ok=True)
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


class HasCiphersMixin(metaclass=trt.MetaHasTraits):
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


class TolerableSingletonMixin(metaclass=trt.MetaHasTraits):
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

    ## Override traitlet loggers that are non-hierarchic.
    @trt.default('log')
    def _log_default(self):
        return get_class_logger(type(self))

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
        `tstamp`
            Print SMTP/IMAP connection messages exchanged (WARN: passwords revealed!).
        `project init/open/append/tstamp`
            Print committed-msg instead of try/false/proj-name (WARN: passwords revealed, see above!).
        `config show` and `config desc`
            Print class-parameters from the whole hierarchy, including those
            from intermediate classes.
          """).tag(config=True)

    ## TODO: Retrofit to force-flags (with code for each specific permission).
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

    def register_validators(self, *validators_and_traits):
        """
        Register validator(s) on class triat(s).

        :param validators_and_traits
            a list of validator-functions or class-traits to be validated;
            they are separated accordingly by this method.
            Validator signature::

                def validate(cls, proposal)
        """
        validators = []
        traits = []
        for vt in validators_and_traits:
            if isinstance(vt, trt.TraitType):
                traits.append(vt)
            else:
                assert callable(vt), vt
                validators.append(vt)

        ntrts = len(traits)
        nvals = len(validators)
        assert nvals and ntrts, (
            "Both validators(%s) and traits(%s) must be given:"
            "  traits    : %s\n"
            "  validators: %s\n" %
            (nvals, ntrts, traits, validators))

        if nvals == 1:
            validate = validators[0]
        else:
            def validate(inst, proposal):
                for v in validators:
                    value = v(inst, proposal)
                    proposal.value = value
                return value

        for t in traits:
            self._register_validator(validate, [t.name])

    ###############
    ## Traitlet @validators to be used by sub-classes
    #  like that::
    #
    #      self.register_validators(<my_class>._warn_deprecated, ['a', ])

    def _is_not_empty(self, proposal):
        value = proposal.value
        if not value:
            myname = type(self).__name__
            raise trt.TraitError("`%s.%s` must not be empty!"
                                 % (myname, proposal.trait.name))
        return proposal.value

    def _is_pure_email_address(self, proposal):
        from validate_email import validate_email

        value = proposal.value
        for value in as_list(value):
            if value and not validate_email(value):
                myname = type(self).__name__
                raise trt.TraitError(
                    "`%s.%s` needs a proper email-address, got: %s"
                    % (myname, proposal.trait.name, value))
        return proposal.value

    def _is_all_latin(self, proposal):
        value = proposal.value
        for value in as_list(value):
            if value and not all(ord(c) < 128 for c in value):
                myname = type(self).__name__
                raise trt.TraitError(
                    '%s.%s must not contain non-ASCII chars: %s'
                    % (myname, proposal.trait.name, value))
        return proposal.value

    def _warn_deprecated(self, proposal):
        t = proposal.trait
        myname = type(self).__name__
        if proposal.value:
            self.log.warning("Trait `%s.%s`: %s" % (myname, t.name, t.help))
        return proposal.value

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

    def __init__(self):
        self._visited_tuples = []

    #: A list of 2-tuples ``(folder, fname(s))`` with loaded config-files
    #: in ascending order (last overrides earlier).
    _visited_tuples = None

    @property
    def config_tuples(self):
        """
        The consolidated list of loaded 2-tuples ``(folder, fname(s))``.

        Sorted in descending order (1st overrides later).
        """
        return self._consolidate(self._visited_tuples)

    @staticmethod
    def _consolidate(visited_tuples):
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
        for b, f in visited_tuples:
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

    def visit_file(self, fpath, miss=False, append=False):
        """
        Invoke this in ascending order for every visited config-file.

        :param miss:
            Loaeded successful?
        :param append:
            set to true to add in descending order (file overriden by above files)
        """
        base, fname = osp.split(fpath)
        pair = (base, None if miss else fname)

        if append:
            self._visited_tuples.append(pair)
        else:
            self._visited_tuples.insert(0, pair)

    def collect_fpaths(self, path_list: List[Text]):
        """
        Collects all (``.json|.py``) files present in the `path_list`, (descending order).

        :param path_list:
            A list of paths (absolute, relative, dir or folders).
        :return:
            fully-normalized paths, with ext
        """
        new_paths = iset()
        default_cfg = default_config_fname()

        def try_json_and_py(basepath):
            found_any = False
            for ext in ('.py', '.json'):
                f = pndlu.ensure_file_ext(basepath, ext)
                if f in new_paths:
                    continue

                if osp.isfile(f):
                    new_paths.add(f)
                    self.visit_file(f, append=True)
                    found_any = True
                else:
                    self.visit_file(f, miss=True, append=True)

            return found_any

        def _derive_config_fpaths(path: Text) -> List[Text]:
            """Return multiple *existent* fpaths for each config-file path (folder/file)."""

            p = pndlu.convpath(path)
            if osp.isdir(p):
                try_json_and_py(osp.join(p, default_cfg))
            else:
                found = try_json_and_py(p)
                ## Do not strip ext if has matched WITH ext.
                if not found:
                    try_json_and_py(osp.splitext(p)[0])

        for cf in path_list:
            _derive_config_fpaths(cf)

        return list(new_paths)

    def head_folder(self):
        """The *last* existing visited folder (if any), even if not containing files."""
        for dirpath, _ in self.config_tuples:
            if osp.exists(dirpath):
                assert osp.isdir(dirpath), ("Expected to be a folder:", dirpath)
                return dirpath


def cmd_line_chain(cmd):
    """Utility returning the cmd-line(str) that launched a :class:`Cmd`."""
    return ' '.join(c.name for c in reversed(cmd.my_cmd_chain()))


class Cmd(TolerableSingletonMixin, trtc.Application, Spec):
    """Common machinery for all (sub-)commands. """
    ## INFO: Do not use it directly; inherit it.
    # See module documentation for developer's guidelines.

    @trt.default('name')
    def _name(self):
        name = class2cmd_name(type(self))
        return name

    ## Override traitlet loggers that are non-hierarchic.
    #  Note that spec's default does not apply due to(?) mro.
    @trt.default('log')
    def _log_default(self):
        return get_class_logger(type(self))

    option_description = trt.Unicode("""
    Options are convenience aliases to configurable class-params,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-params for some <cmd>, use::
        <cmd> --help-all
    or view help for specific parameter using::
        %s config desc <class>.<param>
    """.strip() % APPNAME)

    @trt.default('subcommand_description')
    def _subcmd_msg(self):
        return """
            Subcommands are launched as::
                %(cmd_chain)s <subcmd> [args]
            For information a <subcmd> use::
                %(cmd_chain)s <subcmd> -h
        """ % self._my_text_interpolations()

    #: Whether to raise if configs not found
    #: GUI don't need such validation, dice commands do.
    #: NOTE: HACK to fail early on first AIO launch.
    configs_required = False

    config_paths = PathList(
        default_value=default_config_fpaths(),
        help="""
        Absolute/relative folder/file path(s) to read "static" config-parameters from.

        - Sources for this parameter can either be CLI or ENV-VAR; since the loading
          of config-files depend on this parameter, values specified there are ignored.
        - Multiple values may be given and each one may be separated by '{sep}'.
          Priority is descending, i.e. config-params from the 1st one overrides the rest.
        - For paths resolving to existing folders, the filenames `{appname}_config(.py | .json)`
          are appended and searched (in this order); otherwise, any file-extension
          is ignored, and the mentioned extensions are combined and searched.
        - The 1st *existent* path is important because the *persistent* parameters
          are read (and written) from (and in) it; read `--persist-path`.

        Tips:
          - Use `config paths` to view the actual paths/files loaded.
          - Use `config write` to produce a skeleton of the config-file.

        Examples:
          To read and apply in descending order: [~/my_conf, /tmp/conf.py, ~/.co2dice.json]
          you may issue:
              <cmd> --config-paths=~/my_conf{sep}/tmp/conf.py  --Cmd.config_paths=~/.co2dice.jso
        """.format(appname=APPNAME, sep=osp.pathsep)
    ).tag(config=True, envvar=CONFIG_VAR_NAME)

    persist_path = trt.Unicode(
        None, allow_none=True,
        help="""
        Absolute/relative folder/file path to read/write *persistent* parameters on runtime.

        In practice, when both CLI and ENV-VAR are empty, *persist* file follows
        the location of --config-paths or `{confvar}` envvar, but more precisely:

        - The source for this parameter is a) CLI, b) ENV-VAR, or c) the 1st *existent*
          path collected by the rules of `Cmd.config_paths` parameter (in this order);
          since the loading of config-files depend on this parameter, values specified
          there are ignored.
        - If all sources above result to empty, it defaults to:
               {default}

        - A non-empty path in this parameter gets resolved like this:
          - if it resolves to an existent folder, the filename `{appname}_persist.json`
            gets appended to it;
          - otherwise, the file-extension is assumed to be `.json`; in that case,
            parent folders of the file-path must exist.

        - The 1st *existing* path of `Cmd.config_paths` gets resolved like this::
          - if it resolves to a folder, the filename `<1st-path>/{appname}_persist.json`
            is assumed;
          - if it resolves to a file, the parent-folder of this file is fed into
            the above rule.

        - All *persistent* parameters take precendence over "static" ones.
        - The *persistent* file is written in two occasions:
          - on startup, if an un-encrypted value is met in it;
          - on exit, if *persistent* values have changed.

        Tips:
           - Use `config paths` to view the actual file loaded.
           - Use `{appname} --encrypt` to encrypt all ciphered-prams in peristent file.
        """.format(appname=APPNAME, confvar=CONFIG_VAR_NAME,
                   default=default_persist_fpath())
    ).tag(config=True, envvar=PERSIST_VAR_NAME)

    encrypt = trt.Bool(
        False,
        help="""
        Encrypt ciphered-prams on app startup: True: all classes, False: current-cmd's only

        Sample cmd to encrypt any freshly edited persistent-configs:
            %s --encrypt
        """ % APPNAME
    ).tag(config=True)

    _cfgfiles_registry = CfgFilesRegistry()

    @property
    def loaded_config_files(self):
        return self._cfgfiles_registry.config_tuples

    def _collect_static_fpaths(self):
        """Return fully-normalized paths, with ext."""
        config_paths = self.config_paths
        fpaths = self._cfgfiles_registry.collect_fpaths(config_paths)

        ## NOTE: CO2MPAS-only logic where configs must exist!
        #
        if self.configs_required and not fpaths:
            self.log.error(
                "No DICE-configurations found in %s!\n"
                "  Ask JRC for configs, or copy/adapt them from your old AIO.",
                config_paths)

        return fpaths

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
    def persist_file_resolved(self):
        """Returns the 1st value in config, env, or default-value (might contain ``.json``)."""
        persist_path = self.persist_path
        if persist_path:
            persist_path = pndlu.convpath(persist_path)
        else:
            ## Default and head-folder are already absolute,
            #  don't expand them case they are strange...
            #
            head_folder = self._cfgfiles_registry.head_folder()
            ## Defaults to global default if no head-path.
            persist_path = default_persist_fpath(head_folder)

        return persist_path

    def _read_config_from_persist_file(self):
        persist_path = self.persist_file_resolved
        try:
            persist_path, config = self.load_pconfig(persist_path)
        except Exception as ex:
            self._cfgfiles_registry.visit_file(persist_path)
            if self.raise_config_file_errors:
                raise
            self.log.error("Failed loading persist-file '%s' due to: %s",
                           persist_path, ex, exc_info=True)
        else:
            self.log.debug("Loading persist-file: %s", persist_path)
            self._cfgfiles_registry.visit_file(persist_path, miss=not bool(config))
            if config is None:
                self.log.debug("Missing persist-file: %s", persist_path)
            return config

    def load_configurables_from_files(self) -> Tuple[trtc.Config, trtc.Config]:
        """
        Load :attr:`config_paths`, :attr:`persist_path` and maintain :attr:`config_registry`.

        :return:
            A 2 tuple ``(static_config, persist_config)``, where the 2nd might be `None`.

        Configuration files are read and merged in descending orders:

        1. Static parameters from ``.json`` and/or ``.py`` files
           (see, :attribute:`config_paths`)
        2. Persistent parameters from ``co2dice_persist.json `` file
           (see :attribute:`persist_path`)

        Persistent-parameters override "static" ones.
        """
        ## Code adapted from :meth:`load_config_file` & :meth:`Application._load_config_files`.
        static_paths = self._collect_static_fpaths()
        static_config = self._read_config_from_static_files(static_paths)
        persist_config = self._read_config_from_persist_file()

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

    def _my_text_interpolations(self):
        return {'app_cmd': APPNAME,
                'cmd_chain': cmd_line_chain(self)}

    def emit_description(self):
        ## Overridden for interpolating app-name.
        txt = self.description or self.__doc__
        txt %= self._my_text_interpolations()
        for p in trtc.wrap_paragraphs(txt):
            yield p
            yield ''

    def emit_examples(self):
        ## Overridden for interpolating app-name.
        if self.examples:
            txt = self.examples
            txt = txt.strip() % self._my_text_interpolations()
            yield "Examples"
            yield "--------"
            yield ''
            yield trtc.indent(trtc.dedent(txt))
            yield ''

    def emit_help_epilogue(self, classes=None):
        """Yield the very bottom lines of the help message.

        If classes=False (the default), print `--help-all` msg.
        """
        if not classes:
            interps = self._my_text_interpolations()
            yield "--------"
            yield ("- For all available params, use:\n    %(cmd_chain)s --help-all\n"
                   % interps)
            yield ("- For help on specific classes/params, use:\n    "
                   "%(app_cmd)s config desc [-v] [-c] <class-or-param-1> ...\n"
                   % interps)

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

    # TODO: drop most of cmd-chain inheritance logic - class one is enough.
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
        ## FIXME: can all move o CO2diceCmd and remain here containers only.
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
            persist_path = self.persist_file_resolved
            if ntraits_encrypted:
                self.log.info("Updating persistent config %r with %d auto-encrypted values...",
                              persist_path, ntraits_encrypted)
                self.store_pconfig(persist_path, self.log)

        if static_screams:
            msg = ("Found %d non-encrypted params in static-configs: %s"
                   "\n  Please move them into the \"persistent\" JSON file."
                   % (len(static_screams), list(static_screams)))
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
            res = self.run(*self.extra_args)

            try:
                persist_path = self.persist_file_resolved
                self.store_pconfig(self.persist_file_resolved, self.log)
            except Exception as ex:
                self.log.warning("Failed saving persistent config '%s' due to: %s",
                                 persist_path, ex, exc_info=1)

            return res

        return self.subapp.start()

    def run(self, *args):
        """Leaf sub-commands must inherit this instead of :meth:`start()` without invoking :func:`super()`.

        :param args:
            Invoked by :meth:`start()` with :attr:`extra_args`.

        By default, screams about using sub-cmds, or about doing nothing!
        """
        import ipython_genutils.text as tw

        assert self.subcommands, "Override run() method in cmd subclasses."

        examples = '\n'.join(self.emit_examples()) if self.examples else ''
        if args:
            subcmd_msg = "unknown sub-command `%s`; try one of" % args[0]
        else:
            subcmd_msg = "specify one of its sub-commands"
        msg = tw.dedent(
            """
            %(cmd_chain)s: %(subcmd_msg)s:
                %(subcmds)s
            or type:
                %(cmd_chain)s --help

            %(examples)s
            %(epilogue)s""") % {
                'subcmd_msg': subcmd_msg,
                'cmd_chain': cmd_line_chain(self),
                'subcmds': ', '.join(self.subcommands.keys()),
                'examples': examples,
                'epilogue': '\n'.join(self.emit_help_epilogue()),
        }
        raise CmdException(msg)

    @classmethod
    def make_cmd(cls, argv=None, **kwargs):
        """
        Instanciate, initialize and return application.

        :param argv:
            Like :meth:`initialize()`, if undefined, replaced with ``sys.argv[1:]``.

        - Tip: Apply :func:`pump_cmd()` on return values to process
          generators of :meth:`run()`.
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
        and possibly apply the :func:`pump_cmd()` on its results.

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


class ConsumerBase:
    """Checks if all boolean items (if any) are True, to decide final bool state."""
    any_none_bool = False
    all_true = True

    def __call__(self, item):
        if isinstance(item, bool):
            self.all_true &= item
        else:
            self.any_none_bool = True
        self._emit(item)

    def __bool__(self):
        """
        :return:
            True if all ok, False if any boolean items was false.
        """
        return self.any_none_bool or self.all_true


class PrintConsumer(ConsumerBase):
    """Prints any text-items while checking if all boolean ok."""
    def _emit(self, item):
        if not isinstance(item, bool):
            print(item)


class ListConsumer(ConsumerBase):
    """Collect all items in a list, while checking if all boolean ok."""
    def __init__(self):
        self.items = []

    def _emit(self, item):
        self.items.append(item)


def pump_cmd(cmd_res, consumer=None):
    """
    Sends (possibly lazy) cmd-results to a consumer (by default to STDOUT).

    :param cmd_res:
        Whatever is returnened by a :meth:`Cmd.start()`/`Cmd.run()`.
    :param consumer:
        A callable consuming items and deciding if everything was ok;
        defaults to :class:`PrintConsumer`
    :return:
        ``bool(consumer)``

    - Remember to have logging setup properly before invoking this.
    - This the 2nd half of the replacement for :meth:`Application.launch_instance()`.
    """
    import types

    if not consumer:
        consumer = PrintConsumer()

    if cmd_res is not None:
        if isinstance(cmd_res, types.GeneratorType):
            for i in cmd_res:
                consumer(i)
        elif isinstance(cmd_res, (tuple, list)):
            for i in cmd_res:
                consumer(i)
        else:
            consumer(cmd_res)

    ## NOTE: Enable this code to update `/logconf.yaml`.
    #print('\n'.join(sorted(logging.Logger.manager.loggerDict)))

    return bool(consumer)


def collect_cmd(cmd_res, dont_coalesce=False, assert_ok=False):
    """
    Pumps cmd-result in a new list.

    :param cmd_res:
        A list of items returned by a :meth:`Cmd.start()`/`Cmd.run()`.
        If it is a sole item, it is returned alone without a list.
    :param assert_ok:
        if true, checks :class:`ListConsumer`'s exit-code is not false.
    """
    cons = ListConsumer()
    pump_cmd(cmd_res, cons)
    items = cons.items

    assert not assert_ok or bool(cons), items

    if dont_coalesce:
        return items

    if items:
        if len(items) == 1:
            items = items[0]

        return items
