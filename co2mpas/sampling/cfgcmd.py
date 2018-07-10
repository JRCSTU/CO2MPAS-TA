#!/usr/bin/env pythonw
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""Dice traitlets sub-commands for manipulating configurations"""

from collections import OrderedDict
import os
from typing import Sequence, Text, List, Tuple  # @UnusedImport

from toolz import dicttoolz as dtz

import functools as fnt
import os.path as osp

from . import baseapp, CmdException
from .._vendor import traitlets as trt


def prepare_matcher(terms, is_regex):
    import re

    def matcher(r):
        if is_regex:
            return re.compile(r, re.I).search
        else:
            return lambda w: r.lower() in w.lower()

    matchers = [matcher(t) for t in terms]

    def match(word):
        return any(m(word) for m in matchers)

    return match


def prepare_search_map(all_classes, own_traits):
    """
    :param own_traits:
        bool or None (no traits)
    :return:
        ``{'ClassName.trait_name': (class, trait)`` When `own_traits` not None,
        ``{clsname: class}``) otherwise.
        Note: 1st case might contain None as trait!
    """
    if own_traits is None:
        return OrderedDict([
            (cls.__name__, cls)
            for cls in all_classes])

    if own_traits:
        class_traits = lambda cls: cls.class_own_traits(config=True)
    else:
        class_traits = lambda cls: cls.class_traits(config=True)

    ## Not using comprehension
    #  to work for classes with no traits.
    #
    smap = []
    for cls in all_classes:
        clsname = cls.__name__
        traits = class_traits(cls)
        if not traits:
            smap.append((clsname + '.', (cls, None)))
            continue

        for attr, trait in sorted(traits.items()):
            smap.append(('%s.%s' % (clsname, attr), (cls, trait)))

    return OrderedDict(smap)


def prepare_help_selector(only_class_in_values, verbose):
    from .._vendor.traitlets import config as trtc

    if only_class_in_values:
        if verbose:
            def selector(ne, cls):
                return cls.class_get_help()
        else:
            def selector(ne, cls):
                from ipython_genutils.text import wrap_paragraphs

                help_lines = []
                base_classes = ', '.join(p.__name__ for p in cls.__bases__)
                help_lines.append(u'%s(%s)' % (cls.__name__, base_classes))
                help_lines.append(len(help_lines[0]) * u'-')

                cls_desc = getattr(cls, 'description', None)
                if not isinstance(cls_desc, str):
                    cls_desc = cls.__doc__
                if cls_desc:
                    help_lines.extend(wrap_paragraphs(cls_desc))
                help_lines.append('')

                try:
                    txt = cls.examples.default_value.strip()
                    if txt:
                        help_lines.append("Examples")
                        help_lines.append("--------")
                        help_lines.append(trtc.indent(trtc.dedent(txt)))
                        help_lines.append('')
                except AttributeError:
                    pass

                return '\n'.join(help_lines)

    else:
        def selector(name, v):
            cls, attr = v
            if not attr:
                #
                ## Not verbose and class not owning any trait.
                return "--%s" % name
            else:
                return cls.class_get_trait_help(attr)

    return selector


class ConfigCmd(baseapp.Cmd):
    """
    Commands to manage configuration-options loaded from filesystem, cmd-line or defaults.
    """

    examples = trt.Unicode("""
        - Ask help on parameters affecting the source of the configurations::
              %(cmd_chain)s desc config_paths persist_path

        - Show config-param values for all params containing word "mail"::
              %(cmd_chain)s show  --versbose  mail

        - Show values originating from files::
              %(cmd_chain)s show  --source file

        - Show configuration paths::
              %(cmd_chain)s paths
    """)

    def __init__(self, **kwds):
            super().__init__(
                subcommands=baseapp.build_sub_cmds(*config_subcmds),
                **kwds)


class WriteCmd(baseapp.Cmd):
    """
    Store config defaults into specified path(s); '{confpath}' assumed if none specified.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [<config-path-1>] ...

    - If a path resolves to a folder, the filename '{appname}_config.py' is appended.
    - It OVERWRITES any pre-existing configuration file(s)!
    """

    ## Class-docstring CANNOT contain string-interpolations!
    description = trt.Unicode(__doc__.format(
        confpath=baseapp.default_config_fpaths()[0],
        appname=baseapp.APPNAME))

    examples = trt.Unicode("""
        - Generate a config-file at your home folder::
              %(cmd_chain)s ~/my_conf

        - To re-use the generated custom config-file alone, use the option::
              --config-paths=~/my_conf  ...
    """)

    def run(self, *args):
        ## Prefer to modify `classes` after `initialize()`, or else,
        #  the cmd options would be irrelevant and fatty :-)
        self.classes = self.all_app_configurables()
        args = args or [None]
        for fpath in args:
            self.write_default_config(fpath, self.force)


class PathsCmd(baseapp.Cmd):
    """
    List paths and variables used to load configurations (1st override those below).

    Some of the environment-variables affecting configurations:
        HOME, USERPROFILE,          : where configs & DICE projects are stored
                                      (1st one defined wins)
        CO2DICE_CONFIG_PATHS        : where to read configuration-files.
            CO2DICE_PERSIST_PATH    :
        GNUPGHOME                   : where GPG-keys are stored
                                      (works only if `gpgconf.ctl` is deleted,
                                       see https://goo.gl/j5mwo4)
        GNUPGKEY                    : override which is the master-key
                                      (see `desc master_key`)
        GIT_PYTHON_GIT_EXECUTABLE   : where `git` executable is (if not in PATH)
    """

    def _collect_gpg_paths(self, gpgexe):
        import subprocess as sbp

        gpgconf = osp.join(osp.dirname(gpgexe), 'gpgconf')
        options = '--list-dirs --list-components --list-config'.split()
        res = []
        for option in options:
            try:
                lines = sbp.check_output([gpgconf, option],
                                         universal_newlines=True)
                lines = lines.strip()
                lines = lines.split('\n') if lines else []
                res.append((option, lines))
            except Exception as ex:
                self.log.warning("Failed executing `gpgconf` due to: %s", ex)
        return res

    def _collect_env_vars(self, classes):
        classes = (cls
                   for cls
                   in self._classes_inc_parents(classes))
        return [trait.metadata['envvar']
                for cls in classes
                for trait
                in cls.class_own_traits(envvar=(lambda ev: bool(ev))).values()]


    def run(self, *args):
        if len(args) > 0:
            raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                               % (self.name, len(args), args))

        import sys
        from .. import __version__, __updated__, __dice_report_version__
        from . import project
        from . import crypto

        sep = osp.sep
        l2_yaml_list_sep = '\n    - '

        def sterilize(func, fallback=None):
            try:
                return func()
            except Exception as ex:
                return "<%s due to: %s(%s)>" % (
                    fallback or 'invalid', type(ex).__name__, ex)

        def format_tuple(path, files: List[Text]):
            endpath = sep if path[-1] != sep else ''
            return '    - %s%s: %s' % (path, endpath, files or '')

        # TODO: paths not valid YAML!  ...and renable TC.
        yield "APP:"
        yield "  co2dice_path: %s" % osp.dirname(__file__)
        yield "  python_path: %s" % sys.prefix

        yield "VERSIONS:"
        yield "  co2dice_release: %s" % __version__
        yield "  co2dice_updated: %s" % __updated__
        yield "  dice_report_ver: %s" % __dice_report_version__
        yield "  python_version: %s" % sys.version

        yield "CONFIG:"
        config_paths = l2_yaml_list_sep.join([''] + self.config_paths)
        yield "  config_paths:%s" % (config_paths or 'null')
        yield "  persist_path: %s" % self.persist_file_resolved

        loaded_cfgs = self.loaded_config_files
        if loaded_cfgs:
            yield "  LOADED_CONFIGS:"
            yield from (format_tuple(p, f) for p, f in self.loaded_config_files)
        else:
            yield "  LOADED_CONFIGS: null"

        var_names = """AIODIR HOME HOMEDRIVE HOMEPATH USERPROFILE
                     TRAITLETS_APPLICATION_RAISE_CONFIG_FILE_ERROR
                     GIT_PYTHON_GIT_EXECUTABLE GIT_PYTHON_TRACE GIT_TRACE"""
        yield "ENV_VARS:"
        trait_envvars = self._collect_env_vars(self.all_app_configurables())
        for vname in sorted(set(var_names.split() + trait_envvars)):
            yield "  %s: %s" % (vname, os.environ.get(vname))

        yield "GPG:"
        gpg = crypto.GpgSpec(config=self.config)
        gnupgexe = gpg.gnupgexe_resolved
        yield "  gnupgexe: %s" % gnupgexe
        yield "  gnupghome: %s" % gpg.gnupghome_resolved
        master_key = sterilize(lambda: gpg.master_key_resolved)
        yield "  master_key: %s" % master_key
        for cmd, lines in self._collect_gpg_paths(gnupgexe):
            yield ("  gpgconf%s: |" % cmd[1:]).replace('-', '_')
            for line in lines:
                yield "    %s" % line

        import git
        import shutil

        yield "PROJECTS:"
        git_exe = os.environ.get('GIT_PYTHON_GIT_EXECUTABLE', 'git')
        git_exe = shutil.which(git_exe)
        yield "  git_exe: %s" % git_exe

        repo = project.ProjectsDB.instance(config=self.config)
        yield "  repo_path: %s" % repo.repopath_resolved


class ShowCmd(baseapp.Cmd):
    """
    Print configurations (defaults | files | merged) before any validations.

    SYNTAX
        %(cmd_chain)s [OPTIONS] [--source=(merged | default)] [<search-term-1> ...]
        %(cmd_chain)s [OPTIONS] --source file

    - Search-terms are matched case-insensitively against '<class>.<param>'.
    - Use --verbose to view values for config-params as they apply in the
      whole hierarchy (not
    - Results are sorted in "application order" (later configurations override
      previous ones); use --sort for alphabetical order.
    - Warning: Defaults/merged might not be always accurate!
    - Tip: you may also add `--show-config` global option on any command
      to view configured values accurately on runtime.
    """

    examples = trt.Unicode("""
        - View all "merged" configuration values::
              %(cmd_chain)s

        - View all "default" or "in file" configuration values, respectively::
              %(cmd_chain)s --source defaults
              %(cmd_chain)s --s f

        - View help on specific parameters::
              %(cmd_chain)s tstamp
              %(cmd_chain)s -e 'rec.+wait'

        - List classes matching a regex::
              %(cmd_chain)s -ecl 'rec.*cmd'
    """)

    source = trt.FuzzyEnum(
        'defaults files merged ciphered'.split(),
        default_value='merged',
        allow_none=False,
        help="""
        Show configuration parameters in code, stored on disk files,
        merged or merged and ciphered (encrypted), respectively."""
    ).tag(config=True)

    list = trt.Bool(
        help="Just list any matches."
    ).tag(config=True)

    regex = trt.Bool(
        help="Search terms as regular-expressions."
    ).tag(config=True)

    sort = trt.Bool(
        help="""
        Sort classes alphabetically; by default, classes listed in "application order",
        that is, later configurations override previous ones.
        """
    ).tag(config=True)

    def __init__(self, **kwds):
        import pandalone.utils as pndlu

        kwds.setdefault('cmd_aliases', {
            ('s', 'source'): ('ShowCmd.source',
                              ShowCmd.source.help)
        })
        kwds.setdefault(
            'cmd_flags', {
                ('l', 'list'): (
                    {type(self).__name__: {'list': True}},
                    type(self).list.help
                ),
                ('e', 'regex'): (
                    {type(self).__name__: {'regex': True}},
                    type(self).regex.help
                ),
                ('t', 'sort'): (
                    {type(self).__name__: {'sort': True}},
                    type(self).sort.help
                ),
            }
        )
        kwds.setdefault('encrypt', True)  # Encrypted ALL freshly edited pconfigs.
        kwds.setdefault('raise_config_file_errors', False)
        super().__init__(**kwds)

    def initialize(self, argv=None):
        ## Copied from `Cmd.initialize()`.
        #
        self.parse_command_line(argv)
        static_config, persist_config = self.load_configurables_from_files()
        self._validate_cipher_traits_against_config_files(static_config, persist_config)
        if persist_config:
            static_config.merge(persist_config)
        static_config.merge(self.cli_config)
        ## Stop from applying file-configs - or any trait-validations will scream.

        self._loaded_config = static_config

    def _yield_file_configs(self, config, classes=None):
        assert not classes, (classes, "should be empty")

        for k, v in config.items():
            yield k
            try:
                for kk, vv in v.items():
                    yield '  +--%s = %s' % (kk, vv)
            except Exception:
                yield '  +--%s' % v

    def _yield_configs_and_defaults(self, config, search_terms,
                                    merged: bool, ciphered: bool):
        verbose = self.verbose
        get_classes = (self._classes_inc_parents
                       if verbose else
                       self._classes_with_config_traits)
        all_classes = list(get_classes(self.all_app_configurables()))

        ## Merging needs to visit all hierarchy.
        own_traits = not (verbose or merged)

        search_map = prepare_search_map(all_classes, own_traits)

        if ciphered:
            from . import crypto

            def ciphered_filter(mapval):
                _, trait = mapval
                if isinstance(trait, crypto.Cipher):
                    return mapval

            search_map = dtz.valfilter(ciphered_filter, search_map)

        if search_terms:
            matcher = prepare_matcher(search_terms, self.regex)
            search_map = dtz.keyfilter(matcher, search_map)

        items = search_map.items()
        if self.sort:
            items = sorted(items)  # Sort by class-name (traits always sorted).

        classes_configured = {}
        for key, (cls, trait) in items:
            if self.list:
                yield key
                continue
            if not trait:
                ## Not --verbose and class not owning traits.
                continue

            clsname, trtname = key.split('.')

            ## Print own traits only, even when "merge" visits all.
            #
            sup = super(cls, cls)
            if not verbose and getattr(sup, trtname, None) is trait:
                continue

            ## Instanciate classes once, to merge values.
            #
            obj = classes_configured.get(cls)
            if obj is None:
                try:
                    ## Exceptional rule for Project-zygote.
                    #  TODO: delete when project rule is gone.
                    #
                    if cls.__name__ == 'Project':
                        cls.new_instance('test', None, config)
                    else:
                        obj = cls(config=config)
                except Exception as ex:
                    self.log.warning("Falied initializing class '%s' due to: %r",
                                     clsname, ex)
                    ## Assign config-values as dummy-object's attributes.
                    #  Note: no merging of values now!
                    #
                    class C:
                        pass
                    obj = C()
                    obj.__dict__ = dict(config[clsname])
                classes_configured[cls] = obj

                ## Print 1 class-line for all its traits.
                #
                base_classes = ', '.join(p.__name__ for p in cls.__bases__)
                yield '%s(%s)' % (clsname, base_classes)

            if merged:
                try:
                    val = getattr(obj, trtname, '??')
                except trt.TraitError as ex:
                    self.log.warning("Cannot merge '%s' due to: %r", trtname, ex)
                    val = "<invalid due to: %s>" % ex
            else:
                val = repr(trait.default())
            yield '  +--%s = %s' % (trtname, val)

    def run(self, *args):
        source = self.source.lower()
        self.log.info("Listing '%s' values for search-terms: %s...",
                      source, args)

        if source == 'files':
            if len(args) > 0:
                raise CmdException("Cmd '%s --source files' takes no arguments, received %d: %r!"
                                   % (self.name, len(args), args))

            func = self._yield_file_configs
        elif source == 'defaults':
            func = fnt.partial(self._yield_configs_and_defaults,
                               merged=False, ciphered=False)
        elif source == 'merged':
            func = fnt.partial(self._yield_configs_and_defaults,
                               merged=True, ciphered=False)
        elif source == 'ciphered':
            func = fnt.partial(self._yield_configs_and_defaults,
                               merged=True, ciphered=True)
        else:
            raise AssertionError('Impossible enum: %s' % source)

        config = self._loaded_config

        yield from func(config, args)


class DescCmd(baseapp.Cmd):
    """
    List and print help for configurable classes and parameters.

    SYNTAX
        %(cmd_chain)s [-l] [-c] [-t] [-v] [<search-term> ...]

    - If no search-terms provided, returns all.
    - Search-terms are matched case-insensitively against '<class>.<param>',
      or against '<class>' if --class.
    - Use --verbose (-v) to view config-params from the whole hierarchy, that is,
      including those from intermediate classes.
    - Use --class (-c) to view just the help-text of classes.
    - Results are sorted in "application order" (later configurations override
      previous ones); use --sort for alphabetical order.
    """

    examples = trt.Unicode(r"""
        - Just List::
              %(cmd_chain)s --list         # List configurable parameters.
              %(cmd_chain)s -l --class     # List configurable classes.
              %(cmd_chain)s -l --verbose   # List config params in all hierarchy.

        -  Exploit the fact that <class>.<param> are separated with a dot('.)::
              %(cmd_chain)s -l Cmd.        # List commands and their own params.
              %(cmd_chain)s -lv Cmd.       # List commands including inherited params.
              %(cmd_chain)s -l ceiver.     # List params of TStampReceiver spec class.
              %(cmd_chain)s -l .user       # List parameters starting with 'user' prefix.

        -  Use regular expressions (--regex)::
              %(cmd_chain)s -le  ^t.+cmd   # List params for cmds starting with 't'.
              %(cmd_chain)s -le  date$     # List params ending with 'date'.
              %(cmd_chain)s -le  mail.*\.  # Search 'mail' anywhere in class-names.
              %(cmd_chain)s -le  \..*mail  # Search 'mail' anywhere in param-names.

        Tip:
          Do all of the above and remove -l.
          For instance::
              %(cmd_chain)s -c DescCmd    # View help for this cmd without its parameters.
              %(cmd_chain)s -t Spec.      # View help sorted alphabetically
    """)

    list = trt.Bool(
        help="Just list any matches."
    ).tag(config=True)

    clazz = trt.Bool(
        help="Print class-help only; matching happens also on class-names."
    ).tag(config=True)

    regex = trt.Bool(
        help="""
        Search terms as regular-expressions.

        Example:
             %(cmd_chain)s -e ^DescCmd.regex

        will print the help-text of this parameter (--regex, -e).
        """
    ).tag(config=True)

    sort = trt.Bool(
        help="""
        Sort classes alphabetically; by default, classes listed in "application order",
        that is, later configurations override previous ones.
        """
    ).tag(config=True)

    def __init__(self, **kwds):
        import pandalone.utils as pndlu

        kwds.setdefault(
            'cmd_flags', {
                ('l', 'list'): (
                    {type(self).__name__: {'list': True}},
                    type(self).list.help
                ),
                ('e', 'regex'): (
                    {type(self).__name__: {'regex': True}},
                    type(self).regex.help
                ),
                ('c', 'class'): (
                    {type(self).__name__: {'clazz': True}},
                    type(self).clazz.help
                ),
                ('t', 'sort'): (
                    {type(self).__name__: {'sort': True}},
                    type(self).sort.help
                ),
            }
        )
        super().__init__(**kwds)

    def run(self, *args):
        ## Prefer to modify `class_names` after `initialize()`, or else,
        #  the cmd options would be irrelevant and fatty :-)
        get_classes = (self._classes_inc_parents
                       if self.clazz or self.verbose else
                       self._classes_with_config_traits)
        all_classes = list(get_classes(self.all_app_configurables()))
        own_traits = None if self.clazz else not self.verbose

        search_map = prepare_search_map(all_classes, own_traits)
        if args:
            matcher = prepare_matcher(args, self.regex)
            search_map = dtz.keyfilter(matcher, search_map)
        items = search_map.items()
        if self.sort:
            items = sorted(items)  # Sort by class-name (traits always sorted).

        selector = prepare_help_selector(self.clazz, self.verbose)
        for name, v in items:
            if self.list:
                yield name
            else:
                yield selector(name, v)


config_subcmds = (
    WriteCmd,
    PathsCmd,
    ShowCmd,
    DescCmd,
)
