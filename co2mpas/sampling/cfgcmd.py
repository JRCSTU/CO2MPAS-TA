#!/usr/bin/env pythonw
#
# Copyright 2014-2016 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
"""Dice traitlets sub-commands for manipulating configurations"""

from . import baseapp, CmdException
from typing import Sequence, Text, List, Tuple    # @UnusedImport

import os
import os.path as osp
import traitlets as trt


class FuzzyEnum(trt.Enum):
    """An case-ignoring enum matching choices by unique prefixes/substrings."""

    case_sensitive = False
    #: If True, choices match anywhere in the string, otherwise match prefixes.
    substring = False

    def __init__(self, values, default_value=trt.Undefined,
                 case_sensitive=False, substring=False, **kwargs):
        self.case_sensitive = case_sensitive
        self.substring = substring
        values = [trt.cast_unicode_py2(value) for value in values]
        super().__init__(values, default_value=default_value, **kwargs)

    def validate(self, obj, value):
        if isinstance(value, str):
            value = trt.cast_unicode_py2(value)
        if not isinstance(value, trt.six.string_types):
            self.error(obj, value)

        conv_func = (lambda c: c) if self.case_sensitive else str.lower
        match_func = ((lambda v, c: v in c)
                      if self.substring
                      else (lambda v, c: c.startswith(v)))
        value = conv_func(value)
        choices = self.values
        matches = [match_func(value, conv_func(c)) for c in choices]
        if sum(matches) == 1:
            for v, m in zip(choices, matches):
                if m:
                    return v

        self.error(obj, value)


class ConfigCmd(baseapp.Cmd):
    """
    Commands to manage configuration-options loaded from filesystem.

    Read also the help message for `--config-paths` generic option.
    """

    class InitCmd(baseapp.Cmd):
        """
        Store config defaults into specified path(s); '{confpath}' assumed if none specified.

        - If a path resolves to a folder, the filename '{appname}_config.py' is appended.
        - It OVERWRITES any pre-existing configuration file(s)!

        SYNTAX
            co2dice config init [<config-path-1>] ...
        """

        ## Class-docstring CANNOT contain string-interpolations!
        description = trt.Unicode(__doc__.format(
            confpath=baseapp.default_config_fpaths()[0],
            appname=baseapp.APPNAME))

        examples = trt.Unicode("""
            Generate a config-file at your home folder:
                co2dice config init ~/my_conf

            To re-use this custom config-file alone, use:
                co2dice --config-paths=~/my_conf  ...
            """)

        def run(self, *args):
            ## Prefer to modify `classes` after `initialize()`, or else,
            #  the cmd options would be irrelevant and fatty :-)
            self.classes = self.all_app_configurables()
            args = args or [None]
            for fpath in args:
                self.write_default_config(fpath, self.force)

    class PathsCmd(baseapp.Cmd):
        """List resolved various paths and actual config-files loaded (descending order)."""
        def run(self, *args):
            if len(args) > 0:
                raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                                   % (self.name, len(args), args))

            from . import project
            from . import crypto

            sep = osp.sep

            def format_tuple(path, files: List[Text]):
                endpath = sep if path[-1] != sep else ''
                return '  +--%s%s: %s' % (path, endpath, files or '')

            yield "CONFIG:"
            yield from (format_tuple(p, f) for p, f in self.loaded_config_files)

            yield "PROJECTS:"
            repo = project.ProjectsDB.instance(config=self.config)
            yield "  +--repopath: %s" % repo.repopath_resolved

            yield "GnuPG:"
            gpg = crypto.GpgSpec(config=self.config)
            yield "  +--gnupgexe: %s" % gpg.gnupgexe_resolved
            yield "  +--gnupghome: %s" % gpg.gnupghome_resolved

    class ShowCmd(baseapp.Cmd):
        """
        Print configurations (defaults | files | merged) before any validations.

        - Use --verbose to view config-params on all intermediate classes.
        - Similarly, you may also add `--Cmd.print_config=True` global option
          on any command to view more targeted results.
        """

        source = FuzzyEnum(
            'defaults files merged'.split(), default_value='merged', allow_none=False,
            help="""Show configuration parameters in code, stored on disk files, or merged."""
        ).tag(config=True)

        def __init__(self, **kwds):
                import pandalone.utils as pndlu

                kwds.setdefault('cmd_aliases', {
                    ('s', 'source'): ('ShowCmd.source',
                                      pndlu.first_line(ConfigCmd.ShowCmd.source.help))
                })
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

        def _yield_file_configs(self, config):
            for k, v in config.items():
                yield k
                try:
                    for kk, vv in v.items():
                        yield '  +--%s = %s' % (kk, vv)
                except:
                    yield '  +--%s' % v

        def _yield_configs_and_defaults(self, config, merged: bool):
            ## Prefer to modify `classes` after `initialize()`, or else,
            #  the cmd options would be irrelevant and fatty :-)
            if merged:
                ## Merging below works only if all classes visited.
                self.verbose = True

            self.classes = self.all_app_configurables()
            for cls in self._classes_with_config_traits():
                clsname = cls.__name__
                cls_printed = False

                cls_traits = (cls.class_traits(config=True)
                              if self.verbose else
                              cls.class_own_traits(config=True))
                for name, trait in sorted(cls_traits.items()):
                    key = '%s.%s' % (clsname, name)
                    if merged and key in config:
                        val = config[clsname][name]
                    else:
                        val = trait.default_value_repr()

                    if not cls_printed:
                        base_classes = ','.join(p.__name__ for p in cls.__bases__)
                        yield '%s(%s)' % (clsname, base_classes)
                        cls_printed = True
                    yield '  +--%s = %s' % (name, val)

        def run(self, *args):
            if len(args) > 0:
                raise CmdException('Cmd %r takes no arguments, received %d: %r!'
                                   % (self.name, len(args), args))

            config = self._loaded_config
            source = self.source.lower()
            if source == 'files':
                func = self._yield_file_configs
            elif source == 'default':
                func = lambda cfg: self._yield_configs_and_defaults(cfg, merged=False)
            elif source == 'merged':
                func = lambda cfg: self._yield_configs_and_defaults(cfg, merged=True)
            else:
                raise AssertionError('Impossible enum: %s' % source)

            yield from func(config)

    def __init__(self, **kwds):
            dkwds = {'subcommands': baseapp.build_sub_cmds(*config_subcmds)}
            dkwds.update(kwds)
            super().__init__(**dkwds)


config_subcmds = (
    ConfigCmd.InitCmd,
    ConfigCmd.PathsCmd,
    ConfigCmd.ShowCmd,
)
