# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
r"""
Predict NEDC CO2 emissions from WLTP.

:Home:         http://co2mpas.io/
:Copyright:    2015-2017 European Commission, JRC <https://ec.europa.eu/jrc/>
:License:       EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>

Use the `batch` sub-command to simulate a vehicle contained in an excel-file.


USAGE:
  co2mpas ta          [-f] [-v] [-O=<output-folder>] [<input-path>]...
  co2mpas batch       [-v | -q | --logconf=<conf-file>] [-f]
                      [--use-cache] [--co2mparable=<old-yaml>]
                      [-O=<output-folder>]
                      [--modelconf=<yaml-file>]
                      [-D=<key=value>]... [<input-path>]...
  co2mpas demo        [-v | -q | --logconf=<conf-file>] [-f]
                      [<output-folder>] [--download]
  co2mpas template    [-v | -q | --logconf=<conf-file>] [-f]
                      [<excel-file-path> ...]
  co2mpas ipynb       [-v | -q | --logconf=<conf-file>] [-f] [<output-folder>]
  co2mpas modelgraph  [-v | -q | --logconf=<conf-file>] [-O=<output-folder>]
                      [--modelconf=<yaml-file>]
                      (--list | [--graph-depth=<levels>] [<models> ...])
  co2mpas modelconf   [-v | -q | --logconf=<conf-file>] [-f]
                      [--modelconf=<yaml-file>] [-O=<output-folder>]
  co2mpas gui         [-v | -q | --logconf=<conf-file>]
  co2mpas             [-v | -q | --logconf=<conf-file>] (--version | -V)
  co2mpas             --help

Syntax tip:
  The brackets `[ ]`, parens `( )`, pipes `|` and ellipsis `...` signify
  "optional", "required", "mutually exclusive", and "repeating elements";
  for more syntax-help see: http://docopt.org/


OPTIONS:
  <input-path>                Input xlsx-file or folder. Assumes current-dir if missing.
  -O=<output-folder>          Output folder or file [default: .].
  --download                  Download latest demo files from ALLINONE GitHub project.
  <excel-file-path>           Output file [default: co2mpas_template.xlsx].
  --modelconf=<yaml-file>     Path to a model-configuration file, according to YAML:
                                https://docs.python.org/3.5/library/logging.config.html#logging-config-dictschema
  --use-cache                 Use the cached input file.
  --co2mparable=<old-yaml>    (internal) Enable co2parable generation in tmp-folder and
                              optionally provide an <old-yaml> file to compare with while executing.
                              Overrides CO2MPARE_ENABLED and CO2MPARE_WITH_FPATH env-vars
                              (unless `--co2mparable=` specified).
                              The <old-yaml> may end with '(txt|.yaml)[.xz]' or be <LATEST>.
                              Other env-vars: CO2MPARE_YAML (default: CSV), CO2MPARE_ZIP(yes)
                              [default: <DISABLED>]
  --override, -D=<key=value>  Input data overrides (e.g., `-D fuel_type=diesel`,
                              `-D prediction.nedc_h.vehicle_mass=1000`).
  -l, --list                  List available models.
  --graph-depth=<levels>      An integer to Limit the levels of sub-models plotted.
  -f, --force                 Overwrite output/template/demo excel-file(s).


Model flags (-D flag.xxx, example -D flag.engineering_mode=True):
 engineering_mode=<bool>      Use all data and not only the declaration data.
 soft_validation=<bool>       Relax some Input-data validations, to facilitate experimentation.
 use_selector=<bool>          Select internally the best model to predict both NEDC H/L cycles.
 only_summary=<bool>          Do not save vehicle outputs, just the summary.
 plot_workflow=<bool>         Open workflow-plot in browser, after run finished.
 output_template=<xlsx-file>  Clone the given excel-file and appends results into
                              it. By default, results are appended into an empty
                              excel-file. Use `output_template=-` to use
                              input-file as template.

Miscellaneous:
  -h, --help                  Show this help message and exit.
  -V, --version               Print version of the program, with --verbose
                              list release-date and installation details.
  -v, --verbose               Print more verbosely messages - overridden by --logconf.
  -q, --quiet                 Print less verbosely messages (warnings) - overridden by --logconf.
  --logconf=<conf-file>       Path to a logging-configuration file, according to:
                                https://docs.python.org/3/library/logging.config.html#configuration-file-format
                              If the file-extension is '.yaml' or '.yml', it reads a dict-schema from YAML:
                                https://docs.python.org/3.5/library/logging.config.html#logging-config-dictschema


SUB-COMMANDS:
    gui             Launches co2mpas GUI (DEPRECATED: Use `co2gui` command).
    ta              Simulate vehicle in type approval mode for all <input-path>
                    excel-files & folder. If no <input-path> given, reads all
                    excel-files from current-dir. It reads just the declaration
                    inputs, if it finds some extra input will raise a warning
                    and will not produce any result.
                    Read this for explanations of the param names:
                      http://co2mpas.io/explanation.html#excel-input-data-naming-conventions
    batch           Simulate vehicle in scientific mode for all <input-path>
                    excel-files & folder. If no <input-path> given, reads all
                    excel-files from current-dir. By default reads just the
                    declaration inputs and skip the extra inputs. Thus, it will
                    produce always a result. To read all inputs the flag
                    `engineering_mode` have to be set to True.
                    Read this for explanations of the param names:
                      http://co2mpas.io/explanation.html#excel-input-data-naming-conventions
    demo            Generate demo input-files for co2mpas inside <output-folder>.
    template        Generate "empty" input-file for the `batch` cmd as <excel-file-path>.
    ipynb           Generate IPython notebooks inside <output-folder>; view them with cmd:
                      jupyter --notebook-dir=<output-folder>
    modelgraph      List or plot available models. If no model(s) specified, all assumed.
    modelconf       Save a copy of all model defaults in yaml format.


EXAMPLES::

    # Don't enter lines starting with `#`.

    # View full version specs:
    co2mpas -vV

    # Create an empty vehicle-file inside `input` folder:
    co2mpas  template  input/vehicle_1.xlsx

    # Create work folders and then fill `input` with sample-vehicles:
    md input output
    co2mpas  demo  input

    # View a specific submodel on your browser:
    co2mpas  modelgraph  co2mpas.model.physical.wheels.wheels

    # Run co2mpas with batch cmd plotting the workflow:
    co2mpas  batch  input  -O output  -D flag.plot_workflow=True

    # Run co2mpas with ta cmd:
    co2mpas  batch  input/co2mpas_demo-0.xlsx  -O output

    # or launch the co2mpas GUI:
    co2gui

    # View all model defaults in yaml format:
    co2maps modelconf -O output
"""

from co2mpas import (__version__ as proj_ver, __file__ as proj_file,
                     __updated__ as proj_date)
import collections
import functools as fnt
import glob
import io
import logging
import os.path as osp
import os
import re
import shutil
import sys
import docopt
import yaml
import warnings


proj_name = 'co2mpas'

log = logging.getLogger('co2mpas_main')


class CmdException(Exception):
    """Polite user-message avoiding ``exit(msg)`` when ``main()`` invoked from python."""
    pass


#: Load  this file automatically if it exists in HOME and configure logging,
#: unless overridden with --logconf.
default_logconf_file = osp.expanduser(osp.join('~', '.co2_logconf.yaml'))


def _set_numpy_logging():
    rlog = logging.getLogger()
    if not rlog.isEnabledFor(logging.DEBUG):
        import numpy as np
        np.seterr(divide='ignore', invalid='ignore')


def init_logging(level=None, frmt=None, logconf_file=None,
                 color=False, default_logconf_file=default_logconf_file,
                 not_using_numpy=False, **kwds):
    """
    :param level:
        tip: use :func:`is_any_log_option()` to decide if should be None
        (only if None default HOME ``logconf.yaml`` file is NOT read).
    :param default_logconf_file:
        Read from HOME only if ``(level, frmt, logconf_file)`` are none.
    :param kwds:
        Passed directly to :func:`logging.basicConfig()` (e.g. `filename`);
        used only id default HOME ``logconf.yaml`` file is NOT read.
    """
    ## Only read default logconf file in HOME
    #  if no explicit arguments given.
    #
    no_args = all(i is None for i in [level, frmt, logconf_file])
    if no_args and osp.exists(default_logconf_file):
        logconf_file = default_logconf_file

    if logconf_file:
        from logging import config as lcfg

        logconf_file = osp.expanduser(logconf_file)
        if osp.splitext(logconf_file)[1] in '.yaml' or '.yml':
            with io.open(logconf_file) as fd:
                log_dict = yaml.safe_load(fd)
                lcfg.dictConfig(log_dict)
        else:
            lcfg.fileConfig(logconf_file)

        logconf_src = logconf_file
    else:
        if level is None:
            level = logging.INFO
        if not frmt:
            frmt = "%(asctime)-15s:%(levelname)5.5s:%(name)s:%(message)s"
        logging.basicConfig(level=level, format=frmt, **kwds)
        rlog = logging.getLogger()
        rlog.level = level  # because `basicConfig()` does not reconfig root-logger when re-invoked.

        logging.getLogger('pandalone.xleash.io').setLevel(logging.WARNING)

        if color and sys.stderr.isatty():
            from rainbow_logging_handler import RainbowLoggingHandler

            color_handler = RainbowLoggingHandler(
                sys.stderr,
                color_message_debug=('grey', None, False),
                color_message_info=('blue', None, False),
                color_message_warning=('yellow', None, True),
                color_message_error=('red', None, True),
                color_message_critical=('white', 'red', True),
            )
            formatter = formatter = logging.Formatter(frmt)
            color_handler.setFormatter(formatter)

            ## Be conservative and apply color only when
            #  log-config looks like the "basic".
            #
            if rlog.handlers and isinstance(rlog.handlers[0], logging.StreamHandler):
                rlog.removeHandler(rlog.handlers[0])
                rlog.addHandler(color_handler)
        logconf_src = 'explicit(level=%s)' % level

    if not not_using_numpy:
        _set_numpy_logging()

    logging.captureWarnings(True)

    ## Disable warnings on AIO but not when developing.
    #
    if os.environ.get('AIODIR'):
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        warnings.filterwarnings(action="ignore", module="scipy",
                                message="^internal gelsd")
        warnings.filterwarnings(action="ignore", module="dill",
                                message="^unclosed file")
        warnings.filterwarnings(action="ignore", module="importlib",
                                message="^can't resolve")

    log.debug('Logging-configurations source: %s', logconf_src)


def is_any_log_option(argv):
    """
    Return true if any -v/--verbose/--debug etc options are in `argv`

    :param argv:
        If `None`, use :data:`sys.argv`; use ``[]`` to explicitly use no-args.
    """
    log_opts = '-v --verbose -d --debug --vlevel'.split()
    if argv is None:
        argv = sys.argv
    return argv and set(log_opts) & set(argv)


def exit_with_pride(reason=None,
                    warn_color='\x1b[31;1m', err_color='\x1b[1m',
                    logger=None):
    """
    Return an *exit-code* and logs error/fatal message for ``main()`` methods.

    :param reason:
        - If reason is None, exit-code(0) signifying OK;
        - if exception,  print colorful (if tty) stack-trace, and exit-code(-1);
        - otherwise, prints str(reason) colorfully (if tty) and exit-code(1),
    :param warn_color:
        ansi color sequence for stack-trace (default: red)
    :param err_color:
        ansi color sequence for stack-trace (default: white-on-red)
    :param logger:
        which logger to use to log reason (must support info and fatal).

    :return:
        (0, 1 -1), for reason == (None, str, Exception) respectively.

    Note that returned string from ``main()`` are printed to stderr and
    exit-code set to bool(str) = 1, so print stderr separately and then
    set the exit-code.

    For colors use :meth:`RainbowLoggingHandler.getColor()`, defaults:
    - '\x1b[33;1m': yellow+bold
    - '\x1b[31;1m': red+bold

    Note: it's better to have initialized logging.
    """
    if reason is None:
        return 0
    if not logger:
        logger = log

    if isinstance(reason, BaseException):
        color = err_color
        exit_code = -1
        logmeth = fnt.partial(logger.fatal, exc_info=True)
    else:
        color = warn_color
        exit_code = 1
        logmeth = logger.error

    if sys.stderr.isatty():
        reset = '\x1b[0m'
        reason = '%s%s%s' % (color, reason, reset)

    logmeth(reason)
    return exit_code


def build_version_string(verbose):
    v = '%s-%s' % (proj_name, proj_ver)
    if verbose:
        v_infos = collections.OrderedDict([
            ('co2mpas_version', proj_ver),
            ('co2mpas_rel_date', proj_date),
            ('co2mpas_path', osp.dirname(proj_file)),
            ('python_version', sys.version),
            ('python_path', sys.prefix),
            ('PATH', os.environ.get('PATH', None)),
        ])
        v = ''.join('%s: %s\n' % kv for kv in v_infos.items())
    return v


def print_autocompletions():
    """
    Prints the auto-completions list from docopt in stdout.

    .. Note::
        Must be registered as `setup.py` entry-point.
    """
    from . import docoptutils
    docoptutils.print_wordlist_from_docopt(__doc__)


def _cmd_modelgraph(opts):
    import co2mpas.plot as co2plot
    _init_defaults(opts['--modelconf'])
    if opts['--list']:
        print('\n'.join(co2plot.get_model_paths()))
    else:
        depth = opts['--graph-depth']
        if depth:
            try:
                depth = int(depth)
                if depth < 0:
                    depth = None
            except Exception:
                msg = "The '--graph-depth' must be an integer!  Not %r."
                raise CmdException(msg % depth)
        else:
            depth = None
        dot_graphs = co2plot.plot_model_graphs(opts['<models>'], depth=depth,
                                               output_folder=opts['-O'])
        if not dot_graphs:
            raise CmdException("No models plotted!")


def _generate_files_from_streams(
        dst_folder, file_stream_pairs, force, file_category):
    if not osp.exists(dst_folder):
        if force:
            os.makedirs(dst_folder)
        else:
            raise CmdException(
                "Destination folder '%s' does not exist!  "
                "Use --force to create it." % dst_folder)
    if not osp.isdir(dst_folder):
        raise CmdException(
            "Destination '%s' is not a <output-folder>!" % dst_folder)

    for src_fname, stream_factory in file_stream_pairs:
        dst_fpath = osp.join(dst_folder, src_fname)
        if osp.exists(dst_fpath) and not force:
            msg = "Creating %s file '%s' skipped, already exists! \n  " \
                  "Use --force to overwrite it."
            log.info(msg, file_category, dst_fpath)
        else:
            log.info("Creating %s file '%s'...", file_category, dst_fpath)
            with open(dst_fpath, 'wb') as fd:
                shutil.copyfileobj(stream_factory(), fd, 16 * 1024)


def _download_demos_stream_pairs():
    import requests
    from urllib.request import urlopen
    try:
        res = requests.get(
            'https://api.github.com/repos/JRCSTU/allinone/contents/Archive/'
            'Apps/.co2mpas-demos'
        )
        for url in sorted(v['download_url'] for v in res.json()):
            fname = osp.basename(url)
            log.info('Downloading \'%s\'...' % fname)
            yield fname, fnt.partial(urlopen, url)
    except requests.RequestException as ex:
        raise CmdException("Cannot download demo files due to: %s\n"
                           "  Check you internet connection or download them "
                           "with your browser: https://goo.gl/irbcBj" % ex)


def _cmd_demo(opts):
    dst_folder = opts['<output-folder>'] or '.'
    force = opts['--force']

    if opts['--download']:
        file_stream_pairs = _download_demos_stream_pairs()
    else:
        aio_dir = os.environ.get('AIODIR')
        cache_dir = aio_dir and osp.join(aio_dir, 'Apps', '.co2mpas-demos')
        if cache_dir:
            file_stream_pairs = [
                (osp.basename(fpath), fnt.partial(io.open, fpath, "rb"))
                for fpath in sorted(glob.glob(osp.join(cache_dir, '*.xlsx')))
            ]
        else:
            file_stream_pairs = sorted(
                _get_internal_file_streams('demos', r'.*\.xlsx$').items()
            )

    _generate_files_from_streams(dst_folder, file_stream_pairs,
                                 force, file_category='INPUT-DEMO')

    log.info("You can always download the latest demos with your browser: "
             "https://goo.gl/irbcBj")


def _cmd_ipynb(opts):
    dst_folder = opts['<output-folder>'] or '.'
    force = opts['--force']
    file_category = 'IPYTHON NOTEBOOK'
    file_stream_pairs = _get_internal_file_streams('ipynbs', r'.*\.ipynb$')
    file_stream_pairs = sorted(file_stream_pairs.items())
    _generate_files_from_streams(dst_folder, file_stream_pairs,
                                 force, file_category)


def _get_input_template_fpath():
    import pkg_resources

    fname = 'co2mpas_template.xlsx'
    return pkg_resources.resource_stream(__name__, fname)  # @UndefinedVariable


def _cmd_template(opts):
    dst_fpaths = opts['<excel-file-path>'] or ['co2mpas_template.xlsx']
    if not dst_fpaths:
        raise CmdException('Missing destination filepath for INPUT-TEMPLATE!')

    save_template(dst_fpaths, opts['--force'])


def save_template(dst_fpaths, force):
    for fpath in dst_fpaths:
        if not fpath.endswith('.xlsx'):
            fpath = '%s.xlsx' % fpath
        if osp.exists(fpath) and not force:
            raise CmdException(
                "Writing file '%s' skipped, already exists! "
                "Use --force to overwrite it." % fpath)
        if osp.isdir(fpath):
            raise CmdException(
                "Expecting a file-name instead of directory '%s'!" % fpath)

        log.info("Creating INPUT-TEMPLATE file '%s'...", fpath)
        stream = _get_input_template_fpath()
        with open(fpath, 'wb') as fd:
            shutil.copyfileobj(stream, fd, 16 * 1024)


def _get_internal_file_streams(internal_folder, incl_regex=None):
    """
    :return: a mappings of {filename--> stream-gen-function}.

    REMEMBER: Add internal-files also in `setup.py` & `MANIFEST.in` and
    update checks in `./bin/package.sh`.
    """
    import pkg_resources

    samples = pkg_resources.resource_listdir(__name__,  # @UndefinedVariable
                                             internal_folder)
    if incl_regex:
        incl_regex = re.compile(incl_regex)
    return {f: fnt.partial(pkg_resources.resource_stream,  # @UndefinedVariable
            __name__,
            osp.join(internal_folder, f))
            for f in samples
            if not incl_regex or incl_regex.match(f)}


_input_file_regex = re.compile(r'^\w')


def file_finder(xlsx_fpaths, file_ext='*.xlsx'):
    files = set()
    for f in xlsx_fpaths:
        if osp.isfile(f):
            files.add(f)
        elif osp.isdir(f):
            files.update(glob.glob(osp.join(f, file_ext)))

    return [f for f in sorted(files) if _input_file_regex.match(osp.basename(f))]


_re_override = re.compile(r"^\s*([^=]+)\s*=\s*(.*?)\s*$")


def parse_overrides(override, option_name='--override'):
    res = {}
    for ov in override:
        m = _re_override.match(ov)
        if not m:
            raise CmdException('Wrong %s format %r! ' % (option_name, ov))

        k, v = m.groups()
        if k in res:
            raise CmdException('Duplicated %s key %r!' % (option_name, k))
        res[k] = v

    return res


def _init_defaults(modelconf):
    from co2mpas.conf import defaults
    if modelconf:
        try:
            defaults.load(modelconf)
        except FileNotFoundError:
            msg = "--modelconf: No such file or directory: %s."
            raise CmdException(msg % modelconf)
    return defaults


def _run_batch(opts, **kwargs):
    input_paths = opts['<input-path>'] or ['.']
    output_folder = opts['-O']
    log.info("Processing %r --> %r...", input_paths, output_folder)
    input_paths = file_finder(input_paths)
    if not input_paths:
        cmd = 'ta' if kwargs.get('type_approval_mode') else 'batch'
        raise CmdException("Specify at least one <input-path>!"
                           "\n    read: co2mpas --help"
                           "\n  or try: co2mpas %s <input-fpath>"
                           "\n      or: co2mpas gui" % cmd)

    if not osp.isdir(output_folder):
        if opts['--force']:
            from graphviz.tools import mkdirs
            if not ''.endswith('/'):
                output_folder = '%s/' % output_folder
            mkdirs(output_folder)
        else:
            msg = ("Cannot find '%s' folder!"
                   "\n  Specify an existing folder for '-O' option.")
            raise CmdException(msg % osp.abspath(output_folder))

    _init_defaults(opts['--modelconf'])

    kw = {
        'variation': parse_overrides(opts['--override']),
        'overwrite_cache': not opts['--use-cache'],
        'modelconf': opts['--modelconf']
    }
    kw.update(kwargs)

    from co2mpas.batch import process_folder_files
    process_folder_files(input_paths, output_folder, **kw)


def _cmd_modelconf(opts):
    output_folder = opts['-O']
    if not osp.isdir(output_folder):
        if opts['--force']:
            from graphviz.tools import mkdirs
            if not ''.endswith('/'):
                output_folder = '%s/' % output_folder
            mkdirs(output_folder)
        else:
            msg = ("Cannot find '%s' folder!"
                   "\n  Specify an existing folder for the '-O' option.")
            raise CmdException(msg % osp.abspath(output_folder))

    import datetime
    fname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S-conf.yaml')
    fname = osp.join(output_folder, fname)
    defaults = _init_defaults(opts['--modelconf'])
    defaults.dump(fname)
    log.info('Default model config written into yaml-file(%s)...', fname)


def _cmd_gui(opts):
    from co2mpas import tkui
    log.warning("`co2mpas gui` cmd has been Deprecated, and will be removed in from the next release!"
                "\n  Please use the dedicated `co2gui` command to pass any command-lines options.")
    tkui.main([])


def _main(*args):
    """Throws any exception or (optionally) return an exit-code."""
    argv = args or sys.argv[1:]
    warns = []
    if '--overwrite-cache' in argv:
        argv = [v for v in argv if '--overwrite-cache' != v]
        warns.append(
            '\n`--overwrite-cache` is deprecated and non-functional! '
            'Replaced by `--use-cache`.'
        )
    opts = docopt.docopt(__doc__, argv=argv)

    verbose = opts['--verbose']
    quiet = opts['--quiet']
    if verbose and quiet:
        raise CmdException("Specify one of `verbose` and `quiet` as true!")

    level = None  # Let `init_logging()` decide.
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    init_logging(level=level, logconf_file=opts.get('--logconf'), color=True)

    if warns:
        for w in warns:
            log.warning(w)

    if opts['--version']:
        v = build_version_string(verbose)
        # noinspection PyBroadException
        try:
            sys.stdout.buffer.write(v.encode() + b'\n')
        except Exception:
            print(v)
    elif opts['template']:
        _cmd_template(opts)
    elif opts['demo']:
        _cmd_demo(opts)
    elif opts['ipynb']:
        _cmd_ipynb(opts)
    elif opts['modelgraph']:
        _cmd_modelgraph(opts)
    elif opts['modelconf']:
        _cmd_modelconf(opts)
    elif opts['gui']:
        _cmd_gui(opts)
    elif opts['ta']:
        _run_batch(opts, type_approval_mode=True, overwrite_cache=True)
    else:
        from co2mpas import co2mparable

        with co2mparable.hashing_schedula(opts['--co2mparable']):
            _run_batch(opts)


def main(*args):
    """Handles some exceptions politely and returns the exit-code."""

    if sys.version_info < (3, 5):
        return exit_with_pride(
            "Sorry, Python >= 3.5 is required, found: %s" % sys.version_info)

    try:
        return _main(*args)
    except CmdException as ex:
        log.debug('App exited due to: %r', ex, exc_info=1)
        ## Suppress stack-trace for "expected" errors but exit-code(1).
        return exit_with_pride(str(ex))
    except Exception as ex:
        ## Log in DEBUG not to see exception x2, but log it anyway,
        #  in case log has been redirected to a file.
        log.debug('App failed due to: %r', ex, exc_info=1)
        ## Print stacktrace to stderr and exit-code(-1).
        return exit_with_pride(ex)


if __name__ == '__main__':
    sys.exit(main())
