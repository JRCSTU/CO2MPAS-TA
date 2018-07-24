# -*- coding: utf-8 -*-
##
## Installs co2mpas:
## 		python setup.py install
## or
##		pip install -r requirements.txt
## and then just code from inside this folder.
#
import io
import os
import re
import sys

from setuptools import setup, find_packages


PROJECT = 'co2mpas'

if sys.version_info < (3, 5):
    sys.exit("Sorry, Python >= 3.5 is required to install %s, found: %s" %
             (sys.version_info, PROJECT))


mydir = os.path.dirname(__file__)


def read_text_lines(fname):
    with io.open(os.path.join(mydir, fname), encoding='utf-8') as fd:
        return fd.readlines()


def yield_rst_only_markup(lines):
    """
    :param file_inp:     a `filename` or ``sys.stdin``?
    :param file_out:     a `filename` or ``sys.stdout`?`

    """
    substs = [
        # Selected Sphinx-only Roles.
        #
        (r':abbr:`([^`]+)`', r'\1'),
        (r':ref:`([^`]+)`', r'ref: *\1*'),
        (r':term:`([^`]+)`', r'**\1**'),
        (r':dfn:`([^`]+)`', r'**\1**'),
        (r':(samp|guilabel|menuselection|doc|file):`([^`]+)`', r'``\2``'),

        # Sphinx-only roles:
        #        :foo:`bar`   --> foo(``bar``)
        #        :a:foo:`bar` XXX afoo(``bar``)
        #
        #(r'(:(\w+))?:(\w+):`([^`]*)`', r'\2\3(``\4``)'),
        #(r':(\w+):`([^`]*)`', r'\1(`\2`)'),
        # emphasis
        # literal
        # code
        # math
        # pep-reference
        # rfc-reference
        # strong
        # subscript, sub
        # superscript, sup
        # title-reference


        # Sphinx-only Directives.
        #
        (r'\.\. doctest', r'code-block'),
        (r'\.\. module', r'code-block'),
        (r'\.\. plot::', r'.. '),
        (r'\.\. seealso', r'info'),
        (r'\.\. glossary', r'rubric'),
        (r'\.\. figure::', r'.. '),
        (r'\.\. image::', r'.. '),

        (r'\.\. dispatcher', r'code-block'),

        # Other
        #
        (r'\|version\|', r'x.x.x'),
        (r'\|today\|', r'x.x.x'),
        (r'\.\. include:: AUTHORS', r'see: AUTHORS'),
    ]

    regex_subs = [(re.compile(regex, re.IGNORECASE), sub)
                  for (regex, sub) in substs]

    def clean_line(line):
        try:
            for (regex, sub) in regex_subs:
                line = regex.sub(sub, line)
        except Exception as ex:
            print("ERROR: %s, (line(%s)" % (regex, sub))
            raise ex

        return line

    for line in lines:
        yield clean_line(line)


polyversion = 'polyversion >= 0.2.2a0'  # Workaround buggy git<2.15, envvar: co2mpas_VERION
readme_lines = read_text_lines('README.rst')
description = readme_lines[1]
long_desc = ''.join(yield_rst_only_markup(readme_lines))

setup(
    name=PROJECT,
    ## Include a default for robustness (eg to work on shallow git -clones)
    #  but also for engraves to have their version visible.
    version='0.0.0',
    polyversion=True,
    description="The Type-Approving vehicle simulator predicting NEDC CO2 emissions from WLTP",
    long_description=long_desc,
    download_url='https://github.com/JRCSTU/CO2MPAS-TA/releases/tag/{version}',
    keywords="""
        CO2 fuel-consumption WLTP NEDC vehicle automotive
        EU JRC IET STU correlation back-translation policy monitoring
        M1 N1 simulator engineering scientific
    """.split(),
    url='https://co2mpas.io/',
    license='EUPL 1.1+',
    author='CO2MPAS-Team',
    author_email='JRC-CO2MPAS@ec.europa.eu',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: Implementation :: CPython",
        "Development Status :: 4 - Beta",
        'Natural Language :: English',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        'Environment :: Console',
        'License :: OSI Approved :: European Union Public Licence 1.1 (EUPL 1.1)',
        'Natural Language :: English',
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires='>=3.5',  # http://www.python3statement.org/practicalities/
    setup_requires=[
        # PEP426-field actually not used by `pip`, hence
        # included also in /requirements/developmnet.pip.
        'setuptools',
        'setuptools-git>=0.3',  # Example given like that in PY docs.
        'wheel',
        polyversion,
    ],
    # dev_requires=[
    #     # PEP426-field actually not used by `pip`, hence
    #     # included in /requirements/developmnet.pip.
    #     'sphinx',
    # ],
    install_requires=[
        polyversion,
        'rainbow_logging_handler',
        'pandas',
        'xlsxwriter',
        'scikit-learn',
        'numpy',
        'scipy',
        'lmfit>=0.9.7',
        'matplotlib',
        'networkx',
        'dill!=0.2.7',
        'graphviz',
        'docopt',
        'six',
        'pandalone[xlrd]>=0.2.0', # For datasync pascha-fixes and openpyxl version.
        'regex',
        'schema',
        'tqdm',
        'openpyxl>=2.4.0',
        'PyYAML>=3.12',
        'pip',
        'boltons',
        'wltp',
        'openpyxl>=2.4.0',
        'Pillow',           # for tkui
        'toolz',
        'schedula[plot]>=0.2.1',     # Fix description.
        'formulas>=0.0.10',
        'contextvars',              # for co2mpare, backported for <PY37.
        'xgboost',                  # Pure-python boost also works.

        'ipython_genutils',         # by vendorized `traitlets`
        'python-gnupg',
        ## Win+Cygwin support, new packed-ref header format
        #  (gitpython-developers/GitPython#689)
        'gitpython >= 2.1.8',
        'transitions >= 0.5.0',     # prepare/finally cbs
        'PySocks >= 1.6.7',         # more proxy-error messages (#7)
        'parsedatetime',
        'validate_email',           # dice: distinguish EWS fields
        'Unidecode',                # dice: convert non-ASCII for tstamper.
    ],
    packages=find_packages(exclude=[
        'tests', 'tests.*',
        'doc', 'doc.*',
        'benchmarks'
    ]),
    package_data={
        'co2mpas': [
            'demos/*.xlsx',
            'ipynbs/*.ipynb',
            'icons/*.png',
            'co2mpas_template.xlsx',
            'datasync_template.xlsx',
            'co2mpas_output_template.xlsx',
        ]
    },
    include_package_data=True,
    zip_safe=True,
    test_suite='nose.collector',
    tests_require=['pytest', 'nose>=1.0', 'ddt'],
    entry_points={
        'console_scripts': [
            '%(p)s = %(p)s.__main__:main' % {'p': PROJECT},
            '%(p)s-autocompletions = %(p)s.__main__:print_autocompletions' % {'p': PROJECT},
            'datasync = co2mpas.datasync:main',
            'co2dice = co2mpas.sampling.dice:main ',
            ## Note: launching as gui-scripts DOES NOT WORK
            #  and multiple console-windows flicker on each job launch.
            #  better invoke them with a Windows shortcut "minimized.
            #  Check also: https://github.com/pypa/setuptools/issues/410
            'co2gui = co2mpas.tkui:main'
        ],
    },
    options={
        'bdist_wheel': {
            'universal': True,
        },
    },
    platforms=['any'],
)
