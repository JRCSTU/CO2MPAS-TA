#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import re
import sys
from os import path as osp

from setuptools import find_packages, setup

PROJECT = 'co2sim'

mydir = osp.dirname(osp.abspath(__file__))
os.chdir(mydir)


def read_text_lines(fname):
    with io.open(osp.join(mydir, fname), encoding='utf-8') as fd:
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


def read_pinned_deps(fpath):
    comment_regex = re.compile('^ *#')
    rstrip_regex = re.compile(' *(#.*)?$')

    def procline(line):
        line = line.strip()
        if line and not comment_regex.match(line):
            return line

        return rstrip_regex.sub('', line)

    pinned_deps = []
    with open(fpath) as fp:
        for line in fp:
            if 'CO2MPAS PINNED STOP' in line:
                break

            line = procline(line)
            if line:
                pinned_deps.append(line)

    return pinned_deps



polyver = 'polyversion >= 0.2.2a0'  # Workaround buggy git<2.15, envvar: co2mpas_VERION
readme_lines = read_text_lines('README.rst')
description = readme_lines[1]
long_desc = ''.join(yield_rst_only_markup(readme_lines))
pindeps = read_pinned_deps(osp.join(mydir, 'requirements', 'exe.pip'))

test_requirements = [
    'pytest',
    'pytest-runner',
    'flake8',
    'flake8-builtins',
    'flake8-mutable',
    #'mypy',
    'ddt',
]

setup(
    name=PROJECT,
    ## Include a default for robustness (eg to work on shallow git -clones)
    #  but also for engraves to have their version visible.
    version='3.0.0',
    polyversion=True,
    description="The Type-Approving vehicle simulator predicting NEDC CO2 emissions from WLTP",
    long_description=long_desc,
    download_url='https://pypi.org/project/co2sim/',
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
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
    obsoletes=['co2mpas (< 2.0)'],
    python_requires='>=3.5',
    setup_requires=[
        # PEP426-field actually not used by `pip`, hence
        # included also in /requirements/developmnet.pip.
        'setuptools',
        'setuptools-git>=0.3',  # Example given like that in PY docs.
        'wheel',
        polyver,
    ],
    install_requires=[
        polyver,
        'colorama',
        'formulas>=0.0.10',
        'lmfit>=0.9.7',
        'logutils',
        'numpy',
        'pandas>=0.21.0',       # `axis` kw added in e.g. reindex()
        'regex',
        'schedula[plot]>=0.2.3',
        'schema',
        'scikit-learn',
        'scipy',
        'wltp',
        'xgboost',                  # Pure-python boost would be also ok.
    ],
    extras_require={
        'io': [
            'boltons',
            'contextvars',              # for co2mpare, backported for <PY37
            'cryptography',
            'dill!=0.2.7',
            'docopt',
            'openpyxl>=2.4.0',
            'pandalone[xlrd]>=0.2.0',   # for datasync pascha-fixes and openpyxl version
            'pip',
            'PyYAML>=3.12',
            'toolz',
            'tqdm',
            'xlsxwriter',
        ],
        'plot': [
            'matplotlib',
            'schedula[plot]',
        ],
        'test': test_requirements,
        # Not working yet, due to: https://github.com/pypa/pip/pull/3878
        'pindeps': pindeps,
    },
    tests_require=test_requirements,
    package_dir={'': 'src'},
    packages=find_packages('src'),
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'co2mpas = co2mpas.__main__:main',
            'datasync = co2mpas.datasync:main',
        ],
    },
    package_data={
        'co2mpas': [
            'demos/*.xlsx',
            'ipynbs/*.ipynb',
            'co2mpas_template.xlsx',
            'datasync_template.xlsx',
            'co2mpas_output_template.xlsx',
        ]
    },
    zip_safe=True,
    options={'bdist_wheel': {'universal': True}},
    platforms=['any'],
)
