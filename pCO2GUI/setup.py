#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import re

from setuptools import find_packages, setup


mydir = osp.dirname(osp.abspath(__file__))
os.chdir(mydir)


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
        (r':(\w+):`([^`]*)`', r'\1(`\2`)'),
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
        (r'\.\. currentmodule::', r'currentmodule:'),
        (r'\.\. plot::', r'.. plot:'),
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


with open(osp.join(mydir, 'README.rst')) as readme_file:
    readme = readme_file.readlines()

long_desc = ''.join(yield_rst_only_markup(readme))
polyver = 'polyversion >= 0.2.2a0'
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
    name='co2gui',
    ## Provide a `default_version` for installing eg. in shallow clones,
    #  and `pname` or else it would be `__main__`.
    version='2.0.0',
    polyversion=True,
    description="Polyvers's lib to derive subproject versions from tags on Git monorepos.",
    long_description=long_desc,
    license='EUPL 1.1+',
    author='CO2MPAS-Team',
    author_email='JRC-CO2MPAS@ec.europa.eu',
    url='https://co2mpas.io/',
    download_url='https://pypi.org/project/co2gui/',
    project_urls={
        'Documentation': 'Https://co2mpas.io',
        'Source': 'https://github.com/JRCSTU/CO2MPAS-TA',
        'Tracker': 'https://github.com/JRCSTU/CO2MPAS-TA/issues',
    },
    keywords="""
        CO2 fuel-consumption WLTP NEDC vehicle automotive
        EU JRC IET STU correlation back-translation policy monitoring
        M1 N1 simulator engineering scientific
    """.split(),
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
    setup_requires=['setuptools', 'wheel', polyver],
    install_requires=[
        polyver,
        'co2sim[io]',
        'co2dice',                  # FIXME: retrofit to extra
        'Pillow',                   # for About panel
        'numpy',                    # just for disabling its logging
        'pandalone[xlrd]>=0.2.0',   # For datasync pascha-fixes and openpyxl version.
        'schema',                   # used only for exception checking.
        'PyYAML>=3.12',             # tooltips, logconf
        'boltons',                  # for its sorted set
        'toolz',
        'schedula[plot]>=0.2.3',    # for DispatcherAbort & status-dispatcher.
    ],
    tests_require=test_requirements,
    extras_require={
        'test': test_requirements,
    },
    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data={
        'co2gui': [
            'icons/*.png',
        ]
    },
    test_suite='tests',
    entry_points={
        'console_scripts':
            ## Note: launching as gui-scripts DOES NOT WORK
            #  and multiple console-windows flicker on each job launch.
            #  better invoke them with a Windows shortcut "minimized.
            #  Check also: https://github.com/pypa/setuptools/issues/410
            'co2gui = co2gui.__main__:main'
    },
    zip_safe=True,
    options={'bdist_wheel': {'universal': True}},
    platforms=['any'],
)
