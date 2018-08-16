#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script *polyversion-lib*."""
from __future__ import print_function

import re

from setuptools import setup

import os.path as osp

mydir = osp.dirname(osp.realpath(__file__))


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

test_requirements = [
    'pytest',
    'pytest-runner',
    'pytest-cov',
    'flake8',
    'flake8-builtins',
    'flake8-mutable',
    #'mypy',
]
PROJECT = 'polyversion'


setup(
    name=PROJECT,
    ## Provide a `default_version` for installing eg. in shallow clones,
    #  and `pname` or else it would be `__main__`.
    version='0.0.0',
    polyversion=True,
    description="Polyvers's lib to derive subproject versions from tags on Git monorepos.",
    long_description=long_desc,
    author="Kostis Anagnostopoulos",
    author_email='ankostis@gmail.com',
    url='https://github.com/jrcstu/polyvers',
    download_url='https://pypi.org/project/polyversion/',
    project_urls={
        'Documentation':
        'http://polyvers.readthedocs.io/en/latest/usage-pvlib.html',
        'Source': 'https://github.com/jrcstu/polyvers',
        'Tracker': 'https://github.com/jrcstu/polyvers/issues',
    },
    ## The ``package_dir={'': <sub-dir>}`` arg is needed for `py_modules` to work
    #  when packaging sub-projects. But ``<sub-dir>`` must be relative to launch cwd,
    #  or else, ``pip install -e <subdir>`` and/or ``python setup.py develop``
    #  break.
    #  Also tried chdir(mydir) at the top, but didn't work.
    package_dir={'': osp.relpath(mydir)},
    # packages=find_packages(osp.realpath(osp.join(mydir, 'polyversion')),
    #                        exclude=['tests', 'tests.*']),
    packages=['polyversion'],
    license='MIT',
    zip_safe=True,
    platforms=['any'],
    keywords="version-management configuration-management versioning "
             "git monorepo tool library".split(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    #python_requires='>=3.6',
    setup_requires=['setuptools', 'wheel', 'polyversion >= 0.2.0a2'],
    tests_require=test_requirements,
    extras_require={
        'test': test_requirements,
    },
    entry_points={
        'distutils.setup_keywords': [
            'polyversion = polyversion.setuplugin:init_plugin_kw',
            'polyversion_check_bdist_enabled = polyversion.setuplugin:check_bdist_kw',
        ],
        'console_scripts':
            ['%(p)s = %(p)s.__main__:main' % {'p': PROJECT}]
    },
)
