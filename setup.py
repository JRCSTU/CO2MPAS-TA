#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
CO2MPAS setup.
"""
import io
import os
import collections
import os.path as osp

name = 'co2mpas'
mydir = osp.dirname(__file__)


def read_project_version():
    """
    Version-trick to have version-info in a single place,
    taken from: http://stackoverflow.com/questions/2058802/how-can-i-get-the-
    version-defined-in-setup-py-setuptools-in-my-package

    :return:
        Project version.
    :rtype: str
    """
    fglobals = {}
    with io.open(osp.join(mydir, name, '_version.py'), encoding='UTF-8') as fd:
        exec(fd.read(), fglobals)  # To read __version__
    return fglobals['__version__']


# noinspection PyPackageRequirements
def get_long_description(cleanup=True):
    """
    Return the project long description.

    :param cleanup:
        Clean up the build folder.
    :type cleanup: bool

    :return:
        Project long description.
    :rtype: str
    """
    import shutil
    import tempfile
    from doc.conf import extensions
    from sphinx.application import Sphinx
    from sphinx.util.osutil import abspath
    from sphinxcontrib.writers.rst import RstTranslator
    from sphinx.ext.graphviz import text_visit_graphviz
    RstTranslator.visit_dsp = text_visit_graphviz
    outdir = tempfile.mkdtemp(prefix='setup-', dir='.')
    exclude_patterns = os.listdir(mydir or '.')
    exclude_patterns.remove('pypi.rst')

    # noinspection PyTypeChecker
    app = Sphinx(
        abspath(mydir), osp.join(mydir, 'doc/'), outdir, outdir + '/.doctree',
        'rst', confoverrides={
            'exclude_patterns': exclude_patterns,
            'master_doc': 'pypi',
            'dispatchers_out_dir': abspath(outdir + '/_dispatchers'),
            'extensions': extensions + ['sphinxcontrib.restbuilder']
        }, status=None, warning=None)

    app.build(filenames=[osp.join(app.srcdir, 'pypi.rst')])

    with open(outdir + '/pypi.rst') as file:
        res = file.read()

    if cleanup:
        shutil.rmtree(outdir)
    return res


if __name__ == '__main__':
    import functools
    from setuptools import setup, find_packages

    proj_ver = read_project_version()
    url = 'https://github.com/JRCSTU/%s-ta' % name
    download_url = '%s/tarball/v%s' % (url, proj_ver)
    project_urls = collections.OrderedDict((
        ('Documentation', 'http://%s.readthedocs.io' % name),
        ('Issue tracker', '%s/issues' % url),
    ))

    long_description = ''
    if os.environ.get('ENABLE_SETUP_LONG_DESCRIPTION') == 'TRUE':
        try:
            long_description = get_long_description()
            print('LONG DESCRIPTION ENABLED!')
        except Exception as ex:
            print('LONG DESCRIPTION ERROR:\n %r', ex)

    extras = {
        'cli': ['click', 'click-log'],
        'sync': ['syncing>=1.0.4', 'pandas>=0.21.0'],
        'plot': ['flask', 'regex', 'graphviz', 'Pygments', 'lxml',
                 'beautifulsoup4', 'jinja2', 'docutils', 'plotly'],
        'io': ['pandas>=0.21.0', 'dill', 'regex', 'xlref', 'xlrd', 'asteval']
    }
    extras['dice'] = ['co2mpas_dice>=4.0.5'] + extras['io']
    # noinspection PyTypeChecker
    extras['all'] = list(functools.reduce(set.union, extras.values(), set()))
    extras['dev'] = extras['all'] + [
        'wheel', 'sphinx', 'gitchangelog', 'mako', 'sphinx_rtd_theme',
        'setuptools>=36.0.1', 'sphinxcontrib-restbuilder', 'nose', 'coveralls',
        'ddt', 'sphinx-click'
    ]

    setup(
        name=name,
        version=proj_ver,
        packages=find_packages(exclude=[
            'test', 'test.*',
            'doc', 'doc.*',
            'appveyor'
        ]),
        url=url,
        project_urls=project_urls,
        download_url=download_url,
        license='EUPL 1.1+',
        author='CO2MPAS-Team',
        author_email='JRC-CO2MPAS@ec.europa.eu',
        description='The Type-Approving vehicle simulator predicting NEDC CO2 '
                    'emissions from WLTP',
        long_description=long_description,
        keywords="""CO2 fuel-consumption WLTP NEDC vehicle automotive EU JRC IET 
        STU correlation back-translation policy monitoring M1 N1 simulator 
        engineering scientific
        """.split(),
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: Implementation :: CPython",
            "Development Status :: 4 - Beta",
            'Natural Language :: English',
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Manufacturing",
            'Environment :: Console',
            'License :: OSI Approved :: European Union Public Licence 1.1 '
            '(EUPL 1.1)',
            'Natural Language :: English',
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: OS Independent",
            'Topic :: Scientific/Engineering',
            "Topic :: Scientific/Engineering :: Information Analysis",
        ],
        obsoletes=['co2mpas (< 4.0)'],
        python_requires='>=3.5',
        install_requires=[
            'PyYAML',
            'schedula>=0.3.2',
            'tqdm',
            'scikit-learn',
            'regex',
            'lmfit>=0.9.7',
            'numpy',
            'schema',
            'scipy',
            'wltp',
            'xgboost>=0.90',
            'statsmodels'
        ],
        entry_points={
            'console_scripts': [
                '%(p)s = %(p)s.cli:cli' % {'p': name},
            ],
        },
        extras_require=extras,
        tests_require=['nose>=1.0', 'ddt'],
        test_suite='nose.collector',
        package_data={
            'co2mpas': [
                'demos/*.xlsx',
                'templates/*.xlsx'
            ]
        },
        zip_safe=True,
        options={'bdist_wheel': {'universal': True}},
        platforms=['any'],
    )
