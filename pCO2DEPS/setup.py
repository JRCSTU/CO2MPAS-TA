#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from setuptools import setup, find_packages

import os.path as osp
from polyversion import polyversion

mydir = osp.dirname(osp.realpath(__file__))


with open(osp.join(mydir, 'README.rst')) as rfp:
    readme_lines = rfp.readlines()

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


long_desc = ''.join(readme_lines)
polyver = 'polyversion >= 0.2.2a0'
pindeps = read_pinned_deps(osp.join(
    mydir, '..', 'pCO2SIM', 'requirements', 'exe.pip'))

setup(
    name='co2deps',
    ## Provide a `default_version` for installing eg. in shallow clones,
    #  and `pname` or else it would be `__main__`.
    version='0.0.0',
    polyversion=True,
    description=readme_lines[1],
    long_description=long_desc,
    license='EUPL 1.1+',
    author='CO2MPAS-Team',
    author_email='JRC-CO2MPAS@ec.europa.eu',
    url='https://co2mpas.io/',
    download_url='https://pypi.org/project/co2deps/',
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
    setup_requires=['setuptools', 'wheel', polyver],
    install_requires=[
        polyver,
        'co2sim==%s' % polyversion(pname='co2sim'),
        pindeps,
    ],
    zip_safe=True,
    options={'bdist_wheel': {'universal': True}},
    platforms=['any'],
)
