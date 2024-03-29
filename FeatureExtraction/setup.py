#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages

from setuptools import setup


with open('README.rst', 'rt') as readme_file:
    readme = readme_file.read()


def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv('CIRCLE_BRANCH') in {'master'}:
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='CODEX_FTX',
    use_scm_version={'local_scheme': prerelease_local_scheme},
    description='Channel-level intensity statistics for nuclei in multi-frame CODEX images',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Sam Border',
    author_email='samuel.border@medicine.ufl.edu',
    url='https://github.com/spborder/CODEX_FeatureExtraction/',
    packages=find_packages(exclude=['tests', '*_test']),
    package_dir={
        'CODEX_FTX': 'CODEX_FTX',
    },
    include_package_data=True,
    install_requires=[
        # scientific packages
        'nimfa>=1.3.2',
        'numpy==1.24.3',
        'scipy==1.10.1',
        'Pillow==9.5.0',
        'pandas==2.0.3',
        'imageio==2.29.0',
        'opencv-python==4.8.0.76',
        'scikit-image==0.20.0',
        'tqdm==4.65.0',
        'openpyxl==3.1.2',
        # dask packages
        'dask==2023.9.0',
        'girder-slicer-cli-web',
        'girder-client',
        'matplotlib',
        # cli
        'ctk-cli',
        'wsi-annotations-kit'
    ],
    license='Apache Software License 2.0',
    keywords='CODEX_FTX',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)