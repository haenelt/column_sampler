# -*- coding: utf-8 -*-

import setuptools


INSTALL_REQUIREMENTS = ['numpy==1.20.3',
                        'nibabel==3.2.1',
                        'networkx==2.5',
                        'scipy==1.5.2',
                        'matplotlib==3.4.2',
                        'pyvista==0.31.3',
                        ]

CLASSIFIERS = ["Programming Language :: Python :: 3.8",
               "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
               "Operating System :: OS Independent",
               "Development Status :: 3 - Alpha",
               "Intended Audience :: Science/Research",
               "Topic :: Scientific/Engineering",
               ]

with open("VERSION", "r", encoding="utf8") as fh:
    VERSION = fh.read().strip()

with open("README.md", "r", encoding="utf8") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="column_sampler",
    version=VERSION,
    author="Daniel Haenelt",
    author_email="daniel.haenelt@gmail.com",
    description="Analyze cortical columns on a surface mesh",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/haenelt/column_sampler",
    license='GPL v3',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
    classifiers=CLASSIFIERS,
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
    )
