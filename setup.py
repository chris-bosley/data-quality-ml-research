#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Chris Bosley",
    author_email='chrisbosley@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Project files and data to support impact of degrading data quality on existing ML models.",
    entry_points={
        'console_scripts': [
            'data_quality_ml_research=data_quality_ml_research.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='data_quality_ml_research',
    name='data_quality_ml_research',
    packages=find_packages(include=['data_quality_ml_research', 'data_quality_ml_research.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/c-bosley/data_quality_ml_research',
    version='0.1.0',
    zip_safe=False,
)
