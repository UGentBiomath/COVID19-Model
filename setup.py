from setuptools import find_packages, setup

setup(
    name='covid19model',
    packages=find_packages(exclude=["*.tests"]),
    version='0.1',
    description='COVID modelling package.',
    author='Biomath',
    license='MIT',
)