from setuptools import find_packages, setup

setup(
    name='covid19model',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.1',
    description='COVID modelling package',
    author='Biomath',
    license='MIT',
    extras_require={
        "develop":  ["pytest",
                     "sphinx",
                     "numpydoc",
                     "sphinx_rtd_theme",
                     "recommonmark"],
    }
)