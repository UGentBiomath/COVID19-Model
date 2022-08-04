from setuptools import find_packages, setup

setup(
    name='covid19model',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.1',
    description='COVID-19 modelling package',
    author='Tijs Alleman, BIOMATH, Ghent University',
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'xlrd',
	    'openpyxl',
        'zarr',
        'emcee',
        'xarray',
        'rbfopt',
        'numba'
        'SAlib'
    ],
    extras_require={
        "develop":  ["pytest",
                     "sphinx",
                     "numpydoc",
                     "sphinx_rtd_theme",
                     "myst_parser[sphinx]"],
    }
)
