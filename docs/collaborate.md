## Start to collaborate

When working on the model or the model code, make sure to read the following guidelines and keep them in mind. The main purpose of being rigid about the structure of the repository is to improve the general collaboration while keeping structure of the code in the long run and support reproducibility of the work. Still, it is work in progress and it should not block your work. In case you get stuck on something or you have a suggestion for improvement, feel free to open an [New Issue](https://github.com/UGentBiomath/COVID19-Model/issues/new) on the repository.

### Using data

Data is crucial for both the setup and the evaluation of the models and all the required data is collected in the `data` directory. To organize the data sets, the `data` directory has been split up:

- A `raw` folder contains data as it has been downloaded from the original source, __without any changes__ made (neither manual or using code). For each of the data sets it is crucial to describe the source (URL, contact person,...) from which this data set was derived.
- A `interim` folder contains data as it has been adopted to become useful for the model. Use code to transform the data sets in the `raw` folder and store the result in the `interim` folder. The script or function to do the transformation is kept in the `src/covid19model/data/` folder.

__Remember:__ Don't ever edit the `raw` data, especially not manually, and especially not in Excel. Don't overwrite your raw data. Don't save multiple versions of the raw data. Treat the data (and its format) as immutable. The code you write should move the raw data through a pipeline to your final analysis.

To make sure the `data` directory does not become an unstructured set of data files from which no one knows the origin, the following guidelines apply to all data added to the `data` directory:

- All `raw` data files are stored as the downloaded data file.
- All `interim` data file names are written in lowercase, without spaces (use `_` instead).
- When a (new) data set has been downloaded from a source, store it as such in the `raw` directory and document the origin in the `data/README.md` file.
- The functions to prepare data sets are stored in the `src/covid19model/data/` folder. Add the function to the `data/README.md`./data/README.md document to define on which raw files the function operates and which `interim` files are created by it.

### Notebooks are for exploration and communication

Since notebooks are challenging objects for version control (e.g., diffs of the json are often not human-readable and merging is near impossible), we recommended not collaborating directly with others on Jupyter notebooks. There are two steps we recommend for using notebooks effectively:

- Follow a naming convention that shows the owner and the order the analysis was done in. We propose the format <step>-<ghuser>-<description>.ipynb (e.g., 0.3-twallema-model-network.ipynb).
- Reuse the good parts. Don't write code to do the same task in multiple notebooks. If it's a data preprocessing task, put it in the pipeline at `src/covid19model/data/make_dataset.py` and load data from `data/interim`. If it's useful utility code, refactor it and put it in the appropriate subfolder of the `src/covid19model` folder, e.g. visualisations inside `src/covid19model/visualization`

As the code of the `src/covid19model` folder is a Python package itself (see the `setup.py` file). You can import your code and use it in notebooks without the need of reinstallation. Put the following at the top of your notebook:

```
# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload
# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
%autoreload 2
from covid19model.models import ...
```

__Note:__ Use one of the `notebook/templates` to get started. You can run these online using [Binder](https://mybinder.org/v2/gh/UGentBiomath/COVID19-Model/master?filepath=notebook/templates). To use them locally, copy paste one of the templates to the general notebooks directory, rename it according to the defined format and start working on it.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UGentBiomath/COVID19-Model/master)

### Documentation website

Documentation consists of both the technical matter about the code as well as background information on the models. To keep these up to date and centralized, we use [Sphinx](https://www.sphinx-doc.org/en/master/) which enables us to keep the documentation together on a website.

The Sphinx setup provides the usage of both `.rst` file, i.e. [restructuredtext](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html) as well as `.md` files, i.e. [Markdown](https://www.markdownguide.org/basic-syntax/). The latter is generally experienced as easier to write, while the former provides more advanced functionalities. Existing pages can be adjusted directly (editing them online or on your computer). When you want to create a new page, make sure to add the page to the `index.rst` in order to make the page part of the website.

The website is build automatically using [github actions](https://github.com/UGentBiomath/COVID19-Model/blob/master/.github/workflows/deploy.yml#L22-L24) and the output is deployed to [https://ugentbiomath.github.io/COVID19-Model/](https://ugentbiomath.github.io/COVID19-Model/). In case you want to build the documentation locally, make sure you have the development dependencies installed (`pip install -e ".[develop]"`) to run the sphinx build script. The build sphinx script relies on the [`setuptools` integration of Sphinx](https://www.sphinx-doc.org/en/master/usage/advanced/setuptools.html#setuptools-integration):

```
python setup.py build_sphinx
```

The resulting html-website is created in the directory `build/html`. Double click any of the `html` file in the folder to open the website in your browser (no server required).

### The `covid19model` Python package

The code inside the `src/covid19model` directory is actually a Python package, which provides a number of additional benefits on the maintenance of the code.

Before doing any changes, always make sure your own version of your code (i.e. `fork`) is up to date with the `master` of the [main repository ](https://github.com/UGentBiomath/COVID19-Model). First, check out the __[git workflow](./git_workflow.md)__ for a step-by-step explanation of the proposed workflow. For more information, see also:
- If you are a command line person, check [this workflow](https://gist.github.com/CristinaSolana/1885435)
- If you are not a command line person: [this workflow](https://www.sitepoint.com/quick-tip-sync-your-fork-with-the-original-without-the-cli/) can help you staying up to date.

For each of the functions you write, make sure to add the documentation to the function. We use the [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) format to write documentation. For each function, make sure the following items are defined at least:

- Short summary (top line)
- Parameters
- Returns
- References (if applicable)

__Note:__ When adding new packages makes sure to update both,
- the environment file, [evironment.yml](https://github.com/UGentBiomath/COVID19-Model/blob/master/environment.yml) for binder,
- the setup file, [setup.py](https://github.com/UGentBiomath/COVID19-Model/blob/master/setup.py) file to include this dependency for the installation of the package.

### Repository layout overview

As the previous sections described, each subfolder of the repository has a specific purpose and we would ask to respect the general layout. Still, this is all work in progress, so alterations to it that improve the workflow are certainly possible. Please do your suggestion by creating a [New issue](https://github.com/UGentBiomath/COVID19-Model/issues/new/choose).

__Remember:__ Anyone should be able to reproduce the final products with only the `code` in `src` and the data in `data/raw`!

#### data
```
├── data                                    <- cfr. current set of data sets
│   ├── raw                                 <- data as provided in raw format
│   │   ├── UZGent
│   │   └── economical
│   │   └── google
│   │   └── model_parameters
│   │   └── polymod
│   │   └── sciensano
│   ├── interim                             <- data sets after speicific manipulation
│   │   ├── ...
│   └── README.md                           <- for each data set: what, how to get,...
```

#### code
```
├── src                                     <- all reusable code blocks
│   ├── covid19model
|   │   ├── data                            <- any code required for data reading and manipulation
|   │   ├── models                          <- any code constructing the models
|   │   ├── optimization                    <- code related to parameter callibration
|   │   ├── visualization                   <- code for making figures
|   │   └── __init__.py                     <- structured as lightweight python package
│   ├── tests
|   │   ├── ... .py                         <- all test code during development
```

#### documentation
```
├── docs                                    <- documentation
│   ├── conf.py
│   ├── index.rst                           <- explanations are written inside markdown or st files
│   ├── ... .md                             <- pages of the documentation website
│   ├── Makefile
│   └── _static
│   │   └── figs                            <- figs linked to the documentation
│   │       ├── ...
│   │       └── SEIRSNetworkModel.jpg
```

#### HPC specific code
```
├── hpc
│   ├── calibrate_stochastic.py
│   ├── README.md
│   ├── sim_stochastic.py
│   └── test.sh
```

#### notebooks
```
├── notebooks                               <- notebooks are collected here
│   ├── 0.1-user-content.ipynb              <- notebook per version-user-content
│   ├── templates                           <- demo notebooks that can be used as starting point (and binder intro)
│       ├── SEIRSAgeModel_demo.ipynb
│       └── SEIRSNetworkModel_demo.ipynb
│   ├── scratch                             <- test notebooks

├── reports                                 <- optional (e.g. report from automatic daily rerun)
│   └── figures
```

#### automate stuff
```
├── .github                                 <- Automate specific steps with github actions
│   ├── workflows
│   │   ├── deploy.yml
│   │   └── ...
```

#### other info
```
├── LICENSE
├── environment.yml
├── setup.py
└── README.md                               <- focus on how to get started, setup environment name conventions and
```
