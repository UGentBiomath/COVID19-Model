## Installation

### Get the code

- Create a [`github`](https://github.com/) account if you do not have one already.
- On the [COVID19-Model Github repository page](https://github.com/stijnvanhoey/COVID19-Model) click the `Fork` button.
- From your own repository page (your account) of the `COVID19-Model`, use [`git`](https://git-scm.com/) to download the code to your own computer. See the [Github documentation](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) on how to clone/download a repository.

When all went fine, you should have the code on your computer in a directory called `COVID19-Model`.

### Install Python (conda) and packages

To use the code, make sure you have Python (conda) and the required dependency packages installed. We recommend using `Anaconda` to manage your Python packages. See the [conda installation instructions](https://docs.anaconda.com/anaconda/install/) and make sure you have conda up and running. Next:

- Setup/update the `environment`: Dependencies are collected in the conda `environment.yml` file (inside the root folder), so anybody can recreate the required environment using:

     ```
     conda env create -f environment.yml
     conda activate COVID_MODEL
     ```
     or when the environment was already created,
     ```
     conda activate COVID_MODEL
     conda env update environment.yml
     ```
     
     See the [Anaconda navigator documentation](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment) how you can import the `environment.yml` file using the GUI.

- Install the code developed specifically for the project (lives inside the `src/covid19model` folder) in the environment (in `-e` edit mode):

     ```
     conda activate COVID_MODEL
     pip install -e .
     ```

     __Note:__ This step needs to be done in a terminal or command prompt. Use your favorite terminal or use the [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-anaconda-prompt). Navigate with the `cd` command to the directory where you copied the repository.


_Optional_: To use the code, the general installation instruction outlined above suffice. When you're planning to work on the documentation or the code of the model implementations itself, make sure to also install the development requirements:

```
pip install -e . develop
```
