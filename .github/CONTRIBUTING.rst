Contributing to covid19model
============================

.. _code repository: https://github.com/UGentBiomath/COVID19-Model
.. _new issue: https://github.com/UGentBiomath/COVID19-Model/issues/new
.. _issue list: https://github.com/UGentBiomath/COVID19-Model/issues
.. _documentation: https://github.com/UGentBiomath/COVID19-Model
.. _Sphinx: http://www.sphinx-doc.org/en/master/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _propose a file change: https://help.github.com/articles/editing-files-in-another-user-s-repository/
.. _report an issue: https://github.com/UGentBiomath/COVID19-Model/issues/new
.. _numpy docstring standard: https://numpydoc.readthedocs.io/en/latest/format.html


First of all, thanks for considering contributing to the COVID-19! It's people like you that make it
rewarding for us, the project authors, to work on COVID-19 model.

How you can contribute?
-----------------------

There are several ways you can contribute to this project.

Share the love
^^^^^^^^^^^^^^

Think COVID-19 modelling is useful? Let others discover it by telling them in person, via Twitter_ or a blog post.

.. _Twitter: https://twitter.com/biomathugent

Using COVID-19 model code for a paper you are writing? Consider citing it:

    #TODO

Ask a question ⁉️
^^^^^^^^^^^^^^^^^

Using COVID-19 modelling code and got stuck? Browse the documentation_ to see if you
can find a solution. Still stuck? Post your question as a `new issue`_ on GitHub.
While we cannot offer user support, we'll try to do our best to address it,
as questions often lead to better documentation, new insights or the discovery of bugs.


Propose an idea
^^^^^^^^^^^^^^^^

Have an idea for COVID-19 modelling? Take a look at the documentation_ and
`issue list`_ to see if it isn't included or suggested yet. If not, suggest
your idea as a `new issue`_ on GitHub. While we can't promise to implement
your idea or do your model exercise, it helps to:

* Explain in detail how it would work.
* Keep the scope as narrow as possible.

See :ref:`dev-guidelines`,  if you want to contribute code for your idea as well.

Report a bug
^^^^^^^^^^^^

Using COVID-19 modelling code and discovered a bug? That's annoying! Don't let others have the
same experience and report it as a `new issue`_ so we can fix it. A good bug
report makes it easier for us to do so, so please include:

* Your operating system name and version (e.g. Mac OS 10.13.6).
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

The issue template will provide some guidance on the required information.

Improve the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Noticed a typo on the website? Think a function could use a better example?
Good documentation makes all the difference, so your help to improve it is very
welcome! Maybe you've written a good introduction tutorial or example model case,
these are typically very popular sections for other users.

**The website**

`This website <documentation>`_ is generated with Sphinx_. That means we don't have to
write any html. Content is pulled together from documentation in the code,
reStructuredText_ files and the package ``conf.py`` settings. If you
know your way around *Sphinx*, you can `propose a file change`_ to improve
documentation. If not, report a `new issue`_ and we can point you in the right direction.

**Function documentation**

Functions are described as comments near their code and translated to
documentation using the  `numpy docstring standard`_. If you want to improve a
function description:

1. Go to ``covid19model/`` directory in the `code repository`_.
2. Look for the file with the name of the function.
3. `Propose a file change`_ to update the function documentation in the docstring (in between the triple quotes).


Contribute code
^^^^^^^^^^^^^^^

Care to fix bugs or implement new functionalities? Awesome! Have a
look at the `issue list`_ and leave a comment on the things you want
to work on. See also the development guidelines below.

.. _dev-guidelines:

Development guidelines
-----------------------

Coding guidelines
^^^^^^^^^^^^^^^^^^

The following are some guidelines on how new code should be written. Of course,
there are special cases and there will be exceptions to these rules. However,
following these rules when submitting new code makes the review easier so new
code can be integrated in less time.

Uniformly formatted code makes it easier to share code ownership. The
pydov project tries to closely follow the official Python guidelines
detailed in `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ which detail
how code should be formatted and indented. Please read it and follow it.

In addition, we add the following guidelines:

* Use underscores to separate words in non class names: ``n_samples`` rather than ``nsamples``.
* Avoid multiple statements on one line. Prefer a line return after a control flow statement (\ ``if/for``\ ).
* Please don’t use ``import *`` in any case. It is considered harmful by the official Python recommendations. It makes the code harder to read as the origin of symbols is no longer explicitly referenced, but most important, it prevents using a static analysis tool like pyflakes to automatically find bugs.
* Use the `numpy docstring standard`_ in all your docstrings.
* The attributes for specific classes are Pandas data.frames, please use lowercase names (eventually with `_`) as column names.
