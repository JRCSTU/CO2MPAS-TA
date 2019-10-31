Contributing to co2mpas
=======================

If you want to contribute to |co2mpas| and improve it, your help is very
welcome. Your contribution should be sent by a *pull request*. Next sections
explains how to implement and submit a new functionality:

- clone the repository
- implement a new functionality
- open a pull request

Clone the repository
--------------------
The first step to contribute to |co2mpas| is to clone the repository:

- Create a personal `fork <https://help.github.com/articles/fork-a-repo/
  #fork-an-example-repository>`_ of the `co2mpas <https://github.com/
  JRCSTU/co2mpas-ta>`_ repository on Github.
- `Clone <https://help.github.com/articles/fork-a-repo/
  #step-2-create-a-local-clone-of-your-fork>`_ the fork on your local machine.
  Your remote repo on Github is called ``origin``.
- `Add <https://help.github.com/articles/fork-a-repo/#step-3-configure-git-to
  -sync-your-fork-with-the-original-spoon-knife-repository>`_
  the original repository as a remote called ``upstream``, to maintain updated
  your fork.
- If you created your fork a while ago be sure to pull ``upstream`` changes into
  your local repository.
- Create a new branch to work on! Branch from ``dev``.

How to implement a new functionality
------------------------------------
Test cases are very important. This library uses a data-driven testing approach.
To implement a new function I recommend the `test-driven development cycle
<https://en.wikipedia.org/wiki/Test-driven_development
#Test-driven_development_cycle>`_. Hence, when you think that the code is ready,
add new test in ``test`` folder.

When all test cases are ok (``python setup.py test``), open a pull request.

.. note:: A pull request without new test case will not be taken into
   consideration.

How to open a pull request
--------------------------
Well done! Your contribution is ready to be submitted:

- Squash your commits into a single commit with git's
  `interactive rebase <https://help.github.com/en/github/using-git/about-git-rebase>`_.
  Create a new branch if necessary. Always write your commit messages in the
  present tense. Your commit message should describe what the commit, when
  applied, does to the code – not what you did to the code.
- `Push <https://help.github.com/articles/pushing-to-a-remote/>`_ your branch to
  your fork on Github (i.e., ``git push origin dev``).
- From your fork `open <https://help.github.com/articles/creating-a-pull-
  request-from-a-fork/>`_ a *pull request* in the correct branch.
  Target the project's ``dev`` branch!
- Once the *pull request* is approved and merged you can pull the changes from
  ``upstream`` to your local repo and delete your extra branch(es).

.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS

