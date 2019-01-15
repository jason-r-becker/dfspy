dfspy
=====

|python-Versions| |LICENSE|

``dfspy`` is a Daily Fantasy Sports lineup optimization package for Python.
 The main optimization algorithm scrapes current data from web sources,
 and uses machine learning algorithms and convex optimization to return
 optimal lineups for the current NFL week.

.. contents:: Table of contents
   :backlinks: top
   :local:

Installation
------------

Install Repo
~~~~~~~~~~~~


From terminal:

.. code:: sh

   git clone https://github.com/jason-r-becker/dfspy.git


Set up venv
~~~~~~~~~~~

Using Anaconda, from terminal:

.. code:: sh

   cd dfspy/
   conda create -n dfspy python=3.7
   source activate dfspy
   pip install -U pip
   pip install -r requirements.txt


Scraping Data
-------------

TODO

Contributions
-------------

|GitHub-Commits| |GitHub-Issues| |GitHub-PRs|

All source code is hosted on `GitHub <https://github.com/jason-r-becker/dfspy>`__.
Contributions are welcome.


LICENSE
-------

Open Source (OSI approved): |LICENSE|


Authors
-------

The main developer(s):

- Jason R Becker (`jrbecker <https://github.com/jason-r-becker>`__)
- Jack St. Clair (`JackStC <https://github.com/JackStC>`__`)

.. |GitHub-Status| image:: https://img.shields.io/github/tag/jason-r-becker/dfspy.svg?maxAge=86400
   :target: https://github.com/jason-r-becker/dfspy/releases
.. |GitHub-Forks| image:: https://img.shields.io/github/forks/jason-r-becker/dfspy.svg
   :target: https://github.com/jason-r-becker/dfspy/network
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/jason-r-becker/dfspy.svg
   :target: https://github.com/jason-r-becker/dfspy/stargazers
.. |GitHub-Commits| image:: https://img.shields.io/github/commit-activity/jason-r-becker/dfspy.svg
   :target: https://github.com/jason-r-becker/dfspy/graphs/commit-activity
.. |GitHub-Issues| image:: https://img.shields.io/github/issues-closed/jason-r-becker/dfspy.svg
   :target: https://github.com/jason-r-becker/dfspy/issues
.. |GitHub-PRs| image:: https://img.shields.io/github/issues-pr-closed/jason-r-becker/dfspy.svg
   :target: https://github.com/jason-r-becker/dfspy/pulls
.. |GitHub-Contributions| image:: https://img.shields.io/github/contributors/jason-r-becker/dfspy.svg
   :target: https://github.com/jason-r-becker/dfspy/graphs/contributors
.. |Python-Versions| image:: https://img.shields.io/badge/python-3.7-blue.svg
.. |LICENSE| image:: https://img.shields.io/pypi/l/dfspy.svg
   :target: https://raw.githubusercontent.com/jason-r-becker/dfspy/master/License.txt
