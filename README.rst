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

Directory Organization
~~~~~~~~~~~~~~~~~~~~~~

The data directory organization is shown below, structured
``<season year>/<week>/<player position>/<source>.csv``.
Source name files contain projections from said source whereas ``STATS.csv``
contains the true realized stats. Week 0 refers to full season projections/stats.

.. code:: sh

    data
    ├── 2016
    ├── 2017
    └── 2018
        ├── 0
        ├── 1
        ├── ...
        └── 17
            ├── DST
            ├── K
            ├── QB
            ├── RB
            ├── TE
            └── WR
                ├── CBS.csv
                ├── ESPN.csv
                ├── FFToday.csv
                ├── FantasyPros.csv
                ├── NFL.csv
                ├── RTSports.csv
                ├── STATS.csv
                └── Yahoo.csv

Scraping
~~~~~~~~

Historical projections and stats as well as current projections can be scraped
with the ``scrape_data.py`` module. The following command line options are
used to specify scraping parameters.

========================== ======================= ========================
 Setting                    Command Line Keyword    Default
========================== ======================= ========================
 Sources                    -s, --sources           All Projection Sources
 Week(s) of the season      -w, --weeks             Current Week
 Season Year(s)             -y, --years             Current Season
========================== ======================= ========================

For example, scraping projections for the current week can be accomplished:

.. code:: sh

    python scrape_orderbook.py

To specify historical projections to scrape, command line options can be used.
To scrape full season projections from 2018:

.. code:: sh

    python scrape_orderbook.py -w 0 -y 2018

Similarly for all individual weeks (or specified weeks):

.. code:: sh

    python scrape_orderbook.py -w 1-17 -y 2018

All data (full season and weekly) for given years can also be scaped:

.. code:: sh

    python scrape_orderbook.py -w all -y 2016-2018

Finally, true realized stats can be scraped by specifying the source. Similarly
any individual source can be scraped.

.. code:: sh

    python scrape_orderbook.py -w all -y 2016-2018 -s STATS

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
- Jack St. Clair (`JackStC <https://github.com/JackStC>`__)

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
.. |LICENSE| image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://raw.githubusercontent.com/jason-r-becker/dfspy/master/License.txt
