.. -*- mode: rst -*-

|Azure|_ |CirrusCI|_ |Codecov|_ |CircleCI|_ |Nightly wheels|_ |Black|_ |PythonVersion|_ |PyPi|_ |DOI|_ |Benchmark|_

.. |Azure| image:: https://dev.azure.com/scikit-learn/scikit-learn/_apis/build/status/scikit-learn.scikit-learn?branchName=main
.. _Azure: https://dev.azure.com/scikit-learn/scikit-learn/_build/latest?definitionId=1&branchName=main

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn/scikit-learn/tree/main.svg?style=shield
.. _CircleCI: https://circleci.com/gh/scikit-learn/scikit-learn

.. |CirrusCI| image:: https://img.shields.io/cirrus/github/scikit-learn/scikit-learn/main?label=Cirrus%20CI
.. _CirrusCI: https://cirrus-ci.com/github/scikit-learn/scikit-learn/main

.. |Codecov| image:: https://codecov.io/gh/scikit-learn/scikit-learn/branch/main/graph/badge.svg?token=Pk8G9gg3y9
.. _Codecov: https://codecov.io/gh/scikit-learn/scikit-learn

.. |Nightly wheels| image:: https://github.com/scikit-learn/scikit-learn/workflows/Wheel%20builder/badge.svg?event=schedule
.. _`Nightly wheels`: https://github.com/scikit-learn/scikit-learn/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/scikit-learn.svg
.. _PythonVersion: https://pypi.org/project/scikit-learn/

.. |PyPi| image:: https://img.shields.io/pypi/v/scikit-learn
.. _PyPi: https://pypi.org/project/scikit-learn

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |DOI| image:: https://zenodo.org/badge/21369/scikit-learn/scikit-learn.svg
.. _DOI: https://zenodo.org/badge/latestdoi/21369/scikit-learn/scikit-learn

.. |Benchmark| image:: https://img.shields.io/badge/Benchmarked%20by-asv-blue
.. _`Benchmark`: https://scikit-learn.org/scikit-learn-benchmarks/

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.17.3
.. |SciPyMinVersion| replace:: 1.5.0
.. |JoblibMinVersion| replace:: 1.2.0
.. |ThreadpoolctlMinVersion| replace:: 2.0.0
.. |MatplotlibMinVersion| replace:: 3.3.4
.. |Scikit-ImageMinVersion| replace:: 0.16.2
.. |PandasMinVersion| replace:: 1.0.5
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 7.1.2
.. |PlotlyMinVersion| replace:: 5.14.0

.. image:: https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png
  :target: https://scikit-learn.org/

**robpy** is a Python module for robust statistical methods built on top of scikit-learn,
SciPy and is distributed under the MIT License.

Website: URL

Installation
------------

Dependencies
~~~~~~~~~~~~

scikit-learn requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- scikit-learn (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

=======

Robpy plotting capabilities (i.e., functions start with ``plot_`` and
classes end with ``Display``) require Matplotlib (>= |MatplotlibMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| is required.
A few examples require scikit-image >= |Scikit-ImageMinVersion|, a few examples
require pandas >= |PandasMinVersion|, some examples require seaborn >=
|SeabornMinVersion| and plotly >= |PlotlyMinVersion|.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of NumPy,scikit-learn and SciPy,
the easiest way to install scikit-learn is using ``pip``::

    pip install -U scikit-learn

or ``conda``::

    conda install -c conda-forge scikit-learn

The documentation includes more detailed `installation instructions <https://scikit-learn.org/stable/install.html>`_.


Changelog
---------

See the `changelog <https://scikit-learn.org/dev/whats_new.html>`__
for a history of notable changes to scikit-learn.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: URL
- Download releases: URL
- Issue tracker: URL

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have ``pytest`` >= |PyTestMinVersion| installed)::

    pytest sklearn

See the web page URL
for more information.

    Random number generation can be controlled during testing by setting
    the ``SKLEARN_SEED`` environment variable.

Project History
---------------

The project was started in 2023 by Tim Verdonck, Jakob Raymaekers, Thomas Servotte & Thomas Decorte at the University of Antwerp, imec IDLab. See
the `About us <https://scikit-learn.org/dev/about.html#authors>`__ page
for more information.

The project is currently maintained by a the same team.


Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (development version): INSERT URL
- FAQ: INSERT URL

Citation
~~~~~~~~

If you use robpy in a scientific publication, we would appreciate citations: INSERT URL