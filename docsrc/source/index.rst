.. regtools documentation master file, created by
   cookiecutter-pypi-sphinx.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Regression Tools documentation!
********************************************************************

High-level Regression Utilities

To get started, look here.

.. toctree::
   :caption: Tutorial

   tutorial
   auto_examples/index

An overview of ``regtools``
=============================

This package makes it easier to run various kinds of regressions. Handles
fixed effects, 2+ way clustering, hypothesis testing,
lagged variables, differenced variables, interaction effects,
iteration tools, and producing summaries for a
variety of models including OLS, Logit, Probit, Quantile, and Fama-Macbeth.

Quick Links
------------

Find the source code `on Github <https://github.com/nickderobertis/regtools>`_.


Single Regressions
--------------------

.. autosummarynameonly::

      regtools.reg.reg
      regtools.linmodels.reg.linear_reg
      regtools.differenced.diff_reg
      regtools.quantile.quantile_reg

Produce Summary
----------------

.. autosummarynameonly::

      regtools.summarize.produce_summary

Iteration Tools
-----------------

.. autosummarynameonly::

      regtools.iter.reg_for_each_combo
      regtools.iter.reg_for_each_xvar_set
      regtools.iter.reg_for_each_combo_select_and_produce_summary
      regtools.iter.reg_for_each_xvar_set_and_produce_summary
      regtools.iter.reg_for_each_yvar
      regtools.iter.reg_for_each_yvar_and_produce_summary
      regtools.iter.reg_for_each_lag
      regtools.iter.reg_for_each_lag_and_produce_summary

Hypothesis Testing
--------------------

.. autosummarynameonly::

      regtools.select.select_models

Selecting Models
-------------------

.. autosummarynameonly::

      regtools.hypothesis.lincom.hypothesis_test


API Docs
-----------

.. toctree:: api/modules
   :caption: API Documentation
   :maxdepth: 3

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
