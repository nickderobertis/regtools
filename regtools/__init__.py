"""
High-level tools for running regressions. Handles fixed effects, 2+ way clustering, hypothesis testing,
lagged variables, differenced variables, interaction effects, iteration tools, and producing summaries for a
variety of models including OLS, Logit, Probit, Quantile, and Fama-Macbeth.
"""
from regtools.iter import (
    reg_for_each_combo,
    reg_for_each_xvar_set,
    reg_for_each_combo_select_and_produce_summary,
    reg_for_each_xvar_set_and_produce_summary,
    reg_for_each_yvar,
    reg_for_each_yvar_and_produce_summary,
    reg_for_each_lag,
    reg_for_each_lag_and_produce_summary
)
from regtools.reg import reg
from regtools.linmodels.reg import linear_reg
from regtools.differenced import diff_reg
from regtools.quantile import quantile_reg
from regtools.summarize import produce_summary
from regtools.select import select_models
from regtools.hypothesis.lincom import hypothesis_test
