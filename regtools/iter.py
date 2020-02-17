import itertools
from typing import Sequence, Union, Optional, List

import pandas as pd

from regtools.order import _set_regressor_order
from .chooser import any_reg
from .select import select_models
from .summarize import produce_summary
from .lag import remove_lag_names_from_reg_results


def reg_for_each_combo(df: pd.DataFrame, yvar: str, xvars: Sequence[str], reg_type: str = 'reg', **reg_kwargs):
    """
    Takes each possible combination of xvars (starting from each var individually, then each pair
    of vars, etc. all the way up to all xvars), and regresses yvar on each set of xvars.
    .

    :param df:
    :param yvar: column name of y variable
    :param xvars: column names of x variables
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param reg_kwargs:
    :return: a list of fitted regressions
    """
    reg_list = []
    for i in range(1, len(xvars) + 1):
        for combo in itertools.combinations(xvars, i):
            x = list(combo)
            reg_list.append(any_reg(reg_type, df, yvar, x, **reg_kwargs))

    return reg_list


def reg_for_each_xvar_set(df: pd.DataFrame, yvar: str, xvars_list: Sequence[Sequence[str]],
                          reg_type: str = 'reg', **reg_kwargs):
    """
    Runs regressions on the same y variable for each set of x variables passed. xvars_list
    should be a list of lists, where each individual list is one set of x variables for one model.

    :Notes:

    If fe is passed, should either pass a string to use fe in all models, or a list of strings or
    None of same length as num models

    :param df:
    :param yvar: column name of y variable
    :param xvars_list: sequence where each element is itself a sequence containing column names of x variables
        for that numbered regression
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param reg_kwargs:
    :return: a list of fitted regressions
    """
    fe, interaction_tuples = _pop_and_convert_kwargs_which_are_repeated_across_models(reg_kwargs, len(xvars_list))
    return [any_reg(reg_type, df, yvar, x, fe=fe[i], interaction_tuples=interaction_tuples[i], **reg_kwargs) for i, x in enumerate(xvars_list)]


def reg_for_each_xvar_set_and_produce_summary(df: pd.DataFrame, yvar: str, xvars_list: Sequence[Sequence[str]],
                                              robust: bool = True,
                                              cluster: Union[bool, str, Sequence[str]] = False, stderr: bool = False,
                                              t_stats: bool = True,
                                              fe: Optional[Union[str, Sequence[Optional[str]]]] = None,
                                              float_format: str = '%0.2f',
                                              suppress_other_regressors: bool = False,
                                              regressor_order: Sequence[str] = tuple(), **other_reg_kwargs):
    """
    Convenience function to run regressions for every set of xvars passed
    and present them in a summary format.

    :Notes:

    * Only specify at most one of robust and cluster.
    * Don't set both stderr and t_stats to True

    :param df:
    :param yvar: column name of y variable
    :param xvars_list: sequence where each element is itself a sequence containing column names of x variables
        for that numbered regression
    :param robust: False to not use heteroskedasticity-robust standard errors
    :param cluster: set to a column name to calculate standard errors within clusters
             given by unique values of given column name, or multiple column names for multi-way clustering
    :param stderr: set to True to keep rows for standard errors below coefficient estimates
    :param t_stats: set to True to keep rows for standard errors below coefficient estimates and convert them to t-stats
    :param fe: If fe is passed, should either pass a string to use fe in all models, or a list of strings or
        None of same length as num models
    :param float_format: format string for how to format results in summary
    :param suppress_other_regressors: only used if regressor_order is passed. Then pass True to hide all
        coefficient rows besides those in regressor_order
    :param regressor_order: sequence of column names to put first in the regression results
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param other_reg_kwargs:
    :return: a tuple of (reg_list, summary) where reg_list
    is a list of fitted regression models, and summary is a single dataframe of results.
    """
    reg_list = reg_for_each_xvar_set(df, yvar, xvars_list, robust=robust, cluster=cluster, fe=fe, **other_reg_kwargs)
    regressor_order = _set_regressor_order(regressor_order, other_reg_kwargs)
    summ = produce_summary(reg_list, stderr=stderr, t_stats=t_stats, float_format=float_format,
                           regressor_order=regressor_order, suppress_other_regressors=suppress_other_regressors)
    return reg_list, summ

def reg_for_each_yvar_and_produce_summary(df: pd.DataFrame, yvars: Sequence[str], xvars: Sequence[str],
                                          regressor_order: Sequence[str] = tuple(), reg_type: str = 'reg',
                                          stderr: bool = False, t_stats: bool = True, float_format: str = '%0.2f',
                                          **reg_kwargs):
    """
    Convenience function to run regressions for multiple y variables with the same x variables and
    and present them in a summary format.

    :param df:
    :param yvars: column names of y variables
    :param xvars: column names of x variables
    :param regressor_order: sequence of column names to put first in the regression results
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param stderr: set to True to keep rows for standard errors below coefficient estimates
    :param t_stats: set to True to keep rows for standard errors below coefficient estimates and convert them to t-stats
    :param float_format: format string for how to format results in summary
    :param reg_kwargs:
    :return: a tuple of (reg_list, summary) where reg_list
    is a list of fitted regression models, and summary is a single dataframe of results.
    """
    reg_list = reg_for_each_yvar(df, yvars, xvars, reg_type=reg_type, **reg_kwargs)
    regressor_order = _set_regressor_order(regressor_order, reg_kwargs)
    summ = produce_summary(reg_list, stderr=stderr, t_stats=t_stats, float_format=float_format,
                           regressor_order=regressor_order, suppress_other_regressors=True)

    # Transposed is a better layout since all controls are suppressed and there may be many models
    summ.tables[0] = summ.tables[0].T

    return reg_list, summ



def reg_for_each_yvar(df, yvars, xvars, reg_type='reg', **reg_kwargs):
    """
    Convenience function to run regressions for multiple y variables with the same x variables

    :param df:
    :param yvars: column names of y variables
    :param xvars: column names of x variables
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param reg_kwargs:
    :return: a list of fitted regressions
    """
    return [any_reg(reg_type, df, yvar, xvars, **reg_kwargs) for yvar in yvars]



def reg_for_each_combo_select_and_produce_summary(df: pd.DataFrame, yvar: str, xvars: Sequence[str],
                                                  robust: bool = True, cluster: Union[bool, str, Sequence[str]] = False,
                                                  keepnum: int = 5, stderr: bool = False, t_stats: bool = True,
                                                  float_format: str = '%0.1f',
                                                  regressor_order: Sequence[str] = tuple(),
                                                  **other_reg_kwargs):
    """
    Convenience function to run regressions for every combination of xvars, select the best models,
    and present them in a summary format.

    :param df:
    :param yvar: column name of y variable
    :param xvars: column names of x variables
    :param robust: False to not use heteroskedasticity-robust standard errors
    :param cluster: set to a column name to calculate standard errors within clusters
             given by unique values of given column name, or multiple column names for multi-way clustering
    :param keepnum: number to keep for each amount of x variables. The total number of outputted
             regressions will be roughly keepnum * len(xvars)
    :param stderr: set to True to keep rows for standard errors below coefficient estimates
    :param t_stats: set to True to keep rows for standard errors below coefficient estimates and convert them to t-stats
    :param float_format: format string for how to format results in summary
    :param regressor_order: sequence of column names to put first in the regression results
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param other_reg_kwargs:
    :return: a tuple of (reg_list, summary) where reg_list
    is a list of fitted regression models, and summary is a single dataframe of results
    """
    reg_list = reg_for_each_combo(df, yvar, xvars, robust=robust, cluster=cluster, **other_reg_kwargs)
    regressor_order = _set_regressor_order(regressor_order, other_reg_kwargs)
    outlist = select_models(reg_list, keepnum, xvars)
    summ = produce_summary(outlist, stderr=stderr, t_stats=t_stats,
                           float_format=float_format, regressor_order=regressor_order)
    return outlist, summ

def reg_for_each_lag_and_produce_summary(df, yvar, xvars, regressor_order: Sequence[str] = tuple(),
                                         lag_tuple: Sequence[int] = (1, 2, 3, 4), consolidate_lags: bool = True,
                                         reg_type: str = 'reg',
                                         stderr: bool = False, t_stats: bool = True, float_format: str = '%0.2f',
                                         suppress_other_regressors: bool = False,
                                          **reg_kwargs):
    """
    Convenience function to run regressions with the same y and x variables for every passed number of lags
    and produce a summary.

    :param df:
    :param yvar: column name of y variable
    :param xvars: column names of x variables
    :param regressor_order: sequence of column names to put first in the regression results
    :param lag_tuple: sequence containing how many lags to create and run regressions for each variable
    :param consolidate_lags: True to condense lags for a single variable into a single row
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param stderr: set to True to keep rows for standard errors below coefficient estimates
    :param t_stats: set to True to keep rows for standard errors below coefficient estimates and convert them to t-stats
    :param float_format: format string for how to format results in summary
    :param suppress_other_regressors: only used if regressor_order is passed. Then pass True to hide all
        coefficient rows besides those in regressor_order
    :param reg_kwargs:
    :return: a tuple of (reg_list, summary) where reg_list
    is a list of fitted regression models, and summary is a single dataframe of results
    """
    reg_list = reg_for_each_lag(df, yvar, xvars, lag_tuple=lag_tuple, reg_type=reg_type, **reg_kwargs)

    # If we are consolidating lag coefficients, remove the lag names from the coefficient names
    if consolidate_lags:
        reg_list = remove_lag_names_from_reg_results(reg_list, lag_tuple)
    else:
        # Not consolidating lag coefficients. Therefore regressor order needs to be modified to include
        # renamed lag coefficients.
        # Add to kwargs so regressor order can be set properly
        reg_kwargs.update({'lag_tuple': lag_tuple})

    regressor_order = _set_regressor_order(regressor_order, reg_kwargs)

    model_names = [f'({lag})' for lag in lag_tuple]

    summ = produce_summary(reg_list, stderr=stderr, t_stats=t_stats, float_format=float_format,
                           regressor_order=regressor_order, suppress_other_regressors=suppress_other_regressors,
                           model_names=model_names)

    return reg_list, summ

def reg_for_each_lag(df: pd.DataFrame, yvar: str, xvars: Sequence[str],
                     lag_tuple: Sequence[int] = (1, 2, 3, 4), reg_type: str ='reg', **reg_kwargs):
    """
    Convenience function to run regressions with the same y and x variables for every passed number of lags

    :param df:
    :param yvar: column name of y variable
    :param xvars: column names of x variables
    :param lag_tuple: sequence containing how many lags to create and run regressions for each variable
    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param reg_kwargs:
    :return:
    """
    return [any_reg(reg_type, df, yvar, xvars, num_lags=lag, **reg_kwargs) for lag in lag_tuple]

def _pop_and_convert_kwargs_which_are_repeated_across_models(reg_kwargs, num_models):
    if 'fe' in reg_kwargs:
        fe = reg_kwargs.pop('fe')
    else:
        fe = None

    if 'interaction_tuples' in reg_kwargs:
        interaction_tuples = reg_kwargs.pop('interaction_tuples')
    else:
        interaction_tuples = [None]

    fe = _set_fe(fe, num_models)
    interaction_tuples = _set_interaction_tuples(interaction_tuples, num_models)

    return fe, interaction_tuples


def _set_fe(fe, num_models):
    return _set_for_multiple_models(fe, num_models, param_name='fixed effects')

def _set_interaction_tuples(interaction_tuples, num_models):
    return _set_for_multiple_models(interaction_tuples, num_models, param_name='interaction tuples')

def _set_for_multiple_models(param, num_models, param_name='fixed effects'):
    # Here we are being passed a list of strings or None matching the size of models.
    # This is the correct format so just output
    if (isinstance(param, list)) and (len(param) == num_models):
        out_param = param

    # Here we are being passed a single item or a list with a single item
    # Need to expand to cover all models
    else:
        if (not isinstance(param, list)):
            param = [param]
        if len(param) > 1:
            raise ValueError(
                f'Incorrect shape of items for {param_name} passed. Got {len(param)} items, was expecting {num_models}')
        out_param = [param[0]] * num_models

    # Final input checks
    assert isinstance(out_param, list)

    return out_param
