from statsmodels import api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dero.reg.reg import _create_reg_df_y_x_and_dummies, _estimate_handling_robust_and_cluster
from dero.reg.order import _set_regressor_order

def quantile_reg(df, yvar, xvars, q=0.5, robust=True, cluster=False, cons=True, fe=None, interaction_tuples=None,
        num_lags=0, lag_variables='xvars', lag_period_var='Date', lag_id_var='TICKER'):
    """
    Returns a fitted quantile regression. Takes df, produces a regression df with no missing among needed
    variables, and fits a regression model. If robust is specified, uses heteroskedasticity-
    robust standard errors. If cluster is specified, calculated clustered standard errors
    by the given variable.

    Note: only specify at most one of robust and cluster.

    Required inputs:
    df: pandas dataframe containing regression data
    yvar: str, column name of outcome y variable
    xvars: list of strs, column names of x variables for regression

    Optional inputs:
    q: float between 0 and 1. Quantile of dependent variable to estimate coefficients for
    robust: bool, set to True to use heterskedasticity-robust standard errors
    cluster: False or str, set to a column name to calculate standard errors within clusters
             given by unique values of given column name
    cons: bool, set to False to not include a constant in the regression
    fe: None or str or list of strs. If a str or list of strs is passed, uses these categorical
    variables to construct dummies for fixed effects.
    interaction_tuples: tuple or list of tuples of column names to interact and include as xvars
    num_lags: int, Number of periods to lag variables. Setting to other than 0 will activate lags
    lag_variables: 'all', 'xvars', or list of strs of names of columns to lag for regressions.
    lag_period_var: str, only used if lag_variables is not None. name of column which
                    contains period variable for lagging
    lag_id_var: str, only used if lag_variables is not None. name of column which
                    contains identifier variable for lagging

    Returns:
    If fe=None, returns statsmodels regression result
    if fe is not None, returns a tuple of (statsmodels regression result, dummy_cols_dict)
    """
    regdf, y, X, dummy_cols_dict, lag_variables = _create_reg_df_y_x_and_dummies(df, yvar, xvars, cluster=cluster, cons=cons, fe=fe,
                                                                  interaction_tuples=interaction_tuples, num_lags=num_lags,
                                                                  lag_variables=lag_variables, lag_period_var=lag_period_var,
                                                                  lag_id_var=lag_id_var)

    mod = sm.QuantReg(y, X)

    result = _estimate_handling_robust_and_cluster(regdf, mod, robust, cluster, q=q)

    # Only return dummy_cols_dict when fe is active
    if fe is not None:
        return result, dummy_cols_dict
    else:
        return result


def reg_for_each_quantile_output_plot(main_iv, *reg_args, num_quantiles=8, main_iv_label=None, outpath=None,
                                      clear_figure=True, **reg_kwargs):
    """
    Creates a plot of effect of main_iv on yvar at different quantiles. To be used after
    reg_for_each_quantile_produce_result_df

    :param main_iv: str, column name of independent variable of interest
    :param reg_args: see quantile_reg for args
    :param num_quantiles: number of quantile regressions to run. will be spaced evenly. higher numbers
                          produce smoother graphs but take longer to run
    :param main_iv_label: str, label of independent variable of interest
    :param outpath: str, filepath to output figure. must include matplotlib supported extension such as .pdf or .png
    :param clear_figure: bool, True wipe memory of matplotlib figure after running function
    :param reg_kwargs: see quantile_reg for kwargs
    :return: result_df from reg_for_each_quantile_produce_result_df
    """

    main_iv_label = _validate_quantile_reg_and_plot_args(main_iv, main_iv_label)

    result_df = reg_for_each_quantile_produce_result_df(
        main_iv,
        *reg_args,
        num_quantiles=num_quantiles,
        **reg_kwargs
    )

    quantile_plot_from_quantile_result_df(
        result_df,
        reg_args[1], #yvar
        main_iv_label,
        outpath=outpath,
        clear_figure=clear_figure
    )

    return result_df

def reg_for_each_quantile_produce_result_df(main_iv, *reg_args, num_quantiles=8, **reg_kwargs):
    """
    Produce result DataFrame of running multiple quantile regressions spaced out between the
    (0,1) interval

    :param main_iv: str, column name of independent variable of interest
    :param reg_args: see quantile_reg for args
    :param num_quantiles: number of quantile regressions to run. will be spaced evenly. higher numbers
                          produce smoother graphs but take longer to run
    :param reg_kwargs: see quantile_reg for kwargs
    :return:
    """
    quantiles = _create_quantiles_from_num_quantiles(num_quantiles)
    results = [(quantile_reg(*reg_args, q=q, **reg_kwargs), q) for q in quantiles]
    main_iv = _set_regressor_order([main_iv], reg_kwargs)[0]
    simple_results = [_produce_simplified_result_list(result[0], result[1], main_iv) for result in results]
    return pd.DataFrame(simple_results, columns=['q', 'a', 'b', 'lb', 'ub'])


def _produce_simplified_result_list(res, q, main_iv):

    # Handle possibility of dummy cols dict coming through with result as tuple
    if isinstance(res, tuple):
        res = res[0]

    return [q, res.params['const'], res.params[main_iv]] + \
           res.conf_int().ix[main_iv].tolist()


def _create_quantiles_from_num_quantiles(num_quantiles):
    quant_lower_bound = 1 / (num_quantiles + 1)
    quant_upper_bound = quant_lower_bound * num_quantiles + quant_lower_bound / 2
    return np.arange(quant_lower_bound, quant_upper_bound, quant_lower_bound)


def quantile_plot_from_quantile_result_df(result_df, yvar, main_iv, outpath=None, clear_figure=True):
    """
    Creates a plot of effect of main_iv on yvar at different quantiles. To be used after
    reg_for_each_quantile_produce_result_df

    :param result_df: pd.DataFrame, result from reg_for_each_quantile_produce_result_df
    :param yvar: str, label of dependent variable
    :param main_iv: str, label of independent variable of interest
    :param outpath: str, filepath to output figure. must include matplotlib supported extension such as .pdf or .png
    :param clear_figure: bool, True wipe memory of matplotlib figure after running function
    :return:
    """
    p1 = plt.plot(result_df.q, result_df.b, color='black', label='Quantile Regression Slope')
    p2 = plt.plot(result_df.q, result_df.ub, linestyle='dotted', color='black',
                  label='95% Confidence Interval Lower Bound')
    p3 = plt.plot(result_df.q, result_df.lb, linestyle='dotted', color='black',
                  label='95% Confidence Interval Upper Bound')
    plt.title(f'Effect of {main_iv} on {yvar}')
    plt.ylabel(r'Conditional Coefficient')
    plt.xlabel(f'Quantiles of {yvar}')
    plt.legend()

    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()

    if clear_figure:
        plt.clf()


def _is_quantile_reg_str(reg_str):
    return reg_str in ('quantile', 'quant', 'q', 'quantile reg', 'quantile_reg', 'quant_reg', 'quantreg', 'qreg', 'quantile regression')

def _validate_quantile_reg_and_plot_args(main_iv, main_iv_label):
    main_iv_label = _set_main_iv_label(main_iv, main_iv_label)

    return main_iv_label

def _set_main_iv_label(main_iv, main_iv_label):
    """
    Set default main iv label as main iv
    """
    if main_iv_label is None:
        return main_iv
    else:
        return main_iv_label
