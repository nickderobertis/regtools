from typing import Sequence, Union, Optional, Tuple

import pandas as pd

from regtools.dataprep import _create_reg_df_y_x_and_dummies, _estimate_handling_robust_and_cluster
from regtools.models import get_model_class_by_string, _is_probit_str, _is_logit_str


def reg(df: pd.DataFrame, yvar: str, xvars: Sequence[str], robust: bool = True,
        cluster: Union[bool, str, Sequence[str]] = False, cons: bool = True, fe: Optional[Union[str, Sequence[str]]] = None,
        interaction_tuples: Optional[Union[Tuple[str, str], Sequence[Tuple[str, str]]]] = None,
        num_lags: int = 0, lag_variables: Union[str, Sequence[str]] = 'xvars', lag_period_var: str = 'Date',
        lag_id_var: str = 'TICKER', lag_fill_method: Optional[str] = 'ffill', reg_type: str = 'OLS'):
    """
    Returns a fitted regression. Takes df, produces a regression df with no missing among needed
    variables, and fits a regression model. If robust is specified, uses heteroskedasticity-
    robust standard errors. If cluster is specified, calculated clustered standard errors
    by the given variable.

    :Notes:

    Only specify at most one of robust and cluster.

    :param df:
    :param yvar: column name of outcome y variable
    :param xvars: column names of x variables for regression
    :param robust: set to True to use heterskedasticity-robust standard errors
    :param cluster: set to a column name to calculate standard errors within clusters
         given by unique values of given column name. set to multiple column names for multiway
         clustering following Cameron, Gelbach, and Miller (2011). NOTE: will get exponentially
         slower as more cluster variables are added
    :param cons: set to False to not include a constant in the regression
    :param fe: If a str or list of strs is passed, uses these categorical
        variables to construct dummies for fixed effects.
    :param interaction_tuples: tuple or list of tuples of column names to interact and include as xvars
    :param num_lags: Number of periods to lag variables. Setting to other than 0 will activate lags
    :param lag_variables: 'all', 'xvars', or list of strs of names of columns to lag for regressions.
    :param lag_period_var: only used if lag_variables is not None. name of column which
        contains period variable for lagging
    :param lag_id_var: only used if lag_variables is not None. name of column which
        contains identifier variable for lagging
    :param lag_fill_method: 'ffill' or 'bfill' for which method to use to fill in missing rows when
        creating lag variables. Set to None to not fill and have missing instead.
        See pandas.DataFrame.fillna for more details
    :param reg_type: 'OLS', 'probit', or 'logit' for type of model
    :return: statsmodels regression result.
    """
    regdf, y, X, dummy_cols_dict, lag_variables = _create_reg_df_y_x_and_dummies(df, yvar, xvars, cluster=cluster,
                                                                                 cons=cons, fe=fe,
                                                                                 interaction_tuples=interaction_tuples, num_lags=num_lags,
                                                                                 lag_variables=lag_variables, lag_period_var=lag_period_var,
                                                                                 lag_id_var=lag_id_var,
                                                                                 fill_method=lag_fill_method)

    ModelClass = get_model_class_by_string(reg_type)
    mod = ModelClass(y, X)

    if _is_probit_str(reg_type) or _is_logit_str(reg_type):
        fit_kwargs = dict(
            method='bfgs',
            maxiter=100
        )
    else:
        fit_kwargs = {}

    result = _estimate_handling_robust_and_cluster(regdf, mod, robust, cluster, **fit_kwargs)

    # Only return dummy_cols_dict when fe is active
    if fe is not None:
        result.dummy_cols_dict = dummy_cols_dict
    else:
        result.dummy_cols_dict = None

    if cluster:
        result.cluster_variables = cluster
    else:
        result.cluster_variables = None

    return result


