from typing import Union, Sequence, Optional, Tuple

import pandas as pd

from regtools.linmodels.bindings.modelstr import get_model_class_by_string
from regtools.linmodels.bindings.input import _create_reg_df_y_x_and_lag_vars
from regtools.linmodels.bindings.fit import _estimate_handling_robust_and_cluster
from regtools.linmodels.bindings.result import _convert_linearmodels_result_to_statsmodels_result_format
from regtools.linmodels.bindings.fe import dummy_cols_dict_from_model, linearmodels_fe_kwarg_dict_from_fe


def linear_reg(df: pd.DataFrame, yvar: str, xvars: Sequence[str], entity_var: str, time_var: str, robust: bool = True,
               cluster: Union[bool, str, Sequence[str]] = False, cons: bool = True, fe: Optional[Union[str, Sequence[str]]] = None,
               interaction_tuples: Optional[Union[Tuple[str, str], Sequence[Tuple[str, str]]]] = None,
               num_lags: int = 0, lag_variables: Union[str, Sequence[str]] = 'xvars', lag_period_var: str = 'Date',
               lag_id_var: str = 'TICKER', lag_fill_method: Optional[str] = 'ffill',
               lag_fill_limit: int = None,
               reg_type: str = 'fama macbeth', **fit_kwargs):
    """
    Runs a regression from the linearmodels library, standardizing the output to that of statsmodels

    :param df:
    :param yvar: column name of outcome y variable
    :param xvars: column names of x variables for regression
    :param entity_var: column name of variable representing entities in the data
    :param time_var: column name of variable representing time in the data
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
    :param lag_fill_limit: maximum number of periods to fill with lag_fill_method
    :param reg_type: 'fmb' for type of model
    :param fit_kwargs:
    :return: statsmodels regression result.
    """

    if entity_var is None or time_var is None:
        raise ValueError('must pass both entity_var and time_var')

    if not isinstance(xvars, list):
        xvars = list(xvars)

    regdf, y, X, lag_variables = _create_reg_df_y_x_and_lag_vars(
        df, yvar, xvars, entity_var, time_var,
        cluster=cluster,
        cons=cons, fe=fe,
        interaction_tuples=interaction_tuples,
        num_lags=num_lags,
        lag_variables=lag_variables,
        lag_period_var=lag_period_var,
        lag_id_var=lag_id_var,
        fill_method=lag_fill_method,
        fill_limit=lag_fill_limit
    )

    fe_kwargs = linearmodels_fe_kwarg_dict_from_fe(fe, regdf)

    ModelClass = get_model_class_by_string(reg_type)
    mod = ModelClass(y, X, **fe_kwargs)

    dummy_cols_dict = dummy_cols_dict_from_model(mod, regdf)

    result = _estimate_handling_robust_and_cluster(regdf, mod, robust, cluster, **fit_kwargs)

    _convert_linearmodels_result_to_statsmodels_result_format(result)

    result.dummy_cols_dict = dummy_cols_dict
    result.cluster_variables = cluster

    return result


