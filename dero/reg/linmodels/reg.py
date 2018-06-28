from dero.reg.models import get_model_class_by_string
from dero.reg.linmodels.bindings.input import _create_reg_df_y_x_and_lag_vars
from dero.reg.linmodels.bindings.fit import _estimate_handling_robust_and_cluster
from dero.reg.linmodels.bindings.result import _convert_linearmodels_result_to_statsmodels_result_format

def linear_reg(df, yvar, xvars, entity_var=None, time_var=None, robust=True, cluster=False, cons=True, fe=None, interaction_tuples=None,
        num_lags=0, lag_variables='xvars', lag_period_var='Date', lag_id_var='TICKER',
        model_type='fama macbeth', **fit_kwargs):
    """

    Args:
        df: pandas dataframe containing regression data
        yvar: str, column name of outcome y variable
        xvars: list of strs, column names of x variables for regression
        entity_var: str, name of variable identifying groups for panel
        time_var: str, name of variable identifying time for panel
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
        model_type: str, 'fama macbeth' for type of model

    Returns:

    """

    if entity_var is None or time_var is None:
        raise ValueError('must pass both entity_var and time_var')

    ### TEMP
    # import pdb
    # pdb.set_trace()
    ### END TEMP

    regdf, y, X, lag_variables = _create_reg_df_y_x_and_lag_vars(
        df, yvar, xvars, entity_var, time_var,
        cluster=cluster,
        cons=cons, fe=fe,
        interaction_tuples=interaction_tuples,
        num_lags=num_lags,
        lag_variables=lag_variables,
        lag_period_var=lag_period_var,
        lag_id_var=lag_id_var
    )

    ModelClass = get_model_class_by_string(model_type)
    mod = ModelClass(y, X)

    result = _estimate_handling_robust_and_cluster(regdf, mod, robust, cluster, **fit_kwargs)

    _convert_linearmodels_result_to_statsmodels_result_format(result)

    return result


