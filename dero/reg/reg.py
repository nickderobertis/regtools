from statsmodels import api as sm

from .tools import _to_list_if_str, _to_list_if_tuple
from .fe import fixed_effects_reg_df_and_cols_dict, extract_all_dummy_cols_from_dummy_cols_dict
from .interact import create_interaction_variables, delete_interaction_variables, _collect_variables_from_interaction_tuples
from dero.reg.models import get_model_class_by_string, _is_probit_str, _is_logit_str
from dero.reg.lag import create_lagged_variables, _convert_variable_names, _convert_interaction_tuples, \
    _set_lag_variables


def reg(df, yvar, xvars, robust=True, cluster=False, cons=True, fe=None, interaction_tuples=None,
        num_lags=0, lag_variables='xvars', lag_period_var='Date', lag_id_var='TICKER',
        model_type='OLS'):
    """
    Returns a fitted regression. Takes df, produces a regression df with no missing among needed
    variables, and fits a regression model. If robust is specified, uses heteroskedasticity-
    robust standard errors. If cluster is specified, calculated clustered standard errors
    by the given variable.

    Note: only specify at most one of robust and cluster.

    Required inputs:
    df: pandas dataframe containing regression data
    yvar: str, column name of outcome y variable
    xvars: list of strs, column names of x variables for regression

    Optional inputs:
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
    model_type: str, 'OLS', 'probit', or 'logit' for type of model

    Returns:
    If fe=None, returns statsmodels regression result
    if fe is not None, returns a tuple of (statsmodels regression result, dummy_cols_dict)
    """
    regdf, y, X, dummy_cols_dict, lag_variables = _create_reg_df_y_x_and_dummies(df, yvar, xvars, cluster=cluster,
                                                                  cons=cons, fe=fe,
                                                                  interaction_tuples=interaction_tuples, num_lags=num_lags,
                                                                  lag_variables=lag_variables, lag_period_var=lag_period_var,
                                                                  lag_id_var=lag_id_var)

    ModelClass = get_model_class_by_string(model_type)
    mod = ModelClass(y, X)

    if _is_probit_str(model_type) or _is_logit_str(model_type):
        fit_kwargs = dict(
            method='bfgs',
            maxiter=100
        )
    else:
        fit_kwargs = {}

    result = _estimate_handling_robust_and_cluster(regdf, mod, robust, cluster, **fit_kwargs)

    # Only return dummy_cols_dict when fe is active
    if fe is not None:
        return result, dummy_cols_dict
    else:
        return result

def _create_reg_df_y_x_and_dummies(df, yvar, xvars, cluster=False, cons=True, fe=None, interaction_tuples=None,
                                   num_lags=0, lag_variables='xvars', lag_period_var='Date', lag_id_var='TICKER'):
    # Handle lags
    if num_lags != 0:
        lag_variables = _set_lag_variables(lag_variables, yvar, xvars)
        df = create_lagged_variables(df, lag_variables, id_col=lag_id_var, date_col=lag_period_var, num_lags=num_lags)
        reg_yvar, reg_xvars = _convert_variable_names(yvar, xvars, lag_variables, num_lags=num_lags)
        if interaction_tuples is not None:
            interaction_tuples = _convert_interaction_tuples(interaction_tuples, lag_variables, num_lags=num_lags)
    else:
        reg_yvar, reg_xvars = yvar, xvars

    fe = _set_fe(fe)
    interaction_tuples = _set_interaction_tuples(interaction_tuples)
    regdf, y, X, dummy_cols_dict = _get_reg_df_y_x(df, reg_yvar, reg_xvars, cluster, cons, fe, interaction_tuples)
    return regdf, y, X, dummy_cols_dict, lag_variables

def _post_reg_cleanup(df, num_lags, lag_variables):
    # Cleanup lags
    if num_lags != 0:
        df.drop(lag_variables, axis=1, inplace=True)

def _estimate_handling_robust_and_cluster(regdf, model, robust, cluster, **fit_kwargs):
    assert not (robust and cluster)  # need to pick one of robust or cluster

    if robust:
        return model.fit(cov_type='HC1', **fit_kwargs)

    if cluster:
        groups = regdf[cluster].unique().tolist()
        group_ints = regdf[cluster].apply(lambda x: groups.index(x))
        return model.fit(cov_type='cluster', cov_kwds={'groups': group_ints}, **fit_kwargs)

        ##### TODO: implement this code for multiple clustering - need to ensure cluster is list
        # cluster_cols = ['State', 'Date']
        # reg_df = df.dropna()
        # groups = list(df[cluster_cols].drop_duplicates().itertuples(index=False))
        # group_ints = df[cluster_cols].apply(lambda x: groups.index(tuple(x)), axis=1)
        # result = sm.OLS(df['Return'], df['random']).fit(cov_type='cluster', cov_kwds={'groups': group_ints})
        #####

    return model.fit(**fit_kwargs)

def _get_reg_df_y_x(df, yvar, xvars, cluster, cons, fe, interaction_tuples):
    all_xvars = _collect_all_variables_from_xvars_and_interaction_tuples(xvars, interaction_tuples)
    regdf = _drop_missings_df(df, yvar, all_xvars, cluster, fe)
    y, X, dummy_cols_dict = _y_X_from_df(regdf, yvar, xvars, cons, fe, interaction_tuples)

    return regdf, y, X, dummy_cols_dict

def _drop_missings_df(df, yvar, xvars, cluster, fe):
    drop_set = [yvar] + xvars
    if cluster:
        drop_set += [cluster]
    if fe is not None:
        drop_set += fe

    return df.dropna(subset=drop_set)

def _y_X_from_df(regdf, yvar, xvars, cons, fe, interaction_tuples):

    if fe is not None:
        regdf, dummy_cols_dict = fixed_effects_reg_df_and_cols_dict(regdf, fe)
        model_xvars = xvars + extract_all_dummy_cols_from_dummy_cols_dict(dummy_cols_dict)
    else:
        dummy_cols_dict = None
        model_xvars = xvars.copy()

    if interaction_tuples:
        interaction_vars = create_interaction_variables(regdf, interaction_tuples)
        model_xvars += interaction_vars

    y = regdf[yvar]
    X = regdf.loc[:, model_xvars]

    if cons:
        X = sm.add_constant(X)

    return y, X, dummy_cols_dict

def _set_fe(fe):
    if fe is None:
        return None
    else:
        return _to_list_if_str(fe)

def _set_interaction_tuples(interaction_tuples):
    if interaction_tuples is None:
        return []
    else:
        return _to_list_if_tuple(interaction_tuples)

def _collect_all_variables_from_xvars_and_interaction_tuples(xvars, interaction_tuples):
    interaction_vars = _collect_variables_from_interaction_tuples(interaction_tuples)
    return list(set(xvars + interaction_vars))


def _is_normal_reg_str(reg_str):
    return reg_str in ('reg', 'normal', 'ols') or reg_str == None