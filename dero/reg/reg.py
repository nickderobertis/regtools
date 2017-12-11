from statsmodels import api as sm


def reg(df, yvar, xvars, robust=True, cluster=False):
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
    """
    regdf, y, X = _get_reg_df_y_x(df, yvar, xvars, cluster)

    mod = sm.OLS(y, X)

    return _estimate_handling_robust_and_cluster(regdf, mod, robust, cluster)


def _estimate_handling_robust_and_cluster(regdf, model, robust, cluster):
    assert not (robust and cluster)  # need to pick one of robust or cluster

    if robust:
        return model.fit(cov_type='HC1')

    if cluster:
        groups = regdf[cluster].unique().tolist()
        group_ints = regdf[cluster].apply(lambda x: groups.index(x))
        return model.fit(cov_type='cluster', cov_kwds={'groups': group_ints})

    return model.fit()

def _get_reg_df_y_x(df, yvar, xvars, cluster):
    regdf = _drop_missings_df(df, yvar, xvars, cluster)
    y, X = _y_X_from_df(regdf, yvar, xvars)

    return regdf, y, X

def _drop_missings_df(df, yvar, xvars, cluster):
    drop_set = [yvar] + xvars
    if cluster:
        drop_set += [cluster]

    return df.dropna(subset=drop_set)

def _y_X_from_df(regdf, yvar, xvars):
    y = regdf[yvar]
    X = regdf.loc[:, xvars]
    X = sm.add_constant(X)

    return y, X