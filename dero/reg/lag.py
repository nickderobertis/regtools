from ..ext_pandas.filldata import add_missing_group_rows, drop_missing_group_rows, _to_list_if_str

def create_lagged_variables(df, lag_cols, id_col='TICKER', date_col='Date', num_lags=1):
    """
    Note: partially inplace
    """
    df.sort_values([id_col, date_col], inplace=True)
    df = add_missing_group_rows(df, [id_col, date_col])

    for col in lag_cols:
        _create_lagged_variable(df, col, id_col=id_col, num_lags=num_lags)

    df = drop_missing_group_rows(df, [id_col, date_col])

    return df

def _create_lagged_variable(df, col, id_col='TICKER', num_lags=1):
    """
    Note: inplace
    """
    new_name = varname_to_lagged_varname(col, num_lags=num_lags)
    df[new_name] = df.groupby(id_col)[col].shift(num_lags)

def varname_to_lagged_varname(varname, num_lags=1):
    return varname + f'_{{t - {num_lags}}}'

def _convert_variable_names(yvar, xvars, lag_cols, num_lags=1):
    if yvar in lag_cols:
        yvar = varname_to_lagged_varname(yvar, num_lags=num_lags)

    out_xvars = []
    for xvar in xvars:
        if xvar in lag_cols:
            out_xvars.append(varname_to_lagged_varname(xvar, num_lags=num_lags))
        else:
            out_xvars.append(xvar)

    return yvar, out_xvars

def _convert_interaction_tuples(interaction_tuples, lag_cols, num_lags=1):
    out_tuples = []
    for tup in interaction_tuples:
        out_tuples.append(
            tuple([
                varname_to_lagged_varname(var, num_lags=num_lags)
                if (var in lag_cols) or (var + ' Change' in lag_cols)
                else var
                for var in tup
            ])
        )
    return out_tuples

def _set_lag_variables(lag_variables, yvar, xvars):
    # Already passing a collection of columns, return
    if isinstance(lag_variables, (list, tuple)):
        return lag_variables

    assert isinstance(lag_variables, str)

    #Single str can either be a single column, 'all', or 'xvars'
    if lag_variables == 'xvars':
        lag_variables = xvars.copy()
    elif lag_variables == 'all':
        lag_variables = [yvar] + xvars
    else: #single column passed
        return _to_list_if_str(lag_variables)

    return lag_variables

def _is_special_lag_keyword(lag_variables):
    if isinstance(lag_variables, (list, tuple)):
        return False #list of columns

    special_keywords = ('xvars', 'all')
    return lag_variables in special_keywords