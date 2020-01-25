from typing import Optional
import pandas as pd
from regtools.tools import _to_list_if_str
from pd_utils.filldata import add_missing_group_rows
from regtools.lag.remove import (
    lag_varname_to_varname_and_lag,
    VariableIsNotLaggedVariableException,
)


def create_lagged_variables(
    df,
    lag_cols,
    id_col: Optional[str] = None,
    date_col="Date",
    num_lags=1,
    fill_method="ffill",
    fill_limit: int = None,
):
    """
    Note: partially inplace
    """
    # Handle panel data versus not by whether id_col was passed
    lag_kwargs = dict(num_lags=num_lags)
    if id_col is not None:
        lag_kwargs.update(id_col=id_col)
        lag_func = _create_lagged_variable_panel
        df.sort_values([id_col, date_col], inplace=True)

        # Save original byvars, for outputting df of same shape
        orig_index_df = df[[id_col, date_col]]

        df = add_missing_group_rows(
            df, [id_col], [date_col], fill_method=fill_method, fill_limit=fill_limit
        )
    else:
        lag_func = _create_lagged_variable  # type: ignore

    for col in lag_cols:
        lag_func(df, col, **lag_kwargs)

    if id_col is not None:
        # Don't want to expand size of df
        df = orig_index_df.merge(df, how="left", on=[id_col, date_col])

    return df


def _create_lagged_variable_panel(df, col, id_col="TICKER", num_lags=1):
    """
    Note: inplace
    """
    new_name = varname_to_lagged_varname(col, num_lags=num_lags)
    df[new_name] = df.groupby(id_col)[col].shift(num_lags)


def _create_lagged_variable(df: pd.DataFrame, col: str, num_lags: int = 1) -> None:
    """
    Note: inplace
    """
    new_name = varname_to_lagged_varname(col, num_lags=num_lags)
    df[new_name] = df[col].shift(num_lags)


def varname_to_lagged_varname(varname: str, num_lags: int = 1) -> str:
    if num_lags == 0:
        # No lag string necessary
        return varname

    try:
        base_var, existing_lags = lag_varname_to_varname_and_lag(varname)
    except VariableIsNotLaggedVariableException:
        # Variable is not already lagged, so just add lag portion to str
        return _varname_to_lagged_varname(varname, num_lags)

    # Variable was lagged originally, need to add an additional number of lags and apply to base name
    total_lags = existing_lags + num_lags
    return _varname_to_lagged_varname(base_var, total_lags)


def _varname_to_lagged_varname(varname: str, num_lags: int = 1) -> str:
    return varname + f"$_{{t - {num_lags}}}$"


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
            tuple(
                [
                    varname_to_lagged_varname(var, num_lags=num_lags)
                    if (var in lag_cols) or (var + " Change" in lag_cols)
                    else var
                    for var in tup
                ]
            )
        )
    return out_tuples


def _set_lag_variables(lag_variables, yvar, xvars):
    # Already passing a collection of columns, return
    if isinstance(lag_variables, (list, tuple)):
        return lag_variables

    assert isinstance(lag_variables, str)

    # Single str can either be a single column, 'all', or 'xvars'
    if lag_variables == "xvars":
        lag_variables = xvars.copy()
    elif lag_variables == "all":
        lag_variables = [yvar] + xvars
    else:  # single column passed
        return _to_list_if_str(lag_variables)

    return lag_variables


def _is_special_lag_keyword(lag_variables):
    if isinstance(lag_variables, (list, tuple)):
        return False  # list of columns

    special_keywords = ("xvars", "all")
    return lag_variables in special_keywords
