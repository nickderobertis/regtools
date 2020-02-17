from typing import Sequence, Optional

import pandas as pd

from .reg import reg
from pd_utils.filldata import add_missing_group_rows
from regtools.lag.create import _is_special_lag_keyword


def diff_reg(df: pd.DataFrame, yvar: str, xvars: Sequence[str], id_col: str, date_col: str,
             difference_lag: int = 1, diff_cols: Optional[Sequence[str]] = None,
             diff_fill_method: str = 'ffill', diff_fill_limit: Optional[int] = None, **reg_kwargs):
    """
    Fits a differenced regression.

    :param df:
    :param yvar: column name of outcome y variable
    :param xvars: column names of x variables for regression
    :param id_col: column name of variable representing entities in the data
    :param date_col: column name of variable representing time in the data
    :param difference_lag: Number of lags to use for difference
    :param diff_cols: columns to take differences on
    :param diff_fill_method: pandas fill methods, 'ffill' or 'bfill'
    :param diff_fill_limit: maximum number of periods to fill missing data, default no limit
    :param reg_kwargs:
    :return:
    """

    if not isinstance(xvars, list):
        xvars = list(xvars)

    if diff_cols is None:
        # All by default
        diff_cols = [yvar] + xvars

    df = create_differenced_variables(df, diff_cols, id_col=id_col, date_col=date_col, difference_lag=difference_lag,
                                      fill_method=diff_fill_method, fill_limit=diff_fill_limit)

    # Convert names in lists of variables being passed to reg
    reg_yvar, reg_xvars = _convert_variable_names(yvar, xvars, diff_cols)
    this_reg_kwargs = reg_kwargs.copy()
    if 'interaction_tuples' in reg_kwargs:
        this_reg_kwargs['interaction_tuples'] = _convert_interaction_tuples(reg_kwargs['interaction_tuples'], diff_cols)
    if 'lag_variables' in reg_kwargs:
        this_reg_kwargs['lag_variables'] = _convert_list_of_variables_to_difference_names(reg_kwargs['lag_variables'], diff_cols)


    result = reg(df, reg_yvar, reg_xvars, **this_reg_kwargs)

    differenced_names = [col + ' Change' for col in diff_cols]
    df.drop(differenced_names, axis=1, inplace=True)

    return result



def create_differenced_variables(df, diff_cols, id_col='TICKER', date_col='Date', difference_lag=1,
                                 fill_method='ffill', fill_limit: int = None):
    """
    Note: partially inplace
    """
    df.sort_values([id_col, date_col], inplace=True)

    if fill_method is not None:
        # Save original byvars, for outputting df of same shape
        orig_index_df = df[[id_col, date_col]]

        # Fill in missing data
        df = add_missing_group_rows(df, [id_col], [date_col], fill_method=fill_method, fill_limit=fill_limit)

    for col in diff_cols:
        _create_differenced_variable(df, col, id_col=id_col, difference_lag=difference_lag)

    if fill_method is not None:
        df = orig_index_df.merge(df, how='left', on=[id_col, date_col])

    return df


def _create_differenced_variable(df, col, id_col='TICKER', difference_lag=1, keep_lag=False):
    """
    Note: inplace
    """
    df[col + '_lag'] = df.groupby(id_col)[col].shift(difference_lag)
    df[col + ' Change'] = df[col] - df[col + '_lag']

    if not keep_lag:
        df.drop(col + '_lag', axis=1, inplace=True)

def _convert_variable_names(yvar, xvars, diff_cols):
    if yvar in diff_cols:
        yvar = yvar + ' Change'

    out_xvars = _convert_list_of_variables_to_difference_names(xvars, diff_cols)

    return yvar, out_xvars

def _convert_list_of_variables_to_difference_names(varlist, diff_cols):

    # if 'all' or 'xvars' is passed, no conversion needed
    if _is_special_lag_keyword(varlist):
        return varlist

    out_vars = []
    for var in varlist:
        if var in diff_cols:
            out_vars.append(var + ' Change')
        else:
            out_vars.append(var)
    return out_vars

def _convert_interaction_tuples(interaction_tuples, diff_cols):
    out_tuples = []
    for tup in interaction_tuples:
        out_tuples.append(tuple([var + ' Change' if var in diff_cols else var for var in tup]))

    return out_tuples


def _is_diff_reg_str(reg_str):
    return reg_str in ('diff', 'difference', 'diff_reg', 'diff reg', 'difference reg', 'difference regression')